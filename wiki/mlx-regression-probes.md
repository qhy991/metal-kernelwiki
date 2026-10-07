# MLX 边界探针：异步依赖、singleton GQA、多行 qvm 与缓存生命周期

检查日期：2026-10-07。**四项均已有范围限定的 M4 观测；第 3 项数值门失败，第 4 项发现策略损失；另测 keep=0 mask 有限定失败，服务仍未测**。保留原始设计与后续观测的区别。执行新配置前固定芯片、OS、安装版本、seed、dtype、shape、容差与求值范围，按项目授权和 gate 运行，失败保留原始记录。

本地 r1 只检查了同步 SDPA 的 `D=64`、四组非 singleton 长度，以及另一组 `transpose=True/M>=16` 量化乘法和缓存 API 行为；其结果见 [M4 本地观测](local-mlx-m4.md)，相关方法见 [量化乘法验证](quantized-matmul-validation.md)。r1 结果不覆盖下面四项，独立 capture 导出也没有解析执行路径。后续运行 `2026-10-07-mlx-boundaries` 单独保留了第 1、2 项结果，没有改变 r1 的证据范围。

## 1. async_eval：提交完成与数据依赖分开测（已做限定验证）

三个进程完成以下小输入验证，并增加同一依赖链仅尾部 eval 的 D 路径。四种路径均通过预设 CPU oracle；计时为主机完成区间，存在明显波动，详见 [求值边界观测](mlx-async-evaluation.md)。未测真实生成循环、多流或 GPU 内核时间。

[官方 API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html) 仍标为 experimental。[上游 #4265](https://github.com/ml-explore/mlx/issues/4265) 使用独立调用统一求值，说明需要区分计时范围；其 M3 Ultra 观察不是依赖链加速证据。

小输入：CPU seed=13，float32 `X:[1,256]`、`W:[256,256]=I+normal/64`，`X=0.1*normal`，`f(x)=tanh(x@W)`，32 步。输入和预热在计时外完成，每次重复重新从 X 建图，固定同一 stream。

- A：连续 `x=f(x); mx.eval(x)`，测串行依赖链及每步等待。
- B：预先准备并求值 32 个不同的 `X_i=X+i/1024`，独立计算 `f(X_i)`，最后一次 `mx.eval(outputs)`；它测独立工作完成时间。
- C：连续 `x=f(x); mx.async_eval(x)`，最后 `mx.eval(x)`。分别保存尾部 eval 前区间和包含尾部等待的完成区间；前者也可能含内部等待。
- D：连续构建 `x=f(x)` 的依赖链，最后一次 `mx.eval(x)`。

判据：从实际 float32 输入建立 CPU float64 参考，A/C/D 对照 32 步，B 各自对照一步；预设 `atol=1e-5, rtol=1e-4` 并保留最大误差与非有限值。先确认正确性，再报告重复分布。不能复用已求值输出计时，也不能把 B 的耗时除以 32 称为逐 token decode 延迟。

## 2. singleton GQA：一个 key 的输出应等于 V（已做限定验证）

在原设计上增加 float16、三类 V、四种调用对照，共 48 个不同配置；三个进程每次均精确等于实际存储 V。详见 [Float32 精度边界](mlx-float32-precision.md)。相同输入的重复不是新增测试条件，M4 通过不证明 M5 历史 issue 已修复。

[历史报告 #3953](https://github.com/ml-explore/mlx/issues/3953) 描述 MLX 0.32.0/M5 Max 的 broadcast 精度现象；状态为 Closed，尚未核实修复 commit，不宣称当前版本仍有故障。

原创输入：float32，零 Q `[1,16,1,512]`、零 K `[1,1,1,512]`；CPU 构造 `V=(arange(512)/10000+0.125).astype(float32)` 并保留原值，reshape 为 `[1,1,1,512]`。调用 SDPA，`scale=1, mask="causal"`。单个 score 的 softmax 为 1，因此对每个 query head 检查输出与 V 精确相等，并报告最大误差、非有限值。

再做 `B=2` 控制组及 `ones:[1,16,1,1] @ V` 定位。显式物化 broadcast 仅作诊断对照；非零误差应追踪安装版本的实际路线，不能用 argmax 不变替代该关系检查，也不能由调用方未 repeat 推断后端没有复制。r1 的 `Tq=1,Tk=129,D=64` 不是此 singleton case。

## 3. qvm 多行 stride：复用历史回归范围（已执行，数值门失败）

后续运行 `2026-10-07-mlx-qvm-cache` 用下面 16 个形状/格式组合和两组独立 CPU 种子测试，共 32 个用例；全部未通过预设的正确 CPU 参考和 dense 对照。逐行误差及事后错误跨度诊断见 [量化乘法验证](quantized-matmul-validation.md)。未修改包或容差，失败不能进入性能验收。

[修复 commit 1ea24e1](https://github.com/ml-explore/mlx/pull/3497/commits/1ea24e11f068af5949cda98e5d3eb0ca5f86ea68) 所属 PR #3497 于 2026-05-11 合入；[固定测试 commit 2ecf184](https://github.com/ml-explore/mlx/pull/3497/commits/2ecf184f9150b85b0aa139f7d827751011874d35) 覆盖多行输入。合入日期不能证明安装包包含修复。

原设计沿用上游测试范围（本地实际运行改用已记录的 CPU PCG64 输入，并非原输入重放）：float32 `X:[M,K]`、`W:[K,128]`，分别用独立随机 key 生成 `0.1*normal`；`M={2,3}, K={2048,4096}, bits={4,8}, group_size={32,64}, transpose=False`。保存每格 seed 和 packed W。上游用同一 packed W 的 dequantized dense 输出对照，要求 `max_abs<2e-3`；新增独立 CPU float64 对照时，先按实际 dtype 舍入解码权重并单独标记参考类型。

逐行报告误差，避免总均值隐藏第二、三行错误。此检查不覆盖 5D GQA broadcast、任意 tail 或 speculative decoding 的模型质量；r1 的转置方向和 M 范围不同，不能代替它。

## 4. RotatingKVCache：keep 策略跨 batching 保存（存储路径与另组 mask 已执行，服务未测）

同轮直接 merge/update/extract 接受了 keep=(4,4) 与混合 (0,4)，随后丢失 keep=4 的策略和前缀；(0,0) 对照通过。详见 [缓存生命周期](mlx-cache-lifecycle.md)。未执行 attention/mask、服务或模型质量验证，因此原设计的这些判据仍未满足；没有将 keep 改为 0 后重报通过。

后续 `2026-10-07-mlx-cache-mask` 单独检查 keep=0、不同长度及左填充，经逐 token 旋转后切回多 token 追加，90 项中 4 项 mask/attention 失败，内容与 offset 检查全部通过。详见 [mask 对齐](mlx-cache-mask-alignment.md)。它补充了另一策略的 mask 范围，尚未完成本节原 keep=4 设计的 attention 或服务验收。

[上游 #1631](https://github.com/ml-explore/mlx-lm/issues/1631) 报告外部 supplied cache 经过 merge/extract 后可能丢失 sink policy；本机现已有直接存储路径的限定复现，仍不能推广到每条服务路径。

原设计构造两条 `max_size=8, keep=4` 缓存，K/V 为 `[1,1,T,64]` float32，以 token-id/16 辨识行，先插入 6 行。本轮实际使用请求偏移和逐通道编码，具体公式见生命周期页；其余这里列出的存储范围不变。两条请求使用不同 ID 范围，另存 CPU token-id 参考。通过安装版本实际支持的 merge 路径入 batch，逐 token 追加 10 行，跨越容量后 extract；另做 `keep=0` 对照和 `keep={0,4}` 混合策略负例。

判据：支持该路径时，extract 后仍应 `keep==4`，前四个 sink 行保持原值，有效 tail、offset 与 mask 符合旋转策略；检查有效时间顺序，不把环形存储的物理顺序误当语义顺序。混合策略须明确拒绝或拆开，不能静默统一。若 API 拒绝 keep>0，记录 unsupported，不修改 keep 后报告通过。r1 只观察到空 rotating cache 量化拒绝，未验证这里的生命周期；将 keep 改为 0 还需独立长上下文质量评估。
