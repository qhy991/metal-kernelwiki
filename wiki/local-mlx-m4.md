# M4 本地观测：attention、量化 matmul 与 capture

[English companion](en/local-mlx-m4.md)

证据状态：`locally-measured`，2026-10-07 的一次小型探索。模型未加载，Compiler/Lab 未改动；这不是端到端 LLM 验收、正式实验或系统资格认证。只采用下列域内结论。

## 环境与原始记录

Apple M4、16GB、`applegpu_g16g`，macOS 27.0（26A428）；Python 3.14.3、MLX 0.31.2、mlx-lm 0.31.3、NumPy 2.4.3。已有环境原样使用。所选 CommandLineTools 的 `xcrun --find metal` 和 `xcrun --find xctrace` 均退出 72；这些失败被保留，没有安装、切换或修复工具链。

原始目录（仓库外）：

```text
2026-10-07-mlx-m4-r1  （逻辑运行 ID；原记录在仓库外，未公开）
  validation/probe.py       实际执行的源文件快照
  validation/events.jsonl  逐项原始记录
  validation/results.json  完整结构化结果
  validation/stdout.jsonl  原始标准输出
  validation/stderr.log    原始告警
  capture/                 单独的捕获进程记录与源文件
  capture/attention.gputrace
  profiler-template-check.json  后续工具能力检查的失败记录
```

目录由 `local-mlx-m4-r1` 与 `local-mlx-m4-capture-r1` 两个来源条目的 `artifact_ref` 识别；持有者另存真实路径映射，`get_page.py ... --follow-sources` 只显示元数据。原始结果、脚本和 trace 未随公开仓库提供，外部读者不能据本页独立复验数值。文件保存不等于已建立连续 custody；本页是带范围的历史观察。

## Attention：16 项外部 oracle 检查通过

固定 seed=1729，`B=1,Hq=8,Hkv=2,D=64`；Q/K/V 为标准差 0.3 的正态输入，先转成实际 GPU dtype 再构造 CPU NumPy float64 oracle。四组 `(Tq,Tk)` 为 `(1,129),(17,129),(33,65),(128,128)`；下右对齐 causal mask，scale=1/8。

比较调用者直接传 compact K/V 与每次调用显式 repeat K/V 的两种 `fast.scaled_dot_product_attention` 调用。float32 的预定 `atol=3e-5,rtol=3e-4`，float16 为 `2e-3,2e-2`。共 16 项全部通过；最大绝对误差上界分别为 `8.021e-8` 和 `2.416e-4`。它不覆盖其他 head/dim、任意 mask、大幅极值、长上下文或模型质量。两个路径的误差汇总相同不等于验证了输出逐位相同。

两路径先过数值门，再各预热 3 次、交替 AB/BA 测 11 次；每次新建操作并 `mx.eval` 输出，开始前同步默认 GPU stream。下表单位 ms，为单进程内样本的 **median [min,max]**，不是多个独立 clean-start 的统计。

| dtype 与序列形状 | compact 调用 | 显式 repeat 后调用 |
|---|---|---|
| float32 tq=1 tk=129 | 0.175917 [0.141625, 0.374750] | 0.211041 [0.177334, 0.420375] |
| float32 tq=17 tk=129 | 0.225875 [0.199875, 0.404000] | 0.254083 [0.210208, 0.523083] |
| float32 tq=33 tk=65 | 0.199208 [0.167208, 0.397583] | 0.207584 [0.200375, 0.404208] |
| float32 tq=128 tk=128 | 0.229291 [0.203208, 0.637083] | 0.224791 [0.205250, 0.431916] |
| float16 tq=1 tk=129 | 0.171375 [0.137500, 0.368167] | 0.200208 [0.189541, 0.408041] |
| float16 tq=17 tk=129 | 0.199250 [0.190084, 0.399042] | 0.209167 [0.196000, 0.403209] |
| float16 tq=33 tk=65 | 0.180584 [0.176167, 0.388875] | 0.190750 [0.185416, 0.388709] |
| float16 tq=128 tk=128 | 0.221084 [0.197041, 0.452916] | 0.227750 [0.214375, 0.417917] |

测的是主机 `perf_counter_ns` 的建图→提交→完成等待，含 repeat；不是纯 GPU 时间。热输入/allocator，未清硬件 cache、未控制热状态/功率；只持有协作式 GPU 锁，其他应用仍可运行。样本出现较快和较慢两簇，区间明显重叠；fp32 128×128 中 compact 中位数还略高。因此不发布稳定加速倍率或模型 tok/s 推断。compact 仅描述 API 输入没有显式 repeat，未证明后端无内部复制或走特定 fused kernel。

## QMM：同一输入行会随 M 变化

每种 dtype 固定一份 `X:[128,2560]` 与 `W:[1024,2560]`，W 标准差 0.02；一次 affine 4-bit/group64 量化，测试 `M={16,32,33,64,65,128}`。CPU 独立解包权重，分别用理想 affine 解码值与解码后舍入到输入 dtype 的值做 float64 matmul 参考。18 项输出均有限，CPU 解码再舍入与本地 `mx.dequantize` 完全一致。

三个 dtype 中，前 16 行在 M16/M32 下逐位相同，在 M33/64/65/128 下均不逐位相同。与 M16 相比，所测比较中的最大绝对差为：

| dtype | 最大绝对差 |
|---|---:|
| float32 | 2.503395e-6 |
| float16 | 0.005859375 |
| bfloat16 | 0.046875 |

这只是 batch 数值敏感性的观测，没有预设模型质量门，不能判为 kernel 缺陷或断言 Split-K 根因。各 dtype 使用不同随机输入，不是严格匹配的 dtype 精度比赛。原始记录中的误差/最终舍入比值不是 ULP 或合法误差阈值；BF16 参考是 float64→float32→BF16 两阶段舍入。这里未衡量原始权重的量化损失。候选设计见 [量化 matmul 验证](quantized-matmul-validation.md)。

## 本地 API 与 capture 的覆盖域

- 实际调用 `RotatingKVCache(max_size=16,keep=4).to_quantized(...)` 得到 `NotImplementedError: RotatingKVCache Quantization NYI`。
- 合成模型自有 `make_cache` 时，`make_prompt_cache(...,max_kv_size=16)` 返回它提供的 `KVCache`；不据此声称任意模型有容量上限。
- `generate_step` 的本地 `quantized_kv_start` 默认值为 0；实时 main 文档/实现可能不同。
- 本地 SDPA docstring 签名没有 `force_fused`。未调用该参数测试拒绝，因此只把它作为版本兼容检查，不能直接照搬新版参数。
- 独立进程使用 `MTL_CAPTURE_ENABLED=1`，捕获 fp16/B1/Hq8/Hkv2/Tq128/Tk128/D64 的 3 次调用，成功导出 trace，`exists=true,parsed=false`。它使用新生成输入，不是验证进程的同一张量；未解析 timeline/counter/dispatch，也没有该进程的数值或性能结论。
- 后续直接执行已有 Xcode 的 `xctrace list templates`，在默认 sandbox 下退出 137、无输出；原因未确定。已停止这条能力检查，没有改环境重试。不能据此宣称 Metal System Trace 不存在或 replay 永远不可用。
- MLX API 报告整次 validation 的 peak=24,166,400 bytes、结束时 active=7,880,712、cache=40,395,776；不代表进程/系统总内存，不是每个变体的内存差，也不能相加当峰值。

## 如何继续

持有原始记录时，保留目录不覆写，使用现有 MLX 环境执行其 `probe.py --out <全新仓库外目录>`；capture 必须另启进程，脚本不会安装依赖或修复环境。公开仓库没有该脚本；外部读者可按上述条件另建探针，但须记录为新实验而非原运行的重放。先确认目标和任务适用，逐项记录失败与覆盖限制，不只看退出码。

下一步优先提高计时可解释性、增加输入分布和 dtype/shape 边界，并解析独立 profiler 证据；只有目标模型和工作负载的端到端质量及性能通过，才能声称部署优化收益。
