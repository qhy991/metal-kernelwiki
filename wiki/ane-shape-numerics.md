# ANE 形状与数值契约：validator 上限、布局与饱和边界

[English companion](en/ane-shape-numerics.md)

把一个算子送进 ANE 之前要决定两件事：形状能不能过 validator，以及数值在 fp16 数据通路加宽累加器上会怎样退化。本页把 [ANE 论文](https://arxiv.org/abs/2606.22283) ch17 的 validator 表与布局规则、ch3 的饱和/NaN/denormal/确定性边界、ch19 的宽度切片 ×16 饱和整理成构建前可执行的静态契约；所有数字带条件与 [M] 实测 / [D] 反编译 / [P] 预测标记，本页不发布本机测量。成本侧见 [roofline](ane-roofline-dispatch.md)，压缩权重的形式见 [weights](ane-weights.md)，宣称与可执行面的区别见 [reachability](ane-reachability.md)，编译与可观测边界见 [execution](ane-execution-measurement.md)。

## validator 表：先让形状能编译

| 约束 | 上限 | 超出后果 | 条件 / 标记 |
|---|---|---|---|
| 激活 dtype | 仅 fp16 | int32 cast 与 bf16 程序 I/O 在 M1 被拒 | [M]，表 17.1 |
| rank | ≤5 | rank 6 拒绝；Linear 输入 rank 须 ≤4 | [D]/[M] |
| W / H | 16384；A16/M5 起 65536 | 16385 拒绝（M1 实测的精确一步边界） | [M]，逐轴 |
| C | 65536 | 65537 拒绝 | [M]/[D] |
| conv 核宽 kW | 13（fp16 内核格式） | 14 拒绝；前端预筛 ≤15，16 在前端就拒 | [M] |
| conv 核高 kH | 29，且内核不得高于输入 | 高于输入拒绝 | [M] |
| arg-min/max 归约轴 | 2048 | 2049 拒绝，论文明写是为 fp16 精度 | [M] |
| matmul 收缩深度 | 两侧都为 1 | depth>1 拒绝 | [D] |
| matmul 输出通道 | 64 KB kernel-memory 预算 | 放不下就拒绝 | [D] |
| conv groups | 整除输入与输出通道 | 不整除拒绝；为速度还须整除核数（M1 为 4） | [D] |
| 最大单个活操作数 | 2 MB | 正确但分块、从 DRAM 流式，无报错 | [D]/[M] |

- 50 个 per-layer validator（`_ANECValidate<Op>Layer`）就是后端 legalizer；高级模型分段与手写层编译共用同一份代码，所以“接受”不会与真实编译结果漂移，但它只证明 schema 可达。
- validator 接受不等于能编译：M1 上 sort 与 dynamic-slice 的 validator 接受、codegen 拒绝，top-K 只在禁带之外可编译。真正的判据是 [reachability](ane-reachability.md) 里的 compile-and-run 协议。
- 三条“不拒绝但会变慢或变形”的规则：stride 不在原生集合时分解成大 stride 路径（仍编译）；dilation ≤8、padding ≤16 在 M1 上编译；pooling window 128 及以上接受，但在 2 的幂边界附近有 tile 对齐怪癖会拒绝特定奇数窗。
- depthwise 卷积（groups = 输入通道数）走 channelwise 路径，豁免 group-to-core 偏好；transpose 的 extent 受上限约束（A13–A15 为 16384，A16 起 65536）。

## 布局契约：interleave、granule 与 conv 形状

| 规则 | 要求 | 不满足的后果 |
|---|---|---|
| 张量序 | `[N, D, C, H, W]`，rank ≤5，变换只作用于最后四维 | rank 与转置约束另判 |
| 通道存储 | channel-interleaved；channel 是 interleave 因子的倍数（因子值原文未给） | 补齐并空转 lane：只是警告，不是拒绝 |
| 宽度对齐 | 对齐到 16 的 DMA granule（正文称 16-byte granule，§17.5 检查式按 `width mod 16`） | 补到 granule：警告 |
| conv 形状 | 4-D NCHW（rank 4 或 5，权重与 bias 折叠进算子） | 形状不符走另一条 legalization 分支 |
| linear | 改写成等价的 1×1 卷积才走卷积数据通路 | 保持 MatMul 形态则受 depth=1 等约束 |
| 通道/宽度实例 | `(1,1,64,56,56)` 用满 16-byte lane；`(1,1,64,56,1)` 只用一个 lane | 每个 dispatch 都白付 lane 成本 |

- 2 MB 判据比的是“最大的单个活操作数”，不是输入+权重+输出之和；把它当总和会做错分块。
- 编译器按 2 MB 字段判定（HAL `0x1b8`）；M1/H13 上实测的吞吐拐点在 2.28–2.34 MB（附录 E ch21）。两个口径不要混写成一个数。

## 数值契约：fp16 数据通路 + 更宽的累加器

| 量 | 规则 | 条件 |
|---|---|---|
| 算术宽度 | 激活、权重、输出、per-channel scale/bias、LUT 表项都是 fp16 | 没有 fp16 之外的算术 |
| 累加器 | 原文为 “wide accumulator of **fp32 class**”，物理上比 fp16 宽、每代固定硬件、从未给位宽 | 不要把它记作 fp32 精度的累加器 |
| 舍入点 | 两个：输入（含归约 tile）先舍到 fp16，输出再舍回 fp16 | 首级归约按 4-lane tile 先 fp16 舍入 |
| bf16 | 只是前端标注；11 个程序 I/O dtype 码里没有 bf16 | 框架先 cast 到 fp16 再进引擎段 |
| fp32 / int32 | 前端接受标注、后端不实现、不到硅；M1 上 cast 到 int32 直接拒绝 | 声明更宽 dtype 不换精度 |
| int4 / int8 | int4 没有算术 lane，只是 16 项 fp16 codebook 的调色板索引；int8 权重先重建为 fp16 再做 MAC | 压缩只减少搬运字节，不改 MAC 精度 |

## 饱和与边界：先记 32768，再记 65504

| 边界 | 行为 | 触发条件 | 标记 |
|---|---|---|---|
| fp16 上限 65504 | 超限静默变 +inf，无 fp32 余量 | 任何写出的 fp16 值 | [M] |
| MAC 输出口 2^15 = 32768 | 32752 通过、32768 变 inf；内部 partial 超限时即使后续抵消也溢出 | matmul、linear、任何累积 ≥2 tap 的卷积，与 tap 数无关 | [M] |
| 满量程路径 | 单元素乘在 65504 溢出；单输入通道 pointwise 1×1 允许 fill 60000 | 不经过 MAC 输出口 | [M] |
| reduce-sum | 先舍到 65504 再溢出 | 专用归约路径 | [M] |
| NaN | 输入边界强制成 +inf，引擎不再产生 NaN；relu→inf、sigmoid/tanh→1.0、erf→1.0、max→inf、softmax 把全部质量给该 lane | 例外：layer-norm 内部方差归约把 NaN 当大有限值 | [M] |
| 不定式 | inf−inf、0×inf、sqrt(−1)、rsqrt(−1)、log(−1) 都回 +0 | IEEE 会产生 NaN 的形式 | [M] |
| 输出舍入 | round-half-to-even；2049 在间距 2 的网格上舍到 2048 | M5 对同例返回 2050，是舍入差异还是累加器值不同未定 | [M] |
| denormal | 元素路径保留到 2^−24 ≈ 5.96e−8；MAC 内 M1 flush、M5 保留 | M2–M4 未测，改变代际未知 | [M] |

- 确定性按 lowered program 记：固定图与固定输入跨重跑、跨独立重编译、跨新进程逐位一致，batch 位置也不改变结果。
- 换数学才换位：`(a+b)+c` 对 `a+(b+c)` 约 31% 元素差约 1 个 fp16 ULP，最大绝对差约 7.8e−3，但每种顺序自身可复现——这是图结合顺序的 fp16 非结合性，不是硬件不确定。
- −0 在 reciprocal/rsqrt 之前丢符号：1/(−0) 得 +inf，而 IEEE 给 −inf；sqrt(−0) 仍是 +0。

## M1 宽度切片的 ×16 饱和

- 机制：width 轴上 begin offset 非零的 slice 走 Q.4 定点、固定增益 ×16 的 crop DMA；这是 offset-DMA 的存储格式，不是可借用的通用数值模式。
- 阈值：|v| ≤4094 逐位通过（4094×16 = 65504）；≥4095 在 fp16 网格上舍到 4096，4096×16 = 65536 > 65504，变 +inf。
- 轴与代际：只发生在 width 轴；height/channel/batch 的非零偏移不触发。M1(H13) 与 A14(M2) 上实测饱和；论文把干净路由定在 A15 及以后（A15 是逐家族 lowering 的反编译外推、未在 A15 硅上直接实测），M5 上确认干净。
- 使用约束：必须带非零 last-axis offset 的 slice 把幅值压在 4094 以下；零 offset 完全避开该路径；训练侧 loss-scale 也按此 cap。

## 构建前检查清单

1. rank ≤5；Linear 输入 rank ≤4。
2. dtype：激活 fp16；int32 cast 与 bf16 I/O 先排除。
3. 逐轴 extent：W/H ≤16384（A16/M5 起 65536）、C ≤65536。
4. conv：kW ≤13（前端预筛 15）、kH ≤29 且不高于输入；groups 整除 in/out，为速度整除 4 核。
5. arg-min/max 归约轴 ≤2048。
6. matmul：depth = 1，输出通道放得进 64 KB kernel memory。
7. 最大单个活操作数 ≤2 MB，否则分块流式；不要把 2 MB 当总和。
8. channel 为 interleave 倍数、width 为 granule 倍数（补 padding 只是警告）。
9. 数值：matmul/多 tap conv 的输出量级 <2^15；width slice 零偏移且幅值 ≤4094；cancellation 重的归约交给 CPU/GPU。
10. 最后一步仍是目标芯片上的 compile-and-run，validator 通过不算。

## 采用条件

- 可采用：把本页 validator 表与布局规则做成构建前静态检查；把饱和、NaN、denormal 规则写进 bit-exact oracle；把确定性记在“同一 lowered program”名下。
- 不能推出：M1 的 16384 / 4094 / denormal flush 行为不能直接推到 M5；声明为 fp32/bf16 不会换来精度；跨图的 bitwise 一致不能承诺（换结合顺序约 31% 元素变约 1 ULP）；M5 的 round-half tie 未定。
- 冲突/未定：denormal 的改变代际未知（M2–M4 未测）；2 MB 字段与 2.28–2.34 MB 实测拐点是两个口径；宽度切片干净代际只有 M5 是实测确认，A15 起是反编译外推，发布前应在目标芯片验证。
- 发布前记录：芯片家族、ANECompiler 版本、macOS 版本与具体 shape。

来源 `ane-paper-2606-22283-v2`，arXiv:2606.22283v2 §3.2、§3.5–3.8、§17.1–17.8、§19.1、§19.3、§20.3、附录 A.1 与 B.3；预印本 303 页。
