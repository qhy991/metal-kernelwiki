# ANE 能力面：宣称、可编译与真正可执行

[English companion](en/ane-reachability.md)

给“这个算子能不能跑在 ANE 上”下结论时，能力字节、前端识别、validator 通过都不构成证据；论文的判据是目标芯片上的一次编译并执行。本页按 [ANE 论文](https://arxiv.org/abs/2606.22283) §4.4、附录 A 与附录 E（ch 8、ch 24）整理 187 行操作普查、五个状态标记、分家族地板、宣称但从未 lowering 的集合，以及能编译的操作数信封。数字保留论文的 [M] 实测 / [D] 反编译 / [P] 预测标记；本页不发布本机测量，也不覆盖成本与放置取舍（见 [roofline](ane-roofline-dispatch.md)、[placement](ane-placement.md)）。

## 五个状态标记

论文图例只写了四个标记，表内另有 Compiler-internal 单元格；判读时按五个值处理，其中只有 Native 可以按引擎吞吐做预算。

| 标记 | 操作定义 | 决策后果 |
|---|---|---|
| Native | 在该家族上编译并运行于直连引擎路径 | 可以按引擎吞吐与能耗做预算 |
| Family-gated | 在该家族无路径，从注释指名的家族起 native | 该芯片上按 off-engine 处理 |
| Bridge | 只能经分解、软件回退或 compiler-internal 路由可达，从不独立 codegen | 不得当成独立引擎 kernel 计成本 |
| No path | M1–M5 全家族拒绝，off-engine 计算 | 重写或落主机 |
| Compiler-internal | 已映射、未见独立代码生成，只在包装构造内可达 | 视为包装构造的一部分，不单独承诺 |

## 187 行普查：印刷桶与逐行普查差 5

| 口径 | Native | →M2 | →M3 | Bridge | No-path | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| 论文印刷桶（p.248） | ≈108 | 9 | 4 | — | 37 | 182 |
| 逐行普查（对印刷单元格做算术，工程推断） | 113（含 5 行私有 `ne_`/`pe_` 单元选择，净 108） | 11 | 6 | 22 | 35 | 187 |

- 印刷桶加总 182 与“187 个 IL 操作”差 5；印刷的“37 个全家族拒绝”比逐行 35 个 No-path 多 2。两个口径都保留，不合并成一个数。
- 缺口集中在数据搬运（A.8 的 38 行里 9 行 No-path）与检测/采样（A.14 的 13 行里 9 行 No-path）；注意力/控制/状态（A.11 的 9 行里 7 行 Bridge）说明“状态类”多数只能靠包装构造到达。
- 每格状态只对列出的家族成立：M1/M2/M5 列是物理硅上实测，M3 列与合并的 M4/M5 列中的 M4 半边是反编译外推，逐格状态本身是预测。

## 分家族地板：哪一代解锁什么

| 家族 | 操作面地板 / 解锁 | 证据 |
|---|---|---|
| M1（H13/A13） | ≈108 个 native 操作；无 texture engine；无 sin/cos；IL 路由的 argmin/global-argmin 被门控；int4-LUT 与 sparse 权重可流式；int8 与 blockwise-affine 折叠进描述符 | [M] 附录 A、D.1 |
| M2（H14/A14） | texture engine 到来：crop-resize、resample、affine、硬件 gather；rank/sort bridge（top-k、sort、dynamic-slice）；5 个 resize/upsample 转 native；symmetric/reflect pad native | [M] 附录 A、A.1 |
| M3（H15/A15） | sin/cos native；硬件随机数；IL 路由的整张量 arg 归约；reduction-to-transpose 阈值 192→384；blockwise 与 sparse-blockwise shift-scale 可流式 | [P]（M3 整列是反编译外推） |
| M4/M5（A16/A17） | 同操作集合便携；张量/转置 extent 16384→65536；blockwise 流式 | [M]（M5）、[P]（M4） |
| 更高层（H17d 64 核、H18 fp8、Ultra mesh） | 只在 A16 之上增加核心，不新增操作；例外是少量 N=5/6/7 模板实例（含 fp8），在 H18 以下保持惰性 | [D] |

- texture engine 有无是“同一算子换实现”的分水岭：M1 的 0x81d 为 0，resize 走软件 transpose 回退且舍入不同；不是同一结果只是更慢。
- A15 的 sin/cos 与 blockwise 流式只影响特定图层，不改变 conv/matmul 的 native 地板。
- A16/M5 把 W/H 与转置 extent 抬到 65536；把小代际的 16384 上限套到 M5 会把可编译形状判错。

## 路由决定答案：attention 与 arg 归约两个反例

不写明路由的“M1 支持/不支持”没有意义：conversion（框架/转换）路由与 direct-authoring（直写 netplist）路由用两套不同的门。

| 门 | conversion 路由 | direct-authoring 路由 |
|---|---|---|
| 机制 | 每个 backend 操作带 `MinimumFamily<N>` 下限，共四层 F0–F4，没有高于 F4 的下限 | 读逐芯片 HAL feature byte（`hal[off] & 1`） |
| 家族映射 | F0 = 全家族、F2 = A13+、F3 = A14+、F4 = A15+ | 按目标芯片读字节，同一芯片可与众不同 |
| 代表 | softmax、layernorm、归约、resize、SDPA 在 F2；sin/cos 与 global argmin/argmax 在 F4 | 字节决定同一条 op 在直写路由上是否 native |

- attention：附录 A 把 fused SDPA 记为 M1 native，但注释写明它“跑在 matmul + softmax 路径上、不受 texture engine 门控”；可达的 M1 形态是 matmul + scale + softmax + matmul 的组合。compiler 中确实存在单条 fused SDPA descriptor（4 或 5 个 operand；parser 唯一读的 key 是 `SubtractMax`，默认 false，必须置 true 数值才正确），但高阶转换路径从不发射它；只有直写 netplist 走 tunneled-unit 才能把它送进硅。这是研究用途，不是可交付形态。
- arg 归约：whole-tensor argmin/argmax 在 conversion 路由被 F4 卡住、在 M1 上被拒；而 direct-authoring 的字节 `0x4f2` 从 A13 起就置位，同一颗 M1 上能跑。同一芯片、相反答案，差别只在路由。
- 路由还决定门能否被绕过：sin/cos 没有 direct-authoring bridge，只能走 conversion，M1 上必然被拒；sort/top-K 有 bridge，但 M1 的 code generator 拒绝它。
- 写法约定：任何“M1 不支持”的结论都要带路由与芯片；带不出这两项，就改写成“在 X 路由、Y 芯片上拒绝”。

## 宣称但从未 lowering 的集合

这组条目都在某一层被证实，却在执行层失败；它们是本页的核心反例。

| 项目 | 在哪一层被证实 | 为什么跑不起来 | 决策 |
|---|---|---|---|
| 3-D 卷积 | HAL 能力字节与内核深度 attest（`0x70`，A13 起为 16） | 所有 device mask 上 backend lowering 失败 | 放 CPU/GPU，别进引擎图 |
| 原生 state / ring buffer | compiler 保留操作类、程序描述符有字段 | M1 描述符里 counter-and-event DMA 引擎被 stub，编译失败 | 用共享常驻 buffer 做 KV（见 [execution](ane-execution-measurement.md)） |
| bf16 程序输入/输出 | 文档化的能力 | 11 个 `ANECIRDataType` 码里没有 bf16，编译报 unsupported output dtype | 框架先 cast 到 fp16 再进引擎段 |
| flexible / symbolic shapes | compiler 有完整符号形状系统，master gate 在 M1 上开 | “?” 维能解析，但直连路径不 lowering | 按具体 shape 编译、pad/bucket |
| sort、dynamic-slice | validator 在 M1 上可调用 | M1 的 code generator 拒绝 lowering | 不进引擎图；需要时改写 |
| top-K | validator 通过 | 只在禁带之外可编译；K∈{3,4} 在任何宽度都失败（禁带原文未给出） | 避开 K=3/4，其余按可编译验证 |
| non_maximum_suppression | validator 通过 | engine-only mask 报 not supported on any backend | 落 CPU/GPU |
| CAS、Reverse（authoring 路由） | 有内部语义 validator | 无导出 per-layer 符号 | 直写路由不可达 |

## 操作数信封：能编译的形状

| 约束 | 上限 | 超出后果 | 条件 |
|---|---|---|---|
| 激活 dtype | 仅 fp16 | int32 cast 与 bf16 I/O 在 M1 被拒 | 论文表 17.1 |
| rank | ≤5 | rank 6 拒绝；Linear 要求输入 rank ≤4 | 比通用 5 轴更严 |
| W / H | 16384（M1–A15）；A16/M5 起 65536 | 16385 拒绝 | 逐轴、精确一步边界 |
| C | 65536 | 65537 拒绝 | 各代相同 |
| conv 核宽 kW | 13（fp16 内核格式）；前端预筛 ≤15 | 14 起在 lowering 拒绝；16 前端就拒 | 默认格式大核 29（M1）/32（A14 起）另表 |
| conv 核高 kH | 29，且内核不得高于输入 | 高于输入拒绝 | 输入允许时 |
| arg-min/max 归约轴 | 2048 | 2049 拒绝 | 论文写明是 fp16 精度考虑 |
| matmul 收缩深度 | 两个操作数都为 1 | depth>1 拒绝 | — |
| matmul 输出通道 | 64 KB kernel-memory 预算 | 放不下就拒绝 | 不是 DRAM 预算 |
| 最大单操作数 | 2 MB | 正确但分块、从 DRAM 流式 | 只比最大单个，不算总和 |

## 只有 compile-and-run 才算数

证据阶梯有五层，每层只证明自己那一层：能力字节（HAL `0x48f`–`0x8cc`，约 165 个，按 `hal[off] & 1` 读）→ 前端识别 → per-layer validator（50 个 `_ANECValidate<Op>Layer`，同时是后端 legalizer）→ lowering/codegen → 目标芯片上执行。

- 记录条件：芯片家族、ANECompiler 版本（论文为 9.509.0）、macOS 下限、具体 concrete shape、权重形式；任一改变都可能翻转结论。
- 拒绝要区分干净拒绝与崩溃：dynamic-weight 卷积 batch 1 可编译且对参考余弦 1.0000，batch ≥2 直接以“lost helper application”崩溃，而不是返回错误码。
- 发布任何“支持”结论前，先跑 [shape 契约](ane-shape-numerics.md) 的静态检查，再做目标芯片上的 smoke test；成本与放置结论另按 [roofline](ane-roofline-dispatch.md)、[placement](ane-placement.md) 与 [weights](ane-weights.md) 的口径复核。

## 研究风险：这条路径本身

论文描述的是 Core ML 之下的私有直连路径，仅供测量与研究，不是可交付的产品路径。

- 可达性：普通无 entitlement 的用户进程可以到达；compiler 是沙箱服务；内核只加载 daemon 编译并签名的程序，自建/自编译程序在 load 时以 `0xe00002e2` 拒绝；缺 entitlement 时返回 `0xe00002c7`。
- 脆弱性：私有、未文档化、无 Apple API 契约，绑定 ANECompiler 9.509.0；一次 macOS 更新就可能破坏 dispatch 库或路径，不随 App Store 分发。
- entitlement 不解锁四个硬件门控特性（3-D 卷积、原生 state、bf16 I/O、flexible shapes），只增加 loader 层便利；Core ML 仍是发布路径。
- 没有回退：这套东西在任何平台可安装，但没有 Apple silicon Mac 和编译好的 dispatch 库就完全跑不了。

## 采用条件

- 可采用：把 compile-and-run 作为唯一支持判据；用本页的状态桶与分家族地板规划 off-engine 回退；把 35/37 的差异写成范围而不是单值。
- 不能推出：能力字节或 validator 通过不等于可执行；Predicted 的 M3/M4 不能当实测发布；A16 的 extent 上限不能套到 M1；“按能力表应该有”在 3-D 卷积上是反例；单条 fused SDPA descriptor 的存在不代表转换路径会发射它。
- 待验证：目标芯片、OS 与 compiler 版本组合必须在本地 smoke test 一次；成本、放置、权重形式与执行测量分别移交 [roofline](ane-roofline-dispatch.md)、[placement](ane-placement.md)、[weights](ane-weights.md)、[execution](ane-execution-measurement.md)。

来源 `ane-paper-2606-22283-v2`，arXiv:2606.22283v2 §4.4、§8（ch 8）、§26.1/§26.4、§35.2 的 MinimumFamily 地板表、附录 A（含 A.1）、附录 E（ch 8/ch 24）；预印本 303 页，M3/M4 列由反编译外推。
