# Core ML 只接受沿输出轴分组的权重：ANE 上 sub-8-bit 的结构性边界

[English companion](en/ane-coreml-weight-axis.md)

适用：要把一个投影层放到 ANE 上，手里是 HuggingFace 发布的量化 checkpoint（MLX affine Q4、GPTQ、AWQ、GGUF k-quant），想知道能不能直接喂、以及自己设计一种"硬件感知"的权重格式有没有空间。本页只讲**公共工具链（coremltools 9.0 + Core ML 编译器）实际会把什么放上 ANE**，证据全部是 M1 Pro / M2 / M4 本机实测。不覆盖：ANE 固件与硅本身支持哪些形式（那是[权重形式](ane-weights.md)页的论文证据，结论不同，差异见下文"与论文的张力"）、放置取舍（见[放置](ane-placement.md)）、强度与 dispatch 地板（见 [roofline](ane-roofline-dispatch.md)）。

## 规则：分组的轴决定落点，位宽和分组数都不决定

直接用 MIL 原始算子 `constexpr_lut_to_dense` 构图，LUT 形状 `[gN, gK, 1, 1, 16, 1]`，`gN` 沿输出轴分组、`gK` 沿输入轴，其余不变。conv `[3072, 1024]`，T=256，读 `MLComputePlan` 的 per-op 落点，不执行：

| 沿输出轴 palette 数 | 沿输入轴 palette 数 | 总数 | 落点 |
|---:|---:|---:|---|
| 1（逐张量） | 1 | 1 | ANE |
| 192（每 16 行） | 1 | 192 | ANE |
| 3072（**每行**） | 1 | 3072 | ANE |
| 1 | 16（每 64 输入） | 16 | **CPU** |
| 3072 | 16 | 49152 | **CPU** |

3072 个 palette 沿输出轴被接受，16 个沿输入轴被拒绝，所以**分组数不是变量，轴才是**。`constexpr_blockwise_shift_scale` 的行为一致：per-channel（每输出行一个 scale）留在 ANE，per-block 32/64（每 N 个输入一个 scale）整个 conv 连同后面的 `identity` 一起判给 `MLCPUComputeDevice`。

这条规则此前不可见，是因为 `palettize_weights` 与 `linear_quantize_weights` 两个 API **只暴露输出轴分组**（`per_channel` / `per_grouped_channel`）；原始算子两种都接受，编译器只接受一种。

M4（Mac16,10）实测的完整落点表，`layers.12.mlp.gate_proj` `[6144, 2048]`，T=512，conv layout，预算 1.77e-03（probe 规则：1.05× 跑同一批权重的 MLX Q4 GPU arm）：

| 编码 | MIL 降低 | 落点 | package | wired | 相对 RMSE |
|---|---|---|---:|---:|---:|
| fp16 | `const` | ANE | 25.2 MB | 35.9 MB | 4.98e-04 **PASS** |
| int8 per-channel | `constexpr_blockwise_shift_scale` | ANE | 12.6 MB | 23.5 MB | 8.66e-03 |
| LUT6 per_grouped_channel(16) | `constexpr_lut_to_dense` | ANE | 9.5 MB | 20.2 MB | 2.59e-02 |
| LUT4 per_grouped_channel(16) | `constexpr_lut_to_dense` | ANE | 6.3 MB | 17.1 MB | 1.10e-01 |
| int8 per-block 32 | `constexpr_blockwise_shift_scale` | **CPU** | 13.4 MB | 0.2 MB | 8.15e-03 |
| int8 per-block 64 | `constexpr_blockwise_shift_scale` | **CPU** | 13.0 MB | 0.1 MB | 8.55e-03 |
| uint4 per-block 32 | `constexpr_blockwise_shift_scale` | **CPU** | 7.3 MB | 0.1 MB | 6.12e-02 |

三行 CPU 的 wired 约等于 0，这是内部对照：量的确实是 ANE 的常驻，不是进程开销。掉下 ANE 的代价是 10×——M2 上同一形状 per-block 64 是 0.77 TF/s，留在 ANE 的 int8 是 7.63 TF/s。M1 Pro（2026-10-16）、M2 与 M4 三代同样行为。

## 这条规则关闭了 sub-8-bit，不是留它待调

4 bit 的 palette 只有 16 项。而真实 checkpoint 的值分布是：Clef Flash9B `layers.0.mlp.gate_proj` 的一个 **64 输入组只有 14 个不同值**，一整行（4096 个权重）有 **544 个**，16 行有 1059 个。

所以 4 bit 够用的前提是窄分组；窄分组就是输入轴分组；而输入轴分组正是 ANE 拒绝的那种。三个独立确认：

- **精确 Q4-as-LUT4 跑在 CPU 上。** affine Q4 group-64 的一个组里每个权重都是 `q·scale + bias`、`q ∈ 0..15`，所以 16 项 palette 能逐位复现该组。手工按 (row, 64 输入) 排 palette 建出来确实精确，但 `[12288, 4096]` 上落 CPU，1.12 TF/s，对 fp16 在 ANE 上的 5.46 TF/s；wired 3.1 MB，因为 ANE 上什么都没驻留。
- **在允许的轴上加密 palette 买不到精度。** per-row LUT4 是 1.01e-01，per-4-row 9.84e-02，per-16-row 9.95e-02（Clef 真实 Q4 权重，1024/512 行采样，预算 2.47e-03）。**16 倍的 palette 数量毫无改善**，因为每一行单独看都对 16 个质心太宽。[写法杠杆](ane-write-level-levers.md)在速度侧测到同一个参数的另一半：每行一张表比 `per_grouped_channel(16)` 慢 3–10×。所以加密码本表在允许的轴上既不快也不准。
- **LUT4 是 M4 上实测最快、最省的格子，仍然不可用。** 10.70 TF/s（fp16 8.77）、wired 17.1 MB（fp16 35.9），相对 RMSE 1.10e-01——比它重新编码的那份 Q4 checkpoint 本身还差。

结论：**ANE 上可用的权重编码只有两种**，fp16（2.0 B/weight，对 checkpoint 精确）和 per-row int8（1.0 B/weight，8.8e-03）。任何"为 ANE 设计 sub-8-bit 格式"的方案都要先越过这条编译器约束，而它不是调参能越过的。

## HF 量化格式为什么不能直接喂 ANE

主流发布格式全部沿输入轴分组：MLX affine 每 64 个输入一个 (scale, bias)，GPTQ 与 AWQ 每 128 个，GGUF k-quant 每 32 或 256 个。**没有一种能按发布形式交给 ANE。** 这不是"支持得不好"，是分组轴与编译器要求正交。

实践后果：走 ANE 必须重新编码成 per-row int8（或 fp16）。这一步不是实现者的自由选择，是被这条约束逼出来的，移植时必须保持该性质。重新编码的精度账要分清锚点——若以反量化后的 Q4 权重为真值，per-row int8 相对它是 8.8e-03；若以 bf16 master 为真值，从 master 单次量化的 per-row int8 是 8.4e-03，而 affine Q4 group-64 本身是 9.2e-02，即 **int8 比它要替换的 4-bit checkpoint 准约 11 倍**；从 Q4 二次量化则是 9.2e-02（比 Q4 仅差 0.4%）。

## 规则禁止的是编码，不是图：两个免费的精度增量

输入轴粒度可以从**算子图**拿，编译器允许。K-split 把一个 conv 沿 K 切成若干 part 再求和，每个 part 是独立 conv 带独立权重张量，于是 per-row scale 落在 `K/parts` 的切片上 = 每 (row, K/parts 输入) 一个 scale。

第二个增量：`constexpr_blockwise_shift_scale` 本来就带 shift。affine Q4 的组值围绕非零偏移分布，对称量化要花 range 去覆盖这个偏移。非对称 per-channel 是 `mode='linear'` + uint8，实测**留在 ANE 上，package 大小与对称形式相同**。

Clef `layers.0.mlp.gate_proj` 对自身 Q4 权重的 float64 oracle，1024 行采样，预算 2.47e-03：

| K parts | 每个 scale 覆盖 | 对称 | 非对称 |
|---:|---:|---:|---:|
| 1（现部署） | 4096 输入 | 8.79e-03 | 8.15e-03 |
| 4（`k_chunk` 1024） | 1024 | 7.78e-03 | 7.22e-03 |
| 16 | 256 | 6.83e-03 | 6.08e-03 |
| 64（checkpoint 自身粒度） | 64 | 5.95e-03 | 3.52e-03 |

在 `k_chunk=1024`——本来就因速度被选中的那个 split——两者合起来是现部署的 **1.22×**，字节相同、落点相同、split 已经在图里。**但没有一格过预算**，所以这不能让整个 share 转 int8；它能改的是 int8/fp16 混合拆分的分界线，移动多少未测。

## 比较权重格式时不做 K-split 会高估压缩格式

K-split 帮 fp16、不帮 int8，所以不 split 的对比会把 int8 的优势放大一个 split 的量。两边各自调到最优、同一 T：

| 机器 | 形状 | fp16 | int8 | 不 split 会说 |
|---|---|---:|---:|---:|
| M4 | `down_proj` [2048, 6144], T 1408 | 1.264 ms/512tok | 1.198 | 2.032× |
| M4 | `gate_proj` [6144, 2048], T 512 | 1.132 ms/512tok | 1.106 | 1.201× |
| M2 | [12288, 4096], T 512 | 6.409 ms | 5.783 | 1.571× |

真值 1.02×–1.11×，不 split 读出来是 1.20×–2.03×。**先按落点、精度、常驻选编码，再单独调 K-split**，两件事不要混在一个比较里。

## 与论文的张力

[权重形式](ane-weights.md)页依据 arXiv:2606.22283 §25.2 把 blockwise affine 列为四种受支持形式之一，并按门字节推断它从 A15 起流式。本页实测的是另一层：**Core ML 编译器在 M1 Pro / M2 / M4 上根本不把 blockwise 分组的 conv 放上 ANE**，无论硅是否支持。

两者不矛盾，但必须分开读：论文是对固件与硅能力的反编译，本页是公共工具链的实际策略。硬件能力不等于编译器会用它。对只能用公共 API 的工程来说，有约束力的是后者。同理，论文给的 M1 int4 LUT 流式 2.37× 与本页 LUT4 留在 ANE 一致，但论文没有逐层精度表征，而本页实测 coremltools 的 LUT4 在真实 checkpoint 上是 1.10e-01，不可用——"能流式"与"能用"是两件事。

## 采用条件

- 可采用：判断一个编码能否上 ANE 时，先看它沿哪个轴分组；输出轴（含每行一个 scale/palette）可以，输入轴不行。落点必须用 `MLComputePlan` 逐 op 验证，不能从位宽或 API 名称推断。
- 可采用：HF 发布的量化 checkpoint 一律需要重编码成 per-row int8 或 fp16 才能上 ANE。
- 不能推出：本页不否定 ANE 硅支持 blockwise 或 int4 算术——只说公共 Core ML 路径不发射它们。私有路径不在范围内。
- 精度边界：所有相对 RMSE 都标注了锚点（Q4 oracle 或 bf16 master），两者差一个量级，跨锚点比较无效。预算 1.77e-03/2.47e-03 来自 probe 规则（1.05× 同权重的 GPU arm），不是模型级验收。
- 未建立：int8 误差降 1.22× 后有多少 neuron 能从 fp16 份额移到 int8 份额；单程序的常驻结论在 32 个程序规模上是否成立；非对称 + K-split 的组合未做端到端模型验收。
- 测量边界：全部为单形状、单进程的算子级测量，M2 为共享主机（wired 跨 7 次进程的跨度 90–145 MB），M4 在主机 Metal 锁内测。不构成模型验收，也不是 profiler 分析。

来源 `local-ane-coreml-axis-20261110`。本页数值为 M1 Pro / M2 / M4 上 coremltools 9.0 的本机测量，不外推到其它 OS、coremltools 版本或私有 API 路径。
