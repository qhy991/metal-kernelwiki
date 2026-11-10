# ANE 速查：187 行操作矩阵、每代限制与预算数字

[English companion](en/ane-quick-reference.md)

本页是六个 ANE 主题（[能力面](ane-reachability.md)、[roofline](ane-roofline-dispatch.md)、
[放置](ane-placement.md)、[形状与数值](ane-shape-numerics.md)、[权重](ane-weights.md)、
[执行与测量](ane-execution-measurement.md)）的数据索引：把论文里能查表的数字收成可检索的
[机器可读表](../data/ane-tables.json)，并在这里给出最常用的几张。数据来自第三方逆向工程论文
arXiv:2606.22283v2 的 M1/M5 实测与其余代际反编译，**没有本机复测**；每条都带 `M`（实测）/
`D`（反编译）/`P`（预测）标记，引用时请保留。

## 机器可读表：`data/ane-tables.json`

一个 JSON 对象，七个数据集，共 187 + 24 + 9 + 28 + 119 + 17 + 313 + 67 行；字段与来源在 `meta` 里：

| 数据集 | 行数 | 内容 |
|---|---:|---|
| `ops.rows` | 187 | 操作 × 四个家族（M1/A13、M2/A14、M3/A15、M4M5/A16-A17）的状态与 lowering 注记；`ops.derived` 给出分类计数 |
| `chip_limits.limits` | 24 | 表 A.1 的每代数值限制（kernel、tensor、transpose、指令对齐、KMEM、LUT 预算等） |
| `chip_limits.tiers` | 9 | 表 34.2 的能力档位（维度上限、3D conv、texture engine、sin/cos、dropout） |
| `targets.rows` | 28 | 表 34.1 的编译目标、硅片归属与 NE 核数 |
| `metrics.rows` | 119 | 章节表里的 roofline、能耗、放置、权重形式、数值、坑与执行数据，每行带条件与 locator |
| `hidden_layers.rows` | 17 | 附录 B.1 的原生隐藏层目录：族门与决策陷阱（SDPA `SubtractMax`、TopK K∈{3,4}、Sort/DynamicSlice 的 validator-only 等） |
| `workload_tables.groups` | 313 | ch7–36 的按类对照表：权重形式与延迟、能耗、按类速度/能效、conv 变体与常量、芯片缩放、fp16 分歧轴、decode 放置、训练、数值适配、MAC 几何与存储层次、硅片映射、家族地板、成本模型 clamp 与上层档 |
| `glossary.rows` | 67 | 附录 D 术语表：AFPP/aned/ANEC/CCDMA/DART/DPE/dispatch floor/wide accumulator 等；`core_facts` 含四条更正 |

查询示例（标准库或 `jq` 均可，不需要 GPU/模型）：

```bash
jq '.ops.rows[] | select(.m1=="No path") | .op' data/ane-tables.json
jq '.chip_limits.limits[] | {limit, m1: .m1_a13, m5: .a16_m5}' data/ane-tables.json
jq '.metrics.rows[] | select(.group=="roofline" and (.name|test("ridge")))' data/ane-tables.json
```

`ops.derived` 的分类计数：四个家族都 native 113 行（去掉 5 行私有 `ne_`/`pe_` unit 选择后为 108）、
M1 为 Bridge 后续转 native 10 行、M1 为 family-gated 后续转 native 7 行、永久 Bridge 22 行、
全家族 No path 35 行；合计 187。论文自己的桶合计是 182，差异保留在 `derived.note`。

## 操作矩阵怎么读

- **先按列看代际，不要按行看名字。** 同一行在不同家族可以是从 Bridge 变 Native（resize、topk、
  crop_resize），也可以一直 No path（scatter、gru/lstm/rnn、反三角）。
- **No path 的 35 个操作**是硬边界：`scatter*`、`reduce_prod`、`logical_*`、`mod`、`gru/lstm/rnn`、
  `tan/asin/acos/atanh/asinh/acosh/sinh/cosh`、`random_{bernoulli,categorical,normal}`、`list_*`、
  `reverse_sequence`、`band_part`、`non_zero`、`one_hot`、`shape`、`sliding_windows`、`constexpr_cast`。
  遇到它们先改写（log-sum-exp、minimum、展开循环）再谈编译。
- **M1 上的软边界**：`gather`/`gather_nd` 有软件信封（batch 1、depth 1、三元素 index channel）；
  `sort`/`dynamic-slice` 的 validator 能过、codegen 拒绝；`topk` 只在禁带外可编译，K∈{3,4} 任何宽度都失败。
- **家族台阶**：texture engine 从 A14（crop_resize、resample、affine、硬件 gather、symmetric pad），
  sin/cos 与 hardware random 从 A15，张量/转置 extent 从 A16 抬到 65536；A17/A18 不新增操作。
- **直写路由的隐藏层**（`hidden_layers`）：SDPA 的 `SubtractMax` 默认 false、必须置 true；TopK 的 K∈{3,4}
  任何宽度都失败；Sort/DynamicSlice 在 M1 上 validator 通过而 codegen 拒绝；`GlobalArgMinMax` 转换路由要 A15，
  但直写路由在 M1 可跑。这些都是研究用路由，不是可发布实现。

## 每代限制（表 A.1 摘要）

| 限制 | M1/A13 | A14 | A15 | A16/M5 |
|---|---|---|---|---|
| tensor W/H、D | 16384 | 16384 | 16384 | 65536 |
| tensor C | 65536 | 65536 | 65536 | 65536 |
| tensor N (batch) | 65536 | 65536 | 65536 | 65536 |
| conv kernel W（默认大核 / fp16 大核） | 29 / 13 | 32 / 16 | 32 / 16 | 32 / 16 |
| conv kernel D（大核 / 小核） | 16 / 8 | 16 / 8 | 16 / 8 | 16 / 8 |
| 归约转置阈值 | 192 | 192 | 384 | 384 |
| 指令对齐 | 256 B | 16 B | 16 B | 16 B |
| texture engine | 无 | 有 | 有 | 有 |
| kernel-memory 预算 | 64 KB | 64 KB | 64 KB | 64 KB |
| context-switch 活跃张量 | 2 | ∞ | ∞ | ∞ |

旧代（pre-A13）没有 3D 维度、没有 texture engine、张量 N 只到 4096；`M11` 小核是 A16 特性 + 16384 维上限的混合档。

## 预算数字（M1 口径，M5 另标）

| 量 | 值 | 条件 |
|---|---|---|
| 算力 roof `P` | 12 fp16 TFLOP/s | 融合 matmul 链的边际斜率；单次大 matmul 4.8、单次 conv 1.8 |
| 带宽 roof `B` | 85 GB/s | 内存控制器实测；单行权重流 51、编译器常量 50、单 relu 10 |
| ridge | 141 FLOP/byte | 12/85 口径；M5 的 424 用另一组基准，不可换算 |
| 片上工作集 | 2 MB（M5 4.72 MB） | 比"最大单个活操作数"，不是求和；实测拐点 2.28–2.34 MB |
| dispatch 地板 | 0.23 ms（M5 0.11 ms） | 每评估固定；小图 190 µs 里 98% 是软件+固件 |
| 固件往返 | 130 µs | 只读工具无法再细分 |
| 融合摊销 | 每层 222→6.3 µs（1→32 层）；每样本 196→1.5 µs（batch 512） | 一次 dispatch 付一次地板 |
| 能耗 | 0.37 pJ/FLOP（最优点）、22 pJ/FLOP（地板处） | 另有 0.5 pJ/FLOP 的第三口径，不合并 |

## 按类对照（M1，表 11.1）

| 工作负载 | 引擎 GFLOP/s | 引擎 GFLOP/s/W | GPU GFLOP/s | GPU GFLOP/s/W | 引擎能效优势 |
|---|---:|---:|---:|---:|---:|
| GEMM floor（K=256） | 51 | 17 | 46 | 27 | **0.6×（引擎输）** |
| GEMM bandwidth（K=1024） | 662 | 470 | 1109 | 207 | 2.3× |
| GEMM compute（K=4096） | 3588 | 775 | 7286 | 192 | 4.0× |
| Conv single（C=64） | 99 | 23 | 89 | 18 | 1.3× |
| Conv resnet（d=16） | 9602 | 2063 | 4671 | 142 | 14.5× |
| Attention ViT（S=197） | 1711 | 370 | 1831 | 187 | 2.0× |
| Attention long（S=512） | 1775 | 544 | 3768 | 217 | 2.5× |

注意两行反直觉的结果：**floor 级 GEMM 的能效引擎反而输**（0.6×），**C=64 卷积只有 1.3×**——
"ANE 卷积快 3.8×/9×"只对 256 通道成立。完整 313 行（含权重流延迟表、M2 按类能效、MAC 几何、2 MB 越界吞吐曲线、A1x 驻留缓冲阈值、硅片映射、F0–F4 家族地板与 H18 fp8 上层档）在 `workload_tables.groups`。

## 权重形式（按代际的 stream / fold）

| 形式 | M1 | A14/M2 | A15+/M5 | 实测 |
|---|---|---|---|---|
| int4 LUT（16 项 fp16 codebook，v∈{1,2,4,5}） | stream | stream | stream | M1 2.37× |
| 结构化稀疏（≥50% 零，mask + fp16 值） | stream | stream | stream | M1 1.55–1.64× @0.43× 字节 |
| int8 affine | fold（1.0× 延迟） | stream | stream | A14 深度 0.87/0.60/0.52× |
| blockwise affine | fold | fold（0.985×） | stream（[P]） | M5 四形式 1.6–1.8× |

量化只减搬运字节，MAC 仍是 fp16；int8 编译开关是 weight-only。decode 上 int8 只有 0.99×，因为那一步是 dispatch-bound。

## 放置速查

| 类别 | 去向 | 条件 |
|---|---|---|
| 卷积 / 视觉 / encoder 前段 | ANE | 256 通道 3×3：3.8× 快、9× 省能（M1）；batch 1–256 |
| 短序列 attention（S≈197） | ANE | M1；S≈512 起 GPU 快 2.1×，ANE 仍有 1.8× 能效优势 |
| 大方阵 GEMM | GPU | 2.8× 快、饱和 3.0× |
| 自回归 decode | GPU（每个 batch） | batch 16：2.7× 快、4.6× 省能 |
| 低于 0.23 ms 的琐碎算子 | CPU | 64×256×256 matmul 0.026 ms |
| 服务交叉点 | encoder ≈batch 23、self-attention ≈batch 6 之后转 GPU | M1 |

## 采用条件

- 本页只做索引：结论的条件、反例与论文内部矛盾都在六个主题页与 `metrics.rows` 的 `condition`/`locator` 里；
  速查表省略条件时按主题页的原文执行。
- 数据是第三方论文的 M1/M5 实测与反编译，不是本机结果；M2 只部分实测，M3/M4 与上层档是反编译或预测。
- 不要把 `ops` 的状态当成运行时保证：只有目标芯片上的编译+运行才算确认（见[能力面](ane-reachability.md)）。
- 数据文件是 `data/ane-tables.json`；生成与校验脚本在仓库外（`devlocal/ane-arxiv-2606-22283/scripts/`），
  重新生成不改变本页结论。

来源 `ane-paper-2606-22283-v2`，arXiv:2606.22283v2（2026-10-15）；机械解析自附录 A、附录 C（目标表）、
表 34.1/34.2 与章节表，校验结果见 `devlocal/ane-arxiv-2606-22283/reports/data-validation.md`。
