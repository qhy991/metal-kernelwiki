# ANE 放置：按类别、batch 与能耗在 CPU / GPU / ANE 之间取舍

[English companion](en/ane-placement.md)

Apple Silicon 同一颗 die 上有 CPU、GPU 和 ANE，三者共享 unified DRAM。选谁执行不是一个"谁更快"的问题，
而是**先看类别、再看 batch、最后看能耗**的三个判断。本页把 [ANE 论文](https://arxiv.org/abs/2606.22283)
ch10、ch11、ch13、ch14 的实测结论整理成可执行规则；数字都是论文在 M1/M5 上的第三方测量，不是本机结果，
也没有在 M2–M4 上复核。私有直连路径不在讨论范围，Core ML 的 placement hint 也不是承诺。

## 三张表决定去向

| 工作类别 | 结论 | 论文条件 |
|---|---|---|
| 256 通道 3×3 卷积 | ANE 比同芯片 GPU 快约 3.8×、能效约 9× | M1，ch1/ch13 |
| 16 层 3×3 conv 栈 | ANE 快约 4.2×、能效约 13× | M5，ch11/ch13 |
| 融合五点 stencil（32 次迭代） | ANE 快约 10×、能效约 49× | M5，ch11 |
| 中等方阵 GEMM | ANE | 未越过工作集与形状阈值，ch11 |
| 大方阵 GEMM | GPU 快约 2.8×，饱和时 3.0× | ch11 |
| 短序列 attention（S≈197） | ANE 最快、最省，且 fp16 更准 | ch11 |
| 长序列 attention（S≈512+） | GPU 快约 2.1×，但 ANE 仍有 1.8× 能效优势 | ch11 |
| 自回归 decode | **每个 batch 都给 GPU** | batch 16 时 GPU 快 2.7×、省能 4.6×，ch14 |
| 低于 0.23 ms 的琐碎算子 | CPU（64×256×256 matmul 约 0.026 ms） | ch11 |

服务形态会把界线移动：**encoder block 在 batch 约 23 之后归 GPU，self-attention 在 batch 约 6 之后归 GPU，
视觉卷积在 batch 1–256 都留在 ANE**（§11.6、ch13）。这些交叉点是 M1 的实测，不要外推到别的芯片或别的序列长度。

## decode 为什么几乎总是输

自回归 decode 的算术强度约 1 MAC/weight，远低于 141 FLOP/byte 的 ridge；一层栈还要 40–50 次 dispatch，
每次至少付 0.23 ms。两个代价都落在引擎的弱项上：带宽（权重流）与 dispatch 下限。

把权重从 fp16 换成 int8 只把整步变成 **0.99×**，因为这一步是 dispatch-bound，不是权重带宽-bound——把字节减半
并不能减少 40 多次固件往返。可控的量是 batching：同一程序把 batch 做大，让一次 round trip 摊到更多样本上，
而不是换权重格式（§14.1、§14.4）。

这也解释了为什么"ANE + KV 常驻"不能救 decode：常驻 KV 省的是每步的主机拷贝，而 decode 的墙钟由权重流与
dispatch 次数决定；论文明确说这种别名只省主机调用，端到端速率与逐步 call 相同（§2.6、§6.4）。

## 能效不总是站引擎这边

ANE 在 M1 上对 GPU 有 2–14× 的能效优势，跨 M1/M2/M5 都成立（ch10）。但两类例外要记住：

- **dispatch 受限的极小算子**是引擎唯一明显输能效的类别：同一份工作量下，0.23 ms 下限让引擎空转，
  能效跌到约 22 pJ/FLOP（计算最优点约 0.37 pJ/FLOP，§10.4）。这种算子应该放 CPU 或融进相邻 kernel。
- **量化不一定省瓦**：瞬时功率在 fp16/int8/int4/sparse 之间几乎不变（2.06–2.59 W），省能来自更短延迟；
  稀疏流甚至可能画最多的瓦（§10.6）。用"更快所以更省能"去解释量化收益会得出错误结论。

功率侧还有两个可用于预算的实测：持续卷积 176 s 平在约 1.66 W、最密打包约 4.7 W @6.96 TFLOP/s，
引擎没有 DVFS，也没有片内热逻辑（§31）。所以给 ANE 排期时按**固定功率包络**算，不要指望它自己降频。

## 混合 GPU+ANE 的切分规则

论文的 LLM 案例给出的是一个**前段/后段切分**，而不是任意拆分：prefill、encoder、vision 前段上 ANE，
decode 上 GPU。把这条落成可检查的规则：

1. 只搬**计算受限**的部分：算术强度 ≥ ridge（M1 口径 141 FLOP/byte）、且要被融进同一段。
2. 每个操作数（最大活跃张量）**≤2 MB**；超过就按 tile / 通道 / 空间切，否则跨过阈值后强度塌到带宽斜率。
3. 一段内的算子必须能融进**一个程序**；每多一个段就多一次 launch 加一次跨后端搬运（放置模型给
   engine 段 0.05、跨后端 0.09–0.23 的相对代价，§18.6）。
4. 边界上要付一次**channel-interleaved fp16 的重排 + 主机往返**；段越少越好，边界上不要放小算子。
5. 用 batch / tile 让每段的 compute 项盖过 0.23 ms（M1）或 0.11 ms（M5）的 dispatch 下限。

本仓库的 Clef 列切分（GPU 算前 6656 列、ANE 算其余 5632 列、每层一个 Core ML program）符合第 1、2 条，
但第 5 条需要实测：**若单次 prediction 的 compute 项小于下限**（小 tile 或短序列），32 层串行调用就退化成
纯地板成本，此时应融合相邻层或加大 tile；若 compute 已明显盖过下限，逐层 program 就没有额外损失。
无论是哪种，都要先测一次 per-layer prediction 的真实时延，不能由论文的 M1 常数外推本机结果。

## 与 Metal/MLX 放置判断的关系

roofline 的形式（`min(P, I·B)`、工作集阈值、dispatch 下限）可以迁移到 Metal/MLX 的放置推理，但**常量不可迁移**：
ANE 的 ridge 141 FLOP/byte、2 MB 工作集、0.23 ms 下限都是引擎自己的数；Metal GPU 的带宽与 tile 约束完全不同，
见 [测量](measurement.md) 与 [融合](fusion.md)。真正可复用的只有"先缩工作集、再摊下限、最后才比算力"这个顺序。

## 采用条件

- 每条结论都绑定论文的芯片与形状：卷积/视觉为 M1、conv 栈为 M5，attention 交叉点是 M1、S≈512 附近；
  不要把它当成"所有 Apple 芯片、所有序列长度"的保证。
- M2 只有部分实测，M3/M4 的家族列是反编译或预测；A11/A12 在论文词表以下，不适用。
- 论文的能耗数字来自 shipped device 的测量工具与整机能耗通道，不是本机复测；本机采用前应按
  [测量页](measurement.md) 的口径重测，并把"谁在跑"（Core ML 的 placement 是 hint，不是保证）记录下来。
- 不采用论文的私有直连路径做放置实验：它私有、不受支持、随 OS 更新失效，且不满足 App Store 条件。

来源 `ane-paper-2606-22283-v2`，arXiv:2606.22283v2（2026-10-15），ch10–ch14、§18.6。本页为第三方论文的
整理，**没有本机测量**；"速度/能耗倍数"均为论文在特定 workload 上的单点结果。
