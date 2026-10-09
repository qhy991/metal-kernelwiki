# 量化线性层：反量化成本、常驻内存与 shape 一起比较

[English companion](en/mlx-qmm-path-comparison.md)

证据：2026-10-07 的 M4/MLX 0.31.2 有界观测与官方内存 API。4-bit、M=128 的 dense 路径在三轮中的主机完成中位数均低于 packed；其他配置存在较大波动或排序变化。常驻 dense 增加了 4 MiB 基础占用，本页不能给出整模型的自动切换阈值。

## 比较的是同一份量化权重

上游 [#4621](https://github.com/ml-explore/mlx/issues/4621) 提供了大 M 时比较 dense 路径的线索，但其 M2 Max/MLX 0.32.3、W=[12288,4096] 与本轮不同，不能移植 token 阈值。

| 路径 | 每次计时包含 | 权重驻留 |
|---|---|---|
| A packed | 新建 quantized_matmul 并 eval | packed、scale、bias |
| B percall | 新建 dequantize，再 dense matmul 并 eval | 同 A；dense 在该次操作中产生 |
| C resident | 对已求值 dense 权重新建 matmul 并 eval | 同时保留 packed 与 dense，模拟两种路径均需可用 |

三路都消费相同 packed 文件，C 没有使用量化前原始权重。输入准备与量化在独立进程完成，A/B 测量进程没有原始 GPU dense 权重。B 的反量化不能移到计时外；C 的额外驻留也不能从内存账里删除。

## 输入和预先设定的数值门

Apple M4/16GB、macOS 27.0（26A428）、MLX 0.31.2、NumPy 2.4.3、Python 3.14.3，默认 GPU，`MLX_ENABLE_TF32` 未设置。CPU PCG64 用 seed=31 生成 X、seed=32 生成 W，均为 `0.1*normal` 后实际存为 BF16。`X:[512,1024],W:[2048,1024]`，affine bits={4,8}、group=64、transpose=True，M={1,8,32,128,512} 共用 X 前缀。

CPU 独立解包实际 packed/scales/biases，以 float64 affine 解码，再经 float32→RNE BF16 舍入，最后用实际 BF16 X 做 CPU float64 dot，形成共同参考 R。两个位宽的完整解码矩阵与 native BF16 dequantize 精确数值相等。这个参考是物化 BF16 权重的共同任务语义，不要求 fused QMM 内部必须先逐元素舍入。

运行前固定门槛：shape 正确、全部有限、`max_abs≤0.025`，整体及每一行 `||Y-R||₂/||R||₂≤0.01`。这是该合成尺度的探索目标，不是 MLX 的精度保证或模型质量门。六个独立检查进程、共 30 个 bits/M/路径条件均通过，才启动计时；随后 1080 个计时输出和 90 个内存探针输出也全部通过。

前置检查中，A 的最大绝对误差为 0.01085，最差逐行相对 L2 为 0.007401；B/C 分别为 0.003905 和 0.001758。另存量化前权重、理想 affine 和 BF16 物化三种参考：全 M=512 输出的“理想量化−原权重”相对 L2 为 4-bit 的 0.09104、8-bit 的 0.006862。它们是这组随机输入的量化损失，不是 kernel 误差，更不是模型准确率。

## 主机完成时间与波动

另开 18 个计时进程，每个只测一种 route/bits；每个配置在三个进程重复，轮换六配置顺序，并轮换进程内 M 顺序。每个 M 预热 3 次，采样 12 次，每次重新建操作、计时前同步默认 GPU stream，结束于 eval 完成。下表单位 μs，统计为**三个进程中位数的 median [min,max]**，不是全部 36 个单次样本的范围；原始样本另存。

| bits | M | A packed | B percall | C resident |
|---|---:|---:|---:|---:|
| 4 | 1 | 185.6 [174.4,389.5] | 295.0 [271.7,436.1] | 237.0 [204.3,406.8] |
| 4 | 8 | 351.8 [242.0,444.4] | 525.7 [378.1,700.6] | 359.0 [352.5,402.5] |
| 4 | 32 | 506.9 [476.4,693.9] | 723.2 [456.1,917.5] | 549.4 [269.4,579.9] |
| 4 | 128 | 698.8 [556.4,806.3] | 513.7 [439.8,545.8] | 441.8 [343.1,516.1] |
| 4 | 512 | 1719.1 [1209.4,1910.4] | 1433.7 [1162.4,1441.0] | 1280.7 [1104.9,1697.4] |
| 8 | 1 | 191.0 [178.6,387.7] | 305.2 [295.0,544.2] | 209.2 [205.6,380.8] |
| 8 | 8 | 472.7 [223.1,490.9] | 532.3 [362.0,668.4] | 409.0 [349.5,536.3] |
| 8 | 32 | 408.4 [401.9,545.5] | 511.8 [384.6,691.2] | 402.6 [343.4,564.2] |
| 8 | 128 | 555.8 [481.4,594.2] | 556.1 [450.7,688.5] | 453.0 [343.2,490.6] |
| 8 | 512 | 1349.9 [1291.0,1403.0] | 1384.4 [1193.7,1625.3] | 1204.3 [1100.6,1359.9] |

例如 4-bit、M=512 的 A/B 排序在轮次间反转；不能从汇总中位数写出“超过某 M 固定改用 B”的规则。4-bit、M=128 的观察支持保留 dense 路径作为候选，但仍需目标负载的稳定测量及 profiler 解释。

这些是构图、提交和等待的主机区间，非 GPU 内核时间。样本间验证还包含 MLX 的 GPU float32 转换、CPU 拷贝/检查和日志；虽在计时区间外，仍影响后续状态。没有隔离其他应用、控制热状态/功耗或 flush 硬件缓存。重复用同一输入，增加的是时间重复，不是分布覆盖。

## 内存：基础占用和操作峰值分别看

`get_active_memory` 不包含闲置 allocator 缓存；`get_cache_memory` 记录未归还系统的闲置 buffer。`reset_peak_memory` 只清零计数，不释放数组或缓存；`get_peak_memory` 的重置窗口仍可能包含原有常驻分配。[活跃内存](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html)、[缓存](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_cache_memory.html)、[峰值](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html)、[重置](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.reset_peak_memory.html)

每个 M 另做一次内存探针，在 CPU 输出转换前读取操作窗口峰值。下表为 M=512 的实际 MLX active/peak，单位 MiB（2²⁰ 字节）；每格三个进程一致。全部 M 共用 1 MiB 的最大 X 存储，M=512 输出为 2 MiB。

| bits | 路径 | 操作前 active | 操作窗口 peak（输出仍存活） |
|---|---|---:|---:|
| 4 | A | 2.125 | 4.125 |
| 4 | B | 2.125 | 8.125 |
| 4 | C | 6.125 | 8.125 |
| 8 | A | 3.125 | 5.125 |
| 8 | B | 3.125 | 9.125 |
| 8 | C | 7.125 | 9.125 |

C 的 dense 权重稳定增加 4 MiB 基础占用；B 在这个窗口也出现该占用，不能说“逐次解码无需额外内存”。此表的峰值差额仅针对 M=512；其他 M 的临时分配需单独核查，不能只按权重字节推断。闲置缓存另有非零且随历史变化的读数，原始记录已单列；表中 active/peak 不是进程 RSS、系统统一内存总量或 GPU 硬件 cache。释放后记录的 peak 已包含验证转换，不用来替换本表。

## 应用于部署前还缺什么

先验证真实模型权重/shape、prompt chunk、decode、质量和内存预算，再决定是否保留双路径。若主机区间波动掩盖 kernel 差异，先取得对应 profiler/时间戳证据；不能用本页单次探针生成通用 dispatcher。未测试尾部 K/N、其他 dtype、模型、KV、服务并发或长期稳定性。

本页 transpose=True 与 [已失败的多行 transpose=False QVM](quantized-matmul-validation.md) 属于不同输入域，未修复或覆盖那项失败。来源 `local-mlx-qmm-paths-20261007`，外部逻辑引用 `2026-10-07-mlx-qmm-paths/derived/summary.json`；原始输入、脚本与日志留在仓库外，未随公开库提供。
