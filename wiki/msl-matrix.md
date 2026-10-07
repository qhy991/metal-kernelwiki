# MSL 矩阵乘：SIMD-group、尾块、累加精度与 tile 复用

本页研究 `simdgroup_matrix` 的公开接口与 MLX Steel 的具体写法。2026-10-07 在 M4 / MLX 0.31.2 完成四路径、324项模型外检查：三条 float32 计算路径各81项通过，half 矩阵路径81项均未达到同一个 F32 输出精度合同。**没有性能测量，也没有专用矩阵硬件使用证明。** 原始失败保留，不把这组小矩阵等同于 LLM GEMM 验收。

## 公共接口保证什么

[MSL 4.1 规范 §2.4、§6.8](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)描述从语言2.3起的 half/float 8×8矩阵，bfloat 从3.1起有对应类型。矩阵由 SIMD-group 协作处理，调用须处于组内统一控制流；**矩阵元素到 lane 的映射没有规定**。

`simdgroup_load/store` 支持 device 或 threadgroup 指针，行距以元素计，不是字节。没有有效行列数或 mask 参数，完整 fragment 不能直接写入未填充的边缘输出。标量值构造器表示对角矩阵；要表达全部元素为某值，用 `make_filled_simdgroup_matrix`。本例固定 origin=0、transpose=false，不据接口参数名称猜测其他组合的寻址规则。

规范展示的 `simdgroup_multiply_accumulate(d,a,b,c)` 为 `d=a*b+c`，四个矩阵使用同一个 T。它没有建立“half fragment 输入、float fragment 累加”的混合重载合同；这不等于断言所有编译器或硬件都不支持扩展。本例明确把 F16 buffer 数据转换到 float staging，再用 float 矩阵，避免把存储精度与计算精度混为一谈。

[Apple 能力表（2026-05-21）](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)列 M4 为 Apple9，SIMD-scoped matrix multiply 从 Apple7 起。语言版本、硬件 family、编译与运行仍是分别核对的条件。本轮 shader 元数据记录 `threads_per_simdgroup=32`、`simdgroups_per_threadgroup=1`；没有仅用“一个 group”推算宽度。

## 一个有尾部保护的原创 MSL 函数体

限定 A[M,K]、B[K,N] 为 F16，输出 Y[M,N] 为 F32，正维度且索引可表示；每个完整32线程组负责一个8×8输出 tile。下面是 MLX `metal_kernel` 的 source 函数体，框架生成入口参数。创建时设 `input_names=['a','b']`、`output_names=['y']`、`ensure_row_contiguous=False`；M/K/N 与 TileT 是模板参数。小数组可能被框架绑定为 constant 指针，这里只作合法标量读取，不跨地址空间 cast。

```metal
const uint tid = thread_position_in_threadgroup.x;
const uint row0 = 8 * threadgroup_position_in_grid.y;
const uint col0 = 8 * threadgroup_position_in_grid.x;
threadgroup TileT as[64];
threadgroup TileT bs[64];
threadgroup TileT cs[64];
simdgroup_matrix<TileT,8,8> acc =
    make_filled_simdgroup_matrix<TileT,8,8>(TileT(0));
for (uint k0 = 0; k0 < K; k0 += 8) {
    for (uint i = tid; i < 64; i += 32) {
        const uint r=i/8, c=i%8;
        as[i] = (row0+r<M && k0+c<K) ?
            TileT(a[(row0+r)*a_strides[0]+(k0+c)*a_strides[1]]) : TileT(0);
        bs[i] = (k0+r<K && col0+c<N) ?
            TileT(b[(k0+r)*b_strides[0]+(col0+c)*b_strides[1]]) : TileT(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<TileT,8,8> am, bm;
    simdgroup_load(am, as, 8, ulong2(0), false);
    simdgroup_load(bm, bs, 8, ulong2(0), false);
    simdgroup_matrix<TileT,8,8> next;
    simdgroup_multiply_accumulate(next, am, bm, acc);
    acc = next;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(acc, cs, 8, ulong2(0), false);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i=tid; i<64; i+=32) {
    const uint r=i/8, c=i%8;
    if (row0+r<M && col0+c<N) y[(row0+r)*N+col0+c]=float(cs[i]);
}
```

通过共同数值门的矩阵路径传 `TileT=mx.float32`；本轮也原样测试了 `TileT=mx.float16`，它的失败见下节。调用的输出 shape/dtype 为 `(M,N)` / `mx.float32`，`grid=(32*ceil(N/8),ceil(M/8),1)`，`threadgroup=(32,1,1)`。这里 grid 是总线程数，不能照搬 native `dispatchThreadgroups` 的组数量。[MLX custom 调用契约](https://github.com/ml-explore/mlx/blob/v0.31.2/docs/src/dev/custom_metal_kernels.rst)

三个同步位置对应不同依赖：写完 A/B scratch 后才能 load fragment；下一轮覆盖 scratch 前须完成前轮读取；集体 store 到 C scratch 后才能标量读取。M/N/K 尾部在标量访存处屏蔽，矩阵调用始终统一参与。线程组只含一个完整 SIMD-group，本例保守使用 threadgroup barrier；未测试改用更窄 barrier 或删除任何 barrier 的合法性与速度。

这种写法为清楚表达语义付出代价：每个输出 tile 都装载 A/B，边缘 tile 填零，输出经过 scratch；它没有多 SIMD 间的 tile 复用、异步流水线或寄存器 epilogue。声明的三块64元素数组也不能直接当作编译器实际分配量。示例可用于理解 API，不能当成生产 GEMM 的最优实现。

## F16 存储、F16 矩阵与 F32 输出的区别

四路径共用实际存储的 F16 输入与 F32 输出合同：

| 路径 | 计算与暂存 |
|---|---|
| scalar_f32 | 按输入 strides 读取，转 float 后逐 K 累加。 |
| matrix_f32 | 输入标量转 float，float staging/fragment/accumulator，最终 F32 输出。 |
| matrix_f16 | half staging/fragment/accumulator，写到 half C scratch 后转 F32 输出。 |
| mlx_promote_f32 | 明确先把两个输入转换到 F32，再调用 `mx.matmul`，输出 F32。 |

最后一条是数值对照，不是 F16 原生 matmul 的计时基线；它可能有额外临时量、复制和不同后端分派，本轮没有测这些成本。

M4/16GB、macOS 27.0（26A428）、MLX 0.31.2、NumPy 2.4.3、Python 3.14.3。取得协作 GPU 锁后单进程执行，未改环境、安装软件或加载模型。PCG64 seed=211；九组 `(M,K,N)` 为 `(1,7,5),(1,64,65),(7,9,15),(8,8,8),(9,17,7),(16,32,24),(31,65,33),(32,64,64),(3,257,17)`，包括三维尾部与小 M。

每组测试连续、B 转置 view、A/B 均 step2 三种布局，均从 GPU backing 建 view 并回读逻辑值。分布包括 normal（标准差0.5后存为F16）、交替抵消并含零行、wide（A=256，B按K交替±256）。CPU 参考是**存储舍入后的输入**转 float64 点积；逐元素门预定为 `abs(error)≤1e-4+1e-5*abs(ref)`，另要求 shape/dtype、有限性及零参考精确为零。该门选择 F32 输出质量，不是任意模型的默认容差。

| 路径 | 通过 / 81 | normal 最大绝对误差 | normal 最大混合误差比 |
|---|---:|---:|---:|
| scalar_f32 | 81 | 2.764e-6 | 0.01957 |
| matrix_f32 | 81 | 2.764e-6 | 0.01957 |
| matrix_f16 | 0 | 0.02245 | 140.29 |
| mlx_promote_f32 | 81 | 1.867e-6 | 0.01236 |

混合误差比是逐元素误差除以该元素的容差，≤1才通过。half 路径的 normal 与抵消各27项为有限误差失败，抵消最大绝对误差1.078125、最大混合比约2702.71；wide 的27项均出现非有限结果。其他三条路径的抵消和 wide 输入都与参考精确相等。wide 中单个乘积为±65536，已超出有限 half 范围；结果说明转成 F32 输出不能恢复中间损失，但本轮没有观测具体在哪个乘法、累加或存储阶段发生舍入/溢出，不能猜测内部指令。

81项元数据门通过，包括实际非单例轴 stride 和32线程 SIMD 宽度。合同在运行前已限定只断言长度>1轴的 stride；singleton 轴不会改变唯一逻辑位置，这没有追溯性修改[上一轮](msl-programming.md)的 stride 失败。324项数值记录完整，整体 `passed=false`、退出2；不为让 half 路径通过而放宽门槛，也没有进入性能测试。

独立 CPU 审查用逐元素 `math.fsum` 从81组保存输入重算共同 oracle，复核全部324份输出、原始误差指标及失败集合，均与原记录一致。half 路径 wide 输出共11424个 +Inf，其中7683个位置（12项 case）的参考为0；没有有限输出违反零参考门。计数含布局复用，不是独立随机试验。审查程序成功不改变原实验失败状态。

## 从 Steel 源码提取优化候选

以下是 v0.31.2 源码分析，未在本轮复现其性能；tag URL 可移动，未证明本机 binary 与这些源码逐字对应。

| 已读代码 | 可提取的机制 | 采用前的边界 |
|---|---|---|
| [`mma.h::BaseMMAFrag/BlockMMA`](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/steel/gemm/mma.h) | 手工按 lane 加载 fragment，使用 `thread_elements()`；多个 M/N fragment 复用 A/B，展开 K=8 的工作。 | 不是公共 `simdgroup_load` 路线。lane 映射是实现细节；更大 tile 增加 accumulator 存活量，需逐目标验证。 |
| [`steel_gemm_fused.h::gemm`](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h) | threadgroup staging、`align_M/N/K` 函数常量、整齐与 safe 边界路径、epilogue。 | K余数先计算，改变累加顺序；本轮只审阅 safe loader 的调用及范围传递，没有审计 loader 定义。 |
| [`matmul.cpp`](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/matmul.cpp) | 识别普通/转置布局，对其他 strides 可能先复制；按形状与能力选 GEMV、regular、split-K、NAX。 | 源码 tile 不是通用最优值，源码存在不证明实际 dispatch。比较须包含必要的转换/复制。 |
| [`nax.h::BaseNAXFrag::mma`](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/steel/gemm/nax.h) | Metal 4 MPP cooperative tensors 与 `matmul2d`。 | 是另一套接口，不能复用旧 lane 假设；需单独核对 OS/API/dtype/后端 gate。 |

Steel 的一个 SIMD-group 输出 tile 有 `TM=BM/(8*WM)`、`TN=BN/(8*WN)` 个维向片段，C 的源码每线程存储为 `2*TM*TN*sizeof(AccumType)`；这能提出寄存器压力假设，不能当成实际寄存器计数。当前 host 的 regular 初始候选64×64×16还受架构、shape、dtype分支修改，不宜直接搬入教学例。

下一步应在同一输出合同下比较直接 device 装载与 staging、多 SIMD tile 复用、epilogue 和必要复制。公开接口将 float fragment 直接装载自 device 时需要 F32 buffer；若原输入为 F16，转换成本应计入完整路径，不能把 half 指针直接传给该已文档化重载。之后用真实 K/N、M=1/32/512 分别验证。小 M 填充较多的8×8路线不自动适合 decode。MPP API 支持、F32/TF32选择和实际物理计算单元也要分别查证，见 [Metal tensors](metal-tensors.md)、[GEMM/MoE](gemm-moe.md)、[MSL 调优](msl-optimization.md)。

原始来源 `local-msl-matrix-20261007`，逻辑引用 `2026-10-07-msl-matrix/derived/summary.json`。全部81组输入/CPU参考、324份输出、元数据、源文件及失败在仓库外保存，未随公共库发布。未测 BF16、NaN/Inf输入、任意 origin/transpose 参数、多 SIMD 协作、量化、模型质量、吞吐或其他芯片；本页不支持模型加速或可独立重放的公开实验包声明。
