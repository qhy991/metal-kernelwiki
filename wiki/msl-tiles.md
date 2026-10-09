# MSL tile 优化：直接装载、跨 SIMD 共享与 BK 的代价

[English companion](en/msl-tiles.md)

本页比较四种原创 MSL 写法和原生 MLX matmul：直接从 device 装入矩阵、先放 threadgroup scratch、四个 SIMD-group 共享操作数，以及扩大 K 暂存块。它延续 [矩阵接口与精度](msl-matrix.md)的公开 API 路线。M4 / MLX 0.31.2 的有限结果见下文；输入是 **F32 buffer**，不能外推为 F16 存储或 LLM 加速。

## 改动到底省了什么

| 路径 | 输出 tile / 线程数 / BK | 本轮实际写法 | 源码声明的 scratch |
|---|---|---|---:|
| stage8 | 8×8 / 32 / 8 | A/B 标量装载到共享数组；一个 SIMD-group；C 经 scratch 后有界写出 | 768 B |
| direct8 | 8×8 / 32 / 8 | 合法完整块直接 device `simdgroup_load`，其他块 staging；完整输出 tile 直接 `simdgroup_store` | 768 B |
| coop16_k8 | 16×16 / 128 / 8 | 四个 SIMD-group 各算8×8，共享 A/B，各自写独立 C scratch | 2048 B |
| coop16_k32 | 16×16 / 128 / 32 | 一次装32列/行K，然后执行四轮 K=8 MMA | 5120 B |
| native_f32 | 框架选择 | `mx.matmul`，输入与输出也是 F32 | 未测 |

字节数是三块 float 数组的**源码声明量**，不是 pipeline 实际分配或 occupancy。direct8 保留 fallback 数组，走快路径不等于不占共享资源；它同时改变输入和输出，不能把全部时间变化归因于 direct load。

在完整16×16×8块上，四个独立 stage8 显式读取 `4×(8×8+8×8)=512` 个操作数元素，coop16 读取 `16×8+8×16=256` 个。这说明源码层的复用增加，**不证明 DRAM 事务减半**；缓存、编译器与事务粒度仍影响真实流量。

BK8→BK32 把每组 A/B 声明量从1 KiB增至4 KiB，另外都有1 KiB C scratch。对 K=256，外层 staging 从32轮减为8轮，源码 A/B 的 threadgroup barrier 从64次变为16次；每输出 fragment 仍执行32次 MMA。K=257 时 BK32 会算36个 K=8片段，BK8只需33个，多出的片段为零填充。较少外层同步会换来更多暂存与尾部工作，不能只按循环次数选 BK。

这些是本例的代价分解。上游 [Steel GEMM](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h)也有 staging、同步、整齐/尾部路径；[BlockMMA](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/steel/gemm/mma.h)进一步组织多个 fragment。上游的内部同步和手工 lane 装载与本例不同，本页没有声称复现其性能或实际 host 分派。

## direct load 的布局与边缘合同

[MSL 4.1 §6.8](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)提供 device/threadgroup 指针的矩阵装载重载，行距以元素计。这里采用 origin=0、transpose=false；只让两输入内层 stride=1 且 M/N/K 块完整的情况直接装载，其余统一退回有界 staging。该保守 gate 是本例选择，不是 API 不支持转置的结论。

```metal
const bool direct = DIRECT && TILE==8 && BK==8 &&
    row0+8<=M && col0+8<=N && k0+8<=K &&
    a_strides[1]==1 && b_strides[1]==1;
if (direct) {
    simdgroup_matrix<float,8,8> am, bm, next;
    simdgroup_load(am, a+row0*a_strides[0]+k0,
                   a_strides[0], ulong2(0), false);
    simdgroup_load(bm, b+k0*b_strides[0]+col0,
                   b_strides[0], ulong2(0), false);
    simdgroup_multiply_accumulate(next, am, bm, acc);
    acc=next;
} // else: 全组执行下面的 staging 路径
```

这是函数体片段，变量来自同一完整 kernel 的线程组索引与模板参数。不能把 `half*` 传入本例 float fragment；输入需已是 F32。MLX v0.31.2 对少于8元素的输入生成 constant 指针，本轮明确限制每个输入至少8元素，避免假定 device 重载能接 constant。[生成签名](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp)

M=1 的所有输入块、B 转置及双 step2 的所有输入块都不满足本例 direct-load 条件。K=257 不意味着全 kernel 必须 fallback：完整 M/N tile 的前32个 K块仍可 direct，最后一块 staging。完整 M/N 输出 tile 则独立决定直接 store，即使输入刚才 staging 也可走该输出路径。公开 matrix store 没有 mask，边缘输出必须另行保护。

## 四个 SIMD-group 如何共享一个 tile

设 `TILE=16`、`TG=128`、`sg=simdgroup_index_in_threadgroup`，`sr=sg/2`、`sc=sg%2`。线程组负责以 `row0,col0` 为起点的16×16输出，每组负责其中一个8×8。A scratch 为16×BK行主序，B为BK×16；C scratch 按 SIMD-group 分成四段64元素，不依赖 fragment 的 lane 映射。

```metal
// 全部128线程执行；tid 是 thread_position_in_threadgroup.x。
for (uint i=tid; i<TILE*BK; i+=TG) {
    const uint r=i/BK, c=i%BK;
    ast[i]=(row0+r<M && k0+c<K) ?
        a[(row0+r)*a_strides[0]+(k0+c)*a_strides[1]] : 0.0f;
}
for (uint i=tid; i<BK*TILE; i+=TG) {
    const uint r=i/TILE, c=i%TILE;
    bst[i]=(k0+r<K && col0+c<N) ?
        b[(k0+r)*b_strides[0]+(col0+c)*b_strides[1]] : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint kk=0; kk<BK; kk+=8) {
    simdgroup_matrix<float,8,8> am, bm, next;
    simdgroup_load(am, ast+(8*sr)*BK+kk, BK, ulong2(0), false);
    simdgroup_load(bm, bst+kk*TILE+8*sc, TILE, ulong2(0), false);
    simdgroup_multiply_accumulate(next, am, bm, acc);
    acc=next;
}
threadgroup_barrier(mem_flags::mem_threadgroup); // 下轮覆盖A/B之前
```

以上放在 `for(k0=0;k0<K;k0+=BK)` 中；`acc` 初始为 `make_filled_simdgroup_matrix<float,8,8>(0.0f)`。所有K块结束后，每组 `simdgroup_store(acc,cst+sg*64,8,ulong2(0),false)`，再经全线程组 barrier。最后线程按 `i=tid;i<256;i+=128` 遍历 C scratch，令 `owner=i/64, cell=i%64`，目标行为 `row0+8*(owner/2)+cell/8`、列为 `col0+8*(owner%2)+cell%8`，仅合法 M/N 坐标写出。

声明 `threadgroup float ast[TILE*BK],bst[BK*TILE],cst[TILE*TILE]`。创建 MLX custom kernel 时用 `ensure_row_contiguous=False`，模板提供 M/K/N/TILE/BK/TG/DIRECT；所有矩阵、scratch、累加与输出均为float。co-op 调用的 `grid=(128*ceil(N/16),ceil(M/16),1)`、`threadgroup=(128,1,1)`，grid是总线程数。stage8/direct8则用32和8。编译签名自动注入使用到的 builtin 与 strides，详见 [MLX custom 接口](https://github.com/ml-explore/mlx/blob/v0.31.2/docs/src/dev/custom_metal_kernels.rst)。

即使某组的8×8输出子块全部越界，它仍参与 A/B 合作装载和尾部补零，执行 MMA 和所有全组屏障；这些线程也可能为其他组加载有效操作数。不能在该组自己的合法性分支里跳过 threadgroup barrier。每个 BK 重填尾部零值；不能用上轮 scratch 剩余值充当 padding。上述片段省略了 host 封装，须与明确的 launch、dtype、边缘合同一起使用。

## 有限正确性检查

2026-10-07，Apple M4 /16GB、macOS27.0（26A428）、MLX0.31.2、NumPy2.4.3、Python3.14.3；取得协作 GPU 锁后执行，未改环境。六组 `(M,K,N)` 为 `(1,256,256),(32,256,256),(128,256,256),(31,257,259),(1,17,19),(17,31,33)`。

PCG64 seed307；normal 的标准差0.5，先舍入到F16再存F32。另有交替抵消（M>1含零行，M=1保留非零抵消）与wide（A=256、B按K交替±256）。每组包含连续、B转置GPU view、双输入step2 GPU view；实际值回读并保存。CPU oracle基于已存F32输入，用float64点积；统一逐元素门为 `abs(error)≤1e-4+1e-5*abs(ref)`，另要求shape/dtype、有限性和零参考精确为零。

五路径各54项，共270项通过；32/128线程组的108项元数据检查通过，包括实际SIMD宽度32、组内SIMD数1/4及非单例轴strides。四条自定义路线最大绝对误差约1.133e-5、最大混合误差比0.06012；原生路线分别约9.023e-6和0.05125。抵消/wide均精确匹配。统计含重复布局，不是270个独立随机试验。

独立 CPU 审查从保存输入逐点 `math.fsum` 复算18组不同数值 oracle，另36组布局复用先检查数组相等；270份输出、108条元数据、54条 direct eligibility 均完整，oracle及原始误差指标差异为0。eligibility 由实际stride和源码谓词推导，不能当作GPU dispatch计数。正确性与独立审查通过后才运行下述计时。

## 完成时间与适用范围

计时只取normal的五个配置。共15个顺序进程，每路径三轮；轮间轮换路径次序，每进程也按轮号轮换配置次序，均在同一短时段内完成。每配置先1次first、2次warmup、再7次timed，共75份first、150份warmup、525份timed输出。全部750份数组保留并独立复核通过，没有剔除慢进程。first可能复用全局driver cache，不称为冷编译测量。

准备并求值输入、创建kernel对象在计时外；每个样本先 `mx.synchronize()`，然后从新建调用/计算图开始计时，到 `mx.eval(y)` 返回结束。数据已统一存为F32，各路线都排除前期F16舍入/F32转换。区间包含输出分配、框架调用、提交、GPU工作与等待，原生路径必要的布局处理也包含其中；CPU回读、oracle检查和保存发生在计时后，各样本输出随后释放。这是**主机完成时间**，不是纯kernel/GPU时间；计时外的回读和I/O仍可能影响下个样本的系统状态。

下表单位 **µs**：先取每进程7个timed样本的中位数，再报三进程中位数的 `median [min,max]`；范围不是置信区间。

| M×K×N / 布局 | stage8 | direct8 | coop16_k8 | coop16_k32 | native_f32 |
|---|---:|---:|---:|---:|---:|
| 1×256×256 连续 | 210.2 [206.7, 515.9] | 245.5 [209.3, 518.6] | 408.2 [191.8, 504.1] | 184.5 [174.5, 198.2] | 428.0 [168.1, 478.5] |
| 32×256×256 连续 | 521.2 [211.1, 536.8] | 402.5 [176.9, 508.8] | 524.1 [204.9, 524.6] | 408.2 [400.9, 512.4] | 394.6 [186.5, 456.1] |
| 128×256×256 连续 | 667.5 [603.7, 676.6] | 454.5 [282.4, 560.7] | 465.8 [249.2, 553.1] | 437.4 [244.2, 465.9] | 509.1 [417.0, 524.3] |
| 31×257×259 连续 | 538.8 [524.2, 553.9] | 459.2 [233.2, 520.2] | 425.5 [218.5, 499.2] | 418.2 [195.1, 496.3] | 228.1 [201.0, 411.3] |
| 32×256×256 双step2 | 239.0 [237.5, 523.0] | 508.5 [212.0, 528.6] | 214.5 [203.5, 219.5] | 226.2 [205.5, 404.7] | 212.7 [206.8, 517.1] |

轮间波动足以改变判断。例如32×256×256连续输入，coop16_k32相对coop16_k8在前两轮较短（408.2 vs524.1、512.4 vs524.6 µs），第三轮较长（400.9 vs204.9 µs）。三轮中位数也不能建立稳定的BK赢家。M=1的direct8根本没有触发直接输入/输出路径，不能把该行差异解释为direct-load收益。

128×256×256中stage8的三轮中位数范围高于其他路线；这仅支持这一形状、布局及完成时间范围内的描述。各候选同时改变线程组规模、共享方式或尾部工作，缺少counter不能把变化拆成矩阵吞吐、带宽或同步的单一贡献。未清缓存，未控制温度、电源状态和其他系统负载；保留观察，不为消除波动而修改环境或宣称最优配置。


本轮没有 device timestamp、GPU counters、反汇编或资源分配报告，不建立专用矩阵单元、DRAM 流量、寄存器/spill或occupancy结论。它没有测试模型、量化、F16 buffer、MPP/NAX、异步流水线、拆分K或更大tile，也不证明任一参数全局最优。真实优化仍要对目标的M/K/N、布局和dtype重新比较，并纳入输入转换、下游消费和模型质量。

M=1的另一种工作分解见 [MSL GEMV](msl-gemv.md)：F16存储、F32输出的标量与SIMD归约实验，不与本页F32矩阵时间直接比较，并保留归约顺序导致的精度失败。

本地来源 `local-msl-tiles-20261007`，逻辑引用 `2026-10-07-msl-tiles/derived/summary.json`；源码、54组输入/oracle、全部输出、逐样本时间及独立审查留在仓库外，未随公开仓库发布。保存记录不构成可独立复验的公开实验包或正式资格。
