# MSL 用法：地址空间、向量布局、同步与数值契约

MSL kernel 的第一步是把访问与参与规则写对，再讨论是否更快。本页核对 Apple **MSL 4.1 规范（2026-06-04）**，并用 M4 / MLX 0.31.2 检查两个原创示例。官方规范 URL 可更新；阅读 4.1 文档不代表本机编译器启用了全部 4.1 特性。调优候选另见 [MSL 优化方法](msl-optimization.md)。

## 地址空间要与实际函数签名一致

| 写法 | 应怎样理解 |
|---|---|
| `const device float* x` | buffer 数据，只通过此指针读取；`const` 不把地址空间改为 `constant`。 |
| `constant float* x` / `constant float& scale` | 只读地址空间；不能跨地址空间强转成 device 指针。它也不是 function constant。 |
| `thread float a[...]` | 当前线程私有；局部数组是否驻留寄存器由编译与资源使用决定，可能 spill。 |
| `threadgroup float partial[...]` | 同一线程组共享；先初始化，再按读写依赖同步，不跨线程组共享。 |

依据 [规范 §4](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)。入口的 buffer 索引、dtype、shape、stride、起始 offset 都是调用合同。MLX 会生成入口签名，不能只看 Python 里的“array”：本轮最初把所有输入假定为 device，**N=1 时编译器实际收到 `const constant float*`，packed 转型失败**。[v0.31.2 `write_signature`](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp#L66-L74)按 `arr.size()<8` 选择 constant，否则为 device；这是元素数条件，不是字节数，也不限单元素。调试时检查生成源，而不是删除限定符或跨域强转。

新版本还有 [MSL 4.1 generic pointers](https://developer.apple.com/documentation/metal/writing-reusable-gpu-functions-with-generic-pointers)：覆盖 thread/threadgroup/device，入口资源仍需显式限定，constant 不在这个集合中；无法静态决定地址空间可能增加分支和寄存器成本。规范 §4 仍有旧式绝对措辞，不能据此断言所有版本都没有 generic pointer。本页用显式重载，不要求升级工具链。

## `float4`、packed 与每线程四元素不是一回事

[规范 §2.2、§2.2.3、§2.5](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)给出的字节数如下，本轮编译后的 `sizeof/alignof` 也观察到相同结果。

| 类型 | size | alignment |
|---|---:|---:|
| `float3` | 16 | 16 |
| `packed_float3` | 12 | 4 |
| `float4` | 16 | 16 |
| `packed_float4` | 16 | 4 |

因此不能把三个紧排 float 当作普通 float3 数组，也不能因为元素类型是 float 就断言任意 view 起点满足 float4 的对齐。packed 降低对齐要求，不允许越界，也不把 stride=2 变成连续。源码写了 vector 类型，或一个线程处理四元素，都不证明最终是一条向量 load；需要编译产物或 profiler 验证。

下面是本轮修正后的 MLX `metal_kernel` **header**，显式保留两种合法地址空间：

```metal
inline float4 read_packed4(const device float* p) {
    return float4(*reinterpret_cast<const device packed_float4*>(p));
}
inline float4 read_packed4(const constant float* p) {
    return float4(*reinterpret_cast<const constant packed_float4*>(p));
}
```

对应 **source 函数体**（由 MLX 生成参数与入口）如下。限定 F32、连续一维逻辑输入、至少 N 个元素、输出写满；N 为整数模板参数，`grid=(ceil(N/4),1,1)`，本次 `threadgroup=(64,1,1)`。

```metal
const uint base = 4 * thread_position_in_grid.x;
if (base + 3 < N) {
    device packed_float4* dst =
        reinterpret_cast<device packed_float4*>(y + base);
    const float4 v = read_packed4(x + base);
    *dst = packed_float4(2.0f * v + 0.25f);
} else {
    for (uint j = 0; j < 4 && base + j < N; ++j)
        y[base + j] = 2.0f * x[base + j] + 0.25f;
}
```

`ensure_row_contiguous=False` 表示框架不代你修正布局；此 packed 路径仅用于满足连续性合同的输入。step2 的正确标量路径是 `x[i*x_strides[0]]`，不能直接调用上面连续版本。N 与索引须在整数可表示范围内；本轮仅测试下述有限尺寸，代码不是任意长度/布局的通用库函数。[MLX 调用与模板契约](https://github.com/ml-explore/mlx/blob/v0.31.2/docs/src/dev/custom_metal_kernels.rst)

## SIMD 归约后，跨 SIMD 仍需同步

`simd_sum` 对活动线程的值归约；它没有替代 threadgroup 内存的发布顺序。shuffle 的来源 lane 必须合法且活动，SIMD-group matrix 操作则有统一控制流要求，不能把所有 collective 当成同一种规则。[规范 §6.10、§2.4](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)

下面也是经检查的 MLX source 函数体：一行一个完整线程组，`grid=(TG,M,1)`、`threadgroup=(TG,1,1)`，D/TG 为模板参数，输出 y/info 长度 M。空任务线程以 0 参与，不在 barrier 前提前退出。

```metal
const uint tid = thread_position_in_threadgroup.x;
const uint row = threadgroup_position_in_grid.y;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
threadgroup float partial[TG];
float acc = 0.0f;
for (uint col = tid; col < D; col += TG)
    acc += x[row * x_strides[0] + col * x_strides[1]];
const float s = simd_sum(acc);
if (lane == 0) partial[sg] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    float total = 0.0f;
    for (uint g = 0; g < simdgroups_per_threadgroup; ++g)
        total += partial[g];
    y[row] = total;
    info[row] = simdgroups_per_threadgroup;
}
```

这里用 TG 个 float 为 partial 保守留空间，实际只用每 SIMD 一个；这是便于说明的写法，未证明线程组内存用量最优。最后只有 thread0 读取 partial 并写结果，没有后续复用，所以无需第二次 barrier；若其他线程要读结果，或要覆盖同一 shared 数组，就需重新分析依赖。它不是大型行归约的最佳实现。

barrier 要同时检查参与线程、控制流和内存域。`mem_none` 不为共享 partial 提供所需内存排序；`mem_device` 也不等于整个 grid 的执行屏障。规范 4.1 说明 Apple silicon 已结束的线程不再阻塞 barrier，所以“任何 early return 一律非法”过强；本示例依赖完整组参与及 partial 初始化，不能随意改变它的退出规则。4.1 的新 order/scope 重载需另核对编译目标。

## 数值与原子操作分别建立合同

`precise::exp` 选择这个函数的精度版本，不使其他算术自动逐位等同 CPU。math mode、FP32 函数集、乘加 contraction、中间 dtype 转换是不同设置；safe math 也不能替代固定分步舍入要求。先定义有限输入/特殊值域、存储 dtype、参考与容差，再比较 fast 候选。[规范 §1.6.3、§8](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)，已有 [SwiGLU 舍入观察](mlx-swiglu.md)。

[half算术与类型提升](msl-half-arithmetic.md)进一步区分float字面量、显式窄化、算术FTZ和框架float16_t别名；首轮签名检查失败保持，独立后继28项数值检查通过；F32倒数仍有RNE差异被half输出舍入掩盖，不能把最终值或sizeof当作中间指令精度证明。

原子计数只解决对应对象的原子性，不能代替普通 payload 的发布协议或全 grid barrier。旧版 relaxed 接口的限制不能泛化到 4.1 新增 order/mem_flags 接口；atomic_float 类型存在也不表示所有地址空间都支持同样运算，threadgroup add/sub 支持有版本条件。浮点原子累加仍可能因顺序变化而数值不同。本轮没有执行原子、跨组同步或 4.1 新接口实验。[规范 §6.16](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)

## 本地教学检查与未覆盖范围

2026-10-07，Apple M4/16GB、macOS 27.0（26A428）、MLX 0.31.2、NumPy 2.4.3、Python 3.14.3。先取得协作 GPU 锁，使用已有环境，无模型、安装、profiler 或计时。原始候选 `run` 因上述地址空间转换编译失败，保留源、异常及此前结果；失败发生时该 packed 输出不存在，不能说原始候选保存了全部数组。修正 header 的 `run-r2` 是独立目录，没有改动旧记录或环境。

预先定义 PCG64 seed=107，全部 F32。逐元素 N={1,3,4,31,32,33,255,256,257,4097}；输入为有限整数除16的 dyadic 值，CPU float64 算 `2*x+0.25`，要求精确相等。连续、offset1、step2 均测标量，packed 只测前两类，共50项；N=4 实际覆盖 constant packed 读取，N≥31 覆盖 device。每个布局从 GPU backing 建 view 并回读核对逻辑值；没有独立导出 GPU 指针地址，因此 offset1 不作为实际16字节不对齐地址的测量证据。

归约 M=3，D={1,31,32,33,127,128,129,1023,1024,1025}，TG={32,64,128}，连续/step2，normal/正负抵消含零行，共120项。CPU 从存储后的 F32 输入按 float64 求和，每行要求 `abs(error)≤1e-4+1e-6*abs(ref)`，零参考要求精确为零，另检查 shape/dtype/有限性。记录的 SIMD 组数分别为1/2/4；程序只检查其落在1..TG，不声称独立核验全部设备执行宽度。

**50项逐元素与120项归约数值均通过，整体运行仍为失败。** 归约最大绝对误差 `9.374e-6`，最大混合误差比 `0.06640`。30项元数据检查中29项通过，N=1/step2 的 stride 预期2、实际1，导致 `terminal.passed=false`、进程退出2；未调门或重跑抹去失败。单元素上 stride 不改变唯一元素的逻辑位置，但该观察足以否定“写了 step2 切片就一定向 kernel 暴露 stride2”的测试假设。类型 size/alignment 部分30项均一致。

独立 CPU 审查重新枚举50+120+30项，并从修正版全部170份保存输入/输出重新计算：逐元素公式逐值复算，归约用 `math.fsum`。170项数值仍通过，保存的参考和误差指标与独立计算一致；同时复现唯一元数据不符，保留总体失败。审查程序成功只表示记录与分析一致，不把原运行改成通过。

这组检查不包含 FP16/BF16、NaN/Inf、任意转置/负 stride、非完整归约线程组、原子、function constants 或矩阵/tensor 操作，也没有证明向量访存指令、最优 TG 或速度收益。来源 `local-msl-usage-20261007`，逻辑引用 `2026-10-07-msl-usage/derived/summary.json`；原始记录在仓库外且未公开，公开页面不能独立重放本机结果。

后续[低精度softmax](msl-softmax-lowp.md)另演示half/BF16存储的显式float读取、归约与OT窄化，以及低精度计算中的失败；它有独立输入、容差和原始位模式记录。

矩阵协作的统一参与、无 masked load/store、float/half 中间精度见 [MSL 矩阵乘](msl-matrix.md)，包含独立于本页的 324 项数值检查与失败。

在线max/normalizer状态、空mask与三阶段依赖的具体写法见 [MSL softmax](msl-softmax.md)；其全空行归零是显式合同，不由softmax数学式或原生API名字自动给出。

配合 [线程组与内存](metal-memory-threadgroups.md)、[MSL 优化](msl-optimization.md)、[自定义 RMSNorm](mlx-custom-rms.md)、[测量范围](measurement.md)使用。编译通过、正确性通过、生成预期指令、性能收益是四项不同证据。
