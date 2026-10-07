# MSL 量化 GEMV：4-bit 解包、参数复用与 affine 舍入

单 token decode 常需要在读取压缩权重时完成解包、应用 scale/bias 和归约。本页用原创 MSL 比较逐元素 affine 与按组提取参数的写法，并区分**量化损失、解量化舍入、点积累加误差**。格式和分派来自 MLX v0.31.2；本地结果只覆盖 M4 上的合成数据，不是模型质量、带宽或速度结论。

## 先写出 packed 格式合同

本例计算 `y[n]=sum_k x[k]*W[n,k]`。x、scale、bias 和输出均存 F32；压缩权重是 `uint32 q[N,K/8]`，每 word 八个 unsigned 4-bit 整数，按低位到高位排列。scale/bias 形状均为 `[N,K/G]`，沿每行的 K 维连续分组，`G∈{32,64,128}`，要求正 extents 和 `K%G==0`。[固定版本 API 绑定](https://github.com/ml-explore/mlx/blob/v0.31.2/python/src/ops.cpp#L4353)

```text
word = q[n, k/8]
code = (word >> (4*(k%8))) & 15
group = k/G
W_affine[n,k] = scale[n,group]*code + bias[n,group]
```

bias 是组内 affine 偏移，不是输出层 epilogue bias；code 不是带符号 INT4、FP4 或整数 zero-point。此格式不能仅凭“4-bit”名称与 GGUF Q4_0 互换。G 是八的倍数，故每个 word 不跨组；不能把参数索引误写为每 word 一组或只按 lane 索引。本实验的地址乘积均小于2^31，输入输出不重叠；推广前须在host校验shape、stride、buffer长度及索引类型范围，不把这段uint示例当任意大张量的寻址保证。

MLX 的实际 Metal 量化器包含 scale 符号选择、edge/q0 调整和参数转为 T 的舍入，不能从文档简化的正 scale/min-max 公式重建原 packed 数据。研究应保存**实际 q/scales/biases**；原 W 仅用于单独评估量化损失。[量化源码](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/quantized.h#L2432)

本合同每个逻辑权重的存储量为 `0.5+8/G` byte：4-bit code 加两份 F32 组参数。G32/64/128 分别为 0.75/0.625/0.5625 byte；不包含 x、输出、padding、allocator 或 step2 backing 的额外空间。它是格式容量计算，不是实测 DRAM 流量，也不说明更大 G 的质量可接受。

## 一个完整 SIMD-group 负责一行

以下原创 MLX custom kernel 函数体每 lane 读取一个 packed word、顺序消费其中八个 code，再推进32个 word。每个 word 只在源码中读一次，scale/bias 也在八个 code 外加载；编译后是否保留这种 load 数量尚未观测。

```metal
const uint lane=thread_index_in_simdgroup;
const uint n=threadgroup_position_in_grid.x;
float acc=0.0f;
for (uint j=lane; j<K/8; j+=32) {
    const uint word=q[n*q_strides[0]+j*q_strides[1]];
    const uint group=(8*j)/G;
    const float sc=s[n*s_strides[0]+group*s_strides[1]];
    const float bi=b[n*b_strides[0]+group*b_strides[1]];
    for (uint i=0; i<8; ++i) {
        const uint code=(word>>(4u*i))&0xFu;
        const float w=fma(sc,float(code),bi);
        acc=fma(w,x[(8*j+i)*x_strides[0]],acc);
    }
}
const float total=simd_sum(acc);
if (lane==0) y[n]=total;
```

`uint` 保持无符号移位；shift 只取0…28。显式 `fma` 在 affine 处只舍入一次，随后点积仍有自己的 F32 舍入。使用标量索引，不做地址空间或 vector pointer cast；小于8元素的 MLX 输入可能是 constant 指针，不能假定所有参数都是 device 指针。[MSL 4.1](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) §3.1、§6.10.2；[MLX 参数签名生成](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp)

构造时指定 `input_names=['x','q','s','b']`、`output_names=['y']`、`ensure_row_contiguous=False`，K/N/G 为模板整数。调用边界如下；`grid` 是线程总数，不是线程组数量：

```python
y = kernel(
    inputs=[x, q, scales, biases],
    template=[('K', K), ('N', N), ('G', G)],
    grid=(32*N, 1, 1), threadgroup=(32, 1, 1),
    output_shapes=[(N,)], output_dtypes=[mx.float32],
)[0]
```

本例固定32-lane并记录实际 SIMD 宽度。恰好 N 个完整组，n 天然有效；没有 packed word 的 lane 保持0，也必须到达 `simd_sum`。`K%G==0` 不保证 `K%256==0`，所以 word 循环仍要检查尾部。所有输入 stride 必须按其自身数组读取；连续布局并非关闭自动连续化后的默认保证。

这是寄存器内归约，不需要 threadgroup memory 或 barrier。`simd_sum` 只规定活跃线程求和，不规定浮点树；不能用任意 CPU XOR 树作为它的逐位 oracle。若改成多个输出共享一组或缓存到 threadgroup，需另行确定参与集合、尾部读取和同步合同，不能直接沿用本例的 n 有效性推理。

## 组参数提取：少做 affine，改变舍入位置

实数中同一组的参数可以提到求和之外：

```text
sum_k ((s*q[k]+b)*x[k]) = s*sum_k(q[k]*x[k]) + b*sum_k(x[k])
```

只对同一个 quantization group 成立；不能越过参数边界。对照设计使 `group_literal` 和 `group_factored` 使用完全相同的组到 lane 映射：`group=lane+32*t`，组内按 word、低 nibble 到高 nibble 递增，最后同样 `simd_sum`。

| 路径 | lane 内的计算 | 需要付出的代价或检查 |
|---|---|---|
| word_literal | 每 word 八次显式 affine FMA，再逐项 dot FMA | 同一组参数可被多个 lane 重读；完整32-word块有较多 K 并行 |
| group_literal | 一整个组在一个 lane 内，先形成 group dot，再加到 acc | 参数在 G 项外加载；小 K/G 时工作集中到少数 lane，组内依赖更长 |
| group_factored | 分别积累 `dotq=fma(code,x,dotq)`、`sumx+=x`，组末 `acc+=fma(s,dotq,b*sumx)` | 减少逐元素 affine 表达式，增加一条 sumx 累加链；舍入与抵消位置改变 |

更少源码乘加不等于更快，也不等于更准确。尤其 `b*sumx` 可能先舍入；当它与 `s*dotq` 很接近时，残差相对误差会放大。把 accumulator 写为 float 不能取消这个问题；本页三个原创路径已经全部使用 float。改变组内工作分配还会改变并行度，不能将 word/group 比较称为只改变了解包方式。

MLX v0.31.2 的 QMV 源码也对 bias 项做因式分解，在 lane chunk 上应用参数，而非逐元素先生成 T 权重；它的映射和本例整组分配不同。默认 float 局部值不代表所有由 T 输入组成的子表达式都先升 float，输出最终仍写 T。[QMV 的 load_vector/qdot](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/quantized.h)

## 分派和布局会改变对照含义

[MLX v0.31.2 host](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/quantized.cpp)先处理矩阵连续性，再确定合并后的 M。对本页 `x[1,K] @ W[N,K].T`、4-bit、M=1：K64/128 优先 quad；其余 N%8=0 且 K%512=0 选择 fast，否则普通 QMV。这是源码选择条件，未用 profiler 证明当前 binary 实际执行的 kernel；tag 也可移动。

普通 QMV 与 fast 的每 lane K 工作量不同，quad 又改用四 lane 小组；不能把某个路线的 tile 参数当通用公式。多行 M 的阈值依设备/形状而变；共享权重的 batch 合并也可能改变路线。原生入口会对不符合末两维行连续条件的 x、q、scale、bias 复制，所以 step2 custom 与原生输入相同不代表物理访存相同。

`transpose=False` 的 group 方向和 QVM 路线另有合同。[旧 QVM 多行失败](quantized-matmul-validation.md)仍独立保留，本页的 M1/transpose=True 检查既未复验也未修复它。实际部署的 packed、每次 dequant、常驻 dense 时间与内存取舍见 [三路径比较](mlx-qmm-path-comparison.md)，不能从这轮不计时的数值实验挑性能赢家。

## 本地检查与独立参考

2026-10-07，M4/16GB、macOS27.0（26A428）、MLX0.31.2、NumPy2.4.3、Python3.14.3，`MLX_ENABLE_TF32` 未设置。取得协作 GPU 锁后执行，无模型、安装或环境修改。对每个 G32/64/128，选 `(K,N)=(G,3),(3G,33),(1024,64),(8192,5)`，共12组形状；包括不完整32-word块、少数组参数、N尾部及较长K。

每个形状使用三类数据、两种布局，共72组保存输入：

- `quantized_normal`：PCG64 seed503，x 标准差0.5、原 W 标准差0.1，先存 F32，再保存 MLX 实际量化产生的 q/s/b。
- `codes_sentinel`：code按列、行、group位置取0…15；scale正负交替，scale/bias/x均为dyadic值，输出行1的scale/bias置零。它用于检查解包顺序、跨word/group索引、负scale与精确零输出。
- `affine_cancellation`：code全为7，s=`F32(1/3)`、b=`F32(-7/3)`、x=`1+2^-10`。它是合法的合成 affine 参数压力输入，不代表常见模型分布，也没有可报告量化损失的原 W。

连续布局与 x/q/s/b 全部 step2 的 GPU view 保存相同逻辑值，未使用位置填入不同标记。shader元数据记录实际stride；只对extent>1的轴断言预期stride，singleton轴仍保留实际值。72项元数据均通过，SIMD宽度32、每组1个SIMD-group。实验只含有界有限值，不覆盖subnormal、NaN/Inf输入或隐藏越界读取检测。

共同参考先从保存的packed参数构造 F64 affine，再单次舍入为 F32 的 D，对 D 和实际 x 做高精度点积。另保留未物化 F32 的 affine参考、`sum(abs(D*x))`；仅normal另外计算原 W 的点积。验收门预定为 `abs(error)≤1e-4+1e-5*abs(ref)`，并检查finite、shape/dtype及零参考精确为零。原生API并未承诺这个舍入顺序或误差界，这是实验共同质量门。

原创显式FMA解码器与原生 `mx.dequantize` 的全部144份F32输出均逐元素精确等于D。原生decode精确相等是本轮观测，契约预先将它设为诊断项，而非API必须保证的逐位合同。`dense_native` 使用实际原生解码结果做 `mx.matmul`；`native_qmm` 使用 `x[None,:]`、`transpose=True`、bits4和对应G，输出F32。

| 路径 | 通过 /72 | 最大绝对误差 | 最大混合误差比 |
|---|---:|---:|---:|
| word_literal | 72 | 2.479141e-06 | 0.01797553 |
| group_literal | 72 | 1.561043e-06 | 0.01454584 |
| group_factored | 66 | 0.0006504059 | 6.503264 |
| native_qmm | 66 | 0.0006504059 | 6.503264 |
| dense_native | 72 | 7.265784e-07 | 0.006770276 |

混合比是逐元素误差除以该元素容差，≤1才通过；每列最大值不必来自同一个位置。360项点积检查中348项通过、12项失败；144项decode和72项metadata完成，整体退出2、`passed=false`，没有重跑覆盖。失败均是K8192、N5、affine_cancellation：三种G×两种布局×两条路径，每项5个输出都超限且有限。normal与sentinel全过门；不能由此推断任意量化权重都合格。

独立CPU审查从保存数组重新逐word解包，使用 host `fmaf` 核对单次F32舍入、`math.fsum` 重算36个不同输入的参考，另36个布局副本在逐数组相等后复用。全部360份点积输出、144份decode、72项metadata记录和终态计数一致；参考与原F64 dot的最大差约3.11e-15，没有改变任何判定。180组跨布局输出对照也逐元素相等，未发现非有限输出或精确零违反。审查还核对sentinel构造，12组独立normal输入有11组含负scale。

记录完整性审查通过，原实验数值合同仍失败。元数据审查核对的是保存的shader报告，NPZ中的逻辑数组自身不能独立证明物理GPU stride；这些检查也没有建立生产输入范围内的质量保证。

## 将数值结果用于优化选择

对六组失败输入，以CPU `fmaf` 和精确有理数逐步核对原创 `group_factored`。实际参数为s≈0.3333333432674408、b≈−2.3333332538604736，解码残差为 `5/33554432≈1.490116119e-7`。本数据中它可被F32精确表示，因此未物化的affine参考与物化D的参考相等。

以G32说明误差发生的位置；所有行、组相同：

| 中间量 | 值 |
|---|---:|
| sumx | 32.03125 |
| dotq | 224.21875 |
| 正确的每组点积 | 4.773028194904327e-06 |
| `RN32(b*sumx)`引入的正误差 | 2.5406479835510254e-06 |
| `fma(s,dotq,RN32(b*sumx))` | 7.313676178455353e-06 |

在这些保存数据中，sumx/dotq每步、上述FMA的后续舍入和每lane的组间加法都没有另增误差；总误差恰好是256组乘上bias乘积舍入误差。G64/G128的单组量同比放大而组数减少，得到相同结果。32个lane末值一致，0…32个lane值之和都能由F32精确表示，所以这次结论不依赖假定某一 `simd_sum` 加法树。

| K8192，每个输出 | 值 |
|---|---:|
| affine / materialized参考 | 0.0012218952178955078 |
| CPU因式分解模型、原创group_factored输出 | 0.0018723011016845703 |
| 有符号误差 | +0.0006504058837890625 |
| 原定容差 | 0.00010001221895217896 |

模型逐位复现全部30个原创失败输出；相对参考的误差约53.23%。三个逐项/物化路径在这六组数据上精确等于参考。native_qmm恰好输出同样的错误值，但本轮未观测其binary分派或指令，**数值相等不证明执行了原创模型的同一运算序列**。

这里抵消发生在affine内部的scale项与bias项，最终所有D*x项为正，并不是长dot里正负乘积相消。没有原 W，不能把偏差解释为原始权重量化损失；所有decode又都一致，故它也不是解包或物化误差。小K的某些抵消结果即使相对误差很大，仍可能落在绝对容差内；“通过这个混合门”不等于保证小残差的百分比误差。

因此，参数提取应作为有数值代价的候选：固定实际packed参数、decode舍入、输出dtype和共同门，再比较它与逐项FMA。失败候选保留并返回算术设计阶段；不扩大容差、按已知失败shape静默回退，或用吞吐数字掩盖误差。已经通过的两种原创逐项路径只是这72组数据内可继续研究的起点，尚未得到性能验收或通用误差界。

来源 `local-msl-qmv-20261007`，逻辑引用 `2026-10-07-msl-qmv/derived/summary.json`。实际输入、packed 参数、逐项输出、失败、MSL/Python 源程序和独立 CPU 审查保存在仓库外，未随公开库发布。没有测量本页候选的 GPU 时间、指令、访存、寄存器压力、模型质量或端到端 decode。新目标应重新固定格式、精度、shape、布局与共同质量门，再比较通过候选的实际代价。
