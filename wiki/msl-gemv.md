# MSL GEMV：SIMD 分工、权重布局与归约顺序

[English companion](en/msl-gemv.md)

单 token 的矩阵向量乘需要在输出并行和 K 归约之间分配 lane；8×8矩阵 tile 并非唯一选择。本页给出原创标量/SIMD写法、MLX与llama.cpp的实现线索，以及M4上的有限检查。**F16存储转float计算仍可能因归约顺序损失精度**：本轮8-lane分组在长K抵消输入上失败，原始失败保留。没有测量本页候选的速度，也不将源码中的连续地址或复用直接等同于带宽收益。

## 先固定算式、物理布局与输出精度

记 `y[n]=sum_k x[k]*W[n,k]`，输入为实际存储的F16，输出F32。`W[N,K]` 连续时，相邻K在内存相邻；从 `[K,N]` backing 转置得到同一逻辑W时，相邻输出n在内存相邻。step2 view又是不同的寻址合同。相同shape和数值不等于相同访问代价；自写kernel须使用真实stride，原生框架也可能先复制。

[MLX v0.31.2 host](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/matmul.cpp)在布局处理和batch合并后才以 `min(M,N)==1` 进入GEMV。共享权重的多个单token请求可能被并入M、转走GEMM。对 `x[1,K] @ B[K,N]`，物理B连续通常对应 `gemv_t`，B为行主序W的转置view通常对应 `gemv`；名字由内部矩阵方向决定。通用step2输入不自动成为原生kernel可直接读的布局。

[MLX GEMV源码](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/gemv.metal)把32个lane分成SM×SN，再让每线程处理TM×TN元素。`gemv` 的SN沿K、SM沿输出；`gemv_t` 的SN沿输出、SM沿K。它们的half输入默认float累加，最终仍写回T，即half输出。**把原生half输出再转F32不能恢复末端舍入**；本轮原生数值对照明确先把两个输入升F32再matmul，不能当作与F16存储custom具有相同流量和转换成本的性能基线。tag链接可移动，未证明安装binary与源码逐字一致。

## 四种原创工作划分

| 路径 | 每组输出与K分工 | 代价或精度问题 |
|---|---|---|
| serial_output | 32线程组，每lane一个输出，顺序遍历全部K | 无collective；W转置布局中同轮lane地址相邻；长K串行依赖较长 |
| simd32 | 一个32-lane SIMD-group负责一个输出，lane取 `k=lane+32t` | `simd_sum` 合并；W行连续时同轮lane沿K相邻 |
| subgroups8 | 一组4个输出，每输出8个lane，取 `k=sublane+8t` | XOR shuffle仅在各8-lane集合内求和；相同符号可能集中到同一lane，局部和先变大再相消 |
| two_level128 | 一个128线程组负责一个输出，取 `k=tid+128t` | 4个SIMD partial、16 B声明scratch和全组barrier，再由SG0合并；不是多dispatch split-K |

这些是算法分工，不能据此推断实际load指令、DRAM事务、编译后scratch或occupancy。与 [矩阵fragment](msl-matrix.md)不同，这里明确把标量工作分配给lane，不依赖未规定的矩阵元素到lane映射。

以下是 `subgroups8` 的原始函数体；它用于展示参与和归约规则，**在下文长K抵消输入上未通过质量门，不能作为已验收的通用方案**。

```metal
const uint lane=thread_index_in_simdgroup;
const uint n=4*threadgroup_position_in_grid.x+lane/8;
const uint sublane=lane%8;
float acc=0.0f;
if (n<N) {
    for (uint k=sublane; k<K; k+=8)
        acc=fma(float(x[k*x_strides[0]]),
                float(w[n*w_strides[0]+k*w_strides[1]]),acc);
}
acc+=simd_shuffle_xor(acc,4);
acc+=simd_shuffle_xor(acc,2);
acc+=simd_shuffle_xor(acc,1);
if (sublane==0 && n<N) y[n]=acc;
```

XOR掩码小于8，不跨越对齐的8-lane集合；每步由所有32个lane执行。无效输出槽保持acc=0且继续参与，最后不写出。K尾部只在标量读取处检查。按 [MSL 4.1 §6.10.2](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)，shuffle的mask在整个SIMD-group一致，`simd_sum` 汇总活跃线程；这些接口不提供“与顺序F64点积相同”的数值承诺。本例固定全组参与合同，尤其不能让仍将被shuffle读取的lane提前返回。

MLX custom kernel 用 `input_names=['x','w']`、`output_names=['y']`、`ensure_row_contiguous=False`，模板K/N，输出shape `(N,)`、dtype `mx.float32`。本例 `grid=(32*ceil(N/4),1,1)`、`threadgroup=(32,1,1)`；grid是总线程数。每次乘法前分别转float，`float(xv*wv)`会先在表达式的原类型中相乘，不能代替两个显式转换。K=7的x在MLX0.31.2是constant指针，本例只标量索引，未进行device-only cast。[生成签名与属性注入](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp)

两级归约则先让每个SIMD-group计算自己的 `simd_sum(acc)`，只有其lane0写 `partial[sg]`，全部128线程到达 `threadgroup_barrier(mem_flags::mem_threadgroup)`。之后SG0的lane0..3读取 `partial[lane]`，其余lane贡献0，32个lane共同 `simd_sum`，lane0写输出。不能误读成 `partial[sg]`；第二阶段分支必须对该SIMD-group统一。scratch不复用时本例没有第二个全组barrier。simd32/two_level128分别发恰好N个完整组，故n天然有效；serial_output按 `ceil(N/32)` 发完整组并逐lane保护输出。

## 上游如何选择这些维度

MLX v0.31.2 的 `gemv` 优先对K≤64采用SM/SN=8/4；其后K≥16O时采用BN=8（O为输出长度），扩大组内K并行。仅当前两个条件均未命中时，O≥4096取BM=8，否则BM=4。默认每线程TM/TN=4，极小输出另有修正。`gemv_t` 又有自己的O阈值和方向；`nc`表示batch寻址方式，不承诺矩阵内部任意stride可直接访问。BN的含义依路线而变，不能把参数名当通用策略。[host选择](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/matmul.cpp#L1043)

llama.cpp固定提交 `42b021b4dc42be573f1e1463528532fc8294c650` 的通用dense GEMV在K<32时每lane顺序计算一行；K≥32时整个线程组协作两个输出行，SIMD-group数为 `min(4,ceil(K/128))`，复用同一x片段。在K≥32且K%4=0时选择普通向量类型的 `_4`，先转float4后dot；其他长K走scalar，均有K尾部。共享partial再合并的声明空间为256 B，不能把它与本例16 B直接比较成资源胜负。[kernel](https://github.com/ggml-org/llama.cpp/blob/42b021b4dc42be573f1e1463528532fc8294c650/ggml/src/ggml-metal/kernels/mul_mv.metal)、[pipeline](https://github.com/ggml-org/llama.cpp/blob/42b021b4dc42be573f1e1463528532fc8294c650/ggml/src/ggml-metal/ggml-metal-device.cpp)

该上游通用路径读取行内连续K，行/批偏移通过byte strides表达，不是任意step2模板。short在读取前保护行边界；长路径计算两行、写出时才保护输出，这段源码本身不足以证明奇数输出行的物理输入读取有效性。本次未审查全局分配/padding，不能判定上游整体越界；原创例直接保护每次输入读取。普通half4/float4也不等于packed类型或已证实的向量load，K整除条件不替代任意起始offset的对齐证明。

还要先通过更外层的FWHT/MMA/MM与small-batch选择，才能到达通用GEMV；本页没有审查所有predicate定义，不能据K值宣称实际选中了上述kernel。源码另有小batch复用W服务多个x的 `mul_mv_ext`，量化Q4_0路径则融合block解码与点积，不能只替换dense示例的指针类型。[外层dispatch](https://github.com/ggml-org/llama.cpp/blob/42b021b4dc42be573f1e1463528532fc8294c650/ggml/src/ggml-metal/ggml-metal-ops.cpp)

## 本地检查：长归约揭示的精度边界

2026-10-07，M4/16GB、macOS27.0（26A428）、MLX0.31.2、NumPy2.4.3、Python3.14.3，`MLX_ENABLE_TF32`未设置。取得协作GPU锁后运行，无模型或环境修改。12组 `(K,N)` 是 `(7,5),(31,33),(32,32),(33,31),(63,3),(64,4),(65,5),(257,65),(1024,129),(4096,128),(8193,17),(65537,5)`；最后一组是长归约压力输入，不是典型decode性能配置。

PCG64 seed401，normal标准差0.5后存F16；cancellation令x按K交替16.125/-16、W每行系数为 `1+(n%17)/32`、第1行置零；wide令x=256、W按K交替±256。每组用行连续W、转置GPU view、x/W均step2三种布局，保存实际回读值与shader stride。共同参考为已存F16数据转F64的点积，门预定为 `abs(error)≤1e-4+1e-5*abs(ref)`，另要求有限、shape/dtype和零参考精确为零。没有为任何路径更改门。

五路径共540项，536项通过、4项失败；216项元数据门通过，确认SIMD宽度32、32/128线程组及实际stride。完整运行退出2，`passed=false`。

| 路径 | 通过 /108 | 最大绝对误差 | 最大混合误差比 |
|---|---:|---:|---:|
| serial_output | 108 | 0.000334447 | 0.8659217 |
| simd32 | 108 | 2.191504e-05 | 0.09164022 |
| subgroups8 | 105 | 69.98828 | 1552.66 |
| two_level128 | 108 | 2.876895e-05 | 0.09128173 |
| native_promote | 107 | 0.0003225006 | 1.02327 |

混合比是逐元素绝对误差除以其容差，≤1才通过；最大绝对误差与最大混合比可能来自不同位置。所有失败均为有限数值误差，无overflow或零参考违反。subgroups8在K=65537的cancellation、三种布局都失败于输出2/3；native_promote在同形状normal的w_transposed布局、输出4轻微超限：误差约0.000322501，容差约0.000315167。这是本实验质量门失败，不能据此判定框架违反其公开精度契约。

独立CPU审查从108组保存输入重算36个不同的 `math.fsum` oracle，另72组在数组相等后复用；540份输出、指标及失败集合全部吻合，oracle/原始指标差异为0。四个custom路线各自跨布局输出相同；native在部分同值布局对照中输出不同。后者包含显式升F32的复制/布局处理，未记录其升精度后stride或底层dispatch，不能把差异直接归因于某个GEMV变体。审查完整性通过与原实验数值失败是两件事。


## 从失败定位到新的加法顺序

取保存的K=65537抵消输入，以CPU模拟每lane的F32逐次累加，再按XOR4/2/1合并。F16输入乘积在这个被检查的数据集内能被F32精确表示，故乘积先舍入不改变本次FMA模拟。该模型逐位复现全部5个GPU输出，三种布局一致：

| 输出索引 | F64参考 | 原subgroups8输出 | 有符号误差 |
|---:|---:|---:|---:|
| 0 | 4112.125 | 4112.125 | 0 |
| 1 | 0 | 0 | 0 |
| 2 | 4369.1328125 | 4352.25 | −16.8828125 |
| 3 | 4497.63671875 | 4567.625 | +69.98828125 |
| 4 | 4626.140625 | 4626.125 | −0.015625 |

stride8把正项分给偶数sublane、负项分给奇数sublane，每lane先处理8192次同号加法，sublane0还有最后一项。输出2的误差全来自局部累加；输出3的局部累加误差合计+69.94140625，最终XOR树再增加+0.046875。输出4虽有误差，仍在原门内。这个结果说明减少同一lane的局部和精度损失与最终树形归约同样重要，不能只把误差归结为一次shuffle。

这是对有限保存数据的算术复现，没有观测GPU指令。另模拟的32/128宽XOR树与实际 `simd_sum` 输出略有不同，即使均在门内，也不能据此认定 `simd_sum` 的内部顺序。native的单项超限没有由这个模型解释，原因仍未定。


在独立 `paired2` 运行中，保留每组4输出、每输出8lane、同一shuffle树与F32类型，仅把每lane的K循环改为相邻两项：

```metal
for (uint k=2*sublane; k<K; k+=16) {
    acc=fma(float(x[k*x_strides[0]]),
            float(w[n*w_strides[0]+k*w_strides[1]]),acc);
    if (k+1<K)
        acc=fma(float(x[(k+1)*x_strides[0]]),
                float(w[n*w_strides[0]+(k+1)*w_strides[1]]),acc);
}
```

这仍位于 `if(n<N)` 中，后续三个XOR和store沿用原例。每个K元素恰好被访问一次；不是先在局部把两个乘积相加再加入acc，而是两次连续FMA，舍入顺序不同。算法没有读取符号来选分支，也没有对失败shape做回退；相邻正负项在本构造中更早相消，避免先形成长串同号局部和。

重用并重新保存108组原输入，108项新输出及108项元数据均过相同门，独立CPU审查确认输入相同、oracle与指标一致。normal最大绝对误差约0.000124951、最大混合比0.375720；cancellation和wide均精确等于参考，包括原来的5个长K抵消输出。新候选的 `passed=true` **不改变原五路径运行的 `passed=false`**，也没有消除native那项独立失败。


把相邻工作放在同一lane是可比较的数值与访存设计选择，不是通用抵消修复。不同符号分布、K、更低精度及编译器重排仍需新证据；本例通过范围不证明任意F16输入的误差界。下一轮性能比较还要统一输入/输出dtype、布局转换范围与输出存活期，并用目标shape和可取得的profiler证据判断多输出复用是否支付了寄存器与同步代价。不能把本轮数值改进称为速度改进，不能用该候选隐藏未通过共同门的旧路径。


4-bit压缩权重另见 [MSL量化GEMV](msl-quantized-gemv.md)：uint32低位解包、组参数索引与bias factoring的舍入失败。它使用F32激活/参数/输出和新的量化合同，不是把本页half指针替换为packed指针。

来源 `local-msl-gemv-20261007`，逻辑引用 `2026-10-07-msl-gemv/derived/summary.json`。所有输入、输出、失败、源程序和CPU审查保存在仓库外，未随公开库发布。未测实际GPU时间、指令、缓存、带宽、模型质量或端到端decode；本页说明可验证的写法与数值取舍，不提供生产参数推荐。
