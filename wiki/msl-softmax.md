# MSL softmax：在线归一化、分块合并与 mask 契约

softmax 优化需要一起决定输入读几遍、保留多少局部值、如何合并归约，以及空行输出什么。本页给出原创 MSL 写法、MLX v0.31.2 的源码线索和 M4 有限检查。它只输出给定 logits 的概率，没有实现 QK/PV、KV cache 或完整 FlashAttention，原数值运行未计时；后续[独立计时页](msl-softmax-timing.md)记录同一候选的主机完成成本与统计差异。

## 先定义有效集合，再谈稳定计算

本例 x 为 F32 `[R,K]`，mask 为同形状 uint32 0/1，输出为 F32。只对 mask=1 的集合 A 计算 `m=max(x[A])`、`l=sum(exp(x[A]-m))`、`y[A]=exp(x[A]-m)/l`；mask=0 的位置输出正零。A 为空时整行输出正零，这是**本探针额外规定的语义**，不是从 standalone softmax 或 SDPA 推定的承诺。

输入 logits 全部有限且位于 `[-8192,8192]`，包括被 mask 的位置；只测正 R/K、非重叠输出和下述有界索引。自写 kernel 使用每个数组自己的真实 stride。原型的索引乘积小于2^31；更大张量、负stride、alias、NaN/Inf输入需要新合同。mask 已由调用方构造，本例不会把 `k<=row` 当成通用 causal mask，尤其不能替代有 query offset 的右下对齐规则。

最大值归约的初值取域外有限数 −16385。它只用于本有限输入域的 neutral 值，不能推广为任意 logits 的“负无穷替代”。同时记录有效元素个数；没有数据的lane照常参加归约，全空行跳过指数/除法，明确写0。不能先产生 `-inf-(-inf)` 或 `0/0`，再希望乘0消除NaN。

## 以最大值为原点的状态可以合并

对一个非空块保留 `(m,l,count)`，l 是相对于该块最大值的指数和。两个块的实数合并为：

```text
if a.count == 0: return b
if b.count == 0: return a
m = max(a.m, b.m)
l = a.l * exp(a.m-m) + b.l * exp(b.m-m)
count = a.count + b.count
```

乘上的指数因子把两块的归一化原点移到同一个 m。空块用 `(0,0,0)`，其中m没有数学最大值含义，仅由count决定是否读取。分支必须保护无效计算；不能把可能已求值的 `select` 参数当成短路保护。

[Online normalizer calculation for softmax，v2](https://arxiv.org/pdf/1805.02867v2) 的 Algorithm 3 给出在线归一化，§3.1给出并行状态合并；Algorithm 4另含Top-K，并不是下面的split256。论文的实数合并规律不保证不同F32归约树逐位相同。count、mask、全空行归零是本例扩展，没有把论文的其他硬件速度结果移用到Metal。

以下原创函数体把一行分给一个32-lane SIMD-group，每lane按 `k=lane+32t` 在线更新，再统一原点、合并质量，最后重读输入写概率：

```metal
const uint lane=thread_index_in_simdgroup;
const uint row=threadgroup_position_in_grid.x;
float m=0.0f,l=0.0f; uint count=0;
for (uint k=lane; k<K; k+=32) {
    if (mask[row*mask_strides[0]+k*mask_strides[1]]!=0) {
        const float v=x[row*x_strides[0]+k*x_strides[1]];
        if (count==0) {m=v;l=1.0f;}
        else {
            const float next=max(m,v);
            l=l*precise::exp(m-next)+precise::exp(v-next);
            m=next;
        }
        ++count;
    }
}
const float common=simd_max(count!=0 ? m : -16385.0f);
float mass=0.0f;
if (count!=0) mass=l*precise::exp(m-common);
mass=simd_sum(mass); count=simd_sum(count);
for (uint k=lane; k<K; k+=32) {
    float value=0.0f;
    if (count!=0 && mask[row*mask_strides[0]+k*mask_strides[1]]!=0)
        value=precise::exp(x[row*x_strides[0]+k*x_strides[1]]-common)/mass;
    y[row*K+k]=value;
}
```

MLX `metal_kernel` 使用 `input_names=['x','mask']`、`output_names=['y']`、`ensure_row_contiguous=False`，整数模板K，输出 `(R,K)`/`mx.float32`；发 `grid=(32*R,1,1)`、`threadgroup=(32,1,1)`。grid为线程总数，恰好R个完整组。只要有有效输入，最大值对应的指数为1，因此数学分母不为0；空行则依count写0。所有32lane执行三个collective，K尾部只限制标量访问，不能让仍被归约使用的lane提前退出。

## 三类工作分解改变不同成本

| 路径 | 源码中的工作安排 | 取舍 |
|---|---|---|
| sg32_precise / sg32_fast | 分开max、指数和、输出三次遍历；只改变指数函数命名空间 | 每个有效logit读三次、指数算两次；归约简单，长K的lane内串行工作增多 |
| tg128_precise | 128线程分担一行，先各SIMD归约，再通过共享partial合并 | 缩短每线程K循环；增加跨SIMD共享与四次全组barrier |
| online32_precise | 上述在线状态加输出遍历，两遍读取 | 省独立max读取，增加最大值变化时的重缩放指数与依赖链 |
| split256_precise | 256列块各自产生状态；合并状态；最后写输出 | 增加长行的独立线程组；三个kernel、临时状态及跨阶段依赖都需计入代价 |

这些是原始源程序中的读数和表达式，不是编译后load、DRAM事务、寄存器或速度测量。私有数组保留输入/指数又是不同候选，会减少重读而增加局部存储生命周期；不能仅凭源码存在 `thread` 数组就断言它无spill。

本例TG128只声明四份SIMD partial。四个leader先写max/count，全组barrier；SG0的32lane仅在 `lane<4` 时读partial，其余贡献neutral，leader发布全行max/count，再全组barrier。随后四组计算/写sum，barrier后由SG0合并并发布全行sum，第四次barrier后所有线程写概率。全空行仍经过相同的barrier路径。max/count与sum用不同共享存储，没有在读者结束前复用同一数组。

split256 的第一阶段每块32lane，做局部max/count和指数和；每个尾块、空块都写完整状态。第二阶段每行一个SIMD-group，lane按 `chunk=lane+32t` 合并，再用统一max重缩放并 `simd_sum`；第三阶段按扁平输出索引应用全行状态。`pm/pl/pc` 作为第二阶段真实输入，`gm/gl/gc` 作为第三阶段真实输入，数据依赖表达执行顺序。**threadgroup barrier不跨线程组或kernel**；本例没有实现自旋或全grid屏障。K8193产生33块，覆盖lane0的第二次状态合并。

## `precise` 要说明是哪一个接口

[MSL 4.1](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)区分 `fast::exp`、`precise::exp` 和受编译设置影响的无前缀写法。float precise exp允许至多4 ULP误差，不是正确舍入承诺；fast精度表又不同。显式precise不会把减法、归约、除法和重结合一起变成CPU等价计算，`simd_sum` 也不规定浮点树。阅读4.1规范不代表本机以全部4.1特性编译；本轮记录实际源码拼写，未观察JIT编译选项或机器指令。

MLX v0.31.2 [Python绑定](https://github.com/ml-explore/mlx/blob/v0.31.2/python/src/ops.cpp#L3021)实际接受关键字 `precise`，默认false，但相邻手写签名遗漏它；安装包的 `.pyi` 因此也不能单独作为“不支持该参数”的证据。其 [Metal源码](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/softmax.h#L3)的 `softmax_exp` 始终调用 `fast::exp`，不能把API参数名解释为选择MSL `precise::exp`。

该源码的输入/输出是T，内部统计与数组是AccT，最后转回T。后续补读 [JIT factory](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/jit_kernels.cpp#L290) 和 [静态实例化](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/softmax.metal#L12)，核实 AccT 取 `precise ? float32 : output_dtype`：F32两种flag都是float，half/BF16在false时使用对应类型、true时使用float，输出类型不变。这是源码变量/存储类型，不等于逐条硬件指令的精度证据。本页原数值运行只测F32；后续[低精度softmax](msl-softmax-lowp.md)另保留含F32对照的330项低精度研究检查、26项失败与输入信息损失，未把原F32记录扩写成低精度验收；F32调用传precise=True也不表示启用独立于false的源码累加类型。

## MLX如何在保存与重读之间选择

[v0.31.2 host](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/softmax.cpp)以末维4096为界：较短行用block，较长行用looped。block每线程保存 `ld[N_READS]`，先存输入后改存指数，再直接写输出；looped用有界小数组在线更新统计，归约后重读输入。线程数分别按N_READS向完整SIMD取整和使用pipeline上限。补读 [defines.h](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/defines.h#L14) 核实 `N_READS=4`；固定tag的block组大小为 `32*ceil(K/128)`，故K128对应32线程、K257对应96。K8193走looped；未核实实际pipeline上限或安装binary分派。

原生入口要求contiguous标志且末维stride=1，否则可能先复制。本地native对照还含显式mask操作，因此与custom step2路径不构成相同物理访存的性能A/B。源码tag可移动，也不能当作已确认与安装binary逐字一致。

一个迁移时容易遗漏的前提是：looped源码使用32个共享槽，leader写自己的SIMD槽，随后lane读取相应槽；不能直接复制到只有4个SIMD-group的TG128，再读取全部32槽。本例用 `lane<4` 明确屏蔽无效槽。该审查说明复制代码时必须重建初始化/参与合同，不据此判定上游实际运行存在未初始化读取。

## 本地数值检查

2026-10-07，M4/16GB、macOS27.0（26A428）、MLX0.31.2、NumPy2.4.3、Python3.14.3，`MLX_ENABLE_TF32` 未设置。取得协作GPU锁后用已有环境运行，无模型或安装。K取 `1,7,31,32,33,127,128,129,255,256,257,8193`；每个K有8行，分别覆盖：全有效uniform、全mask、末列唯一有效、prefix有效、首末有效且中间空块、有限4096峰值、seed607的随机dyadic行、该随机行精确加4096。

随机行由PCG64取−128…128整数后除16，mask按列模3构造且首列有效。平移前后保存的F32差值全部精确为4096，mask相同，因此能够检查平移关系；任意大平移可能先改变存储数值，不能据数学不变性要求输出相同。prefix仅是给定mask样本，未声称覆盖不同causal对齐规则。

连续及x/mask均step2的GPU view共24组输入、192行。未使用backing位置填入不同logit和mask标记，按数组自身stride寻址；extent=1时不强制要求被规范化的stride值，仍记录实际元数据。48项metadata检查全部通过，确认32-lane及1/4个SIMD-group的配置。

CPU参考从已保存F32输入出发，只取有效项，使用 host `math.exp` 和 `math.fsum`，不调用框架softmax，也不模拟候选归约树。所有路径共用预定门：逐元素 `abs(error)≤2e-6+2e-5*abs(ref)`，非空行的和与1相差≤2e-5，行L1误差≤5e-5；另要求F32/shape、finite、非负、mask位置与全mask行正零，以及平移对照通过同一混合门。

native对照明确执行 `where(mask,x,-16385)` → `mx.softmax(...,axis=-1,precise=True)` → `where(mask,p,0)`。有限填充值只在本有界域内与有效logit充分分离；末次where定义mask位置归零，全空行的中间softmax可以是uniform。它不是任意输入下的负无穷语义，也不是已证实融合的masked-softmax接口。本机24次 `precise=True` 调用均完成，补充了签名遗漏下的参数可用性证据。

| 路径 | 通过 /24 | 最大误差/逐元素容差 | 最大行和误差 |
|---|---:|---:|---:|
| sg32_precise | 24 | 0.004828739 | 1.026181e-07 |
| sg32_fast | 24 | 0.002893657 | 7.664077e-08 |
| tg128_precise | 24 | 0.004828739 | 1.026181e-07 |
| online32_precise | 24 | 0.004828739 | 1.026181e-07 |
| split256_precise | 24 | 0.004828739 | 1.026181e-07 |
| native_precise_flag | 24 | 0.003430696 | 1.206154e-07 |

144项输出全部通过，整体退出0、`passed=true`。全路径最大概率绝对误差约5.848593e-8，最大行L1误差约1.206459e-7；没有非有限、负概率或被mask位置的非正零结果。表中每列最大值可来自不同输入，综合误差更小不意味着某指数函数在任意参数上更精确，更不建立速度排序。

24份split状态文件保存全部局部/全局max、指数和、count，max/count与独立参考精确相等，空状态符合 `(0,0,0)`。指数和按另行预定的 `2e-5+2e-5*abs(ref)` 门通过：局部最大绝对误差约5.678940e-6，全局约5.134546e-5。这验证了本批数据中的状态合并，包括空块和33块情形；没有用最终概率看似正常来替代中间状态检查。

独立CPU审查重新构造并核对声明的输入模式，用保存数组计算12组参考，另外12组仅在逻辑输入相等后复用。全部144输出、24状态文件、48元数据及终态一致，保存的参考差为0。144项平移对照和72组跨布局输出还观察到逐位相同，这是额外诊断结果，不是推广后的bitwise保证；逻辑NPZ也不能独立证明历史物理stride或无隐藏越界读取。

当前可采用的知识是明确的空集、归约和依赖写法，以及这些F32候选在共同门内通过的有限证据。后续[计时比较](msl-softmax-timing.md)区分单kernel在线路径与三阶段split路径，计入临时状态和mask包装，同时保留布局及输出消费边界。不能把“少一遍源码读取”“更多线程组”或本轮全部通过直接写成更快。

来源 `local-msl-softmax-20261007`，逻辑引用 `2026-10-07-msl-softmax/derived/summary.json`。源程序、输入、输出、中间状态及审查记录在仓库外保留，未随公开库发布。本页数值记录没有GPU时间、指令、寄存器、带宽、模型质量或端到端attention结论；后续主机计时也不提供这些硬件归因。
