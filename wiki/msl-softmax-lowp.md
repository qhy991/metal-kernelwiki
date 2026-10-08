# MSL softmax 低精度：存储、累加与次正规输出

低精度优化要分别说明输入保存为什么类型、统计量在哪里舍入、输出怎样窄化。本页在[softmax 三遍 SIMD 归约](msl-softmax.md)上显式使用 float 状态，并与 MLX 的低精度原生路线比较。M4 的有限检查保留了失败，不能把 `precise=True`、输出dtype或CPU参考通过替代成通用质量与性能保证。

## MSL 类型边界要写出来

MLX v0.31.2 [bf16.h](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/bf16.h#L9) 将 `bfloat16_t` 定义为原生 MSL `bfloat`；`as_type` 帮助函数只重解释位模式。[生成器](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp#L67)按输入/输出dtype生成形参，再插入用户函数体。换输入dtype不会自动把源码中的 `float` 状态改成half/BF16。

本例重用SG32三遍max/指数和/输出结构，关键边界是：

```metal
// x可以是half或bfloat存储；每次读取都明确提升。
const float v = float(x[row*x_strides[0]+k*x_strides[1]]);
// m、l、value均为float，指数实参及SIMD统计也为float。
// ... 依完整的mask/count与SIMD参与规则计算value ...
y[row*K+k] = OT(value);  // 显式转换到声明的输出类型
```

`OT` 是MLX `metal_kernel` 的类型模板参数，须与 `output_dtypes` 对应，例如 `template=[('K', K), ('OT', mx.bfloat16)]` 和 `output_dtypes=[mx.bfloat16]`。grid为 `(32*R,1,1)`，threadgroup为 `(32,1,1)`；每lane都参加collective，尾部只限制标量访问。masked位置和全空行通过count分支直接写正零。这是函数体中的声明与数据流，不是已读取的机器指令精度。

[MSL 4.1](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) §2.23/2.24/8.6区分数值转换与位重解释：half/BF16提升到float无损，但不会恢复输入存储已丢失的信息；float/half转bfloat须显式转换。默认float窄化采用ties-to-even，§1.6.3又允许4.1的RTZ转换选项。本机未捕获JIT语言版本或flags，不能仅凭读了4.1规范就宣称运行时启用了某配置。下面保留真实位模式，并把CPU RNE作为明确的输入构造和输出诊断。

## 原生 `precise` 改变了哪些类型

固定v0.31.2的[JIT factory](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/jit_kernels.cpp#L290)取 `AccT = precise ? float32 : output_dtype`。因此F32两flag都使用float；F16/BF16的false使用对应AccT，true使用float，输出仍为输入类型。

[softmax.h](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/softmax.h)在AccT中保留输入/指数、max、normalizer及倒数，仍调用 `fast::exp`。[BF16重载](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/bf16_math.h#L226)的指数、SIMD sum/max先转float调用内部帮助函数，再转回BF16。这说明存在低精度返回/状态边界，既不能称为每条指令都用BF16，也不能称为全程float。源码tag未与安装binary建立逐字对应，实际归约树和重结合未观察。

## 用实际存储值定义数值合同

2026-10-08，M4/16GB、macOS27.0.1（26A434）、MLX0.31.2、NumPy2.4.3、Python3.14.3，`MLX_ENABLE_TF32`未设置。系统版本与前轮27.0不同，按本次实际环境记录；没有修改或修复环境。已有协作GPU锁、无模型。R=8，K为 `31,32,33,127,128,129,4095,4096,4097,8193,65537`；输入存储F32/F16/BF16，各有连续和x/mask均step2两布局，共66组输入。

八行分别为uniform0、全mask有限8、最后一项有效20、重复dyadic ramp、首项0其余−32、1附近F32小扰动、PCG64(seed=608+K)的dyadic随机行及其F32精确加256的平移行。随机值为−128…128整数除128，mask按列模3构造且首项有效；平移行mask相同。全部logits有限且在 `[-512,512]`，不覆盖NaN/Inf、负stride、alias或任意mask生成器。

先在CPU将共同F32数据按RNE编码为目标存储，再上传uint原始位并用同宽view解释；**不是先把GPU量化当成已验证的输入构造**。保存共同F32、实际存储bits、CPU解码值、mask、step2完整backing，以及两个独立CPU oracle：一个基于实际存储值，另一个基于共同F32，后者只衡量输入信息损失。oracle对有效项用 `math.exp` 与 `math.fsum`，空行输出正零。

| 路线 | 原生／自定义计算 | 输出 |
|---|---|---|
| native_false | 原生softmax，precise=False | 输入存储T |
| native_true | 原生softmax，precise=True | T |
| native_promote_f32 | 显式 `x.astype(float32)` 后原生precise=True | F32 |
| custom_float_out_f32 | 显式float读/状态/precise::exp/归约 | F32 |
| custom_float_out_storage | 同一MSL函数体，只改变OT与输出dtype | T |

三条native路线均含dtype明确的mask填充与最后归零。实际sentinel为F32的−16385或F16/BF16的−16384，远低于本输入域；它不是通用负无穷替代。保存其位模式，并记录原生softmax收到的masked数组dtype。显式F32提升另保存bits并对照CPU解码，不能因此假定它没有转换成本；本轮没有计时。

所有路线对**实际存储oracle**使用输出dtype对应的预定门，同时要求shape/dtype、finite、非负、masked正零：

| 输出 | 逐元素绝对误差门 | 行和误差门 | 行L1门 |
|---|---|---|---|
| F32 | `2e-6 + 2e-5*abs(ref)` | 2e-5 | 5e-5 |
| F16 | `2^-24 + 2^-10*abs(ref)` | 2^-10 | 2^-10 |
| BF16 | `2^-133 + 2^-7*abs(ref)` | 2^-7 | 2^-7 |

这是本研究选定的质量门，不是MLX承诺的通用误差界。另将binary64参考直接RNE到输出格式，记录输出相对它的差异；此诊断没有先转F32造成双重舍入，也不作为“必须正确舍入”的额外验收条件。

## 结果与两类不同的损失

330项输出全部保存，304通过、26未过门，原运行退出2且 `complete=true, passed=false`。每个输入存储/路线都有22项；下表为通过数：

| 输入存储 | native_false | native_true | native_promote_f32 | custom_float_out_f32 | custom_float_out_storage |
|---|---:|---:|---:|---:|---:|
| F32 | 22/22 | 22/22 | 22/22 | 22/22 | 22/22 |
| F16 | 12/22 | 22/22 | 22/22 | 22/22 | 22/22 |
| BF16 | 6/22 | 22/22 | 22/22 | 22/22 | 22/22 |

F16 false失败宽度为31/32/33/128/65537，BF16 false为32/128/129/4095/4096/4097/8193/65537，均涉及两布局。其他四条路线共264项通过；F32中的同类型对照含重复数值路径，不能当成五种独立算法。较小超限表示未满足本研究的预定门，不是已证明违反框架通用契约，也不支持将所有低精度false调用判为错误。

独立CPU审查（Python3.12.14、NumPy2.3.5）重构输入及RNE存储bits，复算两个oracle、所有输出/事件、typed常量、实际masked dtype和提升结果。66输入、66metadata、3常量、465事件一致，保存参考重算差均为0；`record_integrity=true` 而 `contract_acceptance=false`，26失败原样保留。165个跨布局输出对逐位一致是有限诊断，不证明历史访存不存在隐藏问题。直接binary64参考的正确RNE输出在全部330组都满足相应门，说明这些门没有强迫低精度存储达到不可能的F32质量；它仍不取代候选验收。

**可表示的half概率也可能被某条计算路径输出为零。** K65537、F16的uniform0行和near-one行，native_false在两布局均返回65537个 `0x0000`，行和误差与行L1均为1。native_true和custom存F16路线在这些行返回 `0x0100 = 2^-16`，它是F16次正规数，行和为 `1.0000152587890625`，通过相应门。BF16 false的这两行也保留 `2^-16`（BF16 bits为 `0x3780`，属于该格式的正规数）；其配置失败来自其他行。

不能把这两个全零行统称为已证实的分母溢出。独立CPU诊断中，uniform的理想normalizer为65537，超出half最大有限值65504；但near-one从实际存储logits算出的normalizer约65473.048368，低于65504。后者若仅做一次RNE到half为65472，其精确倒数约1.52737048e-5，再窄化得到非零的 `2^-16`。这是CPU假设诊断，没有模拟实际native归约。

[MSL 4.1 §8.1/8.5/8.6](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)区分允许的算术次正规处理和窄化转换；在描述的转换规则中，float→half不能把新生成的half次正规数直接flush为零。这解释了为何要分别检查中间算术与最终转换，但本轮没有读取运行时语言版本、flags、分母或倒数。更强的反例是同一F16 false、K65537配置的dyadic ramp行实际保留了65537个正次正规输出（bits79…587），所以也不能概括成“该路径一律flush次正规数”。现有结果只定位到低精度计算路线的数值风险，未唯一确定溢出、舍入或次正规算术中的具体原因。

**提升不能找回已经量化的logits。** K129的平移行在共同F32中恰好加256，参考概率不变；存成F16后实际差值范围为 `[255.8828125,256.1171875]`，BF16则为 `[255.0078125,256.5]`。它们已经不是同一个常量平移。仅输入存储造成的最大概率差分别约0.003148和0.014702，最大行L1分别约0.053081和0.362050。不能拿未量化F32 oracle去判定这些实际存储输入的kernel错误；同样，precise=True通过实际存储oracle也不表示它恢复了原概率分布。

## 可迁移的优化方法

- 在低精度存储上明确float统计，再单独选择输出精度；为低精度输出保留合理的量化误差门，不能直接套F32容差。
- 对长行检查归约状态的动态范围、倒数的数值范围以及输出转换。最大值稳定化保护指数，不保证低精度normalizer和倒数安全。
- 区分低精度算术中的次正规处理与最终存储转换；看到“half输出”不足以判断中间值是否被flush。
- 将输入信息损失、对实际存储值的执行误差、输出舍入分开。平移不变性只适用于实际logits仍有相同常量差的情形。
- 通过数值门后，再在目标调用图中比较转换、内存、kernel与模型代价。本轮没有速度或端到端收益结论，也不把全部false路线称为普遍不可用。

来源 `local-msl-softmax-lowp-20261008`，逻辑引用 `2026-10-08-msl-softmax-lowp/derived/summary.json`。源文件、预定合同、输入/backing、全部原始输出bits、终态、CPU审查及有限诊断保存在仓库外，未随公开库提供。整体失败没有被其他通过路线覆盖；生成签名、JIT flags、native中间状态、指令、profiler及模型质量均未验证。
