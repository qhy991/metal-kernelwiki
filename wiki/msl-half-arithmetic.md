# MSL half 算术：类型提升、倒数与次正规数

输出buffer是half，并不表示表达式用half计算。本页说明源码类型、生成签名、表达式宽度与最终位模式应如何分开，用来源研究和一次未完成的元数据探针补充[低精度softmax](msl-softmax-lowp.md)的诊断。本次没有得到算术输出，没有读取原生softmax中间状态，也没有测量速度。

## 把运算与输出转换写成两个边界

```metal
const half h = x[i*x_strides[0]];
y[i] = half(half(1) / h);       // half操作数
y[i] = half(1.0f / h);          // float字面量参与，最后窄化
y[i] = half(1.0f / float(h));   // 显式float除法，再窄化
```

这是三种候选赋值的对照，实际kernel每次只选一条。[Apple的标量提升说明](https://developer.apple.com/videos/play/wwdc2016/606/)指出half/float混合表达式采用float计算，即使结果随后赋给half；半精度字面量可写 `1.0h`。此处只引用语言行为，没有把2016年芯片的性能建议迁移到M4。

[MSL4.1 §6.6](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)提供half/float的 `metal::divide(T,T)`，而显式 `fast`/`precise` 变体描述的是单精度。本例用 `metal::fast::divide(1.0f, float(h))` 与 `metal::precise::divide(1.0f, float(h))`，不杜撰 `recip`/`reciprocal` API，也不把qualified调用解释成half硬件指令。`1 / h` 的整数与half混合排名没有在所读规范中单独列成表，因此另保存编译器的 `sizeof(1 / h)` 结果。

## 算术FTZ与存储转换不能混为一谈

MSL4.1 §8.1/8.5允许算术输入、输出次正规数flush为零，flush后的零符号也未规定。§8.2允许算术采用RNE或RTZ；表8.3对Apple GPU Family4及以后half乘法/倒数/除法的正确舍入要求仍受这些条款约束。因此相对CPU RNE有差异，不足以证明违反MSL。

§8.6对转换另有规定：half→float无损，float→half默认ties-to-even，转换新生成的half次正规数不能直接flush；fast math不改变转换精度。4.1还提供显式RTZ浮点转换选项，不能凭文档版本假定运行启用了它。若源表达式先发生了算术冲零，后面的half存储转换也不能恢复其值。

float `precise` 除法与 `fast` 除法有各自误差说明；不要把表8.2中 `1.0/x` 的1 ULP界直接当成所有 `fast::divide` 的界。我们的数值门来自实验合同，不把它冒充规范误差界或框架承诺。

## 生成源码与编译选项的证据范围

MLX v0.31.2的[custom kernel生成器](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp#L301)以 `std::cout`/`std::endl` 输出verbose源码；保留C++输出需要捕获文件描述符1，单独替换Python stdout未必有效。verbose保存的是wrapper，[编译前还会添加](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp#L379) `metal::utils()` 帮助代码，不能称为完整编译单元。调用随后经 [get_library](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/device.cpp#L695) 到 [build_library_](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/device.cpp#L547)，源码显式设置 `fastMathEnabled(false)` 并传入language version。

该版本[语言选择](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/device.cpp#L36)在macOS26+选择MSL4.0，macOS15+选择3.2，否则3.1。这是可变tag的源码策略；没有核对安装binary与tag的对应，也没有截获本次JIT实际options，不能据此宣称本机观察到了MSL4.0、安全数学模式或4.1的RTZ设置。生成签名和 `sizeof` 也不揭示最终机器指令与内部实现精度。

## M4元数据观测：算术矩阵尚未执行

2026-10-08，Apple M4/16GB、macOS27.0.1（26A434）、MLX0.31.2、NumPy2.4.3、Python3.14.3，`MLX_ENABLE_TF32`未设置。使用已有协作GPU锁，未修改环境。预定计划为两个输入域、两种布局、各七条数值路线，共28份算术输出及4份元数据；**实际只完成首组输入和一个元数据调用，算术输出为0**。

首组输入遍历uint16位模式 `0x3c00..0x7bff`，即 `[1,65504]` 内全部16384个可表示正half值。CPU构造bits，GPU按uint上传后以half view解释，逻辑view逐位读回核对；完整backing是CPU构造记录，不是独立的GPU backing导出。元数据使用TG128，返回stride1、SIMD宽度32，以及下列字节数：

| 表达式 | 本次 `sizeof` |
|---|---:|
| `h` | 2 |
| `1 / h` | 2 |
| `half(1) / h` | 2 |
| `1.0f / h` | 4 |
| `1.0f / float(h)` | 4 |

这些是编译器表达式宽度的返回值；`sizeof` 不求表达式的数值，不能据此称倒数已经运行或精度门已经通过。宽度也不证明某种除法指令或中间硬件精度。

随后证据检查失败：实际保存的生成签名含 `const device float16_t* x`，预先写定的检查器只接受 `const device half* x`。生成源码并未丢失，stride/SIMD检查也通过；失败来自检查器对类型拼写的假设。原终态保留 `complete=false, passed=false`，退出1，没有改门、改环境或重跑来覆盖它。静态审查遗漏了这一拼写问题，另保存了事后审查说明。

随后只做源码核对与CPU审查。v0.31.2 [utils.h第13行](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/utils.h#L13)定义 `typedef half float16_t;`，[get_type_string](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/common/compiled.cpp#L47)对float16返回 `float16_t`。这解释了该源码版本的签名拼写，不能据此修写原终态，也没有建立安装binary与tag的逐字对应。后续检查器应核对生成器和类型定义，而非凭猜测拒绝别名。

独立CPU审查（Python3.12.14、NumPy2.3.5）重建首组16384个input/backing bits和binary64参考，保存参考重算最大差为0；检查事件、生成源码及停止位置后，报告已有记录一致、计划未完成、整体合同未通过，算术数值验收为“未评估”。保存的634字节生成源码包含kernel、原始函数体和int64输出声明；唯一不符合检查器的部分是输入类型拼写。数据记录一致不等于历史GPU执行或filesystem custody的独立证明。

未执行部分包括step2布局、所有倒数值输出、正half次正规输入的复制/提升/动态乘加及float窄化比较。不得写成“28项通过”，也不能从元数据判断 `fast::divide` 的误差、half FTZ频率或先前native softmax全零的原因。预定half算术门为 `2^-24 + 2^-10*abs(ref)`，float门为 `2e-6*abs(ref)`，copy/promotion要求精确；这些门本次尚未评估。

## 对LLM kernel优化的用法

- 在normalizer、倒数或小概率需要更大动态范围时，明确写float操作数与状态，最后再选择存储类型；不能只改输出buffer dtype。
- 将输入存储、算术中间结果和窄化分别建对照。先用实际保存的half值定义oracle，再评估与未量化输入的差别。
- 按选定的舍入规则记录零的两种含义：参考正数本来就应舍入为零，或参考舍入仍非零而候选为零。后一类能排除“太小而必须按该规则舍入为零”的解释；非零近似并不表示原值可被精确表示。
- 复制或 `half(float(h))` 通过只能说明最终位模式；编译器可能消去转换。动态buffer操作数减少常量折叠机会，也不证明某条算术指令实际执行。
- 本页无计时、profiler或模型质量结果。候选通过数值门后，仍需在目标调用图比较转换、访存和计算成本，不能把float路线称为普遍更快。

来源 `local-msl-half-arithmetic-20261008`，逻辑引用 `2026-10-08-msl-half-arithmetic/derived/summary.json`。原始输入bits、CPU构造backing、元数据返回值、生成源码、预定合同、环境和独立CPU审查保存在仓库外，未随公开库提供。这次有限元数据观测不能唯一解释先前native softmax的全零行。
