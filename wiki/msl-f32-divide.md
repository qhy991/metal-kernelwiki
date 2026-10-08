# MSL F32 除法：函数选择、舍入与输入来源

[half算术研究](msl-half-arithmetic.md)发现，half输出相同可以掩盖F32舍入差异。本页直接保存F32结果，比较普通除法、显式fast与显式precise。数值验收、正确舍入、规范精度界和部署速度分别判断。

## 选择函数，不靠输出dtype猜精度

在float操作数 `a`、`b` 上，三个候选赋值分别为 `y[i] = a / b;`、`y[i] = metal::fast::divide(a, b);` 和 `y[i] = metal::precise::divide(a, b);`。三条路线的buffer都声明为float；这不自动证明正确舍入或某条机器指令已经执行。

[MSL4.1 §6.6、§8](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)区分函数变体与数值条件。普通精度表对除法要求正确舍入，算术规则允许RNE或RTZ；fast表的一般division界为2.5 ULP，reciprocal另有1 ULP条目，所列分母域为 `[2^-126,2^126]`。不能因 `fast::divide` 的分子为1，就擅自收紧为reciprocal界，也不能从 `/` 的结果猜数学模式。

“距RNE多少个相邻F32值”和“相对无限精确结果的误差是多少ULP”是不同量。前者可在正有限F32域用raw bits整数距离表示；后者需以精确参考定义单位，不能在跨越2的幂时直接套用已舍入参考的spacing。预定相对误差门又是第三个合同。

## 两个设置与未观察的运行时

MSL4.1 §1.6.3分别定义默认FP32函数集（fast／precise）和数学优化模式（fast／relaxed／safe）；函数变体与允许的变换是两回事，乘加contraction另有设置。显式函数选择不冻结所有周边算术。

Apple当前[MTLCompileOptions.mathMode](https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode)明确说明：`fastMathEnabled=false` 请求safe模式＋precise函数集，true请求fast＋fast；后续设置两个新属性可分别覆盖。旧fastMathEnabled页面有与该映射冲突的high-precision措辞，因此这里采用新页面的逐属性说明。

MLX v0.31.2 [Device::build_library_](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/device.cpp#L547)调用 `setFastMathEnabled(false)`，所读函数体没有后续覆盖。因此**若安装binary执行所读路径且设置生效，它请求safe＋precise默认函数**。本轮没有建立安装binary与可变tag的对应，没有截获实际compile options或机器指令；条件式推断不能改写为本机模式观测。生成wrapper也不包含随后添加的全部helper代码。

## 本地对照：相同数学值，不同输入存储

2026-10-08，Apple M4/16GB、macOS27.0.1（26A434）、MLX0.31.2、Python3.14.3、NumPy2.4.3，`MLX_ENABLE_TF32`未设置。使用已有GPU协作锁，在未修改环境的情况下完成一次固定矩阵；本轮没有计时。

三种输入域都使用float局部变量a、b和float输出：

- **half倒数**：buffer为half，位模式 `0x3c00..0x7bff`，覆盖 `[1,65504]` 全部16384个half值；读取后写 `const float a=1.0f; const float b=float(h);`。
- **float同值倒数**：把上述完全相同的数学值无损编码为F32输入bits，直接读取float分母。独立审计逐项确认两个域的分母与精确有理数参考相等。
- **float一般比值**：4097对正正规F32输入，指数范围−20…20，分数字段由固定整数公式产生，另含12组预定边界／有理数案例。实际精确商范围为 `[2^-40,2^40]`；输入和商均远离F32次正规区。

每域分别测试连续和全部输入step2布局，每种布局运行三条除法路线。输入从CPU位模式上传，并核对GPU逻辑view读回的bits；保留的完整backing是CPU构造记录，不是独立GPU物理backing导出。所有source按各自stride寻址，TG128；4097项域有127个guarded尾线程，两个16384项域无尾线程。六组metadata返回符合预期的stride1/2、SIMD32以及三个表达式均为4字节；这些值不证明中间硬件精度。

预定质量门为shape/dtype正确、输出正且有限，以及 `abs(y-ref) <= 2e-6*abs(ref)`，atol为0；ref由实际输入解码为binary64后计算。一次运行得到18份输出、221190个值、6组输入、6份metadata和24份生成wrapper，**18/18通过质量门**。独立CPU审计（Python3.12.14、NumPy2.3.5）重建输入及精确有理数商，再计算RNE、RTZ与误差；该域binary64参考到F32的RNE与精确有理数RNE一致。9对跨布局输出全部逐位相同。

下表每个计数均为**单条路线、单个布局**；合并列出的路线各自得到该结果。审计器的reference-exponent指标以 `2^(floor(log2(q))-23)` 为单位，q和误差均用精确有理数计算。该公式不是规范在恰好可表示的2的幂处的完整ULP定义；另一次仅用保存bits的CPU边界检查确认，两个倒数域各16个、比值域3个这样的数学值，在全部路线／布局中共210项输出均精确无误差，因此此处的相邻spacing约定不改变表中误差或最大值。

| 输入域／路线 | 每路值数 | 不等于RNE | 同时不等于RNE和RTZ | 最大精确参考ULP误差 |
|---|---:|---:|---:|---:|
| half倒数：operator、fast、precise各自 | 16384 | 1248 | 1232 | 0.684364 |
| float同值倒数：operator、precise各自 | 16384 | 0 | 0 | 0.499648 |
| float同值倒数：fast | 16384 | 1248 | 1232 | 0.684364 |
| float一般比值：operator、precise各自 | 4097 | 0 | 0 | 0.499999970 |
| float一般比值：fast | 4097 | 1150 | 779 | 1.422636 |

half倒数的三条路线彼此逐位相同，也与float同值倒数的fast路线逐位相同。它们相对RNE有16项低一个相邻F32值、1232项高一个；最大精确相对误差约 `7.3632691e-8`。float一般比值fast的最大精确相对误差约 `1.1336956e-7`；虽最多距RNE一个相邻值，其相对精确结果的误差可超过1 ULP。这些fast结果在所测域内低于一般division的2.5 ULP界，不能把有限域观测写成全域保证。

## 保留未解释差异，不用质量通过抹掉它

**half输入提升为float后，显式precise的1232项输出既不是精确RNE，也不是精确RTZ。** 这与按所读普通除法精度条款建立的两个正确舍入点预测不符，不能仅用“允许RTZ”解释。所有输入与商均为正规F32，F32 FTZ也不解释这批差异。相同数值改由float buffer读取后，operator与precise在该域全部匹配RNE。这个对照发现了与输入来源相关的结果差异，尚未定位产生差异的编译或执行阶段。

保留的wrapper确认half来源precise实际源码为 `const half h=...; const float a=1.0f; const float b=float(h); y[i]=metal::precise::divide(a,b);`，输出指针为 `device float*`。因此不能把保存的源码解释为half重载或half输出窄化。9个domain×route算术入口和3个metadata入口名称各自唯一；同一入口仅在两种布局重复，完整wrapper相同；模板后缀、函数名、host_name逐项匹配。

另读MLX v0.31.2的[closure与模板命名](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp#L265)及library/kernel缓存路径，未找到明显的源码名称碰撞或误绑定解释。但上游tag源码和verbose文本不能证明安装binary的内部缓存、实际编译选项或最终指令正确，也没有隔离完整编译单元。本页将该差异保留为待解释观察，不宣布MSL违规、某个编译器bug或特定指令已被证实。

独立CPU审计的首版曾误要求算术入口不带MLX模板后缀，报出18项命名错误；该版本完整保留。核对生成器后，单独后继审计按精确的请求名、`__N` 后缀、host_name和模板实参检查，记录一致性通过。修改的是审计器的名称模型；没有修改、修环境或重跑原GPU矩阵。

## 独立后继：先完成F32转换，再构造除法结果

后继 `2026-10-08-msl-divide-provenance` 将问题进一步拆开：差异是否依赖于除法源码中可见的half来源？2026-10-08，仍为M4/16GB、macOS27.0.1（26A434）、MLX和mlx-metal均0.31.2、Python3.14.3、NumPy2.4.3，`MLX_ENABLE_TF32`未设置。只测试上述全部16384个half分母，连续布局、TG128、无尾线程，每条路线执行一次。

- **I，直接提升**：half buffer加载后，在同一kernel内转float并调用precise除法。
- **C，CPU float对照**：CPU从原half位模式无损编码同值F32，再上传并执行float除法。
- **G，GPU先转换**：自定义kernel仅执行 `y[i]=float(x[i*x_strides[0]]);`，写出F32数组p；完成 `mx.eval(p)` 和uint32位模式读回，逐位确认与C相同，随后才构造p的除法结果。

C与G使用**同一个float除法callable、函数名、函数体、模板和launch参数**。G实际传入原先求值完成的p；CPU读回只作检查，没有把读回值重新上传。固定执行顺序为I、C、转换、G，数值失败会保留并继续预定矩阵；转换不精确则停止其依赖的G路线。另有三次独立stride/SIMD检查，不在除法函数体中增加诊断写入。此过程没有更改环境或重跑旧矩阵。

两条lazy Python调用本身不保证已经完成中间数组。[MLX求值机制](mlx-async-evaluation.md)说明应区分构图与求值；这里的求值、已完成的位模式读回和后续构图顺序共同建立**应用可观察的F32中间值**。保存的Python对象关联、源代码和launch参数不等于已观察到Metal资源绑定、物理内存写入或实际dispatch数。没有解析运行时capture，因此不作这些更强的声明。

一次运行保留3份除法输出、1份转换输出、3份metadata、7份wrapper和9个事件。独立CPU审计用精确有理数重新核对全部49152个除法值、16384个转换值及输入；原half输入、C和G的数学分母相同，C/G完整wrapper相同，12项宿主对象关联和保存的launch参数一致，三次metadata均为stride1/SIMD32。三个除法输出均通过原 `2e-6` 相对误差门，转换逐位精确。

下表是每条路线各16384个值，计数不合并：

| 路线 | 不等于RNE | 同时不等于RNE和RTZ | 相对CPU float对照的逐位差异 |
|---|---:|---:|---:|
| I，half在除法内提升 | 1248 | 1232 | 1248 |
| C，CPU提供F32 | 0 | 0 | 0 |
| G，GPU转换并完成F32数组 | 0 | 0 | 0 |

具体见half位模式 `0x3c5b`：分母为 `1115/1024`，精确倒数为 `1024/1115`；RNE和RTZ都为F32位模式 `0x3f6b1b52`。I给出 `0x3f6b1b53`，C与G均给出 `0x3f6b1b52`。这比“全部通过容差”更明确地展示了保留的差异。

**在本次程序与输入域中，完成独立F32转换后再除法，足以消除所观测的差异。** 结果与“除法内可见的half来源影响编译行为”这一解释相容，但没有定位编译阶段、指令、实际选项或运行时pipeline；也没有证明G/C在所有输入、设备、顺序和重复运行中都正确舍入。原先I的差异没有被覆盖或改判。

这是诊断方法，尚未作为优化方案验收。新中间数组的元素数据量为64KiB，实际分配、缓存、传输及延迟未测；为了诊断加入的求值和主机读回边界也不能直接搬进LLM热路径。若要考虑部署，应另用实际调用图验证端到端质量与收益。本轮无计时、profiler或模型结果。

后继来源 `local-msl-divide-provenance-20261008`，逻辑引用 `2026-10-08-msl-divide-provenance/derived/summary.json`；原始输入、转换／除法输出bits、固定合同、生成wrapper、宿主关联和独立CPU审计留在仓库外。记录一致性不建立持续custody或独立的历史GPU执行证明。

## 应用到LLM优化

- 对normalizer倒数或归一化比例，明确验收目标是允许的近似误差还是正确舍入，两者不能互换。
- 比较F32函数语义时直接保留F32结果；仅比较half输出可能看不到差异。
- 将输入类型、数学值、源码表达式和输出存储分别控制；同值不同存储的结果对照不直接证明指令或优化阶段。
- 若考虑用fast替换precise，先按真实调用图检查误差，再测量收益。本页没有计时、profiler、模型质量或通用最快路线结论。

来源 `local-msl-f32-divide-20261008`，逻辑引用 `2026-10-08-msl-f32-divide/derived/summary.json`。原始输入、CPU构造backing、全部输出bits、生成wrapper、合同、环境和独立审计留在仓库外，未随公开库提供。记录一致性不证明持续custody，也不解释原生softmax中间状态。
