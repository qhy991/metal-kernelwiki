# 有限工具环境：运行时验证与 GPU 计时

[English companion](en/runtime-measurement.md)

适用：找不到离线编译器、只有部分开发工具、需要小范围本地验证。本文依据官方接口设计流程，**没有主张任一设备或安装环境已验证通过**。

## 分开检查三条路线

先记录当前开发目录、SDK、框架版本及错误。离线 `metal` 工具、框架现有算子、`MTLDevice.makeLibrary(source:options:)` 运行时编译是不同路线。离线工具查找失败不能证明运行时编译失败；已有算子成功也不证明 custom kernel 成功。按实际需要分别检查导入、设备获取、library、pipeline 和执行，不靠自动安装或切换工具链制造通过。[Metal libraries](https://developer.apple.com/documentation/metal/metal-libraries)；[离线编译](https://developer.apple.com/documentation/metal/building-a-shader-library-by-precompiling-source-files)

已找到二进制仅证明文件存在，不证明工具可运行、模板完整或服务可用。新路线结果与先前环境失败应各自保留。小型加法 kernel 的编译和输出通过，只覆盖该源程序、数据类型与输入，不能升级为模型部署合格。

建议把探针分成仅编译与实际执行两种模式：前者检查源字符串、函数名与 pipeline 创建，后者才分配输入、提交计算并读回。不要把前者成功写成 GPU 正确性通过。框架路线还需记录实际设备和所用算子；同一框架的内置算子与自定义源编译可能走不同实现，结果不应互相替代。

## 明确计时边界与单位

`gpuStartTime`、`gpuEndTime` 是秒；command buffer 完成后才读取，并先检查执行状态和错误。相减得到该 buffer 的 GPU 区间，不能直接命名为其中某个 kernel 的时延。`kernelStartTime`、`kernelEndTime` 实际是 CPU 调度时间。[GPU 时间](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpuendtime)；[调试属性](https://developer.apple.com/documentation/metal/command-buffer-debugging)

零、倒序、非有限值都应标记无效。编译、首次分配、CPU 编码、同步等待与稳态 GPU 时间分开报告。独立 dispatch 的批量平均不是有依赖链的逐 token 延迟。

## Counter 先查支持，再换算

枚举 counter set、counter 和 `supportsCounterSampling`。stage boundary 与 dispatch boundary 不能互换；Apple 文档以 Apple Silicon 的 stage sampling 为例，具体设备仍要查询。一个 compute pass 的起止采样不提供 kernel 内部占用率。[采样边界](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers)

完成后 resolve 结果，检查长度、零值与 `MTLCounterErrorValue`。raw timestamp 通常来自 GPU 时钟，不能直接除以十亿。用前后两组 `sampleTimestamps()` 建立对应关系：

```text
纳秒差 = GPU样本差 / GPU参考跨度 × CPU参考跨度
```

参考跨度也须有效。校准与采样会增加开销，频率应受控，正常基准另跑。[结果检查](https://developer.apple.com/documentation/metal/converting-a-gpus-counter-data-into-a-readable-format)；[时钟换算](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time)

## Capture 导出不等于完成分析

官方文档允许 macOS 14+ 对目标进程设置 `MTL_CAPTURE_ENABLED=1`。这是明确记录的采集选项，应与基准进程分开；不要修改持久环境。查询 `.gpuTraceDocument` 支持后，还须检查 start、实际命令捕获、stop 和导出结果。文件存在不证明能 replay 或得到性能报告；开发工具不完整时分别记录支持、导出与分析失败。[程序化捕获](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-programmatically)

同一环境可以同时出现“当前 `xcrun` 找不到离线工具”和“已安装框架执行并导出 trace 成功”，两者并不矛盾。经 MLX 导出的 trace 只证明该框架捕获路线的相应步骤，不能替代独立的 Swift 源编译探针，也不能证明本机分析工具可用。`parsed=false` 表示尚未解析，不能写成已定位瓶颈；它同样不等于解析曾失败或永远不可用。

若运行被明确限定为 capture-only，则只记录采集链、工作负载与 artifact 状态。未执行 oracle 或未采集基准样本时，正确性和性能结论保持未评估；不能用少量被捕获的执行次数补成性能样本。正常正确性、无采集计时与后续 profiler 分析应保留各自的结果记录。

## 有界探针与证据范围

在已授权和项目 gate 允许的范围内，预先固定小 shape、内存上限、预热数、重复数及总工作量。先外部参考检查，再采样；不运行无上限调参循环。异常、时间无效或内存超预算时停止该探针并保留原因，不让失败触发更大规模重试。

输入应包含可人工核对的简单数值和非整除尾部。每次 GPU 完成后才能读取或复用相关共享资源；有限样本也不能省略正确性。合并多个独立操作用于放大计时信号时，记录合并数量，并保留其与真实请求依赖结构不同这一限制。若工具缺项，完成不依赖它的检查后报告覆盖范围，不把缺失的观察解释为没有性能问题。

保存路线、输入、正确性、计时边界、原始样本、counter 覆盖及采集路径。分别标记未尝试、不支持、环境失败、正确性通过、计时有效、捕获已导出、分析不可用。没有 counter 的计时可验证局部速度差，却不能证明寄存器、带宽或占用率为何变化；模型质量和真实请求收益仍须目标框架验证。
