# Metal 能力与部署边界

[English companion](en/metal-capabilities.md)

适用：选择 LLM 后端、编译自定义 kernel、排查快路径没有启用。**证据是官方能力说明；本文方案未在本机运行。**

## 先确定四层条件

记录芯片与 GPU、OS build、Xcode/SDK、MSL 版本、框架版本或 commit，以及实际选中的计算路线。分别核查：SDK 是否提供符号、OS 是否实现接口、`MTLDevice` 是否支持硬件能力、后端是否支持当前 dtype/shape。`supportsFamily` 与 `@available` 回答不同问题，不能互相替代。[Apple 能力检测](https://developer.apple.com/documentation/metal/detecting-gpu-features-and-metal-software-versions)

2026-05-21 的能力表把 M1/M2/M3-M4/M5 列为 Apple7/8/9/10；SIMD reduction 与 matrix operation、Metal 4 tensor 的可用硬件起点不等于专用矩阵加速器的起点。该表用于建立候选范围，最终以设备查询、当前 SDK 和具体操作限制为准；不要靠芯片名称伪造 family 值，也不要把桌面 Apple Silicon 结论推广到 Intel/AMD Mac。[Apple 能力表](https://developer.apple.com/metal/capabilities/)

## 从症状选择核查方向

- **新 GPU 却没加速**：查看框架 dispatch、实际 pipeline 和 dtype；API 可用不证明选中了加速实现。
- **开发机能编译，部署失败**：检查 SDK 符号、最低部署版本与 runtime guard，再检查硬件能力。
- **某个 dtype 回退**：保留明确支持的旧路线并报告原因，不将回退结果冒充目标路线。
- **算子变快但生成没变快**：分别看 prefill、decode、采样和 CPU/GPU 等待；小矩阵、KV cache、MoE 都会改变瓶颈。

M5 的 GPU Neural Accelerators 位于 shader core，区别于独立 ANE。TensorOps 在旧 GPU 上可用优化 shader 实现；不能据此宣称相同加速幅度。[M5 ML 技术讲座](https://developer.apple.com/videos/play/tech-talks/111432/)

## 检查模板（未执行）

```text
target: chip / GPU / memory / OS build
toolchain: Xcode / SDK / MSL standard
backend: library version or commit / selected pipeline
workload: model / batch / prompt / decode length / dtype
capability: SDK symbol + OS availability + device query
result: supported | explicit fallback | unsupported
```

验证至少包含 pipeline 身份、编译诊断、正确性与目标框架端到端结果。上游实现仅提供复现入口：MLX [PR #2772](https://github.com/ml-explore/mlx/pull/2772) 是引入 Neural Accelerator 支持的已合并记录；它不能证明用户安装版本对任意模型启用了该路径。llama.cpp 提供[能力查询和捕获接口](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-metal.h)，使用前核对安装版本。

准备优化记录时，将“支持条件”和“性能条件”分开：前者决定能否安全编译和执行，后者由指定输入和设备的测量决定。保留支持条件的拒绝原因，不因一个测试通过便扩大到同系列所有设备。若缺少目标机器，可以完成源码选路分析与候选设计，并把设备测试列为未完成；不要把其他芯片的延迟填进目标结果。比较框架版本时，还要固定模型、输入、量化方案和正确性标准，避免把算法变化误判为硬件收益。
