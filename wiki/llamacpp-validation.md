# llama.cpp Metal 验证：防止空跑，定位数值差异

[English companion](en/llamacpp-validation.md)

适用：改 Metal 算子、融合或调度后，需要证明目标用例确实执行，并区分数值失败与计时口径差异。证据级别为 `source-reported`；2026-10-07 核对上游可变 `master`，本地部分仅只读源码观察，未运行测试或 GPU，commit 与二进制映射未知。

后续独立 [FA 限定运行](llamacpp-fa-paths.md)已执行现存 binary 的45项测试，并以SQL完整参数和设备日志排除空跑；它没有补齐构建身份或保存数值数组，也不能追溯性地把本页原只读观察变成已执行验证。

## 先建立执行证据

把接受条件写成“目标设备初始化成功、预期参数用例出现、实际运行数大于零、受支持且通过”，同时保留退出码、stdout 和 stderr。只有 `support` 行、跳过行或总进程成功时，应记录“未验证目标”。上游[测试入口与结果汇总](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)把不匹配设备计入成功；`perf` 外层也没有汇总每例返回值，因此退出码不能单独证明完成。

本机 `llama.cpp/tests/test-backend-ops.cpp` 的只读观察还发现：test 空选择满足 `n_ok == tests_run`；perf 用例集合与 correctness 不同，当前本地 perf 集合没有 `test_rms_norm`。这些是**未运行的源码判断**，不可宣称本机二进制已复现空跑。当前上游还增加了额外 FA vec 检查：操作过滤器决定是否进入，其内部形状循环不接收 `-p`，所以参数过滤不保证缩小全部工作。不能用旧版本行号解释新版全部行为。

以下是本地源码推导的小用例提案，未经执行；先确认实际版本支持相同参数串，再在获准的验证环境使用。设备名也须以真实枚举为准。

```sh
./build/bin/test-backend-ops test -b MTL0 -o RMS_NORM \
  -p '^type=f32,ne=\[64,5,4,3\],v=0,eps=0\.000001,inplace=0$' \
  -j 1 --output csv
```

预期记录必须同时含 MTL0、RMS_NORM 和上述完整参数，不能只检索一个 `OK`。`ctest -R test-backend-ops -V -N` 可查看已配置的命令；`-N` 只列出测试。[官方调试文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/debugging-tests.md)中的 `debug-test.sh` 会构建并执行，不能当作只读发现工具。

## 数值归因先对齐语义

记录 dtype、实际形状与 strides、epsilon、原地写入、输入分布和 oracle。上述普通 RMS_NORM 没有权重乘法，不能直接与加权 RMSNorm 比延迟或误差。测试默认 NMSE 阈值也不等于逐元素 `atol/rtol`；具体用例可以覆盖误差函数与阈值。保留失败位置、非有限值和原始误差，不能放宽阈值后把同一次失败改称通过。相关定义见[测试用例与比较回调](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)。

[后端比较函数](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp)有逐节点执行和整图执行后比较指定张量两条路径。诊断融合时，先说明走了哪条路径、检查了哪些输出；单算子通过仅覆盖该用例的语义与输入域，不能证明融合图与模型输出正确。建议以外部 oracle 加入尾部尺寸、非连续视图、近零及混合量级输入，再回到目标框架验证。该扩展矩阵是验证建议，本页未运行。

## 性能结论保留测量范围

本地 `eval_perf` 在预热后重复图执行，用主机时钟计时并除以重复次数；[同步图接口](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp)包含后端等待。将结果标为“该图的重复执行摊销时间”，不要当作单 kernel GPU 时间、模型 tok/s 或服务 TTFT。保存形状、重复次数、单位、暖机策略和错误日志，确认存在有效计时行后再做同口径 A/B；模型收益转到 [PP/TG 测量](measurement.md)。

## 构建身份限制结论

本机 `build/common/build-info.cpp` 记录 b0/unknown，源码多数未被 Git 跟踪；这次未运行、源码 commit 与现有二进制映射未知。[上游构建脚本](https://github.com/ggml-org/llama.cpp/blob/master/cmake/build-info.cmake)本来就以 0/unknown 为默认值。文件存在或日期接近都不能补齐来源。报告列出实际路径、构建记录和缺失项；此状态只能支持明确限域的探索，不能归因到某个 upstream commit 或宣称版本回归验证。

本地只读记录的逻辑引用为 `2026-10-07-mlx-m4-r1/research/metal-followup-local-tools.md`，实际文件未随公开仓库提供。它不是 GPU 实测证据；无法访问原记录时，应仅采用可读的上游来源并保留本地观察未复核的限制。
