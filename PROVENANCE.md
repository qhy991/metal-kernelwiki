# 来源与复用范围

本库由原 `metal-llm-optimization` 技能整理并改名为 `metal-kernelwiki`。保留其 21 个主题页、检索代码、来源目录和限定范围的本机观测。

仓库与 skill 分离、以薄入口访问 canonical checkout 的组织方式参考 [qhy991/bw1100-kernelwiki](https://github.com/qhy991/bw1100-kernelwiki)，检查时为提交 `52ae9a1`。本库的 `knowledge/` 仅链接知识文件，不链接包含 skill 模板的整份仓库，以避免重复发现。没有复制 BW1100 的知识正文、实验结果、检索实现或硬件结论；本库的 Python 标准库检索与安装入口为此项目编写。

Apple、MLX、MLX-LM、llama.cpp 等一手资料的链接、日期、版本可变性和用途保存在 [catalog](data/catalog.json)。上游文档事实、作者报告、推断、待验证方案与本地测量分别标记，不把 PR 合入或 issue 状态当作本机性能证据。

M4/MLX 的原始验证和独立捕获保存在仓库外，初次运行 ID 为 `2026-10-07-mlx-m4-r1`；后续异步求值与 singleton 精度探针为 `2026-10-07-mlx-boundaries`，包含三个同 seed 进程。多行 QVM 与缓存生命周期运行 `2026-10-07-mlx-qvm-cache` 保留了失败数值门、策略损失及仅用保存数组完成的事后 CPU 诊断。四项 `local-experiment` 来源保留相对于运行集合的 `artifact_ref`；source ID 与真实绝对路径的映射由持有者另存。原记录不随 Git 提交，外部读者不能据此独立复验原运行。公开分享原始记录需另行确定范围。

本仓库公开发布；所参考的 BW1100 仓库可能需要额外访问权限。本库未给第三方资料补造许可，也未将引用链接解释为可重新分发其整份正文或代码的授权。
