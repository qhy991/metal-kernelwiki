# 来源与复用范围

本库由原 `metal-llm-optimization` 技能整理并改名为 `metal-kernelwiki`。保留其 21 个主题页、检索代码、来源目录和限定范围的本机观测。

仓库与 skill 分离、通过 `knowledge` 链接访问 canonical checkout 的组织方式参考 [qhy991/bw1100-kernelwiki](https://github.com/qhy991/bw1100-kernelwiki)，检查时为提交 `52ae9a1`。没有复制 BW1100 的知识正文、实验结果、检索实现或硬件结论；本库的 Python 标准库检索与安装入口为此项目编写。

Apple、MLX、MLX-LM、llama.cpp 等一手资料的链接、日期、版本可变性和用途保存在 [catalog](data/catalog.json)。上游文档事实、作者报告、推断、待验证方案与本地测量分别标记，不把 PR 合入或 issue 状态当作本机性能证据。

M4/MLX 的原始验证和独立捕获保存在仓库外，逻辑运行 ID 为 `2026-10-07-mlx-m4-r1`。两项 `local-experiment` 来源保留相对于运行集合的 `artifact_ref`；source ID 与真实绝对路径的映射由持有者另存。原记录不随 Git 提交，外部读者不能据此独立复验原运行。公开分享原始记录需另行确定范围。

本仓库公开发布；所参考的 BW1100 仓库可能需要额外访问权限。本库未给第三方资料补造许可，也未将引用链接解释为可重新分发其整份正文或代码的授权。
