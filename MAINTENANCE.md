# 来源、版本与维护

本技能借鉴 KernelWiki 的“来源 → 主题页 → 检索”结构。已有本地 MetalKernelWiki 草稿只作为组织方式参考，未将其无来源条目或空索引继承为事实。内容主要覆盖 Apple Silicon 上 Metal/MLX/llama.cpp；不声称全面涵盖 Core ML、独立 ANE、Intel/AMD Metal、分布式推理或每款芯片。

## 单一索引

`data/catalog.json` 的 `schema_version` 为 1；`pages` 注册每个正文，`sources` 注册一手资料，`aliases` 负责中英文关键词与标签映射。

页面字段：`id,title,path,type,engines,tags,symptoms,confidence,sources,summary`。
来源公共字段：`id,title,kind,checked,revision,mutable,note`。远端来源另有 `url`；`kind=local-experiment` 的本地来源只用 `artifact_ref`，格式为 `运行ID/相对文件`，不混用 URL 或本机绝对路径。它是仓库外原记录的逻辑引用，不是仓库内文件。
`checked` 是读取日期，不是发布日；`revision` 如为 `main/master (mutable)` 就不代表固定快照。PR 合入不证明用户已安装它；issue/discussion 中的观察不能升级为已验证事实。

`documented` 指文档描述的机制；`source-reported` 指实现或作者报告；`inferred` 指分析推断；`experimental` 指有待验证的方向；`locally-measured` 指保留原始记录的有限本机观测。一个页面可以包含文档事实与推断，但必须在相应段落明确区分。任何本机测量结论都需独立结果路径与完整配置，不能仅改标签。

本地记录放在源码仓库和技能目录之外，保留执行源文件、输入构造/种子、预先定义的 oracle/容差、逐项结果、失败、版本、计时范围与所有样本。新运行写新目录，不覆盖历史；持续 custody 或正式资格需要其独立制度，文件保存本身不建立这些属性。skill 只保存派生解释和来源路径。

`validate.py` 只验证本地来源字段和引用格式，不 stat/read/hash 外部记录，也不证明它们存在或内容真实；`get_page.py --follow-sources` 只展示元数据。原记录持有者在仓库外保留 source ID 与真实路径的映射，公开仓库不保存用户名、内网位置或该映射。跨机器使用时必须如实保留“原始结果未随仓库发布、未在此机器复验”的边界，不借机自动跑 GPU。检查时间、正确性门、采集成功、解析成功、端到端收益分别记录，失败不得改环境后覆盖成一次成功。

## 刷新方法

1. 根据真实问题选一个主题；重新读 Apple 文档或对应框架官方 docs、源码、PR。优先永久 commit 链接；读不到 revision 时诚实记录可变 URL，不编造 commit。
2. 找到该主题的 API/CLI 支持与条件，区分硬件、OS、SDK、框架版本。代码参数以用户 checkout/安装 help 为准。
3. 只改相关主题和来源的日期/说明，不把一次检索称为全库刷新。新增来源须支持正文具体主张；不增加空页、镜像索引或未阅读链接以充数。
4. 运行 `python3 scripts/validate.py`，并用一条真实中文和一条英文问题检索受影响页面。它仅检查结构和本地引用；不验证 URL 可达、事实真伪、API 能力或速度。
5. 有行为变化时做针对性 forward test：让独立使用者从任务问题检索并给建议，检查是否正确保留版本、质量与证据边界。仅文案小改无需新的硬件实验。

短摘要、技术综合和指向来源的链接足够；不存上游大段手册或源码副本。无需例行 SHA-256 检查。没有新增证据时，不把历史报告升级成当前性能承诺。
