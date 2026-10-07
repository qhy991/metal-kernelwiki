# Metal KernelWiki

面向 Apple Silicon、Metal、MLX/MLX-LM 和 llama.cpp 的 LLM 部署与 kernel 优化知识库。以优化机制、适用条件、代价交换与有范围的观测组织内容，并保留一手来源。

截至 2026-10-07：**28 个主题页、101 个来源条目**，其中 8 项为本地实验记录。M4 原语验证不代表整模型加速；捕获已导出不代表 profiler 已解析。详细范围见 [本地观测](wiki/local-mlx-m4.md)。

## 从哪里开始

- [优化流程](references/optimization-workflow.md)：从部署问题选择候选，再验证正确性和收益。
- [测量与诊断](wiki/measurement.md)：区分 TTFT、prefill、decode、吞吐、主机等待与 GPU 时间。
- [MLX 部署](wiki/mlx-deployment.md)、[KV cache](wiki/mlx-kv-cache.md)、[服务并发](wiki/mlx-serving.md)。
- [llama.cpp 部署](wiki/llamacpp-deployment.md)、[Metal 调优](wiki/llamacpp-metal-tuning.md)、[验证与空跑](wiki/llamacpp-validation.md)。
- [Attention](wiki/attention.md)、[GEMM/MoE](wiki/gemm-moe.md)、[量化 matmul 验证](wiki/quantized-matmul-validation.md)。
- [Metal 能力](wiki/metal-capabilities.md)、[内存与线程组](wiki/metal-memory-threadgroups.md)、[运行时测量](wiki/runtime-measurement.md)。
- [M4 本地记录](wiki/local-mlx-m4.md)、[异步求值边界](wiki/mlx-async-evaluation.md)、[float32 精度诊断](wiki/mlx-float32-precision.md)。
- [量化三路径的时间与内存](wiki/mlx-qmm-path-comparison.md)、[缓存生命周期与 keep 策略](wiki/mlx-cache-lifecycle.md)。
- [旋转缓存 mask 对齐](wiki/mlx-cache-mask-alignment.md)、[缓存保存恢复与续写](wiki/mlx-cache-persistence.md)。
- [残差 RMSNorm：编译与 fast 比较](wiki/mlx-residual-rms.md)。
- [边界探针进度](wiki/mlx-regression-probes.md)、[完整目录](data/catalog.json)。

## 检索

只需 Python 3.9+ 标准库；这些命令不联网、不加载模型、不启动 GPU。

```bash
./mwiki query "MLX 长上下文 decode 慢" --limit 5
./mwiki query --engine llama.cpp --tag flash-attention
./mwiki query "量化 batch数值"
./mwiki get mlx-kv-cache --follow-sources
./mwiki get local-mlx-m4
./mwiki validate
```

`query` 支持中英文关键词及 `--engine/--type/--tag/--symptom` 过滤。`get` 接受页面或来源 ID；跟随来源只输出元数据，不访问外部原始记录。索引是 `data/catalog.json`，无需另装 PyYAML，也没有需要重新生成的第二套页面索引。

## 安装 skill

采用 [BW1100 KernelWiki](https://github.com/qhy991/bw1100-kernelwiki) 的独立知识库加薄 skill 入口方式，技能名称为 **`metal-kernelwiki`**。

```bash
git clone https://github.com/qhy991/metal-kernelwiki.git
cd metal-kernelwiki
python3 scripts/install_skill.py
python3 "${CODEX_HOME:-$HOME/.codex}/skills/metal-kernelwiki/scripts/wiki.py" query "首 token 慢"
```

安装器默认使用 `CODEX_HOME/skills`（未设置时为 `~/.codex/skills`），也可用 `--dest /绝对路径/metal-kernelwiki` 指定技能目录。已存在的目标会被拒绝，不覆盖其他安装。它只复制 `skill/metal-kernelwiki/`，在真实目录 `knowledge/` 中链接 `data/`、`wiki/`、`references/` 及 README、MAINTENANCE、PROVENANCE，并用安装时生成的 `repository.json` 定位当前 checkout。正文不复制，仓库内容更新后自动可见；整个仓库和其中的 skill 模板不会出现在已安装技能的扫描树中。

旧版 whole-repo `knowledge` 链接会让 Codex 发现两个同名 skill。升级安装入口或移动仓库时，先将旧技能目录移到所有技能扫描目录之外保留备份，再重新安装，让知识链接和仓库定位一起更新；只更新 checkout 不会迁移旧安装。不要将整份仓库链接进技能目录。[OpenAI 技能发现规则](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills)说明扫描会跟随符号链接，且同名技能不会合并。直接在 checkout 内运行模板入口也可使用：

```bash
python3 skill/metal-kernelwiki/scripts/wiki.py query "Metal profiling"
```

## 内容与证据

`wiki/` 保存主题页，`references/` 保存流程与使用例，`data/catalog.json` 保存页面、来源、版本、日期与检索别名。`skill/metal-kernelwiki/` 仅承担技能发现与命令入口，不复制整份知识库。

原实验、脚本、日志与 `.gputrace` 留在仓库外；这里保存派生解释和 `artifact_ref` 逻辑引用，真实路径映射由原记录持有者保管。本地记录未随公开仓库提供，外部读者不能据页面独立复验原运行。结构校验不会验证外部记录存在、测量真实性或历史 custody。不要把一次小型数值通过升级成普遍硬件保证。

维护方式见 [MAINTENANCE.md](MAINTENANCE.md)，来源和复用范围见 [PROVENANCE.md](PROVENANCE.md)。提交前运行 `./mwiki validate`，并对变更主题执行实际检索；改动安装或入口时，运行 `python3 scripts/test_install_skill.py`，验证临时安装从其他工作目录能查询，且跟随链接遍历时仅有一个 `SKILL.md`。无需重跑 GPU 来验证文字或打包修改。
