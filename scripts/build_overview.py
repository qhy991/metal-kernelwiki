#!/usr/bin/env python3
"""Build bilingual catalog views using the answer-me-with-html renderer."""

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/qhy991/metal-kernelwiki/blob/main/"


def link(label, path):
    return "[{}]({})".format(label.replace("|", "\\|"), GITHUB + path)


def draft(data, language):
    english = language == "en"
    pages = data["pages"]
    sources = data["sources"]
    local = sum(source["kind"] == "local-experiment" for source in sources)
    translated = sum("en" in page.get("translations", {}) for page in pages)
    output = [
        "---", "template: doc", "theme: shadcn", "lang: " + language,
        "title: Metal KernelWiki",
        "subtitle: " + ("English guide to the skill and its knowledge library" if english else "Metal LLM 部署与 MSL 优化 · 中英文知识总览"),
        "style: 80", "---", "",
        "## A " + ("What is in this skill?" if english else "现在的 skill 有什么？"), "",
        ("The skill retrieves deployment and kernel knowledge for Apple Silicon, MLX and llama.cpp."
         if english else "这个 skill 面向 Apple Silicon、MLX 与 llama.cpp。它帮助检索部署和 kernel 优化知识。"), "",
        "| " + ("Item | Count" if english else "内容 | 数量") + " |",
        "|---|---|",
        "| " + ("Topics" if english else "主题") + " | {} |".format(len(pages)),
        "| " + ("English companions" if english else "英文伴读版") + " | {} |".format(translated),
        "| " + ("Registered sources" if english else "来源条目") + " | {} |".format(len(sources)),
        "| " + ("Local-experiment sources" if english else "本地实验来源") + " | {} |".format(local), "",
        ("English guides cover mechanisms, main observations and limits. Chinese records retain full tables, code and history."
         if english else "所有主题都有英文伴读版，覆盖机制、主要结果与限制。中文保留详细表格、代码和历史记录。"), "",
        ("This is a companion edition, not a line-by-line translation. Shared source notes retain their original language."
         if english else "英文版不是逐句翻译。来源注释保留原语言。"), "",
        ("[中文 / bilingual overview](index.html)" if english else "[English overview](index.en.html)"),
        link("English README", "README.en.md") + " · " + link("中文 README", "README.md"), "",
        "## B " + ("How does the skill work?" if english else "如何使用？"), "",
        "1. " + ("Identify the device, framework, workload and target metric." if english else "确定设备、框架、工作负载和目标指标。"),
        "2. " + ("Retrieve the relevant topics and inspect their sources." if english else "检索相关主题，查看一手来源。"),
        "3. " + ("Compare distinct candidates under the same numerical contract." if english else "用相同数值合同比较不同候选。"),
        "4. " + ("Validate correctness, then measure full call costs." if english else "先验正确性，再测完整调用成本。"), "",
        ("The thin skill points to this repository. It does not duplicate the wiki."
         if english else "薄 skill 入口指向这个仓库，正文无需复制。"), "",
        link("SKILL.md", "skill/metal-kernelwiki/SKILL.md") + " · " + link("Workflow", "references/en/optimization-workflow.md"), "",
    ]
    deployment = {
        "mlx-deployment", "mlx-execution", "mlx-kv-cache", "mlx-serving",
        "llamacpp-deployment", "llamacpp-serving", "llamacpp-metal-tuning",
    }
    common = {
        "measurement", "runtime-measurement", "metal-capabilities", "metal-memory-threadgroups",
        "metal-profiling", "metal-tensors", "attention", "gemm-moe", "quantization", "fusion",
    }
    groups = [
        ("C", "Deployment and serving", "部署与服务", [page for page in pages if page["id"] in deployment]),
        ("D", "MSL programming and optimization", "MSL 用法与优化", [page for page in pages if page["id"].startswith("msl-")]),
        ("E", "Local observations and failure boundaries", "本地观测与失败边界", [page for page in pages if page["id"] not in deployment | common and not page["id"].startswith("msl-")]),
        ("F", "Mechanisms, hardware and measurement", "机制、硬件与测量", [page for page in pages if page["id"] in common]),
    ]
    assert sorted(page["id"] for group in groups for page in group[3]) == sorted(page["id"] for page in pages)
    for ident, en_title, zh_title, rows in groups:
        output += ["## {} {} ({} topics)".format(ident, en_title if english else zh_title, len(rows)), ""]
        if english:
            output += ["| English topic | 中文 record | Evidence basis |", "|---|---|---|"]
        else:
            output += ["| 中文主题 | English guide | 证据来源 |", "|---|---|---|"]
        for page in sorted(rows, key=lambda value: value["id"]):
            en = page["translations"]["en"]
            zh_link = link("中文" if english else page["title"], page["path"])
            en_link = link(en["title"], en["path"])
            output.append("| {} | {} | {} |".format(en_link if english else zh_link, zh_link if english else en_link, page["confidence"]))
        output.append("")
    output += [
        "## G " + ("What does the evidence establish?" if english else "证据能说明什么？"), "",
        "| Label | " + ("Meaning" if english else "含义") + " |", "|---|---|",
        "| documented | " + ("Documented mechanism" if english else "官方说明的机制") + " |",
        "| source-reported | " + ("Upstream implementation or author report" if english else "上游实现或作者报告") + " |",
        "| inferred | " + ("Analysis-based inference" if english else "基于分析的推断") + " |",
        "| experimental | " + ("Candidate awaiting validation" if english else "待验证候选") + " |",
        "| locally-measured | " + ("Bounded local device/version observation" if english else "限定设备和版本的本机观测") + " |", "",
        ("A page may contain several evidence bases. Read the scope of each claim."
         if english else "一页可以包含多种证据。采用结论前，应查看对应范围。"), "",
        ("Primitive success does not establish model quality or deployment speed. Exporting a trace does not establish profiler analysis."
         if english else "原语通过不等于模型质量或部署收益已验收。导出 trace 不等于 profiler 已解析。"), "",
        ("Failures remain in the records. Raw experiments are external and unpublished."
         if english else "失败结果保留在记录中。原始实验存于仓库外，尚未公开。"), "",
        link("Catalog", "data/catalog.json") + " · " + link("Provenance", "PROVENANCE.en.md"), "",
        "## H " + ("Retrieve, install and maintain" if english else "检索、安装与维护"), "",
        "~~~bash",
        './mwiki query "MSL reduction precision" --lang en',
        "./mwiki get msl-f32-divide --lang en --follow-sources",
        "./mwiki get measurement --lang zh",
        "./mwiki validate",
        "python3 scripts/install_skill.py",
        "~~~", "",
        ("Retrieval uses Python's standard library and starts no GPU work. Source following prints metadata only."
         if english else "检索只用 Python 标准库，不启动 GPU。跟随来源只显示元数据。"), "",
        ("Topic links open rendered Markdown on GitHub. This overview also opens locally without a server."
         if english else "主题链接打开 GitHub 的 Markdown 页面。总览可直接在本机打开，无需服务。"), "",
        link("Maintenance", "MAINTENANCE.en.md") + " · " + link("Examples", "references/en/examples.md")
        + " · " + link("Rebuild HTML", "docs/README.md"), "",
    ]
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", type=Path, required=True, help="Path to answer-me-with-html/scripts/am.mjs")
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    data = json.loads((ROOT / "data/catalog.json").read_text(encoding="utf-8"))
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    for language, basename in (("zh", "index"), ("en", "index.en")):
        source = docs / (basename + ".md")
        target = docs / (basename + ".html")
        source.write_text(draft(data, language), encoding="utf-8")
        subprocess.run([args.node, str(args.renderer.expanduser()), "render", str(source),
                        "-o", str(target), "--no-open"], check=True)


if __name__ == "__main__":
    main()
