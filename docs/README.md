# HTML overview / HTML 总览

[中英文总览](index.html) · [English overview](index.en.html)

两份 HTML 是单文件浏览页面，可直接下载或从 checkout 打开。GitHub 文件页显示源码。主题链接指向公开仓库的 Markdown 页面；阅读主题需要网络。总览本身包含样式和交互，无需本地服务。

Both HTML files are standalone browsing views. Open them locally after cloning or downloading; GitHub's file viewer shows source. Topic links open the public repository's rendered Markdown. Reading linked topics requires network access.

`data/catalog.json` remains the sole topic/source index. `scripts/build_overview.py` generates Markdown drafts, then invokes the installed answer-me-with-html skill's renderer. The committed drafts preserve the page content; HTML is generated.

To rebuild, install that skill and pass its renderer path:

~~~bash
python3 scripts/build_overview.py --renderer /absolute/path/answer-me-with-html/scripts/am.mjs
./mwiki validate
~~~

Requires Node.js 20+ for the renderer and Python 3.9+ for the generator. The generator runs no model, GPU experiment or publishing command. It overwrites only generated files under docs/.

After topic/source/translation metadata changes, rebuild both pages. Review the generated content before committing. English companions explain mechanisms, observations and limits; Chinese records preserve full tables, code and history. Generating HTML does not expand experimental evidence.
