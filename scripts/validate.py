#!/usr/bin/env python3
"""Validate skill structure; never open or verify external experiment artifacts."""

import datetime
import re
import sys
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

sys.dont_write_bytecode = True

from _kb import ROOT, KnowledgeBaseError, load_catalog, read_page, safe_path

PAGE_TEXT = ("id", "title", "path", "type", "confidence", "summary")
PAGE_LISTS = ("engines", "tags", "symptoms", "sources")
SOURCE_TEXT = ("id", "title", "kind", "checked", "revision", "note")
SOURCE_KINDS = {"official-doc", "upstream-code", "merged-pr", "issue", "discussion", "local-experiment"}


def nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def valid_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        datetime.date.fromisoformat(value)
        return True
    except ValueError:
        return False


def markdown_links(text):
    """Recognize inline and reference definitions; ignore code and anchor-only links."""
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            run = marker.group(1)
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence):
                fence = None
            continue
        if fence is None:
            lines.append(line)
    text = re.sub(r"(`+).*?\1", "", "\n".join(lines))
    # Parenthesized destinations may themselves contain one balanced pair.
    pattern = r"!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|(?:[^\s()\\]|\\.|\([^()]*\))+)"
    for match in re.finditer(pattern, text):
        yield match.group(1).strip("<>")
    for match in re.finditer(r"^\s{0,3}\[[^\]]+\]:\s*(<[^>\n]+>|\S+)", text, re.M):
        yield match.group(1).strip("<>")


def validate(data):
    errors = []
    if "checked" in data and not valid_date(data["checked"]):
        errors.append("catalog checked must be an ISO calendar date")
    aliases = data.get("aliases", {})
    for canonical, variants in aliases.items():
        if not nonempty_text(canonical) or not isinstance(variants, list) or any(not nonempty_text(x) for x in variants):
            errors.append("invalid alias entry: {!r}".format(canonical))
    seen = set()
    source_ids = set()
    for collection, required in (("pages", PAGE_TEXT), ("sources", SOURCE_TEXT)):
        for number, row in enumerate(data[collection]):
            label = "{}[{}]".format(collection, number)
            for field in required:
                if not nonempty_text(row.get(field)):
                    errors.append("{} requires nonempty {}".format(label, field))
            ident = row.get("id")
            if nonempty_text(ident):
                if ident in seen:
                    errors.append("duplicate id: {}".format(ident))
                seen.add(ident)
                if collection == "sources":
                    source_ids.add(ident)
    for source in data["sources"]:
        label = "source {!r}".format(source.get("id"))
        if source.get("kind") not in SOURCE_KINDS:
            errors.append("{} has invalid kind".format(label))
        if "mutable" in source and not isinstance(source["mutable"], bool):
            errors.append("{} needs boolean mutable".format(label))
        if not valid_date(source.get("checked")):
            errors.append("{} needs ISO checked date".format(label))
        if source.get("kind") == "local-experiment":
            if "url" in source or "artifact_path" in source:
                errors.append("{} uses artifact_ref, without URL or machine-local path".format(label))
            artifact = source.get("artifact_ref")
            # Logical external reference only; never resolve/read external artifacts.
            if not nonempty_text(artifact):
                errors.append("{} needs a relative external artifact_ref".format(label))
            elif ("\x00" in artifact or "\\" in artifact or ":" in artifact
                  or artifact != PurePosixPath(artifact).as_posix()
                  or PurePosixPath(artifact).is_absolute()
                  or ".." in PurePosixPath(artifact).parts
                  or len(PurePosixPath(artifact).parts) < 2):
                errors.append("{} needs a normalized relative artifact_ref with run and file".format(label))
        else:
            if "artifact_path" in source or "artifact_ref" in source:
                errors.append("{} cannot combine remote url with external artifact reference".format(label))
            url = source.get("url", "")
            try:
                parsed = urlsplit(url) if isinstance(url, str) else None
                if parsed is None or parsed.scheme not in ("https", "http") or not parsed.netloc:
                    errors.append("{} needs an absolute HTTP(S) URL".format(label))
            except ValueError:
                errors.append("{} has malformed URL".format(label))
    page_paths = set()
    for page in data["pages"]:
        label = "page {!r}".format(page.get("id"))
        for field in PAGE_LISTS:
            value = page.get(field)
            if not isinstance(value, list) or not value or any(not nonempty_text(x) for x in value):
                errors.append("{} needs nonempty string list {}".format(label, field))
        if page.get("confidence") not in ("documented", "source-reported", "inferred", "experimental", "locally-measured"):
            errors.append("{} has invalid confidence".format(label))
        if page.get("type") not in ("workflow", "deployment", "technique", "hardware", "profiling"):
            errors.append("{} has invalid type".format(label))
        references = page.get("sources", [])
        if isinstance(references, list):
            for ident in references:
                if isinstance(ident, str) and ident not in source_ids:
                    errors.append("{} references unknown source {}".format(label, ident))
        try:
            path = safe_path(page.get("path"))
            if path in page_paths:
                errors.append("duplicate page path: {}".format(page.get("path")))
            page_paths.add(path)
            read_page(page)
        except KnowledgeBaseError as exc:
            errors.append(str(exc))
    wiki = ROOT / "wiki"
    if wiki.is_dir():
        for path in wiki.rglob("*.md"):
            if path.resolve() not in page_paths:
                errors.append("unindexed wiki page: {}".format(path.relative_to(ROOT)))
    # Include SKILL.md, references and other Markdown, not only catalogued pages.
    for path in sorted(ROOT.rglob("*.md")):
        try:
            safe_path(path.relative_to(ROOT).as_posix())
            text = path.read_text(encoding="utf-8")
        except (KnowledgeBaseError, OSError, UnicodeError) as exc:
            errors.append("cannot inspect Markdown {}: {}".format(path, exc))
            continue
        for target in markdown_links(text):
            try:
                target = re.sub(r"\\([()\[\] ])", r"\1", target)
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                local = unquote(parsed.path)
                if local.startswith("/"):
                    errors.append("{} has nonportable absolute file link: {}".format(path.relative_to(ROOT), target))
                    continue
                resolved = (path.parent / local).resolve()
                try:
                    relative = resolved.relative_to(ROOT)
                except ValueError:
                    errors.append("{} link escapes skill: {}".format(path.relative_to(ROOT), target))
                    continue
                if not resolved.exists():
                    errors.append("{} has missing relative link: {}".format(path.relative_to(ROOT), relative))
            except (ValueError, OSError) as exc:
                errors.append("{} has invalid link {!r}: {}".format(path.relative_to(ROOT), target, exc))
    return errors


def main():
    try:
        data = load_catalog()
        errors = validate(data)
    except (KnowledgeBaseError, TypeError, ValueError) as exc:
        errors = [str(exc)]
    if errors:
        for error in errors:
            print("ERROR: " + error, file=sys.stderr)
        print("Validation failed: {} error(s).".format(len(errors)), file=sys.stderr)
        return 1
    print("Validated {} pages, {} sources and local Markdown links.".format(len(data["pages"]), len(data["sources"])))
    print("Structural validation only; external artifacts and local measurements were not opened or verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
