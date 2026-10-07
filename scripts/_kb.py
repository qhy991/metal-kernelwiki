"""Local-only helpers for the Metal knowledge base (Python 3.9+)."""

import json
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "catalog.json"


class KnowledgeBaseError(ValueError):
    pass


def load_catalog():
    try:
        data = json.loads(CATALOG.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KnowledgeBaseError("Cannot read catalog {}: {}".format(CATALOG, exc))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise KnowledgeBaseError("Catalog must be an object with schema_version 1")
    for key in ("pages", "sources"):
        if not isinstance(data.get(key), list):
            raise KnowledgeBaseError("Catalog {} must be a list".format(key))
        if any(not isinstance(row, dict) for row in data[key]):
            raise KnowledgeBaseError("Every {} entry must be an object".format(key))
    if not isinstance(data.get("aliases", {}), dict):
        raise KnowledgeBaseError("Catalog aliases must be an object")
    return data


def safe_path(relative):
    """Resolve one local resource without permitting traversal or symlink escape."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise KnowledgeBaseError("Invalid relative path: {!r}".format(relative))
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or ":" in relative:
        raise KnowledgeBaseError("Path must stay inside the skill: {!r}".format(relative))
    try:
        resolved = (ROOT / relative).resolve()
    except (OSError, RuntimeError) as exc:
        raise KnowledgeBaseError("Cannot resolve path {!r}: {}".format(relative, exc))
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        raise KnowledgeBaseError("Path escapes the skill: {!r}".format(relative))
    return resolved


def read_page(page):
    path = safe_path(page.get("path"))
    if path.suffix.lower() != ".md":
        raise KnowledgeBaseError("Page must be Markdown: {}".format(page.get("path")))
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise KnowledgeBaseError("Cannot read page {}: {}".format(page.get("id"), exc))


def unique_index(rows, label):
    result = {}
    for row in rows:
        ident = row.get("id")
        if not isinstance(ident, str) or not ident:
            raise KnowledgeBaseError("{} needs a nonempty string id".format(label))
        if ident in result:
            raise KnowledgeBaseError("Duplicate {} id: {}".format(label, ident))
        result[ident] = row
    return result


def normalized(value):
    return " ".join(str(value).casefold().split())


def alias_lookup(aliases):
    """Expand exact aliases; shared names accumulate without order precedence."""
    result = {}
    for canonical, variants in aliases.items():
        if not isinstance(canonical, str) or not canonical or not isinstance(variants, list):
            raise KnowledgeBaseError("Alias entries must map nonempty strings to string lists")
        if any(not isinstance(value, str) or not value for value in variants):
            raise KnowledgeBaseError("Alias variants must be nonempty strings")
        names = {normalized(value) for value in [canonical] + variants}
        for name in names:
            result.setdefault(name, set()).update(names)
    return result


def contains(text, term):
    if re.search(r"[\u3400-\u9fff]", term):
        return term in text
    return re.search(r"(?<![a-z0-9_])" + re.escape(term) + r"(?![a-z0-9_])", text) is not None


def query_groups(text, aliases):
    """Keep literal terms and add alias concepts, including mixed Chinese terms."""
    query = normalized(text)
    if not query:
        return []
    groups = []
    for canonical, variants in aliases.items():
        if not isinstance(canonical, str) or not isinstance(variants, list):
            raise KnowledgeBaseError("Alias entries must map strings to string lists")
        if any(not isinstance(v, str) or not v for v in variants):
            raise KnowledgeBaseError("Alias variants must be nonempty strings")
        terms = sorted(set(normalized(v) for v in [canonical] + variants if v))
        if any(contains(query, term) for term in terms):
            groups.append(tuple(terms))
    # Whole phrases help exact requests; individual tokens permit mixed queries.
    tokens = [query] + re.findall(r"[a-z0-9][a-z0-9_+.\-]*|[\u3400-\u9fff]+", query)
    alias_terms = {term for group in groups for term in group}
    for token in tokens:
        if token and token not in alias_terms:
            groups.append((token,))
    return sorted(set(groups))


def matching_resource(data, identifier):
    pages = unique_index(data["pages"], "page")
    sources = unique_index(data["sources"], "source")
    if set(pages) & set(sources):
        raise KnowledgeBaseError("Page and source IDs must be globally unique")
    if identifier in pages:
        return "page", pages[identifier]
    if identifier in sources:
        return "source", sources[identifier]
    matches = [page for page in pages.values() if page.get("path") == identifier]
    if len(matches) == 1:
        return "page", matches[0]
    if len(matches) > 1:
        raise KnowledgeBaseError("Duplicate registered page path: {}".format(identifier))
    raise KnowledgeBaseError("Unknown page/source ID or registered path: {}".format(identifier))
