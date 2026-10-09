#!/usr/bin/env python3
"""Search local metadata and Markdown; never contact a service or run a model."""

import argparse
import json
import sys

sys.dont_write_bytecode = True

from _kb import KnowledgeBaseError, alias_lookup, contains, load_catalog, localized_page, normalized, query_groups, read_page, unique_index


def expanded_values(values, aliases):
    result = set()
    for value in values:
        key = normalized(value)
        result.update(aliases.get(key, {key}))
    return result


def filter_values(values, aliases):
    values = [value for item in values or [] for value in item.split(",") if value.strip()]
    return expanded_values(values, aliases)


def rank(page, body, groups):
    fields = {
        "id": (normalized(page.get("id", "")), 7),
        "title": (normalized(page.get("title", "")), 10),
        "summary": (normalized(page.get("summary", "")), 6),
        "tags": (normalized(" ".join(page.get("tags", []))), 8),
        "symptoms": (normalized(" ".join(page.get("symptoms", []))), 8),
        "engines": (normalized(" ".join(page.get("engines", []))), 3),
        "type": (normalized(page.get("type", "")), 2),
        "body": (normalized(body), 1),
    }
    score = 0
    matched = set()
    for group in groups:
        for name, (content, weight) in fields.items():
            if any(contains(content, term) for term in group):
                score += weight
                matched.add(name)
    return score, sorted(matched)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default="", help="Text or Chinese/English aliases")
    for flag in ("engine", "type", "tag", "symptom"):
        parser.add_argument("--" + flag, action="append", help="Exact value or alias; repeated/comma values are OR, categories are AND")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--lang", choices=("zh", "en"), default="zh", help="Page edition; English is a companion guide")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="Emit a JSON array")
    output.add_argument("--paths-only", action="store_true", help="Emit registered relative page paths")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    try:
        data = load_catalog()
        unique_index(data["pages"], "page")
        groups = query_groups(args.text, data.get("aliases", {}))
        aliases = alias_lookup(data.get("aliases", {}))
        filters = {
            "engines": filter_values(args.engine, aliases),
            "type": filter_values(args.type, aliases),
            "tags": filter_values(args.tag, aliases),
            "symptoms": filter_values(args.symptom, aliases),
        }
        results = []
        for page in data["pages"]:
            page = localized_page(page, args.lang)
            keep = True
            for field, wanted in filters.items():
                values = [page.get(field, "")] if field == "type" else page.get(field, [])
                if wanted and not wanted.intersection(expanded_values(values, aliases)):
                    keep = False
                    break
            if not keep:
                continue
            score, matched = rank(page, read_page(page), groups)
            if groups and score == 0:
                continue
            result = dict(page)
            result.update(score=score, matched_fields=matched)
            results.append(result)
        results.sort(key=lambda row: (-row["score"], row["id"]))
        results = results[:args.limit]
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        elif args.paths_only:
            for result in results:
                print(result["path"])
        elif not results:
            print("No matching pages. Try a different term or remove a filter.")
        else:
            for result in results:
                print("{} | {} | score={} | confidence={}".format(result["id"], result["title"], result["score"], result.get("confidence", "unknown")))
                print("  {}".format(result["path"]))
                print("  {}".format(result.get("summary", "")))
        return 0
    except (KnowledgeBaseError, TypeError, KeyError) as exc:
        print("query: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
