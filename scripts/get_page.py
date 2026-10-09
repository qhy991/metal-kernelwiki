#!/usr/bin/env python3
"""Read a registered page or source; source following prints metadata only."""

import argparse
import json
import sys

sys.dont_write_bytecode = True

from _kb import KnowledgeBaseError, load_catalog, localized_page, matching_resource, read_page, unique_index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identifier", help="Page ID, source ID, or exact registered relative Markdown path")
    parser.add_argument("--follow-sources", action="store_true", help="Append registered source metadata, without any network requests")
    parser.add_argument("--lang", choices=("zh", "en"), default=None, help="Page edition; an exact registered path selects its own edition")
    args = parser.parse_args(argv)
    try:
        data = load_catalog()
        kind, resource = matching_resource(data, args.identifier)
        if kind == "source":
            print(json.dumps(resource, ensure_ascii=False, indent=2))
            return 0
        if args.lang is not None:
            # Resolve the topic again so --lang zh can override an English path.
            resource = localized_page(unique_index(data["pages"], "page")[resource["id"]], args.lang)
        body = read_page(resource)
        followed = []
        if args.follow_sources:
            source_index = unique_index(data["sources"], "source")
            for ident in resource.get("sources", []):
                if ident not in source_index:
                    raise KnowledgeBaseError("Page references unknown source: {}".format(ident))
                followed.append(source_index[ident])
        print(body, end="" if body.endswith("\n") else "\n")
        if args.follow_sources:
            print("\n## Registered sources (metadata only; no network access)\n")
            print(json.dumps(followed, ensure_ascii=False, indent=2))
        return 0
    except (KnowledgeBaseError, TypeError) as exc:
        print("get_page: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
