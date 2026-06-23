#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spacerag_v2_search import DEFAULT_RAG_DIR, SpaceRAGv2Searcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query SpaceRAG_v2 with content-first retrieval and page-linked asset expansion.")
    parser.add_argument("query", help="User query")
    parser.add_argument("--rag-dir", type=Path, default=DEFAULT_RAG_DIR)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--doc-k", type=int, default=8)
    parser.add_argument("--page-k", type=int, default=20)
    parser.add_argument("--chunk-k", type=int, default=40)
    parser.add_argument("--table-k", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    searcher = SpaceRAGv2Searcher(rag_dir=args.rag_dir, env_path=args.env_file)
    results = searcher.search_as_dict(
        args.query,
        top_k=args.top_k,
        doc_k=args.doc_k,
        page_k=args.page_k,
        chunk_k=args.chunk_k,
        table_k=args.table_k,
        timeout=args.timeout,
    )
    searcher.emit_warnings()

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")
    print("")
    for idx, item in enumerate(results, start=1):
        print(
            f"[{idx}] {item['doc_title']} | page={item['page_no']} | "
            f"rerank={item['rerank_score']:.4f} page_score={item['page_score']:.4f}"
        )
        summary = item["page_summary"].replace("\n", " ")
        print(f"    summary: {summary[:260]}")
        for chunk in item["chunks"]:
            snippet = chunk["chunk_text_clean"].replace("\n", " ")
            print(f"    chunk[{chunk['chunk_order']}] score={chunk['score']:.4f}: {snippet[:220]}")
        for table in item["tables"]:
            print(f"    table score={table['score']:.4f}: {table['table_title']} -> {table['asset_path']}")
        for image in item["images"]:
            print(f"    image score={image['score']:.4f}: {image['image_title']} -> {image['image_path']}")
        print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
