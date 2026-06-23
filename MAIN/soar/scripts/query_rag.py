#!/usr/bin/env python3
"""CLI entry point for SpaceRAG V2 (Hierarchical Page-Linked RAG)."""

import argparse
import json
import sys
from pathlib import Path

# Add the project root to sys.path if needed to import from search/
sys.path.append(str(Path(__file__).resolve().parent.parent))

from search.spacerag_v2_search import SpaceRAGv2Searcher, DEFAULT_RAG_DIR


def main():
    parser = argparse.ArgumentParser(description="Query SpaceRAG V2 (Hierarchical Page-Linked RAG).")
    parser.add_argument("query", help="User search query")
    parser.add_argument("--rag-dir", type=Path, default=DEFAULT_RAG_DIR, help="Path to RAG directory (e.g. SpaceRAG_v2)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of pages to return")
    parser.add_argument("--doc-k", type=int, default=8)
    parser.add_argument("--page-k", type=int, default=20)
    parser.add_argument("--chunk-k", type=int, default=40)
    parser.add_argument("--table-k", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    
    args = parser.parse_args()

    if not args.rag_dir.exists():
        print(f"Error: RAG directory not found at {args.rag_dir}", file=sys.stderr)
        return 1

    try:
        searcher = SpaceRAGv2Searcher(rag_dir=args.rag_dir)
    except Exception as e:
        print(f"Error initializing searcher: {e}", file=sys.stderr)
        return 1

    results = searcher.search_as_dict(
        query=args.query,
        top_k=args.top_k,
        doc_k=args.doc_k,
        page_k=args.page_k,
        chunk_k=args.chunk_k,
        table_k=args.table_k,
        timeout=args.timeout
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"\nQuery: {args.query}")
        print(f"RAG Directory: {args.rag_dir}")
        print(f"Results found: {len(results)}\n")
        
        for i, res in enumerate(results, 1):
            print(f"[{i}] {res['doc_title']} | Page {res['page_no']} (Score: {res['rerank_score']:.4f})")
            print(f"    Summary: {res['page_summary'][:200]}...")
            
            if res['chunks']:
                print(f"    Chunks: {len(res['chunks'])} matched")
                for c in res['chunks'][:1]:
                    snippet = c['chunk_text_display'].replace('\n', ' ')
                    print(f"      - {snippet[:150]}...")
            
            if res['tables']:
                print(f"    Tables: {len(res['tables'])} linked")
                for t in res['tables']:
                    print(f"      - {t['table_title']} ({t['table_id']})")
            
            if res['images']:
                print(f"    Images: {len(res['images'])} linked")
                for img in res['images']:
                    print(f"      - {img['image_title']} ({img['image_id']})")
            print("-" * 60)
            
    searcher.emit_warnings()
    return 0


if __name__ == "__main__":
    sys.exit(main())
