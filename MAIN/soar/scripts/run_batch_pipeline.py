#!/usr/bin/env python3
"""Run the document-processing pipeline with Python only.

Stages:
1. normalization.py
2. convert_excel_tables_to_markdown.py
3. extract_document_texts.py
4. extract_document_keywords.py
5. build_rag_from_excel.py

Optional bootstrap mode:
- run extract_document_texts.py first with --bootstrap-workbooks
- auto-create 标注.xlsx and fill worksheet #3 (`文本`)
- then continue with the normal pipeline

The pipeline is data-root driven and works for any industry directory layout
that follows the existing workbook conventions.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from env_utils import resolve_env_value


DEFAULT_DATA_ROOT = Path("source_data")
DEFAULT_OUTPUT_DIR = Path("SpaceRAG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the batch processing pipeline for any data root.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter for OCR/text/keyword stages.")
    parser.add_argument("--rag-python-bin", default=None, help="Optional Python interpreter for the RAG stage.")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--rpm", type=int, default=500)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--table-workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--table-timeout", type=float, default=300.0)
    parser.add_argument("--ocr-lang", default="chi_sim+eng")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--bootstrap-text-first",
        action="store_true",
        help="Optional text-first mode: auto-create missing annotation workbooks and populate worksheet #3 before later stages.",
    )
    parser.add_argument("--skip-normalization", action="store_true")
    parser.add_argument("--skip-tables", action="store_true")
    parser.add_argument("--skip-texts", action="store_true")
    parser.add_argument("--skip-keywords", action="store_true")
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--skip-text-keywords", action="store_true", help="Skip keyword generation inside text extraction.")
    parser.add_argument("--rebuild-keywords", action="store_true", help="Rebuild keyword columns via normalization.py after tables/texts.")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--disable-faiss", action="store_true")
    parser.add_argument("--force-texts", action="store_true")
    parser.add_argument("--force-keywords", action="store_true")
    parser.add_argument("--force-rag", action="store_true")
    parser.add_argument("--keyword-force", action="store_true", help="Force overwrite keywords in normalization.")
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.api_key = resolve_env_value("SILICONFLOW_API_KEY", args.api_key)
    return args


def build_env(api_key: str) -> dict[str, str]:
    env = os.environ.copy()
    if api_key:
        env["SILICONFLOW_API_KEY"] = api_key
    return env


def run_stage(name: str, command: list[str], env: dict[str, str], dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"[Stage] {name}")
    print(f"  Command: {printable}")
    if dry_run:
        return
    subprocess.run(command, check=True, env=env)


def require_api_key(enabled: bool, api_key: str, stage_name: str) -> None:
    if enabled and not api_key:
        raise ValueError(f"Missing SiliconFlow API key for stage: {stage_name}")


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    python_bin = str(Path(args.python_bin).expanduser())
    rag_python_bin = str(Path(args.rag_python_bin).expanduser()) if args.rag_python_bin else python_bin

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    scripts_dir = Path(__file__).resolve().parent
    env = build_env(args.api_key)

    if not args.dry_run:
        if args.bootstrap_text_first and args.skip_texts:
            raise ValueError("--bootstrap-text-first requires the texts stage; do not combine it with --skip-texts.")
        require_api_key(enabled=not args.skip_tables, api_key=args.api_key, stage_name="tables")
        require_api_key(enabled=(not args.skip_keywords) and (not args.rebuild_keywords), api_key=args.api_key, stage_name="keywords")
        require_api_key(enabled=args.rebuild_keywords, api_key=args.api_key, stage_name="keyword rebuild")
        require_api_key(
            enabled=(not args.skip_rag) and (not args.skip_embeddings),
            api_key=args.api_key,
            stage_name="rag embeddings",
        )

    texts_already_run = False

    if args.bootstrap_text_first:
        text_cmd = [
            python_bin,
            str(scripts_dir / "extract_document_texts.py"),
            "--data-root",
            str(data_root),
            "--project-root",
            str(project_root),
            "--max-workers",
            str(args.max_workers),
            "--rpm",
            str(args.rpm),
            "--timeout",
            str(args.timeout),
            "--bootstrap-workbooks",
        ]
        if args.api_key:
            text_cmd.extend(["--api-key", args.api_key])
        if args.skip_text_keywords:
            text_cmd.append("--skip-keywords")
        if args.force_texts:
            text_cmd.append("--force-reextract")
        run_stage("texts-bootstrap", text_cmd, env=env, dry_run=args.dry_run)
        texts_already_run = True

    if not args.skip_normalization:
        normalization_cmd = [
            python_bin,
            str(scripts_dir / "normalization.py"),
            "--data-root",
            str(data_root),
            "--apply",
        ]
        run_stage(
            "normalization",
            normalization_cmd,
            env=env,
            dry_run=args.dry_run,
        )

    if not args.skip_tables:
        table_cmd = [
            python_bin,
            str(scripts_dir / "convert_excel_tables_to_markdown.py"),
            "--data-root",
            str(data_root),
            "--max-workers",
            str(args.table_workers),
            "--rpm",
            str(args.rpm),
            "--timeout",
            str(args.table_timeout),
        ]
        if args.api_key:
            table_cmd.extend(["--api-key", args.api_key])
        run_stage(
            "tables",
            table_cmd,
            env=env,
            dry_run=args.dry_run,
        )

    if not args.skip_texts and not texts_already_run:
        text_cmd = [
            python_bin,
            str(scripts_dir / "extract_document_texts.py"),
            "--data-root",
            str(data_root),
            "--project-root",
            str(project_root),
            "--max-workers",
            str(args.max_workers),
            "--rpm",
            str(args.rpm),
            "--timeout",
            str(args.timeout),
        ]
        if args.api_key:
            text_cmd.extend(["--api-key", args.api_key])
        if args.skip_text_keywords:
            text_cmd.append("--skip-keywords")
        if args.force_texts:
            text_cmd.append("--force-reextract")
        run_stage("texts", text_cmd, env=env, dry_run=args.dry_run)

    if args.rebuild_keywords:
        rebuild_cmd = [
            python_bin,
            str(scripts_dir / "normalization.py"),
            "--data-root",
            str(data_root),
            "--apply",
            "--rebuild-keywords",
            "--keyword-force",
            "--keyword-max-workers",
            str(args.max_workers),
            "--keyword-rpm",
            str(args.rpm),
            "--keyword-timeout",
            str(args.timeout),
        ]
        if args.api_key:
            rebuild_cmd.extend(["--api-key", args.api_key])
        run_stage("keyword-rebuild", rebuild_cmd, env=env, dry_run=args.dry_run)

    if not args.skip_keywords and not args.rebuild_keywords:
        keyword_cmd = [
            python_bin,
            str(scripts_dir / "extract_document_keywords.py"),
            "--data-root",
            str(data_root),
            "--max-workers",
            str(args.max_workers),
            "--rpm",
            str(args.rpm),
            "--timeout",
            str(args.timeout),
            "--save-every",
            str(args.save_every),
            *(["--force"] if args.force_keywords else []),
        ]
        if args.api_key:
            keyword_cmd.extend(["--api-key", args.api_key])
        run_stage(
            "keywords",
            keyword_cmd,
            env=env,
            dry_run=args.dry_run,
        )

    if not args.skip_rag:
        rag_cmd = [
            rag_python_bin,
            str(scripts_dir / "build_rag_from_excel.py"),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--timeout",
            str(args.timeout),
        ]
        if args.api_key:
            rag_cmd.extend(["--api-key", args.api_key])
        if args.force_rag:
            rag_cmd.append("--force-rebuild")
        if args.clear_output:
            rag_cmd.append("--clear-output")
        if args.skip_embeddings:
            rag_cmd.append("--skip-embeddings")
        if args.disable_faiss:
            rag_cmd.append("--disable-faiss")
        run_stage("rag", rag_cmd, env=env, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
