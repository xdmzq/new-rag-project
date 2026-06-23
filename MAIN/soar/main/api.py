from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple
import json
import os

import traceback
import sys
def _apply_project_dotenv(override: bool = False) -> None:
    """Load PROJECT_ROOT/.env for direct main/api.py imports in CLI/tests."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(str(env_path), override=override)
        return
    except Exception:
        pass

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if override or key not in os.environ:
            os.environ[key] = val


_apply_project_dotenv(override=False)


from paths import PROJECT_ROOT
from rag_core import RAG

BASE_DIR = PROJECT_ROOT / "indexes"
DEFAULT_CATEGORY = "space"
CATEGORY_LABELS = {
    "semiconductor": "半导体库",
    "it_software": "IT软件库",
    "ai": "人工智能库",
    "internet": "互联网库",
    "communication_electronics_hardware": "通信与电子硬件库",
    "intelligent_manufacturing": "智能制造库",
    "biotech_health": "生物技术与健康库",
    "new_materials": "新材料库",
    "energy_environmental": "能源环保库",
    "consumer_modern_services": "消费品与现代服务库",
    "military_special": "军工专用库",
    "space": "商业航天库",
    "quantum_computing": "量子计算库",
    "nuclear_fusion": "核聚变库",
}
V2_KB_CONFIG = {
    "semiconductor": {
        "rag_dir": PROJECT_ROOT / "半导体_RAG",
        "media_prefixes": ("半导体_RAG/",),
    },
    "military_special": {
        "rag_dir": PROJECT_ROOT / "军工_RAG",
        "media_prefixes": ("军工_RAG/",),
    },
    "space": {
        "rag_dir": PROJECT_ROOT / "SpaceRAG_v2",
        "media_prefixes": ("SpaceRAG_v2/",),
    },
    "biotech_health": {
        "rag_dir": PROJECT_ROOT / "生物科技与健康_RAG",
        "media_prefixes": ("生物科技与健康_RAG/",),
    },
    "it_software": {
        "rag_dir": PROJECT_ROOT / "ITSoftwareRAG",
        "media_prefixes": ("ITSoftwareRAG/",),
    },
    "internet": {
        "rag_dir": PROJECT_ROOT / "互联网_RAG",
        "media_prefixes": ("互联网_RAG/",),
    },
    "communication_electronics_hardware": {
        "rag_dir": PROJECT_ROOT / "通信与电子硬件_RAG",
        "media_prefixes": ("通信与电子硬件_RAG/",),
    },
    "intelligent_manufacturing": {
        "rag_dir": PROJECT_ROOT / "SmartManufacturingRAG_v2",
        "media_prefixes": ("SmartManufacturingRAG_v2/",),
    },
    "new_materials": {
        "rag_dir": PROJECT_ROOT / "新材料_RAG",
        "media_prefixes": ("新材料_RAG/",),
    },
    "energy_environmental": {
        "rag_dir": PROJECT_ROOT / "能源环保_RAG",
        "media_prefixes": ("能源环保_RAG/",),
    },
    "consumer_modern_services": {
        "rag_dir": PROJECT_ROOT / "消费品与现代服务_RAG",
        "media_prefixes": ("消费品与现代服务_RAG/",),
    },
}
V2_REQUIRED_FILES = (
    "docs.jsonl",
    "pages.jsonl",
    "chunks.jsonl",
    "tables.jsonl",
    "images.jsonl",
    "relations_chunk_asset.jsonl",
    "vector_store/doc_embeddings.npy",
    "vector_store/page_embeddings.npy",
    "vector_store/chunk_embeddings.npy",
    "vector_store/table_embeddings.npy",
)


def _has_v2_kb_assets(rag_dir: Path) -> bool:
    return all((rag_dir / rel_path).exists() for rel_path in V2_REQUIRED_FILES)


ACTIVE_V2_CATEGORIES = {
    category
    for category, cfg in V2_KB_CONFIG.items()
    if _has_v2_kb_assets(cfg["rag_dir"])
}
ACTIVE_KB_CATEGORIES = set(ACTIVE_V2_CATEGORIES)
EMPTY_KB_CATEGORIES = set(CATEGORY_LABELS.keys()) - ACTIVE_KB_CATEGORIES

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from spacerag_v2_search import SpaceRAGv2Searcher
except Exception:
    SpaceRAGv2Searcher = None

print("Initializing multi-knowledge-base RAG engines...")
try:
    ENGINES = {}
    for category in ACTIVE_KB_CATEGORIES:
        index_path = BASE_DIR / category / "faiss_index.index"
        metadata_path = BASE_DIR / category / "faiss_index_metadata.jsonl"
        if not index_path.exists() or not metadata_path.exists():
            continue
        ENGINES[category] = RAG(
            index_path=str(index_path),
            metadata_path=str(metadata_path),
            kb_category=category,
        )
    LOAD_ERROR = None
    print("RAG engines initialized.")
except Exception as e:
    LOAD_ERROR = str(e)
    print(f"RAG engine initialization failed: {e}")
    ENGINES = {}

_V2_SEARCHERS: Dict[str, Any] = {}
_V2_LOAD_ERRORS: Dict[str, str] = {}
_V2_GENERATION_ENGINES: Dict[str, RAG] = {}
_V2_SOURCE_DOC_PATHS: Dict[str, Dict[str, str]] = {}
_V2_SOURCE_TITLE_PATHS: Dict[str, Dict[str, str]] = {}


def _get_v2_config(category: str) -> Dict[str, Any] | None:
    return V2_KB_CONFIG.get((category or "").strip().lower())


def _get_active_kb_display_names() -> str:
    names = [CATEGORY_LABELS[category] for category in CATEGORY_LABELS if category in ACTIVE_KB_CATEGORIES]
    return "、".join(names) if names else "暂无已上线知识库"


def _get_v2_media_whitelist(category: str) -> Tuple[str, ...]:
    config = _get_v2_config(category) or _get_v2_config(DEFAULT_CATEGORY)
    prefixes = tuple(config.get("media_prefixes", ())) if config else ()
    if not prefixes:
        return ("SpaceRAG_v2/",)
    return tuple(p.replace("\\", "/").rstrip("/") + "/" for p in prefixes if p)


def _normalize_v2_source_rel_path(raw_path: str, *, category: str) -> str:
    config = _get_v2_config(category)
    if not config:
        return ""

    path = str(raw_path or "").strip().replace("\\", "/").lstrip("./")
    if not path:
        return ""

    rag_dir = Path(config["rag_dir"]).resolve()
    candidates: List[Path] = []

    raw_candidate = Path(path)
    if raw_candidate.is_absolute():
        candidates.append(raw_candidate)
    else:
        candidates.append((rag_dir / path).resolve())
        if path.lower().startswith(rag_dir.name.lower() + "/"):
            candidates.append((PROJECT_ROOT / path).resolve())
        tail_parts = raw_candidate.parts[-6:]
        if tail_parts:
            candidates.append((rag_dir / Path(*tail_parts)).resolve())
            candidates.append((PROJECT_ROOT / Path(*tail_parts)).resolve())

    for candidate in candidates:
        try:
            rel_path = candidate.relative_to(PROJECT_ROOT)
        except ValueError:
            continue
        if candidate.exists():
            return rel_path.as_posix()

    if path.lower().startswith("assets/"):
        return f"{rag_dir.name}/{path}"
    if path.lower().startswith(rag_dir.name.lower() + "/"):
        return path
    return ""


def _load_v2_source_maps(category: str) -> None:
    category = (category or DEFAULT_CATEGORY).strip().lower()
    if category in _V2_SOURCE_DOC_PATHS:
        return

    doc_map: Dict[str, str] = {}
    title_map: Dict[str, str] = {}
    config = _get_v2_config(category)
    all_texts_path = Path(config["rag_dir"]) / "all_texts.jsonl" if config else None

    if all_texts_path and all_texts_path.is_file():
        try:
            with all_texts_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    source_rel_path = _normalize_v2_source_rel_path(
                        row.get("file_path") or "",
                        category=category,
                    )
                    if not source_rel_path or not _resolve_abs_media_path(source_rel_path):
                        continue

                    doc_id = str(row.get("doc_id") or "").strip()
                    doc_title = str(row.get("doc_title") or "").strip()
                    if doc_id and doc_id not in doc_map:
                        doc_map[doc_id] = source_rel_path
                    if doc_title and doc_title not in title_map:
                        title_map[doc_title] = source_rel_path
        except OSError:
            pass

    _V2_SOURCE_DOC_PATHS[category] = doc_map
    _V2_SOURCE_TITLE_PATHS[category] = title_map


def _get_v2_source_rel_path(doc_id: str, doc_title: str, *, category: str) -> str:
    _load_v2_source_maps(category)
    doc_map = _V2_SOURCE_DOC_PATHS.get(category, {})
    title_map = _V2_SOURCE_TITLE_PATHS.get(category, {})

    doc_id_key = str(doc_id or "").strip()
    if doc_id_key and doc_id_key in doc_map:
        return doc_map[doc_id_key]

    doc_title_key = str(doc_title or "").strip()
    if doc_title_key and doc_title_key in title_map:
        return title_map[doc_title_key]

    return ""


def _normalize_v2_media_rel_path(raw_path: str, *, allowed_prefixes: Tuple[str, ...]) -> str:
    path = str(raw_path or "").strip().replace("\\", "/").lstrip("./")
    if not path:
        return ""
    lowered = path.lower()
    allowed_lower = tuple(p.lower() for p in allowed_prefixes)

    # 1) already project-relative and in whitelist
    for prefix in allowed_prefixes:
        prefix_lower = prefix.lower()
        if lowered.startswith(prefix_lower):
            return prefix.rstrip("/") + "/" + path[len(prefix):] if len(path) > len(prefix) else prefix.rstrip("/")

    # 2) assets shorthand -> map to the first whitelist prefix
    if lowered.startswith("assets/"):
        root = allowed_prefixes[0].rstrip("/")
        return f"{root}/{path}"

    # 3) absolute/foreign path containing any whitelist prefix
    for prefix, prefix_lower in zip(allowed_prefixes, allowed_lower):
        marker = "/" + prefix_lower
        idx = lowered.find(marker)
        if idx >= 0:
            return path[idx + 1 :]
        idx2 = lowered.find(prefix_lower)
        if idx2 >= 0:
            return path[idx2:]

    # Drop non-whitelisted media paths
    return ""


def _resolve_abs_media_path(rel_path: str) -> str:
    if not rel_path:
        return ""
    candidate = (PROJECT_ROOT / rel_path).resolve()
    return str(candidate) if candidate.exists() else ""


def _dedupe_paths(paths: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for p in paths:
        key = str(p or "").strip().replace("\\", "/").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(p).strip())
    return out


def _build_v2_references(
    results: List[Dict[str, Any]],
    *,
    allowed_prefixes: Tuple[str, ...],
    category: str,
) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for idx, item in enumerate(results, start=1):
        page_summary = (item.get("page_summary") or "").strip()
        chunks = item.get("chunks") or []
        chunk_preview = ""
        if chunks:
            top_chunk = chunks[0]
            chunk_preview = (
                (top_chunk.get("chunk_text_display") or top_chunk.get("chunk_text_clean") or "").strip()
            )
        preview = page_summary or chunk_preview
        if len(preview) > 220:
            preview = preview[:220] + "..."

        table_paths = _dedupe_paths(
            [
                _normalize_v2_media_rel_path(
                    t.get("asset_path") or t.get("table_path") or "",
                    allowed_prefixes=allowed_prefixes,
                )
                for t in (item.get("tables") or [])
            ]
        )
        image_paths = _dedupe_paths(
            [
                _normalize_v2_media_rel_path(
                    img.get("image_path") or "",
                    allowed_prefixes=allowed_prefixes,
                )
                for img in (item.get("images") or [])
            ]
        )

        source_rel_path = _get_v2_source_rel_path(
            str(item.get("doc_id") or ""),
            str(item.get("doc_title") or ""),
            category=category,
        )
        source_abs_path = _resolve_abs_media_path(source_rel_path)
        source_ext = Path(source_rel_path).suffix.lower() if source_rel_path else ""

        open_target = ""
        for rel in table_paths + image_paths:
            resolved = _resolve_abs_media_path(rel)
            if resolved:
                open_target = resolved
                break
        if source_abs_path:
            open_target = source_abs_path

        refs.append(
            {
                "ref_id": idx,
                "score": float(item.get("page_score") or 0.0),
                "file_path": open_target,
                "doc_name": f"{item.get('doc_title', 'Unknown')} (第{item.get('page_no', '?')}页)",
                "page_no": int(item.get("page_no") or 0),
                "preview_text": preview,
                "type": "page_bundle",
                "image_paths": image_paths,
                "table_paths": table_paths,
                "source_rel_path": source_rel_path,
                "source_ext": source_ext,
                "source_name": Path(source_rel_path).name if source_rel_path else "",
            }
        )
    return refs


def _build_v2_context(results: List[Dict[str, Any]], *, allowed_prefixes: Tuple[str, ...]) -> str:
    if not results:
        return "<retrieved_chunks>No relevant information found.</retrieved_chunks>"

    parts: List[str] = []
    for idx, item in enumerate(results[:8], start=1):
        doc_title = item.get("doc_title") or "Unknown"
        page_no = item.get("page_no") or "?"
        page_score = float(item.get("page_score") or 0.0)

        chunks = item.get("chunks") or []
        chunk_texts: List[str] = []
        for c in chunks[:3]:
            text = (c.get("chunk_text_display") or c.get("chunk_text_clean") or "").strip()
            if text:
                chunk_texts.append(text)

        page_summary = (item.get("page_summary") or "").strip()
        merged_text = page_summary
        if chunk_texts:
            merged_text = (merged_text + "\n\n" if merged_text else "") + "\n\n".join(chunk_texts)
        if not merged_text:
            merged_text = "No textual content."

        image_paths = _dedupe_paths(
            [
                _normalize_v2_media_rel_path(
                    img.get("image_path") or "",
                    allowed_prefixes=allowed_prefixes,
                )
                for img in (item.get("images") or [])
            ]
        )
        table_paths = _dedupe_paths(
            [
                _normalize_v2_media_rel_path(
                    t.get("asset_path") or t.get("table_path") or "",
                    allowed_prefixes=allowed_prefixes,
                )
                for t in (item.get("tables") or [])
            ]
        )

        if image_paths:
            merged_text += "\n\n[相关图片路径: " + ", ".join(image_paths) + "]"
        if table_paths:
            merged_text += "\n\n[相关表格路径: " + ", ".join(table_paths) + "]"

        chunk_xml = (
            f'<chunk fileId="v2_{idx}" fileName="{doc_title}_p{page_no}" score="{page_score:.4f}">'
            f"{merged_text}"
            f"</chunk>"
        )
        parts.append(chunk_xml)

    context = "<retrieved_chunks>\n" + "\n".join(parts) + "\n</retrieved_chunks>"
    if len(context) > 12000:
        context = context[:12000] + "\n<!-- Context truncated -->"
    return context


def _get_v2_searcher(category: str):
    category = (category or DEFAULT_CATEGORY).strip().lower()
    if category in _V2_SEARCHERS:
        return _V2_SEARCHERS[category]
    if category in _V2_LOAD_ERRORS:
        raise RuntimeError(_V2_LOAD_ERRORS[category])
    if SpaceRAGv2Searcher is None:
        err = "未找到 spacerag_v2_search.py 或其依赖未安装。"
        _V2_LOAD_ERRORS[category] = err
        raise RuntimeError(err)
    config = _get_v2_config(category)
    if not config:
        err = f"未配置 {category!r} 对应的 v2 知识库目录。"
        _V2_LOAD_ERRORS[category] = err
        raise RuntimeError(err)
    try:
        searcher = SpaceRAGv2Searcher(rag_dir=config["rag_dir"])
        _V2_SEARCHERS[category] = searcher
        return searcher
    except Exception as exc:
        err = f"{CATEGORY_LABELS.get(category, category)} v2 初始化失败: {exc}"
        _V2_LOAD_ERRORS[category] = err
        raise RuntimeError(err) from exc


def _get_v2_generation_engine(category: str) -> RAG:
    category = (category or DEFAULT_CATEGORY).strip().lower()
    engine = _V2_GENERATION_ENGINES.get(category)
    if engine is not None:
        return engine

    index_path = BASE_DIR / category / "faiss_index.index"
    metadata_path = BASE_DIR / category / "faiss_index_metadata.jsonl"
    engine = RAG(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        kb_category=category,
        skip_load=not (index_path.exists() and metadata_path.exists()),
    )
    _V2_GENERATION_ENGINES[category] = engine
    return engine


def _prepare_v2_rag_payload(prompt: str, *, category: str = "space") -> Tuple[RAG, str, List[Dict[str, Any]]]:
    searcher = _get_v2_searcher(category)
    allowed_prefixes = _get_v2_media_whitelist(category)
    raw_results = searcher.search_as_dict(
        prompt,
        top_k=5,
        doc_k=10,
        page_k=24,
        chunk_k=50,
        table_k=18,
        timeout=25.0,
    )
    refs = _build_v2_references(raw_results, allowed_prefixes=allowed_prefixes, category=category)
    context = _build_v2_context(raw_results, allowed_prefixes=allowed_prefixes)
    current_engine = _get_v2_generation_engine(category)
    return current_engine, context, refs


def _run_v2_full_rag(prompt: str, *, category: str = "space") -> Tuple[str, List[Dict[str, Any]], List[str]]:
    current_engine, context, refs = _prepare_v2_rag_payload(prompt, category=category)
    answer = current_engine._ask_llm_dashscope(context, prompt)
    related_qs = current_engine._generate_suggestions_dashscope(context, prompt)
    return answer, refs, related_qs


def run_rag_session(
    prompt: str,
    category: str = DEFAULT_CATEGORY,
    retrieval_mode: str = "legacy",
) -> Generator[Dict[str, Any], None, None]:
    if LOAD_ERROR:
        yield {
            "phase": "error",
            "gen": None,
            "message": f"System initialization failed: {LOAD_ERROR}",
        }
        return

    category = (category or DEFAULT_CATEGORY).strip().lower()
    kb_display = CATEGORY_LABELS.get(category, "未知知识库")
    retrieval_mode = (retrieval_mode or "legacy").strip().lower()

    yield {
        "phase": "meta",
        "gen": None,
        "message": f"收到问题：{prompt}（正在检索：{kb_display}，模式：{retrieval_mode}）",
        "best_fitness": None,
        "best_plan": None,
    }

    if category not in CATEGORY_LABELS:
        yield {
            "phase": "error",
            "gen": None,
            "message": f"错误：不支持的知识库类别 {category!r}。",
            "best_fitness": None,
            "best_plan": None,
        }
        return

    if category in EMPTY_KB_CATEGORIES:
        answer = (
            f"{kb_display}当前暂无可用知识库内容。"
            f"目前已上线：{_get_active_kb_display_names()}。"
        )
        yield {
            "phase": "retrieval",
            "gen": None,
            "message": f"{kb_display}暂为空库，未检索到可用内容。",
            "best_fitness": None,
            "best_plan": None,
        }
        yield {
            "phase": "answer",
            "gen": None,
            "message": answer,
            "best_fitness": None,
            "best_plan": None,
        }
        structured = {
            "answer": answer,
            "references": [],
            "related_questions": [],
        }
        yield {
            "phase": "result",
            "gen": None,
            "message": json.dumps(structured, ensure_ascii=False),
            "best_fitness": None,
            "best_plan": None,
        }
        return

    if retrieval_mode == "v2" and category in ACTIVE_V2_CATEGORIES:
        try:
            current_engine, context, refs = _prepare_v2_rag_payload(prompt, category=category)

            if refs:
                for ref in refs:
                    score_display = f"RRF:{ref.get('score', 0):.4f}"
                    msg = (
                        f"检索到片段[{ref['ref_id']}] {score_display}: "
                        f"{ref.get('doc_name')} (预览: {ref.get('preview_text', '')})"
                    )
                    yield {
                        "phase": "retrieval",
                        "gen": ref["ref_id"],
                        "message": msg,
                        "best_fitness": None,
                        "best_plan": None,
                    }
            else:
                yield {
                    "phase": "retrieval",
                    "gen": None,
                    "message": f"{kb_display} v2 未检索到足够相关片段。",
                    "best_fitness": None,
                    "best_plan": None,
                }

            answer = ""
            for partial_answer in current_engine.iter_answer_dashscope(context, prompt):
                answer = partial_answer or answer
                yield {
                    "phase": "answer_chunk",
                    "gen": None,
                    "message": json.dumps({"text": partial_answer}, ensure_ascii=False),
                    "best_fitness": None,
                    "best_plan": None,
                }

            if not answer:
                answer = "未找到明确答案，请尝试更换提问方式。"
                yield {
                    "phase": "answer_chunk",
                    "gen": None,
                    "message": json.dumps({"text": answer}, ensure_ascii=False),
                    "best_fitness": None,
                    "best_plan": None,
                }

            related_qs = current_engine._generate_suggestions_dashscope(context, prompt)

            structured = {
                "answer": answer,
                "references": refs,
                "related_questions": related_qs,
            }
            yield {
                "phase": "result",
                "gen": None,
                "message": json.dumps(structured, ensure_ascii=False),
                "best_fitness": None,
                "best_plan": None,
            }
            return
        except Exception as e:
            traceback.print_exc()
            yield {
                "phase": "error",
                "gen": None,
                "message": f"{kb_display} v2 执行出错: {str(e)}",
                "best_fitness": None,
                "best_plan": None,
            }
            return

    if retrieval_mode == "v2" and category not in ACTIVE_V2_CATEGORIES:
        yield {
            "phase": "retrieval",
            "gen": None,
            "message": "当前类别尚未接入 v2 检索，已自动回退到 legacy 模式。",
            "best_fitness": None,
            "best_plan": None,
        }

    current_engine = ENGINES.get(category)
    if not current_engine:
        yield {
            "phase": "error",
            "gen": None,
            "message": f"错误：未找到类别 {category!r} 对应的知识库引擎。",
        }
        return

    try:
        result = current_engine.chat(prompt)
        refs = result.get("references", [])
        answer = result.get("answer", "")
        related_qs = result.get("related_questions", [])

        if refs:
            for ref in refs:
                score_display = f"RRF:{ref.get('score', 0):.4f}"
                preview = ref.get("preview_text", "")
                if ref.get("type") == "image_caption":
                    preview = "[图片内容匹配]" + preview
                msg = (
                    f"检索到片段[{ref['ref_id']}] {score_display}: "
                    f"{ref.get('file_path')} (预览: {preview})"
                )
                yield {
                    "phase": "retrieval",
                    "gen": ref["ref_id"],
                    "message": msg,
                    "best_fitness": None,
                    "best_plan": None,
                }
        else:
            yield {
                "phase": "retrieval",
                "gen": None,
                "message": "未检索到足够相关片段，将尝试直接作答。",
                "best_fitness": None,
                "best_plan": None,
            }

        yield {
            "phase": "answer",
            "gen": None,
            "message": answer,
            "best_fitness": None,
            "best_plan": None,
        }

        structured = {
            "answer": answer,
            "references": refs,
            "related_questions": related_qs,
        }
        yield {
            "phase": "result",
            "gen": None,
            "message": json.dumps(structured, ensure_ascii=False),
            "best_fitness": None,
            "best_plan": None,
        }
    except Exception as e:
        traceback.print_exc()
        yield {
            "phase": "error",
            "gen": None,
            "message": f"RAG 执行出错: {str(e)}",
            "best_fitness": None,
            "best_plan": None,
        }


if __name__ == "__main__":
    print("--- 测试商业航天库 ---")
    for step in run_rag_session("火箭发射市场融资情况？", category="space"):
        print(step)

    print("\n--- 测试空库分类 ---")
    for step in run_rag_session("半导体行业有哪些代表企业？", category="semiconductor"):
        print(step)
