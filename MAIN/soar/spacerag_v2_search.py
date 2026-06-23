from __future__ import annotations

import json
import math
import os
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests


DEFAULT_RAG_DIR = Path("SpaceRAG_v2")
DEFAULT_API_BASE = "https://api.siliconflow.com/v1"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_RERANK_MODEL = "Qwen/Qwen3-Reranker-8B"
DEFAULT_DOC_ALPHA = 0.65
DEFAULT_PAGE_ALPHA = 0.6
DEFAULT_CHUNK_ALPHA = 0.25
DEFAULT_TABLE_ALPHA = 0.6
DOC_PAGE_EXPAND_LIMIT = 2
LOW_SIGNAL_PAGE_PENALTY = 0.35
LOW_CONFIDENCE_RERANK_THRESHOLD = 0.12
IMAGE_MIN_SCORE = 0.3
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*")
CN_RE = re.compile(r"[\u4e00-\u9fff]+")
REFERENCE_SIGNAL_RE = re.compile(r"(如下图|如下表|见下图|见下表|图\s*\d+|表\s*\d+)")
LOW_SIGNAL_PAGE_SUMMARY_RE = re.compile(r"^page\s+\d+\s+of\s+.+$", re.IGNORECASE)
QUERY_ALIAS_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "中电金信": ("东方金信",),
}
LOW_SIGNAL_TEXT_PATTERNS = (
    "这张图片是空白的",
    "这张图片是纯白色的",
    "没有包含任何文字内容",
    "没有包含任何可见的文字内容",
    "没有任何可见的文字内容",
    "图中没有文字内容",
    "图中没有可见的文字内容",
    "图片中未检测到文字",
)


@dataclass
class SearchResult:
    doc_id: str
    doc_title: str
    page_id: str
    page_no: int
    page_score: float
    rerank_score: float
    page_summary: str
    chunks: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    images: list[dict[str, Any]]


class BM25Index:
    def __init__(self, docs: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in docs]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.tf: list[dict[str, int]] = []
        self.df: dict[str, int] = {}

        for doc in docs:
            tf: dict[str, int] = {}
            for token in doc:
                tf[token] = tf.get(token, 0) + 1
            self.tf.append(tf)
            for token in tf:
                self.df[token] = self.df.get(token, 0) + 1

        n_docs = max(len(docs), 1)
        self.idf = {
            token: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in self.df.items()
        }

    def score(self, query_tokens: list[str], candidate_ids: list[int] | None = None) -> dict[int, float]:
        if not query_tokens:
            return {}
        doc_ids = candidate_ids if candidate_ids is not None else list(range(len(self.tf)))
        scores: dict[int, float] = {}
        for doc_id in doc_ids:
            tf = self.tf[doc_id]
            dl = self.doc_len[doc_id]
            score = 0.0
            for token in query_tokens:
                freq = tf.get(token, 0)
                if freq <= 0:
                    continue
                idf = self.idf.get(token, 0.0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score += idf * (freq * (self.k1 + 1.0) / max(denom, 1e-9))
            if score > 0:
                scores[doc_id] = score
        return scores


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace("\r\n", "\n").replace("\r", "\n").strip()


def tokenize_sparse(text: str) -> list[str]:
    normalized = normalize_text(text).lower()
    tokens: list[str] = []
    tokens.extend(TOKEN_RE.findall(normalized))
    for segment in CN_RE.findall(normalized):
        if len(segment) == 1:
            tokens.append(segment)
            continue
        tokens.append(segment)
        for i in range(len(segment) - 1):
            tokens.append(segment[i : i + 2])
        for i in range(len(segment) - 2):
            tokens.append(segment[i : i + 3])
    return tokens


def normalize_scores(score_map: dict[int, float]) -> dict[int, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return {key: 1.0 for key in score_map}
    return {key: (value - lo) / (hi - lo) for key, value in score_map.items()}


def dot_search(matrix: np.ndarray, query_vector: np.ndarray, top_k: int, candidate_ids: list[int] | None = None) -> dict[int, float]:
    if matrix.size == 0:
        return {}
    if candidate_ids is not None:
        subset = matrix[candidate_ids]
        scores = subset @ query_vector
        order = np.argsort(scores)[::-1][:top_k]
        return {candidate_ids[int(i)]: float(scores[int(i)]) for i in order}
    scores = matrix @ query_vector
    order = np.argsort(scores)[::-1][:top_k]
    return {int(i): float(scores[int(i)]) for i in order}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
        elif line.startswith("sk-") and "SILICONFLOW_API_KEY" not in os.environ:
            os.environ["SILICONFLOW_API_KEY"] = line


def keyword_overlap_score(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    common = len(left_set & right_set)
    if common == 0:
        return 0.0
    return common / max(min(len(left_set), len(right_set)), 1)


class SpaceRAGv2Searcher:
    def __init__(self, rag_dir: Path = DEFAULT_RAG_DIR, env_path: Path | None = None) -> None:
        self.rag_dir = rag_dir.resolve()
        if env_path is None:
            env_path = self.rag_dir.parent / ".env"
        load_env_file(env_path)
        self.api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
        self.api_base = DEFAULT_API_BASE
        self.embedding_model = DEFAULT_EMBEDDING_MODEL
        self.rerank_model = DEFAULT_RERANK_MODEL
        self.session = requests.Session()
        self.runtime_warnings: list[str] = []

        self.docs = self._read_jsonl("docs.jsonl")
        self.pages = self._read_jsonl("pages.jsonl")
        self.chunks = self._read_jsonl("chunks.jsonl")
        self.tables = self._read_jsonl("tables.jsonl")
        self.images = self._read_jsonl("images.jsonl")
        self.relations = self._read_jsonl("relations_chunk_asset.jsonl")

        self.doc_matrix = self._load_matrix("doc_embeddings.npy")
        self.page_matrix = self._load_matrix("page_embeddings.npy")
        self.chunk_matrix = self._load_matrix("chunk_embeddings.npy")
        self.table_matrix = self._load_matrix("table_embeddings.npy")

        self.doc_by_id = {row["doc_id"]: row for row in self.docs}
        self.page_by_id = {row["page_id"]: row for row in self.pages}
        self.chunk_by_id = {row["chunk_id"]: row for row in self.chunks}
        self.table_by_id = {row["table_id"]: row for row in self.tables}
        self.image_by_id = {row["image_id"]: row for row in self.images}

        self.page_index_by_id = {row["page_id"]: idx for idx, row in enumerate(self.pages)}
        self.doc_index_by_id = {row["doc_id"]: idx for idx, row in enumerate(self.docs)}
        self.chunk_index_by_id = {row["chunk_id"]: idx for idx, row in enumerate(self.chunks)}
        self.table_index_by_id = {row["table_id"]: idx for idx, row in enumerate(self.tables)}

        self.chunk_ids_by_page: dict[str, list[str]] = {}
        for page in self.pages:
            self.chunk_ids_by_page[page["page_id"]] = list(page.get("chunk_ids", []) or [])

        self.table_ids_by_page: dict[str, list[str]] = {}
        self.image_ids_by_page: dict[str, list[str]] = {}
        for page in self.pages:
            self.table_ids_by_page[page["page_id"]] = list(page.get("table_ids", []) or [])
            self.image_ids_by_page[page["page_id"]] = list(page.get("image_ids", []) or [])

        self.relations_by_chunk: dict[str, list[dict[str, Any]]] = {}
        for row in self.relations:
            self.relations_by_chunk.setdefault(row["chunk_id"], []).append(row)

        self.doc_texts = [self._build_doc_text(row) for row in self.docs]
        self.page_texts = [self._build_page_text(row) for row in self.pages]
        self.chunk_texts = [self._build_chunk_text(row) for row in self.chunks]
        self.table_texts = [self._build_table_text(row) for row in self.tables]

        self.doc_bm25 = BM25Index([tokenize_sparse(text) for text in self.doc_texts])
        self.page_bm25 = BM25Index([tokenize_sparse(text) for text in self.page_texts])
        self.chunk_bm25 = BM25Index([tokenize_sparse(text) for text in self.chunk_texts])
        self.table_bm25 = BM25Index([tokenize_sparse(text) for text in self.table_texts])

    def _read_jsonl(self, name: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        path = self.rag_dir / name
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _load_matrix(self, name: str) -> np.ndarray:
        matrix = np.load(self.rag_dir / "vector_store" / name)
        matrix = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _build_doc_text(self, row: dict[str, Any]) -> str:
        return normalize_text(
            "\n".join(
                [
                    row.get("doc_title", ""),
                    "关键词: " + "、".join(row.get("top_keywords", []) or []),
                    row.get("doc_summary", ""),
                ]
            )
        )

    def _build_page_text(self, row: dict[str, Any]) -> str:
        return normalize_text(
            "\n".join(
                [
                    row.get("doc_title", ""),
                    f"页码: {row.get('page_no', 0)}",
                    "关键词: " + "、".join(row.get("page_keywords", []) or []),
                    row.get("page_summary", ""),
                ]
            )
        )

    def _build_chunk_text(self, row: dict[str, Any]) -> str:
        return normalize_text(
            "\n".join(
                [
                    row.get("doc_title", ""),
                    row.get("heading", ""),
                    "关键词: " + "、".join(row.get("keywords", []) or []),
                    row.get("chunk_text_clean", ""),
                ]
            )
        )

    def _build_table_text(self, row: dict[str, Any]) -> str:
        headers = row.get("table_headers", []) or []
        return normalize_text(
            "\n".join(
                [
                    row.get("doc_title", ""),
                    row.get("table_title", ""),
                    "关键词: " + "、".join(row.get("keywords", []) or []),
                    "列名: " + "、".join(headers),
                    row.get("table_summary", ""),
                ]
            )
        )

    def _embed_query(self, query: str, timeout: float) -> np.ndarray | None:
        if not self.api_key:
            return None
        endpoint = self.api_base.rstrip("/") + "/embeddings"
        payload = {
            "model": self.embedding_model,
            "input": [query],
            "encoding_format": "float",
            "dimensions": 4096,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            response = self.session.post(endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            vector = np.array(response.json()["data"][0]["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            return vector
        except requests.RequestException as exc:
            self.runtime_warnings.append(f"Embedding request failed, fallback to lexical-only retrieval: {exc}")
            return None

    def _rerank_documents(self, query: str, documents: list[str], timeout: float) -> list[tuple[int, float]]:
        if not documents:
            return []
        if not self.api_key:
            return [(idx, float(len(documents) - idx)) for idx in range(len(documents))]
        endpoint = self.api_base.rstrip("/") + "/rerank"
        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            response = self.session.post(endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            results = []
            for item in response.json()["results"]:
                results.append((int(item["index"]), float(item["relevance_score"])))
            return results
        except requests.RequestException as exc:
            self.runtime_warnings.append(f"Rerank request failed, fallback to heuristic bundle ranking: {exc}")
            return [(idx, float(len(documents) - idx)) for idx in range(len(documents))]

    def _combine_scores(self, sparse: dict[int, float], dense: dict[int, float], alpha: float) -> dict[int, float]:
        sparse_norm = normalize_scores(sparse)
        dense_norm = normalize_scores(dense)
        ids = set(sparse_norm) | set(dense_norm)
        return {idx: alpha * dense_norm.get(idx, 0.0) + (1.0 - alpha) * sparse_norm.get(idx, 0.0) for idx in ids}

    def _detect_table_bias(self, query: str) -> float:
        query_lower = normalize_text(query).lower()
        hot_terms = ("市场规模", "收入", "增速", "毛利", "净利", "成本", "价格", "对比", "统计", "金额", "占比", "cagr")
        return 0.15 if any(term in query_lower for term in hot_terms) else 0.0

    def _expand_query_aliases(self, query: str) -> str:
        expanded = normalize_text(query)
        if not expanded:
            return expanded
        for alias, expansions in QUERY_ALIAS_EXPANSIONS.items():
            if alias not in expanded:
                continue
            missing = [item for item in expansions if item not in expanded]
            if missing:
                expanded = normalize_text(expanded + " " + " ".join(missing))
        return expanded

    def _is_low_signal_text(self, text: str) -> bool:
        normalized = normalize_text(text)
        if not normalized:
            return True
        return any(pattern in normalized for pattern in LOW_SIGNAL_TEXT_PATTERNS)

    def _is_low_signal_page(self, page: dict[str, Any], chunk_ids: list[str]) -> bool:
        page_summary = normalize_text(page.get("page_summary", ""))
        if not LOW_SIGNAL_PAGE_SUMMARY_RE.fullmatch(page_summary):
            return False
        if not chunk_ids:
            return True
        for chunk_id in chunk_ids:
            chunk = self.chunk_by_id.get(chunk_id)
            if not chunk:
                continue
            chunk_text = chunk.get("chunk_text_clean") or chunk.get("chunk_text_display") or ""
            if not self._is_low_signal_text(chunk_text):
                return False
        return True

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        doc_k: int = 8,
        page_k: int = 20,
        chunk_k: int = 40,
        table_k: int = 16,
        timeout: float = 120.0,
    ) -> list[SearchResult]:
        retrieval_query = self._expand_query_aliases(query)
        query_tokens = tokenize_sparse(retrieval_query)
        query_vector = self._embed_query(retrieval_query, timeout=timeout)

        doc_sparse = self.doc_bm25.score(query_tokens)
        doc_dense = dot_search(self.doc_matrix, query_vector, doc_k * 2) if query_vector is not None else {}
        doc_scores = self._combine_scores(doc_sparse, doc_dense, DEFAULT_DOC_ALPHA)

        page_sparse = self.page_bm25.score(query_tokens)
        page_dense = dot_search(self.page_matrix, query_vector, page_k * 2) if query_vector is not None else {}
        page_scores = self._combine_scores(page_sparse, page_dense, DEFAULT_PAGE_ALPHA)

        table_sparse = self.table_bm25.score(query_tokens)
        table_dense = dot_search(self.table_matrix, query_vector, table_k * 2) if query_vector is not None else {}
        table_scores = self._combine_scores(table_sparse, table_dense, min(0.75, DEFAULT_TABLE_ALPHA + self._detect_table_bias(retrieval_query)))

        chunk_sparse = self.chunk_bm25.score(query_tokens)
        chunk_dense = dot_search(self.chunk_matrix, query_vector, chunk_k * 2) if query_vector is not None else {}
        chunk_scores = self._combine_scores(chunk_sparse, chunk_dense, DEFAULT_CHUNK_ALPHA)

        top_doc_ids = [self.docs[idx]["doc_id"] for idx in sorted(doc_scores, key=doc_scores.get, reverse=True)[:doc_k]]
        top_page_ids = [self.pages[idx]["page_id"] for idx in sorted(page_scores, key=page_scores.get, reverse=True)[:page_k]]
        top_table_ids = [self.tables[idx]["table_id"] for idx in sorted(table_scores, key=table_scores.get, reverse=True)[:table_k]]
        top_chunk_ids = [self.chunks[idx]["chunk_id"] for idx in sorted(chunk_scores, key=chunk_scores.get, reverse=True)[:chunk_k]]

        candidate_doc_ids = set(top_doc_ids)
        candidate_page_ids = set(top_page_ids)

        for table_id in top_table_ids:
            candidate_page_ids.add(self.table_by_id[table_id]["page_id"])
            candidate_doc_ids.add(self.table_by_id[table_id]["doc_id"])
        for chunk_id in top_chunk_ids:
            candidate_page_ids.add(self.chunk_by_id[chunk_id]["page_id"])
            candidate_doc_ids.add(self.chunk_by_id[chunk_id]["doc_id"])

        # Expand pages from candidate docs to avoid losing relevant neighboring pages.
        for doc_id in list(candidate_doc_ids):
            doc = self.doc_by_id.get(doc_id)
            if not doc:
                continue
            for page_id in doc.get("page_ids", [])[: min(DOC_PAGE_EXPAND_LIMIT, len(doc.get("page_ids", [])))]:
                if page_id in self.page_by_id:
                    candidate_page_ids.add(page_id)

        # Compute a bundle score per page from page/chunk/table/doc evidence.
        page_bundle_scores: dict[str, float] = {}
        page_best_chunk_scores: dict[str, dict[str, float]] = defaultdict(dict)
        page_best_table_scores: dict[str, dict[str, float]] = defaultdict(dict)

        for page_id in candidate_page_ids:
            page = self.page_by_id[page_id]
            chunk_ids = self.chunk_ids_by_page.get(page_id, [])
            table_ids = self.table_ids_by_page.get(page_id, [])
            image_ids = self.image_ids_by_page.get(page_id, [])
            low_signal_page = self._is_low_signal_page(page, chunk_ids)
            if low_signal_page and not table_ids and not image_ids:
                continue

            score = page_scores.get(self.page_index_by_id[page_id], 0.0)
            doc_score = doc_scores.get(self.doc_index_by_id[page["doc_id"]], 0.0) if page["doc_id"] in self.doc_index_by_id else 0.0
            score = max(score, 0.45 * score + 0.25 * doc_score)

            chunk_page_best = 0.0
            for chunk_id in chunk_ids:
                if chunk_id not in self.chunk_index_by_id:
                    continue
                idx = self.chunk_index_by_id[chunk_id]
                chunk_score = chunk_scores.get(idx, 0.0)
                if chunk_score > 0:
                    page_best_chunk_scores[page_id][chunk_id] = chunk_score
                    chunk_page_best = max(chunk_page_best, chunk_score)
            score += 0.55 * chunk_page_best

            table_page_best = 0.0
            for table_id in table_ids:
                if table_id not in self.table_index_by_id:
                    continue
                idx = self.table_index_by_id[table_id]
                table_score = table_scores.get(idx, 0.0)
                if table_score > 0:
                    page_best_table_scores[page_id][table_id] = table_score
                    table_page_best = max(table_page_best, table_score)
            score += 0.35 * table_page_best
            if low_signal_page:
                score *= LOW_SIGNAL_PAGE_PENALTY
            page_bundle_scores[page_id] = score

        ranked_page_ids = sorted(page_bundle_scores, key=page_bundle_scores.get, reverse=True)[: max(top_k * 3, 12)]

        bundle_texts: list[str] = []
        for page_id in ranked_page_ids:
            page = self.page_by_id[page_id]
            chunk_ids = sorted(page_best_chunk_scores.get(page_id, {}), key=page_best_chunk_scores[page_id].get, reverse=True)[:3]
            table_ids = sorted(page_best_table_scores.get(page_id, {}), key=page_best_table_scores[page_id].get, reverse=True)[:2]
            chunk_text = "\n\n".join(self.chunk_by_id[cid]["chunk_text_display"] for cid in chunk_ids)
            table_text = "\n\n".join(self.table_by_id[tid]["table_summary"] for tid in table_ids)
            image_ids = self.image_ids_by_page.get(page_id, [])
            image_text = "\n".join(
                f"{self.image_by_id[iid]['image_title']} | 关键词: {'、'.join(self.image_by_id[iid].get('keywords', []))}"
                for iid in image_ids[:2]
            )
            bundle_texts.append(
                normalize_text(
                    "\n".join(
                        [
                            f"文档: {page['doc_title']}",
                            f"页码: {page['page_no']}",
                            "页面摘要:",
                            page["page_summary"],
                            "",
                            "候选正文:",
                            chunk_text,
                            "",
                            "候选表格:",
                            table_text,
                            "",
                            "候选图片:",
                            image_text,
                        ]
                    )
                )
            )

        reranked = self._rerank_documents(retrieval_query, bundle_texts, timeout=timeout)
        rerank_score_by_page = {
            ranked_page_ids[idx]: score for idx, score in reranked if idx < len(ranked_page_ids)
        }
        final_page_ids = sorted(
            ranked_page_ids,
            key=lambda pid: (rerank_score_by_page.get(pid, 0.0), page_bundle_scores.get(pid, 0.0)),
            reverse=True,
        )[:top_k]
        if final_page_ids:
            top_rerank_score = max(rerank_score_by_page.get(pid, 0.0) for pid in final_page_ids)
            if 0.0 <= top_rerank_score <= 1.0 and top_rerank_score < LOW_CONFIDENCE_RERANK_THRESHOLD:
                self.runtime_warnings.append(
                    f"Low-confidence retrieval filtered for query {query!r}: top rerank score {top_rerank_score:.4f}"
                )
                return []

        results: list[SearchResult] = []
        for page_id in final_page_ids:
            page = self.page_by_id[page_id]
            selected_chunk_ids = sorted(
                page_best_chunk_scores.get(page_id, {}),
                key=page_best_chunk_scores[page_id].get,
                reverse=True,
            )[:3]
            selected_chunks = []
            for chunk_id in selected_chunk_ids:
                chunk = dict(self.chunk_by_id[chunk_id])
                chunk["score"] = page_best_chunk_scores[page_id][chunk_id]
                selected_chunks.append(chunk)

            selected_table_ids = set(
                sorted(
                    page_best_table_scores.get(page_id, {}),
                    key=page_best_table_scores[page_id].get,
                    reverse=True,
                )[:2]
            )

            image_candidate_scores: dict[str, float] = {}
            table_candidate_scores: dict[str, float] = {
                table_id: page_best_table_scores.get(page_id, {}).get(table_id, 0.0) for table_id in self.table_ids_by_page.get(page_id, [])
            }

            for chunk in selected_chunks:
                relations = self.relations_by_chunk.get(chunk["chunk_id"], [])
                for relation in relations:
                    asset_id = relation["asset_id"]
                    relation_score = float(relation.get("relation_score", 0.0))
                    if relation["asset_type"] == "table":
                        table = self.table_by_id.get(asset_id)
                        if not table:
                            continue
                        overlap = keyword_overlap_score(chunk.get("keywords", []), table.get("keywords", []))
                        score = 0.45 * relation_score + 0.35 * overlap + 0.20 * page_bundle_scores.get(page_id, 0.0)
                        table_candidate_scores[asset_id] = max(table_candidate_scores.get(asset_id, 0.0), score)
                    else:
                        image = self.image_by_id.get(asset_id)
                        if not image:
                            continue
                        overlap = keyword_overlap_score(chunk.get("keywords", []), image.get("keywords", []))
                        signal = 0.1 if REFERENCE_SIGNAL_RE.search(chunk.get("chunk_text_clean", "")) else 0.0
                        score = 0.45 * relation_score + 0.30 * overlap + 0.15 * page_bundle_scores.get(page_id, 0.0) + signal
                        image_candidate_scores[asset_id] = max(image_candidate_scores.get(asset_id, 0.0), score)

            selected_tables = []
            for table_id in sorted(table_candidate_scores, key=table_candidate_scores.get, reverse=True):
                score = table_candidate_scores[table_id]
                if score < 0.18 and table_id not in selected_table_ids:
                    continue
                table = dict(self.table_by_id[table_id])
                table["score"] = score
                selected_tables.append(table)
                if len(selected_tables) >= 3:
                    break

            selected_images = []
            for image_id in sorted(image_candidate_scores, key=image_candidate_scores.get, reverse=True):
                score = image_candidate_scores[image_id]
                if score < IMAGE_MIN_SCORE:
                    continue
                image = dict(self.image_by_id[image_id])
                image["score"] = score
                selected_images.append(image)
                if len(selected_images) >= 3:
                    break

            results.append(
                SearchResult(
                    doc_id=page["doc_id"],
                    doc_title=page["doc_title"],
                    page_id=page_id,
                    page_no=int(page["page_no"]),
                    page_score=float(page_bundle_scores.get(page_id, 0.0)),
                    rerank_score=float(rerank_score_by_page.get(page_id, 0.0)),
                    page_summary=page["page_summary"],
                    chunks=selected_chunks,
                    tables=selected_tables,
                    images=selected_images,
                )
            )
        return results

    def search_as_dict(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        results = self.search(query, **kwargs)
        payload: list[dict[str, Any]] = []
        for item in results:
            payload.append(
                {
                    "doc_id": item.doc_id,
                    "doc_title": item.doc_title,
                    "page_id": item.page_id,
                    "page_no": item.page_no,
                    "page_score": item.page_score,
                    "rerank_score": item.rerank_score,
                    "page_summary": item.page_summary,
                    "chunks": item.chunks,
                    "tables": item.tables,
                    "images": item.images,
                }
            )
        return payload

    def emit_warnings(self) -> None:
        for warning in self.runtime_warnings:
            print(f"[WARN] {warning}", file=sys.stderr)
