import json
from pathlib import Path

import faiss
import numpy as np

from paths import PROJECT_ROOT
INPUT_JSONL = str(PROJECT_ROOT / "pre_data" / "employee" / "bge-m3.jsonl")
VECTOR_INDEX_PATH = str(PROJECT_ROOT / "indexes" / "employee" / "faiss_index.index")


def load_embeddings_from_jsonl(input_jsonl: str):
    """Load embeddings and metadata from a JSONL file."""
    embeddings = []
    metadata = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            emb = record.get("embedding")
            if emb is not None:
                embeddings.append(emb)
                metadata.append(record)
    return np.array(embeddings), metadata


def build_faiss_index(embeddings: np.ndarray):
    """Build cosine-similarity index via L2 normalization + IndexFlatIP."""
    if embeddings.size == 0 or embeddings.ndim != 2:
        raise ValueError(
            "No embeddings were loaded. Check embedding output file and model availability."
        )
    dim = embeddings.shape[1]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings[np.isnan(embeddings)] = 0.0
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_faiss_index(index, index_path: str):
    """Persist FAISS index to disk."""
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to: {index_path}")


def main():
    print("[INFO] Loading embeddings and metadata...")
    embeddings, metadata = load_embeddings_from_jsonl(INPUT_JSONL)

    print("[INFO] Building FAISS cosine index...")
    index = build_faiss_index(embeddings)

    print("[INFO] Saving FAISS index...")
    save_faiss_index(index, VECTOR_INDEX_PATH)

    metadata_path = VECTOR_INDEX_PATH.replace(".index", "_metadata.jsonl")
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        for record in metadata:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
