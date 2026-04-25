"""RAG retrieval — Phase 4.

Top-5 cosine similarity from ChromaDB. FAISS pickle fallback for Streamlit Cloud.
"""
from __future__ import annotations

from pathlib import Path

import chromadb

from src.rag.embedding import embed_texts
from src.rag.ingest import COLLECTION_NAME, _get_client, _get_collection
from src.utils import VECTOR_DB_DIR

_collection: chromadb.Collection | None = None


def _get_cached_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = _get_client(Path(VECTOR_DB_DIR))
        _collection = _get_collection(client)
    return _collection


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Return top-k chunks most similar to query.

    Each item: {"text": str, "source": str, "score": float (cosine similarity)}.
    Falls back to FAISS pickle if ChromaDB unavailable.
    """
    try:
        collection = _get_cached_collection()
        if collection.count() == 0:
            raise RuntimeError("Collection empty")
        embedding = embed_texts([query])[0]
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                {
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "score": round(1.0 - float(dist), 4),
                }
            )
        return chunks
    except Exception:
        return _faiss_fallback(query, k)


def _faiss_fallback(query: str, k: int) -> list[dict]:
    import pickle

    import faiss
    import numpy as np

    index_path = VECTOR_DB_DIR / "faiss.index"
    meta_path = VECTOR_DB_DIR / "faiss_meta.pkl"

    if not index_path.exists():
        return []

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    q_emb = np.array(embed_texts([query]), dtype="float32")
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, k)

    chunks = []
    for idx, score in zip(indices[0], distances[0]):
        if idx < 0:
            continue
        chunks.append(
            {
                "text": meta["texts"][idx],
                "source": meta["sources"][idx],
                "score": round(float(score), 4),
            }
        )
    return chunks
