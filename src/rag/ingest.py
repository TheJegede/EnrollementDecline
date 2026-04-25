"""RAG ingestion pipeline — Phase 4.

Walk data/corpus/ markdown files, chunk, embed, persist to ChromaDB at data/vector_db/.
Idempotent — safe to re-run; skips chunks already in the store.
Also exports a FAISS pickle as Streamlit Cloud fallback.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from urllib.parse import unquote

import chromadb
import faiss
import numpy as np

from src.rag.embedding import EMBEDDING_DIM, chunk_documents, embed_texts
from src.utils import CORPUS_DIR, VECTOR_DB_DIR

COLLECTION_NAME = "admissions_corpus"
BATCH_SIZE = 256


def _get_client(vector_db_dir: Path) -> chromadb.PersistentClient:
    vector_db_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(vector_db_dir))


def _get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_corpus(
    corpus_dir: Path | None = None,
    vector_db_dir: Path | None = None,
    reset: bool = False,
) -> int:
    """Ingest all markdown files from corpus_dir into ChromaDB. Returns total chunk count."""
    corpus_dir = Path(corpus_dir or CORPUS_DIR)
    vector_db_dir = Path(vector_db_dir or VECTOR_DB_DIR)

    client = _get_client(vector_db_dir)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Existing collection deleted.")
        except Exception:
            pass

    collection = _get_collection(client)
    existing_ids = set(collection.get(include=[])["ids"])

    md_files = sorted(corpus_dir.glob("*.md"))
    texts, metas = [], []
    for path in md_files:
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(content) < 200:
            continue
        source = unquote(path.stem)
        texts.append(content)
        metas.append({"source": source, "filename": path.name})

    print(f"Loaded {len(texts)} documents from corpus.")
    chunks, chunk_metas = chunk_documents(texts, metas)
    print(f"Split into {len(chunks)} chunks.")

    new_chunks, new_metas, new_ids = [], [], []
    for chunk, meta in zip(chunks, chunk_metas):
        chunk_id = hashlib.md5(
            (meta.get("source", "") + "|" + str(meta.get("chunk_index", 0)) + "|" + chunk[:80]).encode()
        ).hexdigest()
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_metas.append(meta)
            new_ids.append(chunk_id)

    if not new_chunks:
        total = len(existing_ids)
        print(f"Already up to date — {total} chunks in store.")
        return total

    for i in range(0, len(new_chunks), BATCH_SIZE):
        b_chunks = new_chunks[i : i + BATCH_SIZE]
        b_metas = new_metas[i : i + BATCH_SIZE]
        b_ids = new_ids[i : i + BATCH_SIZE]
        b_embeddings = embed_texts(b_chunks)
        collection.add(
            ids=b_ids,
            documents=b_chunks,
            embeddings=b_embeddings,
            metadatas=b_metas,
        )
        print(f"  Batch {i // BATCH_SIZE + 1}: added {len(b_chunks)} chunks.")

    total = len(existing_ids) + len(new_chunks)
    print(f"Done. Total chunks in store: {total}")

    export_faiss(collection, vector_db_dir)
    return total


def export_faiss(
    collection: chromadb.Collection | None = None,
    vector_db_dir: Path | None = None,
) -> None:
    """Export ChromaDB collection to FAISS flat-IP index + metadata pickle."""
    vector_db_dir = Path(vector_db_dir or VECTOR_DB_DIR)

    if collection is None:
        client = _get_client(vector_db_dir)
        collection = _get_collection(client)

    result = collection.get(include=["documents", "embeddings", "metadatas"])
    docs = result["documents"]
    embeddings = result["embeddings"]
    metas = result["metadatas"]

    if not docs:
        print("Collection empty — skipping FAISS export.")
        return

    mat = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(mat)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(mat)

    faiss.write_index(index, str(vector_db_dir / "faiss.index"))
    with open(vector_db_dir / "faiss_meta.pkl", "wb") as f:
        pickle.dump(
            {"texts": docs, "sources": [m.get("source", "") for m in metas]}, f
        )
    print(f"FAISS index exported: {index.ntotal} vectors.")


if __name__ == "__main__":
    ingest_corpus()
