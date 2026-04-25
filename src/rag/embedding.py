"""RAG embedding — Phase 4.

sentence-transformers all-MiniLM-L6-v2 (384-dim).
Chunk size 500 tokens, 50-token overlap via LangChain RecursiveCharacterTextSplitter.
"""
from __future__ import annotations

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_DIM = 384

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def chunk_documents(
    texts: list[str], metadatas: list[dict] | None = None
) -> tuple[list[str], list[dict]]:
    """Split texts into overlapping chunks. Returns (chunks, per-chunk metadatas)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_chunks: list[str] = []
    all_metas: list[dict] = []
    for i, text in enumerate(texts):
        chunks = splitter.split_text(text)
        meta = metadatas[i] if metadatas else {}
        all_chunks.extend(chunks)
        all_metas.extend([{**meta, "chunk_index": j} for j, _ in enumerate(chunks)])
    return all_chunks, all_metas


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with all-MiniLM-L6-v2. Returns list of 384-dim unit vectors."""
    model = _get_model()
    embeddings: np.ndarray = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
        batch_size=64,
    )
    return embeddings.tolist()
