"""RAG module — Phase 4. Pipeline: ingest → embed → retrieve → generate."""
from src.rag.generation import generate
from src.rag.ingest import ingest_corpus
from src.rag.retrieval import retrieve

__all__ = ["ingest_corpus", "retrieve", "generate"]
