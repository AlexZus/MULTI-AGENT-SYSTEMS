"""
Hybrid retrieval module.

Combines semantic search (FAISS vector DB) + BM25 (lexical) with
cross-encoder reranking for maximum relevance.
"""

import os
import pickle

# Point HuggingFace cache to a writable project-local directory
_HF_CACHE = os.path.join(os.path.dirname(__file__), ".hf_cache")
os.makedirs(_HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", _HF_CACHE)

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import Settings

settings = Settings()

# Module-level singletons (lazy-loaded on first search call)
_embed_model: SentenceTransformer | None = None
_rerank_model: CrossEncoder | None = None
_faiss_index = None
_chunks: list[dict] | None = None
_bm25: BM25Okapi | None = None


def _load_resources() -> None:
    """Load all retrieval resources from disk (idempotent)."""
    global _embed_model, _rerank_model, _faiss_index, _chunks, _bm25

    if _faiss_index is not None:
        return  # already loaded

    faiss_path = os.path.join(settings.index_dir, "faiss.index")
    chunks_path = os.path.join(settings.index_dir, "chunks.pkl")

    if not os.path.exists(faiss_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"Index not found in '{settings.index_dir}/'. "
            "Run `python ingest.py` first to build the knowledge base."
        )

    _faiss_index = faiss.read_index(faiss_path)

    with open(chunks_path, "rb") as f:
        _chunks = pickle.load(f)

    # BM25 over whitespace-tokenized chunks
    tokenized = [c["text"].lower().split() for c in _chunks]
    _bm25 = BM25Okapi(tokenized)

    _embed_model = SentenceTransformer(settings.embedding_model)
    _rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def search(query: str, top_k: int | None = None, top_n: int | None = None) -> list[dict]:
    """
    Hybrid search with reranking.

    Steps:
      1. Semantic search  — top-k chunks by cosine similarity (FAISS)
      2. BM25 search      — top-k chunks by lexical score
      3. Ensemble         — union of both candidate sets
      4. Cross-encoder    — rerank all candidates, return top-n

    Returns a list of chunk dicts: {"text", "source", "page"}.
    """
    _load_resources()

    k = top_k or settings.retrieval_top_k
    n = top_n or settings.rerank_top_n

    # --- 1. Semantic search ---
    query_emb = _embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_emb)
    _, indices = _faiss_index.search(query_emb, k)
    semantic_hits = {int(i) for i in indices[0] if i >= 0}

    # --- 2. BM25 search ---
    bm25_scores = _bm25.get_scores(query.lower().split())
    bm25_top = set(int(i) for i in np.argsort(bm25_scores)[::-1][:k])

    # --- 3. Ensemble (union) ---
    candidate_indices = list(semantic_hits | bm25_top)
    candidates = [_chunks[i] for i in candidate_indices]

    # --- 4. Cross-encoder reranking ---
    pairs = [(query, c["text"]) for c in candidates]
    scores = _rerank_model.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    return [chunk for _, chunk in ranked[:n]]
