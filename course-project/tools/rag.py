"""RAG tool — hybrid FAISS + BM25 retrieval with HTTP embedding service.

Auto-rebuilds the index when source documents are newer than the cached index.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

RAG_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "knowledge_search",
        "description": (
            "Search the local knowledge base for coding standards, Python patterns, "
            "FastAPI best practices, and project guidelines. "
            "Always try this before searching the web."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

# Default paths (overridable via env vars)
_DEFAULT_DOCS_DIR = Path(__file__).parent.parent / "rag_docs"
_DEFAULT_INDEX_DIR = _DEFAULT_DOCS_DIR / ".index"

_EMBEDDING_URL = None  # set on first call from config


def _get_embedding_url() -> str:
    global _EMBEDDING_URL
    if _EMBEDDING_URL is None:
        _EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8084/v1/embeddings")
    return _EMBEDDING_URL


def embed(texts: list[str]) -> list[list[float]]:
    """Call embedding HTTP service and return vector list."""
    url = _get_embedding_url()
    resp = httpx.post(url, json={"input": texts}, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in data["data"]]


async def embed_async(texts: list[str]) -> list[list[float]]:
    """Async version of embed()."""
    url = _get_embedding_url()
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json={"input": texts}, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks by words."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def _index_is_stale(docs_dir: Path, index_dir: Path) -> bool:
    """Return True if any .md/.txt doc is newer than the saved index."""
    faiss_path = index_dir / "index.faiss"
    if not faiss_path.exists():
        return True
    index_mtime = faiss_path.stat().st_mtime
    for ext in ("*.md", "*.txt"):
        for f in docs_dir.glob(ext):
            if f.stat().st_mtime > index_mtime:
                return True
    return False


def build_index(docs_dir: Path | None = None, index_dir: Path | None = None) -> None:
    """Chunk all .md/.txt docs, embed them, build FAISS + BM25 index."""
    import faiss
    import numpy as np
    from rank_bm25 import BM25Okapi

    docs_dir = docs_dir or _DEFAULT_DOCS_DIR
    index_dir = index_dir or _DEFAULT_INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)

    docs: list[dict] = []  # [{text, source}]
    for ext in ("*.md", "*.txt"):
        for f in sorted(docs_dir.glob(ext)):
            text = f.read_text(encoding="utf-8")
            for chunk in _chunk_text(text):
                docs.append({"text": chunk, "source": f.name})

    if not docs:
        print("No documents found in", docs_dir)
        return

    print(f"Embedding {len(docs)} chunks from {docs_dir}…")
    texts = [d["text"] for d in docs]
    # Batch to avoid large requests
    batch_size = 32
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        all_vectors.extend(embed(texts[i: i + batch_size]))

    vectors = np.array(all_vectors, dtype="float32")
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    faiss.write_index(index, str(index_dir / "index.faiss"))

    # BM25
    tokenised = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenised)

    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(index_dir / "docs.json", "w") as f:
        json.dump(docs, f)

    print(f"Index built: {len(docs)} chunks, dim={dim}")


def _load_index(index_dir: Path):
    import faiss
    from rank_bm25 import BM25Okapi

    faiss_index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(index_dir / "docs.json") as f:
        docs = json.load(f)
    return faiss_index, bm25, docs


def knowledge_search(query: str, top_k: int = 5, docs_dir: Path | None = None) -> str:
    """Hybrid RRF retrieval from FAISS + BM25 index.

    Auto-rebuilds index if stale.
    """
    import faiss
    import numpy as np

    docs_dir = docs_dir or _DEFAULT_DOCS_DIR
    index_dir = docs_dir / ".index"

    if _index_is_stale(docs_dir, index_dir):
        build_index(docs_dir, index_dir)

    try:
        faiss_index, bm25, docs = _load_index(index_dir)
    except Exception as e:
        return f"RAG index not available: {e}"

    # Dense retrieval
    try:
        q_vec = np.array(embed([query]), dtype="float32")
        faiss.normalize_L2(q_vec)
        k = min(top_k * 3, faiss_index.ntotal)
        _, dense_indices = faiss_index.search(q_vec, k)
        dense_ranks = {int(idx): rank for rank, idx in enumerate(dense_indices[0]) if idx >= 0}
    except Exception as e:
        dense_ranks = {}

    # Sparse (BM25) retrieval
    scores = bm25.get_scores(query.lower().split())
    bm25_top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: top_k * 3]
    bm25_ranks = {idx: rank for rank, idx in enumerate(bm25_top)}

    # Reciprocal Rank Fusion
    k_rrf = 60
    all_ids = set(dense_ranks) | set(bm25_ranks)
    rrf_scores = {
        i: (1 / (k_rrf + dense_ranks.get(i, len(docs))) + 1 / (k_rrf + bm25_ranks.get(i, len(docs))))
        for i in all_ids
    }
    top_ids = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_ids:
        doc = docs[idx]
        results.append(f"[{doc['source']}]\n{doc['text']}")

    return "\n\n---\n\n".join(results) if results else "No relevant documents found."


async def knowledge_search_async(query: str, top_k: int = 5) -> str:
    """Async wrapper for knowledge_search."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, knowledge_search, query, top_k)


# ---------------------------------------------------------------------------
# CLI entry point: python -m tools.rag build
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    else:
        print("Usage: python -m tools.rag build")
