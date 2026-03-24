"""
Knowledge ingestion pipeline.

Loads documents from data/ directory, splits into chunks,
generates embeddings, and saves the index to disk.

Usage: python ingest.py
"""

import os
import pickle
import re
from pathlib import Path

# Point HuggingFace cache to a writable project-local directory
_HF_CACHE = os.path.join(os.path.dirname(__file__), ".hf_cache")
os.makedirs(_HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", _HF_CACHE)

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import Settings

settings = Settings()


def load_documents(data_dir: str) -> list[dict]:
    """Load all PDF and TXT/MD files from data_dir, returning one entry per page."""
    documents = []
    data_path = Path(data_dir)

    for file_path in sorted(data_path.iterdir()):
        if file_path.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(str(file_path))
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append({
                            "text": text,
                            "source": file_path.name,
                            "page": page_num,
                        })
            except Exception as e:
                print(f"Warning: could not read {file_path.name}: {e}")
        elif file_path.suffix.lower() in (".txt", ".md"):
            try:
                text = file_path.read_text(encoding="utf-8")
                if text.strip():
                    documents.append({
                        "text": text,
                        "source": file_path.name,
                        "page": 1,
                    })
            except Exception as e:
                print(f"Warning: could not read {file_path.name}: {e}")

    return documents


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', '?' boundaries."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in parts if s.strip()]


def split_into_chunks(documents: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Split document texts into chunks whose boundaries align with sentence endings."""
    chunks = []

    for doc in documents:
        sentences = _split_sentences(doc["text"])

        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            s_len = len(sentence)
            # If a single sentence exceeds chunk_size, emit it as its own chunk
            if not current and s_len >= chunk_size:
                chunks.append({"text": sentence, "source": doc["source"], "page": doc["page"]})
                continue

            if current_len + (1 if current else 0) + s_len > chunk_size and current:
                chunks.append({
                    "text": " ".join(current),
                    "source": doc["source"],
                    "page": doc["page"],
                })
                # Roll back sentences for overlap
                overlap_len = 0
                overlap: list[str] = []
                for s in reversed(current):
                    if overlap_len + len(s) > chunk_overlap:
                        break
                    overlap.insert(0, s)
                    overlap_len += len(s) + 1
                current = overlap
                current_len = overlap_len

            current.append(sentence)
            current_len += (1 if len(current) > 1 else 0) + s_len

        if current:
            chunks.append({"text": " ".join(current), "source": doc["source"], "page": doc["page"]})

    return chunks


def ingest():
    os.makedirs(settings.index_dir, exist_ok=True)

    print(f"Loading documents from '{settings.data_dir}'...")
    documents = load_documents(settings.data_dir)
    sources = sorted(set(d["source"] for d in documents))
    print(f"Loaded {len(documents)} pages from {len(sources)} file(s): {sources}")

    print(f"Splitting into chunks (size={settings.chunk_size}, overlap={settings.chunk_overlap})...")
    chunks = split_into_chunks(documents, settings.chunk_size, settings.chunk_overlap)
    print(f"Created {len(chunks)} chunks")

    print(f"Generating embeddings with '{settings.embedding_model}'...")
    model = SentenceTransformer(settings.embedding_model)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Normalize for cosine similarity via inner-product index
    faiss.normalize_L2(embeddings)

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"Index contains {index.ntotal} vectors (dim={dimension})")

    faiss_path = os.path.join(settings.index_dir, "faiss.index")
    chunks_path = os.path.join(settings.index_dir, "chunks.pkl")

    faiss.write_index(index, faiss_path)
    print(f"Saved FAISS index → {faiss_path}")

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved chunks      → {chunks_path}")

    print("\nIngestion complete!")


if __name__ == "__main__":
    ingest()
