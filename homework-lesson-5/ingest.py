"""
Knowledge ingestion pipeline.

Loads documents from data/ directory, splits into chunks,
generates embeddings, and saves the index to disk.

Usage: python ingest.py
"""


def ingest():
    # TODO:
    # 1. Load documents from config.data_dir (PDF, TXT, MD)
    # 2. Split into chunks using TextSplitter
    # 3. Generate embeddings
    # 4. Build vector store (FAISS, Qdrant, Chroma, etc.)
    # 5. Save index to config.index_dir
    # 6. Save chunks for BM25 retriever (pickle or JSON)
    pass


if __name__ == "__main__":
    ingest()
