"""Integration tests for embedding service — requires running embedding HTTP service."""

import os

import pytest
import pytest_asyncio

from tools.rag import embed, embed_async

pytestmark = pytest.mark.asyncio

EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8084/v1/embeddings")


class TestEmbedding:
    def test_single_embed(self):
        result = embed(["Hello world"])
        assert len(result) == 1
        vec = result[0]
        assert len(vec) == 768  # all-mpnet-base-v2

    def test_batch_embed(self):
        texts = ["Python programming", "FastAPI tutorial", "Machine learning basics"]
        result = embed(texts)
        assert len(result) == 3
        for vec in result:
            assert len(vec) == 768

    def test_vectors_differ(self):
        """Different texts should produce different embeddings."""
        result = embed(["cat", "quantum mechanics"])
        assert result[0] != result[1]

    def test_same_text_consistent(self):
        """Same text should produce the same embedding."""
        r1 = embed(["test consistency"])
        r2 = embed(["test consistency"])
        assert r1[0] == pytest.approx(r2[0], abs=1e-5)

    async def test_async_embed(self):
        result = await embed_async(["async test"])
        assert len(result) == 1
        assert len(result[0]) == 768
