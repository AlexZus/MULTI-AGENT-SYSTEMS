"""Integration tests for RAG tool — requires running embedding service."""

import os
import tempfile
from pathlib import Path

import pytest

from tools.rag import _index_is_stale, build_index, knowledge_search

pytestmark = pytest.mark.asyncio


class TestRagIndex:
    def test_stale_when_no_index(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("some content")
        assert _index_is_stale(docs_dir, docs_dir / ".index") is True

    def test_not_stale_after_build(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("FastAPI is a web framework for Python.")
        index_dir = docs_dir / ".index"
        build_index(docs_dir, index_dir)
        assert _index_is_stale(docs_dir, index_dir) is False

    def test_stale_after_doc_update(self, tmp_path):
        import time
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        doc = docs_dir / "test.md"
        doc.write_text("original content")
        index_dir = docs_dir / ".index"
        build_index(docs_dir, index_dir)

        time.sleep(0.05)  # ensure mtime differs
        doc.write_text("updated content")
        assert _index_is_stale(docs_dir, index_dir) is True


class TestRagSearch:
    def test_basic_search(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "fastapi.md").write_text(
            "FastAPI is a modern, fast web framework for building APIs with Python. "
            "It uses Pydantic for data validation and OpenAPI for documentation."
        )
        (docs_dir / "unrelated.md").write_text(
            "Recipes for cooking pasta: boil water, add salt, cook 8 minutes."
        )
        result = knowledge_search("FastAPI web framework", top_k=1, docs_dir=docs_dir)
        assert "FastAPI" in result or "web framework" in result.lower()

    def test_auto_rebuild_on_stale(self, tmp_path):
        """knowledge_search auto-rebuilds index when docs change."""
        import time
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        doc = docs_dir / "content.md"
        doc.write_text("Python decorators are a way to modify function behaviour.")
        # First search builds index
        knowledge_search("decorators", top_k=1, docs_dir=docs_dir)

        time.sleep(0.05)
        doc.write_text("Pydantic provides runtime type checking for Python.")
        # Second search should auto-rebuild without error
        result = knowledge_search("Pydantic type checking", top_k=1, docs_dir=docs_dir)
        assert isinstance(result, str)

    def test_no_results_graceful(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "empty.md").write_text("x")
        result = knowledge_search("quantum entanglement black holes", top_k=3, docs_dir=docs_dir)
        assert isinstance(result, str)
        assert len(result) > 0
