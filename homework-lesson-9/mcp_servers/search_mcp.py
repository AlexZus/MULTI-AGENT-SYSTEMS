"""SearchMCP — MCP server exposing web_search, read_url, knowledge_search tools."""

import json
import os
import sys

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP

import retriever
import trafilatura
from config import Settings
from ddgs import DDGS

settings = Settings()
mcp = FastMCP(name="SearchMCP")


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool
def web_search(query: str) -> str:
    """Search the internet for information using DuckDuckGo.

    Returns a JSON array of results, each with 'title', 'url', and 'snippet'
    fields. Snippets are short — use read_url on a promising result to read
    the full page content.
    """
    try:
        results = DDGS().text(query, max_results=settings.max_search_results)
        items = [
            {"title": r["title"], "url": r["href"], "snippet": r["body"]}
            for r in results
        ]
        return json.dumps(items, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps([{"error": f"Search failed: {e}"}])


@mcp.tool
def read_url(url: str) -> str:
    """Fetch the full text content of a web page by URL.

    Use this to read the complete content of a page found via web_search.
    Returns the extracted text, truncated to avoid filling the context window.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Error: Could not fetch content from {url}"
        text = trafilatura.extract(downloaded)
        if not text:
            return f"Error: Could not extract readable text from {url}"
        if len(text) > settings.max_url_content_length:
            text = text[: settings.max_url_content_length] + "\n\n[Content truncated...]"
        return text
    except Exception as e:
        return f"Error reading {url}: {e}"


@mcp.tool
def knowledge_search(query: str) -> str:
    """Search the local knowledge base of ingested PDF/TXT documents.

    Use this for questions that may be answered by locally stored research
    papers or documents. Returns the most relevant excerpts with source
    and page references. Always try this before going to the web.
    """
    try:
        results = retriever.search(query)
    except FileNotFoundError as e:
        return f"Knowledge base not available: {e}"
    if not results:
        return "No relevant documents found in the knowledge base."
    parts = []
    for i, chunk in enumerate(results, start=1):
        parts.append(
            f"[{i}] Source: {chunk['source']}, Page {chunk['page']}\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("resource://knowledge-base-stats")
def knowledge_base_stats() -> str:
    """Statistics about the local knowledge base index."""
    import pickle
    from datetime import datetime

    chunks_path = os.path.join(settings.index_dir, "chunks.pkl")
    if not os.path.exists(chunks_path):
        return json.dumps({"doc_count": 0, "last_updated": None, "status": "not built"})

    stat = os.stat(chunks_path)
    last_updated = datetime.fromtimestamp(stat.st_mtime).isoformat()

    try:
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        doc_count = len(set(c.get("source", "") for c in chunks))
        chunk_count = len(chunks)
    except Exception:
        doc_count = 0
        chunk_count = 0

    return json.dumps({
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "last_updated": last_updated,
        "index_dir": os.path.abspath(settings.index_dir),
    })


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8901)
