import os

import trafilatura
from ddgs import DDGS
from langchain.tools import tool

from config import Settings
import retriever

settings = Settings()


# ── Tool implementations ───────────────────────────────────────────────────────

def _web_search(query: str) -> list[dict]:
    try:
        results = DDGS().text(query, max_results=settings.max_search_results)
        return [
            {"title": r["title"], "url": r["href"], "snippet": r["body"]}
            for r in results
        ]
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


def _read_url(url: str) -> str:
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
        return f"Error reading {url}: {str(e)}"


def _knowledge_search(query: str) -> str:
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


def _save_report(filename: str, content: str) -> str:
    os.makedirs(settings.output_dir, exist_ok=True)
    if not filename.endswith(".md"):
        filename += ".md"
    path = os.path.join(settings.output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Report saved to {os.path.abspath(path)}"


# ── LangChain @tool versions ──────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the internet for information using DuckDuckGo.

    Returns a JSON array of results, each with 'title', 'url', and 'snippet' fields.
    Snippets are short — use web_fetch(url) on a promising result to read the
    full page content.
    """
    import json
    return json.dumps(_web_search(query), ensure_ascii=False, indent=2)


@tool
def web_fetch(url: str) -> str:
    """Fetch the full text content of a web page by URL.

    Use this to read the complete content of a page found via web_search.
    web_search returns only short snippets — use web_fetch when you need the
    full article, documentation, or page body.
    Returns the extracted text, truncated to avoid filling the context window.
    """
    return _read_url(url)


@tool
def knowledge_search(query: str) -> str:
    """Search the local knowledge base of ingested PDF/TXT documents.

    Use this for questions that may be answered by locally stored research
    papers or documents. Returns the most relevant excerpts with source
    and page references. Always try this before going to the web.
    """
    return _knowledge_search(query)


@tool
def save_report(filename: str, content: str) -> str:
    """Save a Markdown research report to a file in the output directory.

    Call this as the final step when the research has been approved by the Critic.
    The file will be saved in the output/ directory.
    """
    return _save_report(filename, content)

