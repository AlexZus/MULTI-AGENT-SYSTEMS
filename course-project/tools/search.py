"""Web search tool using DuckDuckGo."""

from __future__ import annotations

import json

SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web using DuckDuckGo. Returns a list of results with title, URL, and snippet. "
            "Use this for finding documentation, examples, and up-to-date information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def web_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo and return formatted results."""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        if not results:
            return "No results found."
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Search error: {e}"


async def web_search_async(query: str, max_results: int = 5) -> str:
    """Async wrapper for web_search (runs in executor to avoid blocking)."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, web_search, query, max_results)
