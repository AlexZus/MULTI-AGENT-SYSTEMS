import os

import trafilatura
from ddgs import DDGS

from config import Settings

settings = Settings()


# ── Tool implementations ───────────────────────────────────────────────────────

def web_search(query: str) -> list[dict]:
    """Search the internet for information using DuckDuckGo."""
    try:
        results = DDGS().text(query, max_results=settings.max_search_results)
        return [
            {"title": r["title"], "url": r["href"], "snippet": r["body"]}
            for r in results
        ]
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


def read_url(url: str) -> str:
    """Fetch and extract the full text content from a web page URL."""
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


def write_report(filename: str, content: str) -> str:
    """Save a Markdown research report to a file in the output directory."""
    os.makedirs(settings.output_dir, exist_ok=True)
    if not filename.endswith(".md"):
        filename += ".md"
    path = os.path.join(settings.output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Report saved to {os.path.abspath(path)}"


# ── JSON Schema tool definitions for the API ──────────────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the internet for information using DuckDuckGo. "
                "Returns a list of results, each with 'title', 'url', and 'snippet' fields. "
                "Use this to discover relevant sources and get an overview of a topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": (
                "Fetch and extract the full text content from a web page URL. "
                "Use this after web_search to read the full content of a promising page. "
                "Returns the extracted text, truncated to avoid filling the context window."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the web page to fetch.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_report",
            "description": (
                "Save a Markdown research report to a file in the output directory. "
                "Call this when the user asks for a report or when the research is complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The file name (e.g. 'report.md'). Will be saved in the output/ directory.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full Markdown content of the report.",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
]

# Dispatch map: name -> callable
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "read_url": read_url,
    "write_report": write_report,
}
