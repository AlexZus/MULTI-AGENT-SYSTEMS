import os

import trafilatura
from ddgs import DDGS
from langchain_core.tools import tool

from config import Settings

settings = Settings()


@tool
def web_search(query: str) -> list[dict]:
    """Search the internet for information using DuckDuckGo.

    Returns a list of results, each with 'title', 'url', and 'snippet' fields.
    Use this to discover relevant sources and get an overview of a topic.
    """
    try:
        results = DDGS().text(query, max_results=settings.max_search_results)
        return [
            {"title": r["title"], "url": r["href"], "snippet": r["body"]}
            for r in results
        ]
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


@tool
def read_url(url: str) -> str:
    """Fetch and extract the full text content from a web page URL.

    Use this after web_search to read the full content of a promising page.
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
        return f"Error reading {url}: {str(e)}"


@tool
def write_report(filename: str, content: str) -> str:
    """Save a Markdown research report to a file in the output directory.

    Args:
        filename: The file name (e.g. 'report.md'). Will be saved in the output/ directory.
        content: The full Markdown content of the report.

    Returns a confirmation message with the full path to the saved file.
    """
    os.makedirs(settings.output_dir, exist_ok=True)
    if not filename.endswith(".md"):
        filename += ".md"
    path = os.path.join(settings.output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Report saved to {os.path.abspath(path)}"
