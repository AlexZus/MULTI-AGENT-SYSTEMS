"""ReportMCP — MCP server exposing the save_report tool."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP

from config import Settings

settings = Settings()
mcp = FastMCP(name="ReportMCP")


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool
def save_report(filename: str, content: str) -> str:
    """Save a Markdown research report to a file in the output directory.

    Call this as the final step when the research has been approved by the Critic.
    The file will be saved in the output/ directory.
    """
    os.makedirs(settings.output_dir, exist_ok=True)
    if not filename.endswith(".md"):
        filename += ".md"
    path = os.path.join(settings.output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Report saved to {os.path.abspath(path)}"


# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("resource://output-dir")
def output_dir_info() -> str:
    """Information about the report output directory and its contents."""
    abs_path = os.path.abspath(settings.output_dir)
    if not os.path.exists(abs_path):
        return json.dumps({"path": abs_path, "reports": [], "status": "directory does not exist"})

    reports = sorted(
        f for f in os.listdir(abs_path) if f.endswith(".md")
    )
    return json.dumps({"path": abs_path, "reports": reports})


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8902)
