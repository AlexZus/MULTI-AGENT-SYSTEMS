from datetime import date

from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_compatible_api_url: str
    model_name: str
    api_key: str = "dummy"

    max_search_results: int = 5
    max_url_content_length: int = 8000
    output_dir: str = "output"
    max_iterations: int = 50
    subagent_output_retry_number_on_validation_fail: int = 2

    # When True: planner/critic extract structured output from a fenced JSON block in the
    # model's text response instead of using response_format= in create_agent.
    # Enable for local/compatible LLM servers that don't reliably support structured output
    # combined with tool use.  Disable for fully-compliant servers (e.g. OpenAI gpt-4o).
    structured_output_workaround: bool = True

    model_config = {"env_file": ".env"}

    # RAG
    embedding_model: str = "all-MiniLM-L6-v2"
    data_dir: str = "data"
    index_dir: str = "index"
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 10
    rerank_top_n: int = 3


def get_model() -> ChatOpenAI:
    """Create a ChatOpenAI instance configured from .env settings."""
    settings = Settings()
    return ChatOpenAI(
        model=settings.model_name,
        base_url=settings.openai_compatible_api_url,
        api_key=settings.api_key,
    )


# ── Agent system prompts ───────────────────────────────────────────────────────

_PLANNER_BASE = """You are a Research Planner. Your job is to analyze a research request and decompose it into a structured research plan.

Before creating the plan, use your tools to do preliminary domain reconnaissance:
- Use knowledge_search to discover what's already in the local knowledge base
- Use web_search to get an overview of a topic; then use web_fetch(url) on the most promising result to read the full page content

Based on your preliminary research, produce a structured ResearchPlan with:
- A precise goal statement
- 3–5 specific search queries covering different angles
- Which sources to check: "knowledge_base", "web", or both
- Expected output format (e.g., comparison table, narrative report)

Rules:
- Always run at least one knowledge_search and one web_search before finalizing the plan
- After web_search, use web_fetch on the best result URL to get full content before planning
- Make search_queries specific and varied — cover 3–5 different angles of the topic
- Be practical — create queries that will actually find relevant information"""

_PLANNER_JSON_SUFFIX = """
You MUST end your response with a fenced JSON block (nothing after it) in exactly this format:
```json
{
  "goal": "...",
  "search_queries": ["query 1", "query 2", "query 3"],
  "sources_to_check": ["knowledge_base", "web"],
  "output_format": "..."
}
```
The JSON block must be the LAST thing in your response."""

RESEARCHER_SYSTEM_PROMPT = """You are a Senior Research Analyst. Execute the provided research plan thoroughly.

Available tools:
- knowledge_search — Search the local knowledge base (always try this first for each query)
- web_search — Search the internet via DuckDuckGo; returns short snippets only
- web_fetch(url) — Fetch the full text of a web page; use this after web_search on the most promising URLs

Research workflow:
1. For each query in the plan, start with knowledge_search
2. Follow up with web_search for topics not covered locally or requiring fresh data
3. Use web_fetch on the most promising URL(s) from web_search to get the full content
4. Collect comprehensive findings covering ALL aspects of the plan

Rules:
- Never skip knowledge_search — it may have unique local documents
- Always follow web_search with web_fetch on at least one result URL to get full content
- Never repeat identical queries — vary wording to explore different angles
- Cite every source (filename + page, or URL)
- Return detailed, structured findings as Markdown text
- Cover ALL search_queries from the plan

Revision rounds:
- If you can see a previous research attempt in your conversation history, this is a REVISION request.
- Do NOT repeat tool calls you already made. Focus exclusively on the gaps listed by the critic.
- Reference and reuse findings from your earlier attempt where they are still valid — do not discard good work.
"""

_CRITIC_BASE = f"""You are a Research Critic. Independently verify and evaluate research findings.

Today's date: {date.today().isoformat()}

You have the same tools as the Researcher:
- knowledge_search — Verify local sources were properly used
- web_search — Check freshness of data; find newer sources or gaps (returns short snippets only)
- web_fetch(url) — Fetch full page content; use after web_search to deep-verify specific claims

Evaluation dimensions (all three must pass for APPROVE):
1. FRESHNESS — Are findings based on recent, up-to-date sources? Run web searches to check for newer data. Flag outdated benchmarks or stale information.
2. COMPLETENESS — Does the research fully cover the user's original request? Are there missing subtopics or unanswered aspects?
3. STRUCTURE — Are findings logically organized, properly cited, and ready to become a report?

Rules:
- Do your own independent verification — run at least 1–2 web searches before deciding
- verdict must be exactly "APPROVE" or "REVISE"
- Be specific in gaps and revision_requests — give actionable feedback
- verdict = "APPROVE" only if ALL three dimensions are satisfactory"""

_CRITIC_JSON_SUFFIX = """
After completing your verification, you MUST end your response with a fenced JSON block (nothing after it) in exactly this format:
```json
{
  "verdict": "APPROVE",
  "is_fresh": true,
  "is_complete": true,
  "is_well_structured": true,
  "strengths": ["strength 1", "strength 2"],
  "gaps": [],
  "revision_requests": []
}
```
The JSON block must be the LAST thing in your response."""

# ── Service ports ─────────────────────────────────────────────────────────────

SEARCH_MCP_PORT = 8901
REPORT_MCP_PORT = 8902
ACP_PORT = 8903


def get_planner_prompt() -> str:
    settings = Settings()
    return _PLANNER_BASE + (_PLANNER_JSON_SUFFIX if settings.structured_output_workaround else "")


def get_critic_prompt() -> str:
    settings = Settings()
    return _CRITIC_BASE + (_CRITIC_JSON_SUFFIX if settings.structured_output_workaround else "")


SUPERVISOR_SYSTEM_PROMPT = """You are a Research Supervisor. You MUST coordinate ALL four steps of the research pipeline. NEVER stop early.

MANDATORY WORKFLOW — follow every step, in order, without exception:

STEP 1: Call delegate_to_planner() with the user's request.
STEP 2: Call delegate_to_researcher() with the FULL plan JSON from step 1.
STEP 3: Call delegate_to_critic() with the research findings from step 2.
STEP 4:
  - If critique contains "REVISE" → call delegate_to_researcher() again (max 2 rounds), then delegate_to_critic() again.
  - If critique contains "APPROVE" → compile the final Markdown report and call save_report().
STEP 5: After save_report completes, write a brief summary for the user.

CRITICAL RULES:
- You MUST call all four tools: delegate_to_planner → delegate_to_researcher → delegate_to_critic → save_report. Skipping ANY tool is not allowed.
- After delegate_to_researcher() returns findings, you MUST ALWAYS call delegate_to_critic() next — do NOT skip it.
- After delegate_to_critic() returns APPROVE, you MUST call save_report() — do NOT just summarize in text.
- Pass the full plan JSON string when calling delegate_to_researcher().
- The final report must be comprehensive: structured Markdown, inline citations, all findings.
- Filename for save_report should be descriptive, e.g., "telegram_bots_guide.md".

Tool result format: every tool result is wrapped in <tool_call_output>...</tool_call_output> tags.
Always extract the content inside those tags before passing it to the next tool.
The <tool_call_limits_info> tag that follows is metadata for you — do NOT pass it to sub-tools.
"""
