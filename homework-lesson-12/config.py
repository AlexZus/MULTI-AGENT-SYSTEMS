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

    model_config = {"env_file": ".env", "extra": "ignore"}

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


# ── Structured-output suffixes (technical workaround, not part of agent persona) ──────

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


# ── Prompt loading from Langfuse ─────────────────────────────────────────────

def _load_langfuse_prompt(name: str, label: str = "production", **variables: str) -> str:
    """Load a prompt from Langfuse Prompt Management. Raises RuntimeError on failure."""
    from langfuse import get_client
    lf = get_client()
    prompt = lf.get_prompt(name, label=label)
    return prompt.compile(**variables) if variables else prompt.prompt


def get_planner_prompt() -> str:
    """Load planner system prompt from Langfuse."""
    text = _load_langfuse_prompt("mas-planner")
    settings = Settings()
    if settings.structured_output_workaround:
        text += _PLANNER_JSON_SUFFIX
    return text


def get_researcher_prompt() -> str:
    """Load researcher system prompt from Langfuse."""
    return _load_langfuse_prompt("mas-researcher")


def get_critic_prompt() -> str:
    """Load critic system prompt from Langfuse."""
    text = _load_langfuse_prompt("mas-critic", today_date=date.today().isoformat())
    settings = Settings()
    if settings.structured_output_workaround:
        text += _CRITIC_JSON_SUFFIX
    return text


def get_supervisor_prompt() -> str:
    """Load supervisor system prompt from Langfuse."""
    return _load_langfuse_prompt("mas-supervisor")


