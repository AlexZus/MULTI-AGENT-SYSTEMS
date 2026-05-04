"""Application configuration — all settings loaded from .env."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openai_compatible_api_url: str
    model_name: str
    api_key: str

    # MCP servers
    mcp_filesystem_url: str
    mcp_repl_url: str

    # Embedding
    embedding_url: str

    # MongoDB / tracevault
    mongodb_url: str
    mongodb_db: str

    # Ports
    tracevault_port: int
    app_port: int

    # Pipeline
    max_qa_iterations: int
    structured_output_workaround: bool
    tasks_state_dir: str
    workspace_dir: str

    # Agent loop limits — not in .env, reasonable defaults kept here
    max_agent_iterations: int = 50
    max_tool_calls_per_agent: int = 30
    invalid_tool_call_max_retries: int = 7

    model_config = {"env_file": ".env"}
