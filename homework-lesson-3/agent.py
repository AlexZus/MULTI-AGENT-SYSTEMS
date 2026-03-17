import json
import uuid

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from config import Settings, SYSTEM_PROMPT
from tools import web_search, read_url, write_report

settings = Settings()


def _build_schema_lookup(tools: list) -> dict:
    """Map frozenset(required_arg_names) -> tool_name for schema-based matching."""
    lookup = {}
    for t in tools:
        if not t.args_schema:
            continue
        schema = t.args_schema.model_json_schema()
        required = frozenset(
            schema.get("required") or list(schema.get("properties", {}).keys())
        )
        if required:
            lookup[required] = t.name
    return lookup


def _try_parse_tool_call(content: str, schema_lookup: dict) -> dict | None:
    """Try to match a JSON string to a tool call via parameter schema."""
    content = content.strip()
    if not content.startswith("{"):
        return None
    try:
        args = json.loads(content)
        if not isinstance(args, dict):
            return None
    except (json.JSONDecodeError, ValueError):
        return None

    keys = frozenset(args.keys())
    if keys in schema_lookup:
        return {
            "name": schema_lookup[keys],
            "args": args,
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        }
    return None


class _ToolCallFixingRunnable(Runnable):
    """Wraps a bound model to convert JSON text content to proper tool_calls.

    Some models (e.g. run via vLLM without --enable-auto-tool-choice) output
    tool call arguments as plain JSON in the content field instead of in the
    tool_calls field. This wrapper detects that pattern and fixes it.
    """

    def __init__(self, inner: Runnable, schema_lookup: dict):
        self.inner = inner
        self.schema_lookup = schema_lookup

    def _fix(self, msg: AIMessage) -> AIMessage:
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            if isinstance(msg.content, str):
                call = _try_parse_tool_call(msg.content, self.schema_lookup)
                if call:
                    return AIMessage(
                        content="",
                        tool_calls=[call],
                        response_metadata=msg.response_metadata,
                        id=msg.id,
                    )
        return msg

    def invoke(self, input, config=None, **kwargs):
        return self._fix(self.inner.invoke(input, config=config, **kwargs))

    async def ainvoke(self, input, config=None, **kwargs):
        return self._fix(await self.inner.ainvoke(input, config=config, **kwargs))

    def stream(self, input, config=None, **kwargs):
        # Accumulate the full response, then fix and yield once.
        full = None
        for chunk in self.inner.stream(input, config=config, **kwargs):
            full = chunk if full is None else full + chunk
        if full is not None:
            yield self._fix(full)

    async def astream(self, input, config=None, **kwargs):
        full = None
        async for chunk in self.inner.astream(input, config=config, **kwargs):
            full = chunk if full is None else full + chunk
        if full is not None:
            yield self._fix(full)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class ToolCallAwareChatOpenAI(ChatOpenAI):
    """ChatOpenAI that handles models outputting tool call JSON as text content."""

    def bind_tools(self, tools, **kwargs):
        bound = super().bind_tools(tools, **kwargs)
        return _ToolCallFixingRunnable(bound, _build_schema_lookup(tools))


ALL_TOOLS = [web_search, read_url, write_report]

llm = ToolCallAwareChatOpenAI(
    base_url=settings.openai_compatible_api_url,
    api_key=settings.api_key,
    model=settings.model_name,
    temperature=0,
)

memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=ALL_TOOLS,
    checkpointer=memory,
    prompt=SYSTEM_PROMPT,
)
