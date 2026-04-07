import json
import re
import uuid

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from config import Settings, SYSTEM_PROMPT
from tools import TOOLS_SCHEMA, TOOL_FUNCTIONS

settings = Settings()

client = OpenAI(
    base_url=settings.openai_compatible_api_url,
    api_key=settings.api_key,
)

# Map frozenset(required_param_names) -> tool_name for text-JSON detection
_SCHEMA_LOOKUP: dict[frozenset, str] = {}
_ALL_TOOL_NAMES: set[str] = set()
for _t in TOOLS_SCHEMA:
    _fn = _t["function"]
    _all_tool_names_local = _fn["name"]
    _ALL_TOOL_NAMES.add(_all_tool_names_local)
    _required = frozenset(
        _fn["parameters"].get("required")
        or list(_fn["parameters"].get("properties", {}).keys())
    )
    if _required:
        _SCHEMA_LOOKUP[_required] = _fn["name"]

# Regex patterns that suggest the model wrote a tool call as plain text
_TEXT_TOOL_CALL_PATTERNS = [
    # {"query": "..."} or {"url": "...", ...}  — bare JSON object
    re.compile(r'^\s*\{'),
    # tool_name(... or tool_name\n{
    re.compile(r'\b(' + '|'.join(re.escape(n) for n in _ALL_TOOL_NAMES) + r')\s*[\(\{]', re.IGNORECASE),
    # tool_name: { or "tool_name": or tool":"tool_name
    re.compile(r'\b(' + '|'.join(re.escape(n) for n in _ALL_TOOL_NAMES) + r')\b.*[\{\(]', re.IGNORECASE),
]

_TEXT_TOOL_CALL_ERROR = (
    "ERROR: You tried to call a tool by writing it as plain text in your response. "
    "This does NOT execute the tool. "
    "You MUST use the structured tool-call mechanism provided by the API "
    "(the 'tool_calls' field in the assistant message) — not free-form text. "
    "Do not write tool names or JSON arguments in your response text. "
    "Instead, issue the tool call through the API and wait for the result."
)


def _looks_like_text_tool_call(content: str) -> bool:
    """Return True if the content appears to contain a tool call written as plain text."""
    for pattern in _TEXT_TOOL_CALL_PATTERNS:
        if pattern.search(content):
            return True
    return False


def _try_fix_tool_call(message: ChatCompletionMessage) -> ChatCompletionMessage:
    """If the model put a bare JSON tool-call in the content field, convert it to tool_calls."""
    if message.tool_calls:
        return message
    content = (message.content or "").strip()
    if not content.startswith("{"):
        return message
    try:
        args = json.loads(content)
    except json.JSONDecodeError:
        return message
    if not isinstance(args, dict):
        return message
    tool_name = _SCHEMA_LOOKUP.get(frozenset(args.keys()))
    if not tool_name:
        return message

    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall, Function
    )
    fake_tc = ChatCompletionMessageToolCall(
        id=f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=Function(name=tool_name, arguments=json.dumps(args)),
    )
    message.tool_calls = [fake_tc]
    message.content = ""
    return message


class ResearchAgent:
    """Research agent with a custom ReAct loop and persistent dialogue memory."""

    def __init__(self):
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def _execute_tool(self, name: str, arguments: dict) -> str:
        fn = TOOL_FUNCTIONS.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'"
        try:
            result = fn(**arguments)
            return json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
        except Exception as e:
            return f"Error executing {name}: {e}"

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for iteration in range(settings.max_iterations):
            response = client.chat.completions.create(
                model=settings.model_name,
                messages=self.messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
            )

            message = _try_fix_tool_call(response.choices[0].message)
            content = message.content or ""

            # Add assistant message to history
            self.messages.append(message.model_dump(exclude_unset=False))

            if not message.tool_calls:
                # Check if the model wrote a tool call as plain text instead of using the API
                if _looks_like_text_tool_call(content):
                    print(f"\n⚠️  Model wrote a tool call as plain text — sending correction")
                    self.messages.append({
                        "role": "user",
                        "content": _TEXT_TOOL_CALL_ERROR,
                    })
                    continue

                # Genuine final answer
                return content

            # Execute each tool call
            for tc in message.tool_calls:
                tool_name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                args_str = ", ".join(
                    f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                    for k, v in arguments.items()
                )
                print(f"\n🔧 Tool call: {tool_name}({args_str[:120]})")

                result = self._execute_tool(tool_name, arguments)

                preview = result[:200] + "..." if len(result) > 200 else result
                print(f"📎 Result: {preview}")

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return "Error: reached maximum iteration limit without a final answer."


agent = ResearchAgent()
