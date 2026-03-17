import json
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
for _t in TOOLS_SCHEMA:
    _fn = _t["function"]
    _required = frozenset(
        _fn["parameters"].get("required")
        or list(_fn["parameters"].get("properties", {}).keys())
    )
    if _required:
        _SCHEMA_LOOKUP[_required] = _fn["name"]


def _try_fix_tool_call(message: ChatCompletionMessage) -> ChatCompletionMessage:
    """If the model put tool-call JSON in the content field, convert it to tool_calls."""
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

    # Reconstruct a fake tool_call on the message dict and re-parse
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
            finish_reason = response.choices[0].finish_reason

            # Add assistant message to history
            self.messages.append(message.model_dump(exclude_unset=False))

            # No tool calls → final answer
            if not message.tool_calls:
                return message.content or ""

            # Execute each tool call and collect results
            for tc in message.tool_calls:
                tool_name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                args_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in arguments.items())
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
