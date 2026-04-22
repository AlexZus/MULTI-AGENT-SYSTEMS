"""Multi-agent research system REPL with HITL interrupt/resume loop."""

import json
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Interrupt
from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler

from agents.middleware import _tool_budget
from config import Settings
from supervisor import create_supervisor

_settings = Settings()
_langfuse = get_client()
_langfuse_handler = CallbackHandler()

# ── Pretty-printing helpers ────────────────────────────────────────────────────

def _print_tool_call(message: AIMessage) -> None:
    """Print tool calls from an AI message."""
    if not message.tool_calls:
        return
    for tc in message.tool_calls:
        args_preview = json.dumps(tc["args"], ensure_ascii=False)
        if len(args_preview) > 120:
            args_preview = args_preview[:120] + "..."
        print(f"\n  🔧 {tc['name']}({args_preview})")


def _print_tool_result(message: ToolMessage) -> None:
    """Print a tool result with a short preview."""
    preview = str(message.content)
    if len(preview) > 300:
        preview = preview[:300] + "..."
    print(f"  📎 {preview}")


def _print_interrupt(interrupts: tuple[Interrupt, ...]) -> None:
    """Display interrupt details for HITL approval."""
    print(f"\n{'=' * 60}")
    print("  ⏸️  ACTION REQUIRES APPROVAL")
    print(f"{'=' * 60}")
    for interrupt in interrupts:
        value = interrupt.value if isinstance(interrupt.value, dict) else {}
        for req in value.get("action_requests", []):
            print(f"  Tool:  {req.get('action', 'N/A')}")
            args = req.get("args", {})
            # Show filename and content preview
            if "filename" in args:
                print(f"  File:  {args['filename']}")
            if "content" in args:
                preview = args["content"][:400]
                print(f"  Preview:\n{preview}{'...' if len(args['content']) > 400 else ''}")
            else:
                print(f"  Args:  {json.dumps(args, indent=2, ensure_ascii=False)[:300]}")
    print()


def _stream_until_interrupt(
    supervisor,
    input_data: dict,
    config: dict,
) -> tuple[bool, tuple[Interrupt, ...]]:
    """Stream supervisor events and pretty-print progress. Returns (interrupted, interrupts)."""
    interrupts: tuple[Interrupt, ...] = ()

    for chunk in supervisor.stream(input_data, config=config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interrupts = chunk["__interrupt__"]
            _print_interrupt(interrupts)
            return True, interrupts

        for node_name, node_updates in chunk.items():
            if node_name.startswith("__"):
                continue
            messages = node_updates.get("messages", []) if isinstance(node_updates, dict) else []
            for msg in messages:
                if isinstance(msg, AIMessage):
                    _print_tool_call(msg)
                    # Only print final text responses (not intermediate reasoning)
                    if msg.content and not msg.tool_calls:
                        print(f"\n[{node_name}] {msg.content[:800]}")
                elif isinstance(msg, ToolMessage):
                    _print_tool_result(msg)

    return False, interrupts


def _handle_hitl(supervisor, interrupts: tuple[Interrupt, ...], config: dict) -> None:
    """Handle the approve / edit / reject loop for a HITL interrupt."""
    while True:
        try:
            action = input("\n👉 approve / edit / reject: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            action = "reject"

        if action == "approve":
            resume_cmd = Command(
                resume={"decisions": [{"type": "approve"}]}
            )
            interrupted, new_interrupts = _stream_until_interrupt(
                supervisor, resume_cmd, config
            )
            if interrupted:
                interrupts = new_interrupts
                _print_interrupt(interrupts)
                continue
            print("\n  ✅ Report saved successfully!")
            break

        elif action == "edit":
            try:
                feedback = input("✏️  Your feedback: ").strip()
            except (EOFError, KeyboardInterrupt):
                feedback = ""
            resume_cmd = Command(
                resume={"decisions": [{"type": "edit", "edited_action": {"feedback": feedback}}]}
            )
            interrupted, new_interrupts = _stream_until_interrupt(
                supervisor, resume_cmd, config
            )
            if interrupted:
                interrupts = new_interrupts
                _print_interrupt(interrupts)
                continue
            break

        elif action == "reject":
            resume_cmd = Command(
                resume={"decisions": [{"type": "reject", "message": "User rejected the report."}]}
            )
            _stream_until_interrupt(supervisor, resume_cmd, config)
            print("\n  ❌ Report rejected. No file was saved.")
            break

        else:
            print("  Please enter 'approve', 'edit', or 'reject'.")


# ── Main REPL ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("Multi-Agent Research System (type 'exit' to quit)")
    print("Pipeline: Supervisor → Planner → Researcher → Critic → (iterate) → Report")
    print("-" * 60)

    supervisor = create_supervisor()

    # One stable session for the whole REPL run
    session_id = f"mas-session-{uuid.uuid4().hex[:8]}"
    user_id = "homework-12-user"
    print(f"[Langfuse session: {session_id}]")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # New thread per conversation turn
        thread_id = str(uuid.uuid4())
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [_langfuse_handler],
        }
        input_data = {"messages": [{"role": "user", "content": user_input}]}

        print(f"\n[Supervisor starting — thread {thread_id[:8]}]")

        @observe(name="mas-run")
        def run_turn() -> None:
            with propagate_attributes(
                session_id=session_id,
                user_id=user_id,
                tags=["multi-agent", "research", "homework-12"],
                metadata={"thread_id": thread_id, "query": user_input[:120]},
            ):
                token = _tool_budget.set({"remaining": _settings.max_iterations})
                try:
                    interrupted, interrupts = _stream_until_interrupt(supervisor, input_data, config)
                    if interrupted:
                        _handle_hitl(supervisor, interrupts, config)
                finally:
                    _tool_budget.reset(token)

        run_turn()

        # Flush any pending spans
        _langfuse.flush()

        # Print final supervisor message if present
        try:
            state = supervisor.get_state(config)
            final_msgs = state.values.get("messages", [])
            if final_msgs:
                last = final_msgs[-1]
                if isinstance(last, AIMessage) and last.content:
                    print(f"\nAgent: {last.content}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
