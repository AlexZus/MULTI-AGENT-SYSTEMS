"""Run the MAS with several queries to generate Langfuse traces.
Auto-approves every HITL interrupt so the script is non-interactive.
"""

import json
import uuid
from dotenv import load_dotenv
load_dotenv(".env")

from langchain_core.messages import AIMessage, ToolMessage
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

QUERIES = [
    "What are the main differences between LangGraph and AutoGen for building multi-agent systems?",
    "What is retrieval-augmented generation (RAG) and what are best practices for implementing it?",
    "Compare Python asyncio vs threading for IO-bound tasks",
]

SESSION_ID = f"mas-session-{uuid.uuid4().hex[:8]}"
USER_ID = "homework-12-user"


def _stream_and_autoapprove(supervisor, input_data, config):
    """Stream until done, auto-approving any HITL interrupt."""
    while True:
        interrupted = False
        for chunk in supervisor.stream(input_data, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                interrupted = True
                interrupts = chunk["__interrupt__"]
                # Show what would be approved
                for interrupt in interrupts:
                    value = interrupt.value if isinstance(interrupt.value, dict) else {}
                    for req in value.get("action_requests", []):
                        fname = req.get("args", {}).get("filename", "?")
                        print(f"  [HITL] Auto-approving save_report: {fname}")
                break  # exit the for-loop to handle interrupt

            for node_name, node_updates in chunk.items():
                if node_name.startswith("__"):
                    continue
                messages = (
                    node_updates.get("messages", [])
                    if isinstance(node_updates, dict)
                    else []
                )
                for msg in messages:
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"  [{node_name}] -> {tc['name']}()")
                    elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                        preview = str(msg.content)[:120].replace("\n", " ")
                        print(f"  [{node_name}] {preview}")

        if not interrupted:
            break  # graph finished normally

        # Auto-approve
        input_data = Command(resume={"decisions": [{"type": "approve"}]})


@observe(name="mas-run")
def run_query(supervisor, query: str, thread_id: str) -> None:
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [_langfuse_handler],
    }
    with propagate_attributes(
        session_id=SESSION_ID,
        user_id=USER_ID,
        tags=["multi-agent", "research", "homework-12"],
        metadata={"thread_id": thread_id, "query": query[:120]},
    ):
        token = _tool_budget.set({"remaining": _settings.max_iterations})
        try:
            _stream_and_autoapprove(
                supervisor,
                {"messages": [{"role": "user", "content": query}]},
                config,
            )
        finally:
            _tool_budget.reset(token)


def main():
    print(f"Langfuse session: {SESSION_ID}")
    print(f"Running {len(QUERIES)} queries to generate traces...\n")

    supervisor = create_supervisor()

    for i, query in enumerate(QUERIES, 1):
        thread_id = str(uuid.uuid4())
        print(f"\n{'='*60}")
        print(f"Query {i}/{len(QUERIES)}: {query[:70]}...")
        print(f"Thread: {thread_id[:8]}")
        print('='*60)

        try:
            run_query(supervisor, query, thread_id)
            print(f"  Query {i} complete.")
        except Exception as e:
            print(f"  Query {i} error: {e}")

        _langfuse.flush()

    print(f"\nDone. Check Langfuse UI → session '{SESSION_ID}'")


if __name__ == "__main__":
    main()
