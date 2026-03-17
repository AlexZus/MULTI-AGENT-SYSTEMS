from agent import agent

THREAD_CONFIG = {"configurable": {"thread_id": "session-1"}}


def main():
    print("Research Agent (type 'exit' to quit)")
    print("-" * 40)

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

        for chunk in agent.stream(
            {"messages": [("user", user_input)]},
            config=THREAD_CONFIG,
        ):
            # Tool execution updates
            if "tools" in chunk:
                for msg in chunk["tools"].get("messages", []):
                    tool_name = getattr(msg, "name", "tool")
                    print(f"  [{tool_name}] done")

            # Agent (LLM) updates
            if "agent" in chunk:
                for msg in chunk["agent"].get("messages", []):
                    # Show tool calls being made
                    for tc in getattr(msg, "tool_calls", []):
                        args_preview = str(tc.get("args", ""))[:80]
                        print(f"  → {tc['name']}({args_preview})")
                    # Show final text response
                    if isinstance(msg.content, str) and msg.content:
                        print(f"\nAgent: {msg.content}")


if __name__ == "__main__":
    main()
