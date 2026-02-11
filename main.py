"""CLI entry-point for the LangGraph chatbot."""

import os
import sys

# ── Make the src/ package importable ────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chatbot.config import get_llm  # noqa: E402
from chatbot.graph import build_graph  # noqa: E402


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "cli-session"}}

    # Show connection info at startup
    llm = get_llm()
    print(f"Model  : {llm.model_name}")
    print(f"Server : {llm.openai_api_base}")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break

        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
        )

        # The last message in the state is the assistant reply
        assistant_msg = result["messages"][-1]
        print(f"AI: {assistant_msg.content}\n")


if __name__ == "__main__":
    main()
