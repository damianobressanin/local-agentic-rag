"""CLI entry-point for the LangGraph chatbot."""

import os
import sys
from uuid import uuid4

# ── Make the src/ package importable ────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chatbot.config import get_langfuse_handler, get_langfuse_status, get_llm, shutdown_langfuse  # noqa: E402
from chatbot.graph import build_graph  # noqa: E402


def main() -> None:
    graph = build_graph()

    # Session id: reuse across messages so Langfuse groups them together
    session_id = os.environ.get("CHAT_SESSION_ID") or f"cli-{uuid4().hex[:8]}"

    # Base LangGraph config (thread_id keeps in-memory conversation history)
    config: dict = {"configurable": {"thread_id": session_id}}

    # Attach Langfuse callback + session metadata if available
    langfuse_handler = get_langfuse_handler()
    if langfuse_handler is not None:
        config["callbacks"] = [langfuse_handler]
        config["metadata"] = {"langfuse_session_id": session_id}

    # Show connection info at startup
    llm = get_llm()
    langfuse_on, langfuse_info = get_langfuse_status()
    print(f"Model   : {llm.model_name}")
    print(f"Server  : {llm.openai_api_base}")
    print(f"Langfuse: {'ON' if langfuse_on else 'OFF'} ({langfuse_info})")
    print(f"Session : {session_id}")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    try:
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
    finally:
        shutdown_langfuse()


if __name__ == "__main__":
    main()
