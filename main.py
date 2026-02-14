"""CLI entry-point for the LangGraph chatbot."""

import os
import sys
from uuid import uuid4

# ── Make the src/ package importable ────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chatbot.config import get_langfuse_handler, get_langfuse_status, shutdown_langfuse  # noqa: E402
from chatbot.graph import build_graph  # noqa: E402
from chatbot.models import get_settings  # noqa: E402

RAG_SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "rag_summary.txt")


def _load_rag_summary() -> str | None:
    """Read the RAG summary file if it exists."""
    if os.path.isfile(RAG_SUMMARY_PATH):
        with open(RAG_SUMMARY_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def main() -> None:
    graph = build_graph()

    # Session id: reuse across messages so Langfuse groups them together
    s = get_settings()
    session_id = s.chat_session_id or f"cli-{uuid4().hex[:8]}"

    # Base LangGraph config (thread_id keeps in-memory conversation history)
    config: dict = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 50,
    }

    # Attach Langfuse callback + session metadata if available
    langfuse_handler = get_langfuse_handler()
    if langfuse_handler is not None:
        config["callbacks"] = [langfuse_handler]
        config["metadata"] = {"langfuse_session_id": session_id}

    # Build system prompt with RAG context awareness
    rag_summary = _load_rag_summary()
    system_parts: list[str] = [
        "You are a helpful AI assistant with access to tools.",
        "",
        "IMPORTANT RULES:",
        "1. Focus more on the user's LATEST messages. Ignore too much old topics "
        "unless the user explicitly refers back to them.",
        "2. When the user's question relates to ANY topic covered in your "
        "knowledge base (see summary below), you MUST call search_documents "
        "to find relevant information before answering.",
        "3. You may call search_documents AT MOST ONCE per user message.",
        "4. After receiving search results, answer immediately using those "
        "results. NEVER call search_documents again with a rephrased query.",
        "5. When citing search results, include the document name and "
        "page number from the [Source: ...] headers.",
        "6. If the search results are not useful, say so honestly and "
        "answer from your own knowledge instead. Do NOT search again.",
        "7. NEVER expose your internal reasoning process in your response. "
        "Reply directly and naturally.",
    ]
    if rag_summary:
        system_parts.append(
            "\nBelow is a summary of the documents available in your knowledge base. "
            "Use this to decide when the search_documents tool is relevant:\n\n"
            + rag_summary
        )

    system_prompt = "\n".join(system_parts)

    # Show connection info at startup
    langfuse_on, langfuse_info = get_langfuse_status()
    print(f"Model   : {s.llm_model}")
    print(f"Server  : {s.llm_base_url}")
    print(f"Langfuse: {'ON' if langfuse_on else 'OFF'} ({langfuse_info})")
    print(f"RAG     : {'summary loaded' if rag_summary else 'no summary (run ingest.py first)'}")
    print(f"Session : {session_id}")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    try:
        first_turn = True
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

            messages: list[dict] = []
            if first_turn:
                messages.append({"role": "system", "content": system_prompt})
                first_turn = False
            messages.append({"role": "user", "content": user_input})

            result = graph.invoke({"messages": messages}, config)

            # The last message in the state is the assistant reply
            assistant_msg = result["messages"][-1]
            print(f"AI: {assistant_msg.content}\n")
    finally:
        shutdown_langfuse()


if __name__ == "__main__":
    main()
