"""Graph nodes – each function is a step in the LangGraph pipeline."""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from chatbot.config import get_llm
from chatbot.state import State
from chatbot.tools import ALL_TOOLS

# ANSI colors for terminal output
_DIM = "\033[2m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def _log(icon: str, msg: str) -> None:
    """Print a dimmed status line so it's visually distinct from AI output."""
    print(f"{_DIM}{icon}  {msg}{_RESET}")


def call_model(state: State, config: RunnableConfig) -> dict:
    """Invoke the LLM with tools bound.

    The LLM decides whether to call a tool (e.g. search_documents) or
    answer directly.  If it produces tool_calls, the graph will route
    to the tool node and loop back here with the results.
    """
    _log("🤖", "Thinking...")
    llm = get_llm().bind_tools(ALL_TOOLS)

    try:
        response: AIMessage = llm.invoke(state["messages"], config=config)
    except Exception as exc:
        _log("❌", f"{_YELLOW}LLM call failed: {exc}{_RESET}")
        error_msg = AIMessage(
            content=(
                "I'm sorry, I couldn't process your request. "
                "The language model is currently unavailable. "
                "Please check that your LLM server is running."
            )
        )
        return {"messages": [error_msg]}

    # Log what the LLM decided to do
    if response.tool_calls:
        for tc in response.tool_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
            _log("🔧", f"Calling tool: {_CYAN}{tc['name']}{_RESET}{_DIM}({args})")
    else:
        _log("💬", "Responding directly (no tool call)")

    return {"messages": [response]}
