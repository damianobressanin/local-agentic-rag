"""Graph nodes – each function is a step in the LangGraph pipeline."""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from chatbot.config import get_llm
from chatbot.state import State


def call_model(state: State, config: RunnableConfig) -> dict:
    """Invoke the LLM with the full conversation history.

    Returns a dict whose ``messages`` key contains the assistant reply.
    The ``add_messages`` reducer in ``State`` will append it automatically.
    """
    llm = get_llm()
    response: AIMessage = llm.invoke(state["messages"], config=config)
    return {"messages": [response]}
