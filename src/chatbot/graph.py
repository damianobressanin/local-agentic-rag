"""Build and compile the LangGraph conversation graph."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from chatbot.nodes import call_model
from chatbot.state import State


def build_graph() -> StateGraph:
    """Create a compiled graph with in-memory conversation persistence.

    Graph topology (for now)::

        START ──▶ call_model ──▶ END

    The ``InMemorySaver`` checkpointer keeps conversation history
    across invocations within the same process.  Swap it for
    ``PostgresSaver`` when you need production-grade persistence.
    """
    builder = StateGraph(State)

    builder.add_node("call_model", call_model)

    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)
