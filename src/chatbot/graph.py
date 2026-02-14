"""Build and compile the LangGraph conversation graph."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from chatbot.nodes import call_model
from chatbot.state import State
from chatbot.tools import ALL_TOOLS


def build_graph() -> CompiledStateGraph:
    """Create a compiled graph with tool-calling and in-memory persistence.

    Graph topology::

        START ──▶ call_model ──▶ [tools_condition] ──▶ tool_calls? ──▶ tools ──▶ call_model (loop)
                                                   └──▶ no tool_calls ──▶ END

    The LLM decides whether to call a tool (e.g. search_documents)
    or answer directly.  ``ToolNode`` executes tool calls automatically
    and ``tools_condition`` routes based on whether the last AI message
    contains tool_calls.
    """
    builder = StateGraph(State)

    # Nodes
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(ALL_TOOLS))

    # Edges
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)
