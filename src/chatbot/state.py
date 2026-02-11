"""Graph state definition for the chatbot."""

from langgraph.graph import MessagesState


class State(MessagesState):
    """Conversation state.

    Inherits a ``messages`` list with the ``add_messages`` reducer from
    ``MessagesState``, which automatically appends new messages instead
    of overwriting the list.

    Extend this class when you need extra fields (e.g. retrieved
    documents for RAG, tool results, etc.).
    """
