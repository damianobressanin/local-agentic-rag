"""Configuration module - loads env vars and exposes the LLM instance."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # reads .env from project root


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance pointed at the configured endpoint.

    All settings come from environment variables so switching from
    LMStudio to vLLM (or OpenAI cloud) is just an env change.
    """
    return ChatOpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
    )
