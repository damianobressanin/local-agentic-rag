"""Configuration module – exposes the LLM, embeddings, vector store, and Langfuse.

All settings are loaded via ``AppSettings`` (Pydantic BaseSettings)
which reads ``.env`` automatically and validates types at startup.
"""

import logging
import os
import urllib.request
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from chatbot.models import get_settings


def _ensure_langfuse_env() -> None:
    """Propagate Langfuse settings to ``os.environ``.

    The Langfuse SDK reads ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``
    and ``LANGFUSE_BASE_URL`` directly from the environment.  Since
    ``pydantic-settings`` loads ``.env`` into the model but does **not**
    populate ``os.environ``, we bridge the gap here.
    """
    s = get_settings()
    mapping = {
        "LANGFUSE_PUBLIC_KEY": s.langfuse_public_key,
        "LANGFUSE_SECRET_KEY": s.langfuse_secret_key,
        "LANGFUSE_BASE_URL": s.langfuse_base_url,
    }
    for key, value in mapping.items():
        if value and key not in os.environ:
            os.environ[key] = value


# ── LLM ────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance pointed at the configured endpoint.

    All settings come from ``AppSettings`` so switching from
    LMStudio to vLLM (or OpenAI cloud) is just an env change.
    """
    s = get_settings()
    return ChatOpenAI(
        base_url=s.llm_base_url,
        model=s.llm_model,
        api_key=s.llm_api_key,
        temperature=s.llm_temperature,
    )


# ── Embeddings ─────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Return an OpenAIEmbeddings instance pointed at the configured endpoint.

    Uses the same OpenAI-compatible pattern as the LLM so you can
    run the embedding model on LMStudio, vLLM, or OpenAI cloud.
    """
    s = get_settings()
    return OpenAIEmbeddings(
        base_url=s.embedding_base_url,
        model=s.embedding_model,
        api_key=s.llm_api_key,
        check_embedding_ctx_length=False,  # LMStudio expects plain strings, not token arrays
    )


# ── Vector store (Qdrant) ──────────────────────────────────────
@lru_cache(maxsize=1)
def get_vector_store():
    """Return a QdrantVectorStore connected to the local Qdrant instance."""
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    s = get_settings()
    client = QdrantClient(url=s.qdrant_url)

    return QdrantVectorStore(
        client=client,
        collection_name=s.qdrant_collection,
        embedding=get_embeddings(),
    )


# ── Langfuse (optional) ────────────────────────────────────────
# In SDK v3 the CallbackHandler takes NO constructor args.
# All config (keys, base_url) comes from env vars:
#   LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
# Session/user ids are passed via LangGraph config metadata.

def _langfuse_server_reachable(base_url: str, timeout: float = 2.0) -> bool:
    """Quick health check – try to connect to the Langfuse server."""
    try:
        url = base_url.rstrip("/") + "/api/public/health"
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def _silence_otel_errors() -> None:
    """Suppress noisy OpenTelemetry export warnings when Langfuse is down."""
    for name in (
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        "opentelemetry.sdk._shared_internal",
    ):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def get_langfuse_handler():
    """Return a Langfuse CallbackHandler if the package, keys, and server are available."""
    s = get_settings()
    if not s.langfuse_public_key or not s.langfuse_secret_key:
        return None

    if not _langfuse_server_reachable(s.langfuse_base_url):
        _silence_otel_errors()
        return None

    try:
        _ensure_langfuse_env()
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except Exception:
        return None


def get_langfuse_status() -> tuple[bool, str]:
    """Return (enabled, description) for startup diagnostics."""
    s = get_settings()
    if not s.langfuse_public_key or not s.langfuse_secret_key:
        return False, "missing LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY"

    try:
        from langfuse.langchain import CallbackHandler  # noqa: F401
    except ImportError:
        return False, "langfuse package not installed"

    if not _langfuse_server_reachable(s.langfuse_base_url):
        return False, f"server unreachable at {s.langfuse_base_url}"

    return True, f"base_url={s.langfuse_base_url}"


def shutdown_langfuse() -> None:
    """Flush pending Langfuse events before the process exits."""
    try:
        from langfuse import get_client
        get_client().shutdown()
    except Exception:
        pass
