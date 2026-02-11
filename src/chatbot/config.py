"""Configuration module – loads env vars and exposes the LLM and Langfuse."""

import logging
import os
import urllib.request

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # reads .env from project root


# ── LLM ────────────────────────────────────────────────────────
def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance pointed at the configured endpoint.

    All settings come from environment variables so switching from
    LMStudio to vLLM (or OpenAI cloud) is just an env change.
    """
    return ChatOpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
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
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None

    base_url = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
    if not _langfuse_server_reachable(base_url):
        _silence_otel_errors()
        return None

    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except Exception:
        return None


def get_langfuse_status() -> tuple[bool, str]:
    """Return (enabled, description) for startup diagnostics."""
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return False, "missing LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY"

    try:
        from langfuse.langchain import CallbackHandler  # noqa: F401
    except ImportError:
        return False, "langfuse package not installed"

    base_url = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
    if not _langfuse_server_reachable(base_url):
        return False, f"server unreachable at {base_url}"

    return True, f"base_url={base_url}"


def shutdown_langfuse() -> None:
    """Flush pending Langfuse events before the process exits."""
    try:
        from langfuse import get_client
        get_client().shutdown()
    except Exception:
        pass
