"""Pydantic models for configuration, tool schemas, and data structures.

Uses ``pydantic-settings`` so that all configuration is loaded from
environment variables / ``.env`` with automatic type coercion and
validation at startup.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Application Settings ───────────────────────────────────────
class AppSettings(BaseSettings):
    """Centralised, validated configuration for the entire application.

    Values are loaded automatically from environment variables and
    the ``.env`` file in the project root.  A ``ValidationError`` is
    raised at startup if any required field is missing or has an
    invalid value — no more silent ``KeyError`` buried deep in code.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # ignore env vars we don't define (e.g. PATH)
    )

    # ── LLM ─────────────────────────────────────────────────────
    llm_base_url: str = Field(
        description="Base URL of the OpenAI-compatible chat endpoint.",
    )
    llm_model: str = Field(
        description="Model name or identifier (e.g. 'openai/gpt-oss-20b').",
    )
    llm_api_key: str = Field(
        default="lm-studio",
        description="API key for the LLM provider.",
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0 = deterministic, 2 = max creativity).",
    )

    # ── Embeddings ──────────────────────────────────────────────
    embedding_base_url: str = Field(
        default="http://127.0.0.1:1234/v1",
        description="Base URL for the embedding model endpoint.",
    )
    embedding_model: str = Field(
        default="text-embedding-nomic-embed-text-v1.5@f32",
        description="Embedding model name.",
    )

    # ── Qdrant ──────────────────────────────────────────────────
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="URL of the Qdrant vector database.",
    )
    qdrant_collection: str = Field(
        default="rag_documents",
        description="Name of the Qdrant collection for RAG documents.",
    )

    # ── RAG retrieval pipeline ──────────────────────────────────
    rag_retrieval_k: int = Field(
        default=12,
        ge=1,
        description="Number of chunks retrieved from vector search before reranking.",
    )
    rag_score_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine-similarity score. Chunks below this are discarded "
            "before reranking. Note: dense embedding models like nomic-embed-text "
            "have a baseline similarity of ~0.55-0.60 even for unrelated texts, "
            "so values below 0.6 are rarely useful."
        ),
    )
    rag_rerank_top_n: int = Field(
        default=6,
        ge=1,
        description="Number of chunks kept after listwise reranking.",
    )
    rag_max_context_chunks: int = Field(
        default=4,
        ge=1,
        description="Maximum number of relevant chunks passed to the answer model.",
    )
    rag_verbose_chunk_logging: bool = Field(
        default=True,
        description="Whether to log per-chunk retrieval/rerank/filter decisions.",
    )
    flashrank_cache_dir: str = Field(
        default="models/flashrank",
        description=(
            "Directory containing the pre-downloaded FlashRank reranker model. "
            "Relative paths are resolved from the project root. "
            "Run the setup script once while online to populate this directory."
        ),
    )
    flashrank_model: str = Field(
        default="ms-marco-MultiBERT-L-12",
        description="FlashRank model name (must exist inside flashrank_cache_dir).",
    )

    # ── Langfuse (optional) ─────────────────────────────────────
    langfuse_public_key: str | None = Field(
        default=None,
        description="Langfuse public key (leave empty to disable tracing).",
    )
    langfuse_secret_key: str | None = Field(
        default=None,
        description="Langfuse secret key.",
    )
    langfuse_base_url: str = Field(
        default="http://localhost:3000",
        description="Langfuse server URL.",
    )

    # ── Chat ────────────────────────────────────────────────────
    chat_session_id: str | None = Field(
        default=None,
        description="Fixed session ID. If unset, a random one is generated each run.",
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the singleton ``AppSettings`` instance.

    Cached with ``lru_cache`` so the ``.env`` file is read only once.
    """
    return AppSettings()  # type: ignore[call-arg]


# ── Tool input schemas ─────────────────────────────────────────
class SearchDocumentsInput(BaseModel):
    """Validated input schema for the ``search_documents`` tool."""

    query: str = Field(
        min_length=1,
        max_length=1000,
        description=(
            "The search query to find relevant documents in the knowledge base. "
            "Be specific to get the best results."
        ),
    )


# ── Data models for Qdrant payloads ────────────────────────────
class ChunkMetadata(BaseModel):
    """Typed representation of the metadata stored with each chunk in Qdrant."""

    source: str = Field(description="Full file path of the original document.")
    page: int = Field(default=0, description="0-indexed page number.")
    start_index: int | None = Field(
        default=None,
        description="Character offset where the chunk starts in the original page.",
    )


class ChunkPayload(BaseModel):
    """Typed representation of a full Qdrant point payload."""

    page_content: str = Field(description="The text content of the chunk.")
    metadata: ChunkMetadata
