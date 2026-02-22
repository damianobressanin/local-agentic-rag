"""LangChain tools available to the agent."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.tools import tool

from chatbot.config import get_vector_store
from chatbot.models import SearchDocumentsInput, get_settings

# Resolve project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ANSI colors for terminal output
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def _log(icon: str, msg: str) -> None:
    print(f"{_DIM}{icon}  {msg}{_RESET}")


def _doc_label(doc: Document) -> str:
    source_name = os.path.basename(str(doc.metadata.get("source", "unknown")))
    page = doc.metadata.get("page", None)
    page_str = "?"
    if page is not None:
        try:
            page_str = str(int(page) + 1)  # 0-indexed -> 1-indexed
        except (TypeError, ValueError):
            page_str = str(page)
    return f"{source_name} p.{page_str}"


def _one_indexed_page(page: object) -> int | None:
    if page is None:
        return None
    try:
        return int(page) + 1
    except (TypeError, ValueError):
        return None


def _rerank_documents(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """Rerank retrieved chunks using FlashRank (local cross-encoder).

    The model is loaded from a **local, pre-downloaded** directory
    (``models/flashrank/`` by default) so that no network access is
    ever required — not even on first boot.

    Falls back to retrieval order if FlashRank is not installed or
    the model files are missing.
    """
    if not docs:
        return []

    try:
        from flashrank import Ranker
        from langchain_community.document_compressors import FlashrankRerank

        s = get_settings()

        # Resolve cache_dir: support both absolute and project-relative paths
        cache_path = Path(s.flashrank_cache_dir)
        if not cache_path.is_absolute():
            cache_path = _PROJECT_ROOT / cache_path

        model_path = cache_path / s.flashrank_model
        if not model_path.is_dir():
            _log(
                "⚠️",
                f"{_YELLOW}FlashRank model not found at {model_path}. "
                f"Run the setup script to download it. Using retrieval-order fallback.{_RESET}",
            )
            return docs[:top_n]

        ranker = Ranker(
            model_name=s.flashrank_model,
            cache_dir=str(cache_path),
        )
        reranker = FlashrankRerank(client=ranker, top_n=top_n)
        reranked = reranker.compress_documents(documents=docs, query=query)
        _log("🧭", f"FlashRank reranked {_GREEN}{len(docs)}{_RESET}{_DIM} → {_GREEN}{len(reranked)}{_RESET}{_DIM} chunk(s)")
        return list(reranked)
    except ImportError:
        _log("⚠️", f"{_YELLOW}flashrank not installed — using retrieval order{_RESET}")
        return docs[:top_n]
    except Exception as exc:
        _log("⚠️", f"{_YELLOW}Reranking failed ({exc}). Using retrieval-order fallback.{_RESET}")
        return docs[:top_n]


@tool(args_schema=SearchDocumentsInput)
def search_documents(query: str) -> str:
    """Search the local knowledge base ONCE for relevant documents.

    Call this tool a SINGLE time per user question. Do NOT call it
    again with a rephrased query — use the results from the first
    call to answer.  Results include [Source: filename, Page N]
    headers that you MUST cite in your response.
    """
    s = get_settings()

    _log("🔍", f"Searching Qdrant for: {_GREEN}{query!r}{_RESET}")
    results_with_scores = get_vector_store().similarity_search_with_score(
        query, k=s.rag_retrieval_k,
    )

    if not results_with_scores:
        _log("⚠️", f"{_YELLOW}No relevant documents found.{_RESET}")
        return "No relevant documents found."

    _log("📄", f"Retrieved {_GREEN}{len(results_with_scores)}{_RESET}{_DIM} chunk(s) (k={s.rag_retrieval_k})")

    # ── Score-threshold filter ──────────────────────────────────
    above: list[Document] = []
    below_count = 0
    for rank, (doc, score) in enumerate(results_with_scores, 1):
        if s.rag_verbose_chunk_logging:
            preview = doc.page_content[:80].replace("\n", " ")
            tag = f"{_GREEN}✓ PASS{_RESET}{_DIM}" if score >= s.rag_score_threshold else f"{_YELLOW}✗ SKIP{_RESET}{_DIM}"
            _log("  ", f"  [R{rank}] {_doc_label(doc)} (sim={score:.4f}) {tag} — \"{preview}...\"")
        if score >= s.rag_score_threshold:
            doc.metadata["retrieval_score"] = score
            above.append(doc)
        else:
            below_count += 1

    if not above:
        _log(
            "⚠️",
            f"{_YELLOW}No chunks above score threshold ({s.rag_score_threshold}). "
            f"All {len(results_with_scores)} retrieved chunk(s) were below threshold.{_RESET}",
        )
        return (
            "No relevant documents found in the knowledge base "
            f"(all results below relevance threshold of {s.rag_score_threshold})."
        )

    if below_count:
        _log("🗑️", f"Filtered out {_YELLOW}{below_count}{_RESET}{_DIM} chunk(s) below threshold {s.rag_score_threshold}")

    # Rerank with FlashRank, then take the top results
    rerank_top_n = min(s.rag_rerank_top_n, len(above))
    reranked_docs = _rerank_documents(query, above, top_n=rerank_top_n)

    # Cap to max context chunks
    passed_docs = reranked_docs[: s.rag_max_context_chunks]

    _log(
        "🧮",
        f"RAG pipeline: retrieved={len(results_with_scores)}, "
        f"above_threshold={len(above)} (sim≥{s.rag_score_threshold}), "
        f"reranked={len(reranked_docs)}, passed={len(passed_docs)}",
    )

    if s.rag_verbose_chunk_logging:
        for rank, doc in enumerate(passed_docs, 1):
            rerank_score = doc.metadata.get("relevance_score", "n/a")
            sim_score = doc.metadata.get("retrieval_score", "n/a")
            rerank_fmt = f"{rerank_score:.4f}" if isinstance(rerank_score, float) else rerank_score
            sim_fmt = f"{sim_score:.4f}" if isinstance(sim_score, float) else sim_score
            _log("✅", f"  [{rank}] {_doc_label(doc)} (sim={sim_fmt}, rerank={rerank_fmt})")

    # Build response with source metadata so the LLM can cite documents
    parts: list[str] = []
    for doc in passed_docs:
        source_name = os.path.basename(str(doc.metadata.get("source", "unknown")))
        page = _one_indexed_page(doc.metadata.get("page", None))
        header = f"[Source: {source_name}"
        if page is not None:
            header += f", Page {page}"
        header += "]"
        parts.append(f"{header}\n{doc.page_content}")

    return (
        "[SEARCH COMPLETE — Do NOT search again. Use these results to answer now.]\n\n"
        + "\n\n---\n\n".join(parts)
    )


@tool
def list_tools() -> str:
    """List all tools available to the agent with their descriptions.

    Use this tool when the user asks what you can do or what tools
    you have access to.
    """
    descriptions = []
    for t in ALL_TOOLS:
        descriptions.append(f"- **{t.name}**: {t.description}")
    return "\n".join(descriptions)


# ── Master list of all tools (imported by nodes.py) ────────────
ALL_TOOLS = [search_documents, list_tools]
