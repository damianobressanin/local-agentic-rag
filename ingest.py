"""Manage documents in the Qdrant vector store.

Usage:
    python ingest.py ingest          Ingest new documents from rag_documents/ (skips duplicates)
    python ingest.py list            List all ingested documents
    python ingest.py delete <name>   Delete a single document by filename
    python ingest.py delete --all    Delete ALL documents (wipes the collection)
"""

import argparse
import os
import sys

# ── Make the src/ package importable ────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, Filter, FieldCondition, MatchValue, VectorParams

from chatbot.config import get_embeddings
from chatbot.models import ChunkPayload, get_settings

# Directory containing the documents to ingest
RAG_DOCS_DIR = os.path.join(os.path.dirname(__file__), "rag_documents")

# nomic-embed-text-v1.5 produces 768-dimensional vectors
EMBEDDING_DIMENSION = 768


def _get_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def _get_collection() -> str:
    return get_settings().qdrant_collection


def _ensure_collection(client: QdrantClient, collection: str) -> None:
    """Create the collection if it doesn't exist yet."""
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        print(f"Collection '{collection}' created ({EMBEDDING_DIMENSION}d, cosine)")


def _get_ingested_sources(client: QdrantClient, collection: str) -> set[str]:
    """Return the set of 'source' values already present in the collection."""
    sources: set[str] = set()
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            src = (point.payload or {}).get("metadata", {}).get("source", "")
            if src:
                sources.add(src)
        if offset is None:
            break
    return sources


def _source_to_name(source: str) -> str:
    """Extract a short display name from a full source path."""
    return os.path.basename(source)


RAG_SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "rag_summary.txt")


def _update_rag_summary(client: QdrantClient, collection: str) -> None:
    """Regenerate rag_summary.txt from the first chunk of each document.

    The entire first chunk is used so that the summary captures the real
    topic of the document, not just the first few characters.
    """
    if not client.collection_exists(collection):
        # Collection gone → remove stale summary
        if os.path.exists(RAG_SUMMARY_PATH):
            os.remove(RAG_SUMMARY_PATH)
            print("\nrag_summary.txt removed (collection empty).")
        return

    # Collect ALL chunks, grouped by source (validated via ChunkPayload)
    docs_by_source: dict[str, list[ChunkPayload]] = {}
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=collection, limit=100, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for point in results:
            try:
                chunk = ChunkPayload.model_validate(point.payload or {})
            except Exception:
                continue  # skip malformed payloads
            docs_by_source.setdefault(chunk.metadata.source, []).append(chunk)
        if offset is None:
            break

    if not docs_by_source:
        if os.path.exists(RAG_SUMMARY_PATH):
            os.remove(RAG_SUMMARY_PATH)
        print("\nrag_summary.txt removed (no documents).")
        return

    lines: list[str] = [
        "# RAG Knowledge Base Summary",
        f"# Generated automatically — {len(docs_by_source)} document(s)\n",
    ]

    for src in sorted(docs_by_source):
        name = _source_to_name(src)
        # Sort by page so we always pick the real first chunk
        chunks = sorted(docs_by_source[src], key=lambda c: c.metadata.page)
        first_chunk = chunks[0].page_content.strip()
        total_chunks = len(chunks)
        pages = sorted({c.metadata.page for c in chunks})
        page_range = f"pages {min(pages)+1}–{max(pages)+1}" if len(pages) > 1 else f"page {min(pages)+1}"

        lines.append(f"## {name}")
        lines.append(f"   Chunks: {total_chunks} | {page_range}")
        lines.append(f"   Content of first chunk (full):")
        lines.append(f"   {first_chunk}")
        lines.append("")

    with open(RAG_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nrag_summary.txt updated ({len(docs_by_source)} document(s))")


# ── INGEST ──────────────────────────────────────────────────────
def load_documents() -> list:
    """Load .txt and .pdf files from rag_documents/."""
    docs = []

    txt_loader = DirectoryLoader(
        RAG_DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True,
    )
    txt_docs = txt_loader.load()
    docs.extend(txt_docs)
    print(f"  Loaded {len(txt_docs)} .txt file(s)")

    pdf_loader = DirectoryLoader(
        RAG_DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True,
    )
    pdf_docs = pdf_loader.load()
    docs.extend(pdf_docs)
    print(f"  Loaded {len(pdf_docs)} .pdf page(s)")

    return docs


def cmd_ingest() -> None:
    print(f"Loading documents from: {RAG_DOCS_DIR}\n")

    # 1. Load
    docs = load_documents()
    if not docs:
        print("\nNo documents found. Add .txt or .pdf files to rag_documents/ and retry.")
        sys.exit(1)

    # 2. Check for duplicates
    client = _get_client()
    collection = _get_collection()
    _ensure_collection(client, collection)

    existing_sources = _get_ingested_sources(client, collection)
    new_docs = [d for d in docs if d.metadata.get("source", "") not in existing_sources]
    skipped = len(docs) - len(new_docs)

    if skipped:
        skipped_names = {_source_to_name(d.metadata["source"])
                         for d in docs if d.metadata.get("source", "") in existing_sources}
        print(f"\nSkipping {skipped} page(s) from already-ingested file(s): {', '.join(sorted(skipped_names))}")

    if not new_docs:
        print("Nothing new to ingest.")
        return

    # 3. Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True,
    )
    chunks = splitter.split_documents(new_docs)
    print(f"\nSplit into {len(chunks)} new chunk(s)")

    # 4. Embed & store
    s = get_settings()
    print(f"\nEmbedding model: {s.embedding_model}")
    print(f"Qdrant URL     : {s.qdrant_url}")
    print(f"Collection     : {collection}")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=get_embeddings(),
    )
    vector_store.add_documents(documents=chunks)

    print(f"\nDone! {len(chunks)} chunk(s) ingested. ({skipped} page(s) skipped as duplicates)")

    # Update the summary file for the agent's system prompt
    _update_rag_summary(client, collection)


# ── LIST ────────────────────────────────────────────────────────
def cmd_list() -> None:
    client = _get_client()
    collection = _get_collection()

    if not client.collection_exists(collection):
        print("Collection does not exist yet. Run 'python ingest.py ingest' first.")
        return

    sources = _get_ingested_sources(client, collection)
    if not sources:
        print("No documents in the collection.")
        return

    # Count chunks per source
    chunk_counts: dict[str, int] = {}
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=collection, limit=100, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for point in results:
            src = (point.payload or {}).get("metadata", {}).get("source", "unknown")
            chunk_counts[src] = chunk_counts.get(src, 0) + 1
        if offset is None:
            break

    total_chunks = sum(chunk_counts.values())
    print(f"\nCollection: {collection}  ({total_chunks} chunk(s) total)\n")
    print(f"  {'Document':<50} {'Chunks':>8}")
    print(f"  {'─' * 50} {'─' * 8}")
    for src in sorted(chunk_counts):
        print(f"  {_source_to_name(src):<50} {chunk_counts[src]:>8}")


# ── DELETE ──────────────────────────────────────────────────────
def cmd_delete(name: str | None, delete_all: bool) -> None:
    client = _get_client()
    collection = _get_collection()

    if not client.collection_exists(collection):
        print("Collection does not exist.")
        return

    if delete_all:
        info = client.get_collection(collection)
        count = info.points_count
        client.delete_collection(collection)
        print(f"Deleted collection '{collection}' ({count} chunk(s) removed).")
        _update_rag_summary(client, collection)
        return

    if not name:
        print("Specify a document name or use --all. See 'python ingest.py delete -h'.")
        return

    # Find the full source path(s) matching the given filename
    all_sources = _get_ingested_sources(client, collection)
    matching = [s for s in all_sources if _source_to_name(s) == name]

    if not matching:
        print(f"No document named '{name}' found in the collection.")
        print("Use 'python ingest.py list' to see ingested documents.")
        return

    for source in matching:
        client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=source),
                )]
            ),
        )
        print(f"Deleted all chunks from: {_source_to_name(source)}")

    _update_rag_summary(client, collection)
    print("Done.")


# ── CLI ─────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage documents in the Qdrant vector store.",
    )
    sub = parser.add_subparsers(dest="command")

    # ingest
    sub.add_parser("ingest", help="Ingest new documents (skips duplicates)")

    # list
    sub.add_parser("list", help="List all ingested documents")

    # delete
    del_parser = sub.add_parser("delete", help="Delete documents")
    del_parser.add_argument("name", nargs="?", default=None,
                            help="Filename to delete (e.g. 'report.pdf')")
    del_parser.add_argument("--all", action="store_true", dest="delete_all",
                            help="Delete ALL documents (wipes the collection)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)
    elif args.command == "ingest":
        cmd_ingest()
    elif args.command == "list":
        cmd_list()
    elif args.command == "delete":
        cmd_delete(args.name, args.delete_all)


if __name__ == "__main__":
    main()
