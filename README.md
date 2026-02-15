# 🔒 Local Agentic RAG

> A local-first, privacy-first Retrieval-Augmented Generation system with an autonomous AI agent that decides **when and what** to retrieve — powered by LangGraph, Qdrant, and an OpenAI-compatible chat endpoint that supports tool calling.


[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0-green.svg)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20DB-red.svg)](https://qdrant.tech/)
[![Status](https://img.shields.io/badge/status-WIP-orange.svg)]()

---

## What Is This?

Most RAG systems blindly retrieve documents on every query. This one doesn't.

**Local Agentic RAG** is a conversational AI assistant where an LLM agent _autonomously decides_ whether to search a local knowledge base or answer from its own knowledge. The retrieval pipeline uses a **4-stage filtering process** (vector search → score threshold → cross-encoder reranking → context cap) to maximize relevance and minimize noise.

Everything can run on your machine, and **no data leaves localhost**, as long as the chat model, embedding model, and Langfuse all point to local endpoints. The architecture is OpenAI API–compatible: the chat model and embeddings are instantiated via LangChain’s ChatOpenAI / OpenAIEmbeddings, so switching between a local server (e.g., LM Studio) and OpenAI Cloud is typically a .env change. For agentic RAG, the chat endpoint must implement OpenAI-style tool calling. This is not a universal multi-provider abstraction (Anthropic/Gemini/Bedrock would require different LangChain clients/adapters).


### Why I Built This

This is a personal learning project built to **deepen and consolidate my knowledge** of agentic RAG systems, LangGraph, vector databases, local LLM inference, and the surrounding ecosystem. It's a concrete demonstration of my interest and passion for AI/LLM engineering.

A secondary goal was to **stress-test my new hardware** with real local inference workloads (see [Development Environment](#development-environment)). The test went very well — the setup handles a good LLM model with tool-calling and real-time RAG retrieval smoothly, and I will use this machine for more ambitious projects going forward.

### Key Features

- **Agentic retrieval** — the LLM calls `search_documents` only when relevant, via LangGraph tool-calling
- **4-stage RAG pipeline** — vector search (Qdrant) → cosine threshold filter → FlashRank cross-encoder reranking → context cap
- **Can run 100% local** — LLM (LM Studio / gpt-oss-20b), embeddings (nomic-embed-text-v1.5), Qdrant, FlashRank — all on-device (tested in this configuration)
- **OpenAI-compatible backend** — developed and tested with LM Studio. You can point the LLM to OpenAI Cloud by changing LLM_BASE_URL, LLM_MODEL, and LLM_API_KEY in .env. Other OpenAI-compatible servers (e.g., vLLM) should work if they support OpenAI-style tool calling (required for agentic retrieval)
- **Domain-agnostic** — the system supports unrelated documents. During ingestion, a `rag_summary.txt` is auto-generated from a first page chunk of each document, giving the agent an overview of what the knowledge base contains. This lets the LLM make informed decisions about _when_ to search (when relevant data is likely to exist) and avoids wasteful retrievals on topics not covered by the indexed documents
- **Observable** — optional Langfuse integration for tracing every agent decision and retrieval step
- **Document management CLI** — ingest, list, and delete documents with deduplication by source path and auto-generated knowledge base summaries

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (CLI REPL)                      │
│  System prompt + RAG summary → first turn only                  │
│  User input → graph.invoke() → print AI response                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Agent Loop                         │
│                                                                 │
│  ┌───────────┐    tool_calls?    ┌────────┐                     │
│  │call_model │───────YES────────▶│ tools  │──┐                  │
│  │  (LLM)    │                   │ (exec) │  │                  │
│  └─────▲─────┘                   └────────┘  │                  │
│        │              ┌──────────────────────┘                  │
│        └──────────────┘  (results fed back)                     │
│        │                                                        │
│        NO tool_calls                                            │
│        │                                                        │
│        ▼                                                        │
│      [END] → return response                                    │
│                                                                 │
│  State: MessagesState (append-only) + InMemorySaver checkpoint  │
└─────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          search_documents           list_tools
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  4-Stage Retrieval Pipeline                     │
│                                                                 │
│  Query ──▶ Embedding (nomic-embed-text-v1.5, 768d)              │
│        ──▶ Qdrant vector search (k=12, cosine similarity)       │
│        ──▶ Score threshold filter (≥ 0.65)                      │
│        ──▶ FlashRank cross-encoder reranking (top 6)            │
│        ──▶ Context cap (max 4 chunks to LLM)                    │
│                                                                 │
│  Each chunk returned with [Source: filename, Page N] for citing │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Role |
|---|---|---|
| Agent orchestration | **LangGraph** (StateGraph + ToolNode) | Reactive loop with conditional tool-calling |
| LLM | **gpt-oss-20b** via LM Studio (OpenAI-compatible API) | Chat completions with tool use. This is a Mixture-of-Experts (MoE) model that activates fewer parameters during inference, fitting in 16 GB VRAM (the model has 21B total parameters with only 3.6B active at a time). Replaceable with any **OpenAI-compatible chat endpoint that supports tool calling** |
| Embeddings | **nomic-embed-text-v1.5** (768d) | Dense vector representations |
| Vector database | **Qdrant** (local, persistent storage) | Similarity search with cosine distance |
| Reranking | **FlashRank** (CPU cross-encoder) | Reranking to improve precision |
| Configuration | **Pydantic Settings** | Validated, type-safe `.env` loading |
| Observability | **Langfuse** (optional, self-hosted) | Tracing agent decisions, latencies, token usage |
| Document loading | LangChain `PyPDFLoader` + `TextLoader` | PDF and plain text ingestion |
| Text splitting | `RecursiveCharacterTextSplitter` | 1000-char chunks with 200-char overlap |

---

## Project Structure

```
local-agentic-rag/
├── main.py                  # CLI entry point (REPL chat loop)
├── ingest.py                # Document management CLI (ingest/list/delete)
├── rag_summary.txt          # Auto-generated KB summary (used in system prompt)
├── requirements.txt         # Python dependencies
├── .env.example             # Configuration template
├── rag_documents/           # Drop your PDFs and TXT files here (create if missing)
├── qdrant_storage/          # Qdrant persistent data (auto-generated)
└── src/chatbot/
    ├── config.py            # Factory singletons (LLM, embeddings, vector store, Langfuse)
    ├── graph.py             # LangGraph topology (2 nodes, conditional loop)
    ├── models.py            # Pydantic models (AppSettings, tool schemas, data models)
    ├── nodes.py             # Graph nodes (call_model with error handling)
    ├── state.py             # Conversation state (MessagesState + append reducer)
    └── tools.py             # Agent tools (search_documents, list_tools)
```

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- **Qdrant** running locally (default: `http://localhost:6333`)
- **LM Studio** (or any OpenAI-compatible server) with:
  - A chat model loaded (e.g., `openai/gpt-oss-20b`)
  - An embedding model loaded (e.g., `nomic-embed-text-v1.5`)
- **Langfuse** _(optional)_ — self-hosted instance for observability and tracing (see [langfuse.com](https://langfuse.com))

### 1. Clone and install

```bash
git clone https://github.com/damianobressanin/local-agentic-rag.git
cd local-agentic-rag

# Create a virtual environment (conda or venv)
conda create -n local-agentic-rag-env python=3.12 -y
conda activate local-agentic-rag-env

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your LLM server URL and model name
```

### 3. Start Qdrant

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 4. Ingest documents

```bash
# Place your PDF/TXT files in rag_documents/ (create if missing)
python ingest.py ingest    # Embed and store (skips duplicates by source path; if a file changes, delete and re-ingest)
python ingest.py list      # Verify what's indexed
```

### 5. Chat

```bash
python main.py
```

```
Model   : openai/gpt-oss-20b
Server  : http://127.0.0.1:1234/v1
Langfuse: ON (base_url=http://localhost:3000)
RAG     : summary loaded
Session : cli-a1b2c3d4
Type 'exit' or 'quit' to end the conversation.

You: What does the NASA handbook say about systems engineering?
🤖  Thinking...
🔧  Calling tool: search_documents(query='NASA systems engineering handbook')
🔍  Searching Qdrant for: 'NASA systems engineering handbook'
📄  Retrieved 12 chunk(s) (k=12)
🗑️  Filtered out 3 chunk(s) below threshold 0.65
🧭  FlashRank reranked 9 → 6 chunk(s)
🧮  RAG pipeline: retrieved=12, above_threshold=9 (sim≥0.65), reranked=6, passed=4
💬  Responding directly (no tool call)
AI: According to the NASA Systems Engineering Handbook...
```

### Switching to a cloud provider

To switch the **LLM only** (keep local embeddings + existing Qdrant data), just update `.env`:

> **Note:** Your LLM endpoint must support OpenAI-style tool calling.


```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=your-model-here
LLM_API_KEY=sk-your-key-here
```

To switch **both LLM and embeddings** to a cloud provider:

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=your-model-here
LLM_API_KEY=sk-your-key-here

EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
```

> **⚠️ Re-ingestion required when changing embedding model.** If you switch to a different embedding model, you **must** delete the existing collection and re-ingest all documents, because the old vectors were produced by a different model and live in a different vector space. Also note that `EMBEDDING_DIMENSION` is currently hardcoded to `768` in `ingest.py` — if the new model uses a different dimensionality (e.g., OpenAI's `text-embedding-3-small` uses 1536d), you'll need to update that constant before re-ingesting:
>
> ```bash
> python ingest.py delete --all   # wipe old vectors
> # update EMBEDDING_DIMENSION in ingest.py if needed
> python ingest.py ingest          # re-embed with the new model
> ```

> **Note on embeddings compatibility:** `check_embedding_ctx_length=False` ([config.py](src/chatbot/config.py)) disables client-side token-length checking/auto-splitting; chunks are small and inputs are sent as **strings** for maximum OpenAI-compatible endpoint compatibility.

---

## Configuration Reference

All settings are loaded from `.env` and validated at startup via Pydantic Settings (`AppSettings` in `models.py`). See [.env.example](.env.example) for the full template.

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_BASE_URL` | **yes** | — | OpenAI-compatible chat endpoint |
| `LLM_MODEL` | **yes** | — | Model identifier (e.g. `openai/gpt-oss-20b`) |
| `LLM_API_KEY` | no | `lm-studio` | API key for the LLM provider |
| `LLM_TEMPERATURE` | no | `0.7` | Sampling temperature (0–2) |
| `EMBEDDING_BASE_URL` | no | `http://127.0.0.1:1234/v1` | Embedding model endpoint |
| `EMBEDDING_MODEL` | no | `text-embedding-nomic-embed-text-v1.5@f32` | Embedding model name |
| `QDRANT_URL` | no | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION` | no | `rag_documents` | Qdrant collection name |
| `RAG_RETRIEVAL_K` | no | `12` | Chunks retrieved from vector search before filtering |
| `RAG_SCORE_THRESHOLD` | no | `0.65` | Minimum cosine similarity to keep a chunk |
| `RAG_RERANK_TOP_N` | no | `6` | Chunks kept after cross-encoder reranking |
| `RAG_MAX_CONTEXT_CHUNKS` | no | `4` | Max chunks passed to the LLM |
| `RAG_VERBOSE_CHUNK_LOGGING` | no | `true` | Log per-chunk retrieval/rerank decisions |
| `LANGFUSE_PUBLIC_KEY` | no | — | Langfuse public key (omit to disable tracing) |
| `LANGFUSE_SECRET_KEY` | no | — | Langfuse secret key |
| `LANGFUSE_BASE_URL` | no | `http://localhost:3000` | Langfuse server URL |
| `CHAT_SESSION_ID` | no | _(random)_ | Fixed session ID (useful for Langfuse grouping) |

**Note:** LLM_API_KEY is currently reused for the embeddings client as well (see get_embeddings() in src/chatbot/config.py). If your embedding endpoint requires a different key, you’ll need a small code change (e.g., add EMBEDDING_API_KEY).

---

## How the RAG Pipeline Works

1. **Vector search** — the query is embedded with nomic-embed-text-v1.5 and matched against Qdrant (top-k=12 by cosine similarity)
2. **Score threshold** — chunks below 0.65 cosine similarity are discarded (0.65 is a heuristic that worked well for the default embedding model/corpus; tune as needed)
3. **Cross-encoder reranking** — FlashRank (a lightweight, CPU-only cross-encoder) re-scores the surviving chunks and keeps the top 6
4. **Context cap** — at most 4 chunks are passed to the LLM, each tagged with `[Source: filename, Page N]` for citation

The agent sees a summary of all indexed documents in its system prompt, so it can make an informed decision about _when_ to search.

---

## Development Environment

This project was developed and tested on:

| Component | Spec |
|---|---|
| OS | Pop!_OS 22.04 LTS |
| GPU | NVIDIA RTX 5080 16 GB VRAM |
| CPU | AMD Ryzen 9 9950X3D |
| RAM | Corsair 96 GB DDR5  |
| LLM Server | LM Studio (local inference) |
| IDE | VSCode |

Running an LLM model locally with tool-calling and real-time RAG retrieval.

---

## Roadmap

This is a work-in-progress demo. Planned improvements:

- [ ] **Evaluation pipeline (RAGAS)** — automated RAG quality metrics with RAGAS
- [ ] **LLM evaluation** — test different models and use LLM-as-a-Judge (via Langfuse) to compare quality
- [ ] **Prompt engineering** — experiment with different system prompts and retrieval instructions
- [ ] **Improved RAG summary generation** — better heuristics for building `rag_summary.txt` (e.g. LLM-generated summaries instead of first-chunk extraction)
- [ ] **vLLM** — test vLLM as an alternative local inference server
- [ ] **MCP server** — expose tools via a Model Context Protocol server
- [ ] **Tests** — add unit and integration tests for the Python codebase
- [ ] **Docker Compose deployment** — one-command setup
- [ ] **RAG variants** — experiment with alternative retrieval strategies and different RAG settings (query decomposition, hybrid search, HyDE, etc.)
- [ ] **More document formats** — DOCX, HTML
- [ ] **UI** — a simple frontend as an alternative to the CLI

---

## Acknowledgments

Code development was AI-accelerated with **Claude Opus 4.6** and **GPT-5.3 Codex**. The project is based on known RAG approaches; I was responsible for component selection, integration choices, tradeoffs, manual validation, and final implementation quality.



---


**Author:** [Damiano Bressanin](https://github.com/damianobressanin) · [LinkedIn](https://www.linkedin.com/in/damianobressanin)