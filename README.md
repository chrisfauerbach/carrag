# Carrag

Local RAG (Retrieval-Augmented Generation) system that runs entirely on your machine. Upload documents, ask questions, get answers grounded in your own data — no API keys, no cloud services.

Built with FastAPI, Elasticsearch, Ollama, and React.

## How It Works

```
Upload PDF / TXT / URL
        │
        ▼
  Parse ──▶ Auto-tag (LLM) ──▶ Chunk ──▶ Embed (Ollama) ──▶ Store (Elasticsearch)

Ask a question
        │
        ▼
  Embed query ──▶ Hybrid search (BM25 + kNN) ──▶ RRF fusion ──▶ Rerank ──▶ Expand context ──▶ LLM
```

Documents are parsed, automatically tagged by the LLM, split into small overlapping chunks (500 chars), embedded as 768-dimensional vectors, and stored in Elasticsearch. At query time, your question is embedded with the same model, a large pool of candidate chunks is retrieved via hybrid search (keyword + semantic), fused with Reciprocal Rank Fusion, then a cross-encoder reranker scores each candidate against your exact question and picks the best matches. Those top chunks are expanded with their neighbors to give the LLM enough surrounding text, and the LLM generates an answer grounded in those sources.

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- ~8 GB free RAM (Elasticsearch + Ollama models)
- NVIDIA GPU recommended for faster inference (works on CPU too)

### Run

```bash
git clone https://github.com/chrisfauerbach/carrag.git
cd carrag
cp .env.example .env    # optional — defaults work out of the box
docker compose up --build -d
```

On first boot, Ollama pulls the embedding and LLM models (~2 GB). This takes about 30 seconds on a decent connection.

Once healthy, open **http://localhost:3000** for the web UI or **http://localhost:8000/docs** for the Swagger API docs.

### Stop

```bash
docker compose down
```

Data persists in Docker volumes (`es_data`, `ollama_models`) across restarts.

## Usage

### Web UI (port 3000)

- **Query tab** — Chat-style interface for asking questions about your ingested documents. Supports multi-turn conversation history, streaming responses, model selection, tag filtering, and shows source citations with similarity scores.
- **Upload tab** — Drag-and-drop files (PDF, TXT, MD) or paste a URL to ingest web pages. Documents are automatically tagged by the LLM (make, model, year for automotive docs). You can also add manual tags.
- **Documents tab** — View, tag, and delete ingested documents. Click any document row to browse its stored chunks. Includes a document similarity visualization.

### API (port 8000)

```bash
# Upload a file (auto-tagged by LLM + optional manual tags)
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@manual.pdf" -F "tags=ford, f-150"

# Ingest a web page
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article", "tags": ["research"]}'

# Ask a question (optionally filter by tags)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "top_k": 5, "tags": ["ford"]}'

# List documents (includes auto-generated tags)
curl http://localhost:8000/documents

# View a document's chunks
curl http://localhost:8000/documents/{document_id}/chunks

# Update tags on a document
curl -X PATCH http://localhost:8000/documents/{document_id}/tags \
  -H "Content-Type: application/json" \
  -d '{"tags": ["ford", "f-150", "2019"]}'

# Delete a document
curl -X DELETE http://localhost:8000/documents/{document_id}
```

Full API documentation available at http://localhost:8000/docs (Swagger UI).

## Architecture

```
frontend/       React SPA (Vite + nginx)
  ├── nginx reverse-proxies /ingest, /query, /documents, /chats → FastAPI
  └── served on port 3000

app/             FastAPI backend
  ├── api/routes/        HTTP endpoints (ingest, query, documents, chats)
  ├── services/
  │   ├── parsers/       PDF (PyMuPDF), text, web (trafilatura)
  │   ├── chunker.py     Recursive text splitting with overlap
  │   ├── embeddings.py  Ollama embedding client
  │   ├── elasticsearch.py  Index management, bulk insert, hybrid search
  │   ├── rag.py         RAG orchestration + LLM auto-tag generation
  │   ├── reranker.py    Flashrank cross-encoder reranking
  │   ├── chat.py        Chat session persistence in ES
  │   └── similarity.py  Document similarity (centroid cosine)
  └── models/schemas.py  Pydantic request/response models
```

### Stack

| Service | Port | Purpose |
|---------|------|---------|
| **Frontend** (nginx) | 3000 | React SPA + API reverse proxy |
| **FastAPI** | 8000 | REST API, RAG orchestration |
| **Elasticsearch** 8.15.0 | 9200 | Vector storage, kNN search (cosine similarity) |
| **Ollama** | 11434 | `nomic-embed-text` for 768-dim embeddings, `llama3.2` for generation |

### Key Design Decisions

- **Auto-tagging** — At ingest, the LLM reads the first 8000 chars + filename and generates up to 5 descriptive tags (automotive-focused: make, model, year). Auto-tags merge with user-supplied tags. Failures are swallowed — tagging never blocks ingestion.
- **Partial tag matching** — Tags are stored as `text` (not `keyword`) in ES, so searching for tag "ford" matches documents tagged "ford lincoln manual".
- **Hybrid search + RRF + reranking** — Queries over-retrieve candidates (3x the final count) via concurrent BM25 + kNN search, fuse with RRF, then a cross-encoder reranker (flashrank) scores each candidate against the exact question to pick the best matches. This two-stage approach — fast retrieval followed by precise reranking — gives much better results than retrieval alone.
- **Small chunks + context expansion** — Documents are split into small 500-char chunks so each embedding represents a focused concept. After reranking picks the best chunks, their immediate neighbors (chunk before + chunk after) are fetched and merged back in, giving the LLM ~1500 chars of context per match — precise retrieval with sufficient surrounding text.
- **Recursive chunking** — Splits on paragraph breaks first, then sentences, then words, with configurable overlap (default 100 chars) to preserve context across chunk boundaries.
- **Batched embeddings** — Chunks are embedded in batches of 32 to balance throughput and memory.
- **Streaming responses** — SSE streaming delivers tokens in real-time as the LLM generates answers.
- **Persistent chat sessions** — Conversations are stored in a dedicated ES index, supporting multi-turn history.
- **Grounded generation** — The LLM prompt includes a system message instructing it to answer only from the provided context and cite sources.

## Configuration

All settings can be overridden via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `ES_URL` | `http://elasticsearch:9200` | Elasticsearch URL |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model (768-dim) |
| `LLM_MODEL` | `llama3.2` | LLM for answer generation |
| `CHUNK_SIZE` | `500` | Max characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| `ES_INDEX` | `carrag_chunks` | Elasticsearch index name |
| `RERANK_ENABLED` | `true` | Enable cross-encoder reranking |
| `RERANK_MODEL` | `ms-marco-MiniLM-L-12-v2` | Flashrank reranker model |
| `RETRIEVAL_K_MULTIPLIER` | `3` | Over-retrieval factor (retrieves top_k * multiplier candidates for reranking) |
| `CONTEXT_EXPANSION_ENABLED` | `true` | Fetch neighboring chunks after reranking |

## Development

### Frontend (hot reload)

```bash
cd frontend
npm install
npm run dev    # Vite dev server on :5173, proxies API to :8000
```

### Backend

The backend runs inside Docker. View logs with:

```bash
docker compose logs -f app
```

### Tests

```bash
python3 -m pytest -v
```

The test suite covers all API endpoints, services (Elasticsearch, embeddings, RAG), parsers (PDF, text, web), the chunker, and Pydantic schemas. Tests use mocked Elasticsearch and Ollama clients — no running services required.

## License

MIT
