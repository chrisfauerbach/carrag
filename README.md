# Carrag

Local RAG (Retrieval-Augmented Generation) system that runs entirely on your machine. Upload documents, ask questions, get answers grounded in your own data — no API keys, no cloud services.

Built with FastAPI, Elasticsearch, Ollama, and React.

## How It Works

```
Upload PDF / TXT / URL
        │
        ▼
  Parse ──▶ Chunk ──▶ Embed (Ollama) ──▶ Store (Elasticsearch)

Ask a question
        │
        ▼
  Embed query ──▶ kNN search ──▶ Build prompt with context ──▶ LLM generates answer
```

Documents are parsed, split into overlapping chunks, embedded as 768-dimensional vectors, and stored in Elasticsearch. At query time, your question is embedded with the same model, the most similar chunks are retrieved via kNN search, and a local LLM generates an answer grounded in those sources.

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- ~8 GB free RAM (Elasticsearch + Ollama models)
- NVIDIA GPU recommended for faster inference (works on CPU too)

### Run

```bash
git clone https://github.com/YOUR_USERNAME/carrag.git
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

- **Query tab** — Ask questions about your ingested documents. Shows the answer, model info, response time, and source citations with similarity scores.
- **Upload tab** — Drag-and-drop files (PDF, TXT, MD) or paste a URL to ingest web pages.
- **Documents tab** — View and delete ingested documents.

### API (port 8000)

```bash
# Upload a file
curl -X POST http://localhost:8000/ingest/file -F "file=@paper.pdf"

# Ingest a web page
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "top_k": 5}'

# List documents
curl http://localhost:8000/documents

# Delete a document
curl -X DELETE http://localhost:8000/documents/{document_id}
```

Full API documentation available at http://localhost:8000/docs (Swagger UI).

## Architecture

```
frontend/       React SPA (Vite + nginx)
  ├── nginx reverse-proxies /ingest, /query, /documents → FastAPI
  └── served on port 3000

app/             FastAPI backend
  ├── api/routes/        HTTP endpoints
  ├── services/
  │   ├── parsers/       PDF (PyMuPDF), text, web (trafilatura)
  │   ├── chunker.py     Recursive text splitting with overlap
  │   ├── embeddings.py  Ollama embedding client
  │   ├── elasticsearch.py  Index management, bulk insert, kNN search
  │   └── rag.py         Orchestration: embed → retrieve → generate
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

- **Recursive chunking** — Splits on paragraph breaks first, then sentences, then words, with configurable overlap (default 200 chars) to preserve context across chunk boundaries.
- **Batched embeddings** — Chunks are embedded in batches of 32 to balance throughput and memory.
- **kNN with cosine similarity** — Elasticsearch's HNSW index enables fast approximate nearest-neighbor search over dense vectors.
- **Grounded generation** — The LLM prompt includes a system message instructing it to answer only from the provided context and cite sources.

## Configuration

All settings can be overridden via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `ES_URL` | `http://elasticsearch:9200` | Elasticsearch URL |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model (768-dim) |
| `LLM_MODEL` | `llama3.2` | LLM for answer generation |
| `CHUNK_SIZE` | `2000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `ES_INDEX` | `carrag_chunks` | Elasticsearch index name |

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

## License

MIT
