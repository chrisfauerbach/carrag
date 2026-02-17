# Carrag

Local RAG (Retrieval-Augmented Generation) system using Elasticsearch for vector storage and Ollama for embeddings + LLM generation. Optimized for automotive manuals and documents.

## Stack

- **FastAPI** app on port 8000
- **React** frontend (Vite + nginx) on port 3000
- **Elasticsearch 8.15.0** on port 9200 (security disabled, dense_vector/kNN)
- **Ollama** on port 11434 (nomic-embed-text for 768-dim embeddings, llama3.2 for generation)
- All runs via Docker Compose

## Commands

```bash
docker compose up --build -d   # Start all services (detached)
docker compose down             # Stop all services
docker compose down -v          # Stop and wipe volumes (ES data + Ollama models)
docker compose logs app         # View FastAPI logs
docker compose logs -f app      # Follow FastAPI logs
```

The app waits for ES + Ollama health checks before starting. On first boot it pulls both Ollama models (stream: false to ensure completion).

## Project Structure

```
app/
├── main.py                    # FastAPI app, lifespan (init ES index, pull models)
├── config.py                  # Pydantic Settings from env vars
├── api/routes/
│   ├── ingest.py              # POST /ingest/file, POST /ingest/url (auto-tags on ingest)
│   ├── query.py               # POST /query, POST /query/stream, GET /query/models
│   ├── documents.py           # GET/PATCH/DELETE /documents, similarity
│   └── chats.py               # CRUD for persistent chat sessions
├── services/
│   ├── embeddings.py          # Ollama /api/embed client
│   ├── elasticsearch.py       # Index management, bulk insert, hybrid search
│   ├── chunker.py             # Recursive text splitting with overlap
│   ├── rag.py                 # RAG orchestration + LLM auto-tag generation
│   ├── chat.py                # Chat session persistence in ES
│   ├── similarity.py          # Document similarity (centroid-based cosine)
│   └── parsers/
│       ├── pdf.py             # PyMuPDF extraction
│       ├── text.py            # .txt/.md reading
│       └── web.py             # trafilatura + BeautifulSoup fallback
└── models/
    └── schemas.py             # Pydantic request/response models
```

## API Endpoints

- `GET /health` — health check
- `POST /ingest/file` — upload PDF/text (multipart form, optional `tags` field)
- `POST /ingest/url` — ingest web page `{"url": "...", "tags": [...]}`
- `POST /query` — RAG query `{"question": "...", "top_k": 5, "tags": [...]}`
- `POST /query/stream` — SSE streaming RAG query (sources → tokens → done)
- `GET /query/models` — list available Ollama LLM models
- `GET /documents` — list ingested documents (with tags)
- `GET /documents/similarity` — pairwise document similarity graph
- `GET /documents/{id}` — document details
- `GET /documents/{id}/chunks` — browse document chunks
- `PATCH /documents/{id}/tags` — update document tags
- `DELETE /documents/{id}` — delete document and chunks
- `POST /chats` — create chat session
- `GET /chats` — list chats
- `GET /chats/{id}` — get chat with message history
- `PATCH /chats/{id}` — rename chat
- `DELETE /chats/{id}` — delete chat
- Swagger UI at http://localhost:8000/docs

## Key Config (env vars / .env)

| Variable | Default | Purpose |
|---|---|---|
| ES_URL | http://elasticsearch:9200 | Elasticsearch URL |
| OLLAMA_URL | http://ollama:11434 | Ollama URL |
| EMBEDDING_MODEL | nomic-embed-text | Embedding model (768-dim) |
| LLM_MODEL | llama3.2 | Generation model |
| CHUNK_SIZE | 2000 | Characters per chunk |
| CHUNK_OVERLAP | 200 | Overlap between chunks |
| ES_INDEX | carrag_chunks | Elasticsearch index name |

## Architecture Notes

- Singleton service instances (`es_service`, `embedding_service`, `chat_service`) created at module level, initialized during FastAPI lifespan
- Embeddings batched in groups of 32 during ingestion
- Chunker uses recursive splitting: paragraphs → sentences → words → hard character split
- ES index uses `dense_vector` with cosine similarity for kNN search
- Query uses hybrid search: BM25 + kNN fused via manual Reciprocal Rank Fusion (RRF)
- Tags field is `text` type (not keyword) to support partial matching (e.g. "ford" matches "ford lincoln manual")
- Auto-tagging at ingest: LLM generates up to 5 tags from first 8000 chars + filename; automotive-focused prompt prioritizes make/model/year; merged with user-supplied tags; failures are swallowed (never blocks ingestion)
- RAG prompt includes system message instructing the LLM to only use provided context
- Streaming support via SSE for real-time token delivery
- Chat sessions persisted in a separate ES index (`carrag_chats`)
- Ollama healthcheck uses `ollama list` (no curl in the image)
- Ollama model pulls use `stream: false` to block until download completes
