# Carrag

Local RAG (Retrieval-Augmented Generation) system using Elasticsearch for vector storage and Ollama for embeddings + LLM generation.

## Stack

- **FastAPI** app on port 8000
- **Elasticsearch 8.15.0** on port 9200 (security disabled, dense_vector/kNN)
- **Ollama** on port 11434 (nomic-embed-text for 768-dim embeddings, llama3.2 for generation)
- All runs via Docker Compose

## Commands

```bash
docker compose up --build -d   # Start all services (detached)
docker compose down             # Stop all services
docker compose logs app         # View FastAPI logs
docker compose logs -f app      # Follow FastAPI logs
```

The app waits for ES + Ollama health checks before starting. On first boot it pulls both Ollama models (~30s).

## Project Structure

```
app/
├── main.py                    # FastAPI app, lifespan (init ES index, pull models)
├── config.py                  # Pydantic Settings from env vars
├── api/routes/
│   ├── ingest.py              # POST /ingest/file, POST /ingest/url
│   ├── query.py               # POST /query
│   └── documents.py           # GET/DELETE /documents
├── services/
│   ├── embeddings.py          # Ollama /api/embed client
│   ├── elasticsearch.py       # Index management, bulk insert, kNN search
│   ├── chunker.py             # Recursive text splitting with overlap
│   ├── rag.py                 # RAG orchestration (embed → retrieve → generate)
│   └── parsers/
│       ├── pdf.py             # PyMuPDF extraction
│       ├── text.py            # .txt/.md reading
│       └── web.py             # trafilatura + BeautifulSoup fallback
└── models/
    └── schemas.py             # Pydantic request/response models
```

## API Endpoints

- `GET /health` — health check
- `POST /ingest/file` — upload PDF/text (multipart form)
- `POST /ingest/url` — ingest web page `{"url": "..."}`
- `POST /query` — RAG query `{"question": "...", "top_k": 5}`
- `GET /documents` — list ingested documents
- `GET /documents/{id}` — document details
- `DELETE /documents/{id}` — delete document and chunks
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

- Singleton service instances (`es_service`, `embedding_service`) created at module level, initialized during FastAPI lifespan
- Embeddings batched in groups of 32 during ingestion
- Chunker uses recursive splitting: paragraphs → sentences → words → hard character split
- ES index uses `dense_vector` with cosine similarity for kNN search
- RAG prompt includes system message instructing the LLM to only use provided context
- Ollama healthcheck uses `ollama list` (no curl in the image)
