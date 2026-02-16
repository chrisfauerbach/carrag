import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.services.elasticsearch import es_service
from app.services.embeddings import embedding_service
from app.api.routes import ingest, query, documents, chats
from app.services.chat import chat_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting up — initializing services...")

    await es_service.init()
    await chat_service.init_index()
    await embedding_service.ensure_model()

    # Pull LLM model too
    await _pull_ollama_model(settings.llm_model)

    logger.info("Startup complete.")
    yield

    await es_service.close()
    await embedding_service.close()
    logger.info("Shutdown complete.")


async def _pull_ollama_model(model: str):
    """Pull an Ollama model if not already available."""
    import httpx

    async with httpx.AsyncClient(base_url=settings.ollama_url, timeout=600) as client:
        resp = await client.get("/api/tags")
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(model in m for m in models):
            logger.info(f"Pulling Ollama model: {model}")
            resp = await client.post("/api/pull", json={"name": model})
            resp.raise_for_status()
            logger.info(f"Model {model} pulled successfully.")
        else:
            logger.info(f"Model {model} already available.")


app = FastAPI(title="Carrag", description="Local RAG with Elasticsearch + Ollama", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(query.router, tags=["query"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chats.router, prefix="/chats", tags=["chats"])


@app.get("/health")
async def health():
    return {"status": "ok"}
