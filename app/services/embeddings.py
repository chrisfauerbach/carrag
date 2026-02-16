import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=settings.ollama_url,
                timeout=120,
            )
        return self._client

    async def ensure_model(self):
        """Pull the embedding model if it isn't already available."""
        resp = await self.client.get("/api/tags")
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        model = settings.embedding_model
        if not any(model in m for m in models):
            logger.info(f"Pulling embedding model: {model}")
            resp = await self.client.post("/api/pull", json={"name": model}, timeout=600)
            resp.raise_for_status()
            logger.info(f"Model {model} pulled.")
        else:
            logger.info(f"Embedding model {model} already available.")

    async def embed(self, texts: list[str], prefix: str = "") -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Uses the Ollama /api/embed endpoint with batch support.
        The prefix param supports nomic-embed-text task prefixes
        ('search_document: ' for indexing, 'search_query: ' for queries).
        Returns a list of embedding vectors.
        """
        prefixed = [prefix + t for t in texts] if prefix else texts
        resp = await self.client.post(
            "/api/embed",
            json={"model": settings.embedding_model, "input": prefixed},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    async def embed_single(self, text: str, prefix: str = "") -> list[float]:
        """Generate an embedding for a single text."""
        embeddings = await self.embed([text], prefix=prefix)
        return embeddings[0]

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


embedding_service = EmbeddingService()
