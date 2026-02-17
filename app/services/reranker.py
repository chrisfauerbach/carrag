import logging

from app.config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self):
        self._ranker = None
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._ranker is not None

    def init(self):
        """Load the flashrank model. Synchronous — call at startup."""
        if not settings.rerank_enabled:
            logger.info("Reranker disabled by config.")
            return

        try:
            from flashrank import Ranker

            self._ranker = Ranker(
                model_name=settings.rerank_model,
                cache_dir="/app/data/flashrank_cache",
            )
            self._enabled = True
            logger.info(f"Loading reranker model: {settings.rerank_model}")
        except Exception:
            logger.warning("Failed to load reranker model", exc_info=True)
            self._enabled = False

    def rerank(self, query: str, passages: list[dict], top_k: int) -> list[dict]:
        """Rerank passages by relevance to query. Returns top_k results.

        Each passage dict must have a 'content' key.
        Falls back to passages[:top_k] if disabled or on error.
        """
        if not self.enabled or not passages:
            return passages[:top_k]

        try:
            from flashrank import RerankRequest

            rerank_input = [
                {"id": i, "text": p["content"], "meta": p}
                for i, p in enumerate(passages)
            ]

            request = RerankRequest(query=query, passages=rerank_input)
            results = self._ranker.rerank(request)

            reranked = []
            for r in results[:top_k]:
                passage = r["meta"]
                passage["rerank_score"] = r["score"]
                reranked.append(passage)

            return reranked
        except Exception:
            logger.warning("Reranking failed, falling back to original order", exc_info=True)
            return passages[:top_k]


reranker_service = RerankerService()
