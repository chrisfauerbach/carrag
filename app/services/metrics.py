import asyncio
import logging
from datetime import datetime, timezone

from app.config import settings
from app.services.elasticsearch import es_service

logger = logging.getLogger(__name__)

METRICS_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "timestamp": {"type": "date"},
            "event_type": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "model": {"type": "keyword"},
            "prompt_tokens": {"type": "integer"},
            "completion_tokens": {"type": "integer"},
            "total_tokens": {"type": "integer"},
            "duration_ms": {"type": "float"},
            "ollama_prompt_eval_ms": {"type": "float"},
            "ollama_eval_ms": {"type": "float"},
            "ollama_load_ms": {"type": "float"},
            "ollama_total_ms": {"type": "float"},
            "metadata": {"type": "object"},
        }
    }
}


def extract_ollama_metrics(result: dict) -> dict:
    """Extract standard metrics from an Ollama API response.

    Works for both /api/generate and /api/embed responses.
    Converts nanosecond durations to milliseconds.
    Returns empty dict for missing fields.
    """
    metrics = {}

    if "prompt_eval_count" in result:
        metrics["prompt_tokens"] = result["prompt_eval_count"]
    if "eval_count" in result:
        metrics["completion_tokens"] = result["eval_count"]

    prompt_tokens = metrics.get("prompt_tokens", 0)
    completion_tokens = metrics.get("completion_tokens", 0)
    if prompt_tokens or completion_tokens:
        metrics["total_tokens"] = prompt_tokens + completion_tokens

    ns_to_ms = {
        "prompt_eval_duration": "ollama_prompt_eval_ms",
        "eval_duration": "ollama_eval_ms",
        "load_duration": "ollama_load_ms",
        "total_duration": "ollama_total_ms",
    }
    for ns_key, ms_key in ns_to_ms.items():
        if ns_key in result:
            metrics[ms_key] = result[ns_key] / 1_000_000

    return metrics


class MetricsService:
    @property
    def client(self):
        return es_service.client

    async def init_index(self):
        """Create the metrics index if it doesn't exist."""
        exists = await self.client.indices.exists(index=settings.es_metrics_index)
        if not exists:
            await self.client.indices.create(
                index=settings.es_metrics_index, body=METRICS_INDEX_MAPPING
            )
            logger.info(f"Created metrics index: {settings.es_metrics_index}")
        else:
            logger.info(f"Metrics index {settings.es_metrics_index} already exists.")

    async def record(self, event_type: str, model: str, **kwargs):
        """Record a metrics event to Elasticsearch.

        Never raises — metrics recording should never break the caller.
        """
        try:
            doc = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "model": model,
            }
            for key, value in kwargs.items():
                if value is not None:
                    doc[key] = value
            await self.client.index(
                index=settings.es_metrics_index, body=doc
            )
        except Exception:
            logger.warning("Failed to record metrics event", exc_info=True)

    async def query(self, minutes: int = 60) -> list[dict]:
        """Retrieve metrics events from the last N minutes."""
        body = {
            "query": {
                "range": {
                    "timestamp": {
                        "gte": f"now-{minutes}m",
                    }
                }
            },
            "sort": [{"timestamp": {"order": "asc"}}],
            "size": 1000,
        }
        resp = await self.client.search(index=settings.es_metrics_index, body=body)
        return [hit["_source"] for hit in resp["hits"]["hits"]]

    def record_background(self, event_type: str, model: str, **kwargs):
        """Fire-and-forget metrics recording via asyncio.create_task."""
        asyncio.create_task(self.record(event_type, model, **kwargs))


metrics_service = MetricsService()
