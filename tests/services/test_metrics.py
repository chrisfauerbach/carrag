"""Tests for app.services.metrics — MetricsService + extract_ollama_metrics."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.metrics import MetricsService, extract_ollama_metrics


@pytest.fixture
def mock_es_client():
    """Mock Elasticsearch client."""
    client = AsyncMock()
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock()
    client.indices.create = AsyncMock()
    client.index = AsyncMock()
    return client


@pytest.fixture
def service(mock_es_client):
    """MetricsService with mocked ES client."""
    svc = MetricsService()
    with patch("app.services.metrics.es_service") as mock_es_svc:
        mock_es_svc.client = mock_es_client
        yield svc, mock_es_client


class TestInitIndex:
    async def test_creates_when_not_exists(self, service):
        svc, mock_client = service
        mock_client.indices.exists.return_value = False

        await svc.init_index()

        mock_client.indices.create.assert_called_once()
        call_kwargs = mock_client.indices.create.call_args[1]
        assert call_kwargs["index"] == "carrag_metrics"

    async def test_skips_when_exists(self, service):
        svc, mock_client = service
        mock_client.indices.exists.return_value = True

        await svc.init_index()

        mock_client.indices.create.assert_not_called()


class TestRecord:
    async def test_indexes_with_correct_fields(self, service):
        svc, mock_client = service

        await svc.record("query", "llama3.2", duration_ms=123.4, prompt_tokens=50)

        mock_client.index.assert_called_once()
        call_kwargs = mock_client.index.call_args[1]
        assert call_kwargs["index"] == "carrag_metrics"
        doc = call_kwargs["body"]
        assert doc["event_type"] == "query"
        assert doc["model"] == "llama3.2"
        assert doc["duration_ms"] == 123.4
        assert doc["prompt_tokens"] == 50
        assert "timestamp" in doc

    async def test_omits_none_values(self, service):
        svc, mock_client = service

        await svc.record("embedding", "nomic-embed-text", duration_ms=50.0, document_id=None)

        doc = mock_client.index.call_args[1]["body"]
        assert "document_id" not in doc
        assert doc["duration_ms"] == 50.0

    async def test_includes_metadata(self, service):
        svc, mock_client = service

        await svc.record("ingest", "", metadata={"filename": "test.pdf", "chunk_count": 5})

        doc = mock_client.index.call_args[1]["body"]
        assert doc["metadata"] == {"filename": "test.pdf", "chunk_count": 5}

    async def test_never_raises_on_es_error(self, service):
        svc, mock_client = service
        mock_client.index.side_effect = Exception("ES connection refused")

        # Should not raise
        await svc.record("query", "llama3.2", duration_ms=100.0)

    async def test_uses_correct_index_name(self, service):
        svc, mock_client = service

        await svc.record("embedding", "nomic-embed-text")

        call_kwargs = mock_client.index.call_args[1]
        assert call_kwargs["index"] == "carrag_metrics"


class TestQuery:
    async def test_builds_correct_es_query(self, service):
        svc, mock_client = service
        mock_client.search = AsyncMock(return_value={"hits": {"hits": []}})

        await svc.query(30)

        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["index"] == "carrag_metrics"
        body = call_kwargs["body"]
        assert body["query"]["range"]["timestamp"]["gte"] == "now-30m"
        assert body["sort"] == [{"timestamp": {"order": "asc"}}]
        assert body["size"] == 1000

    async def test_returns_source_dicts(self, service):
        svc, mock_client = service
        mock_client.search = AsyncMock(return_value={
            "hits": {
                "hits": [
                    {"_source": {"event_type": "query", "model": "llama3.2", "timestamp": "2026-01-01T00:00:00Z"}},
                    {"_source": {"event_type": "embedding", "model": "nomic-embed-text", "timestamp": "2026-01-01T00:01:00Z"}},
                ]
            }
        })

        results = await svc.query(60)

        assert len(results) == 2
        assert results[0]["event_type"] == "query"
        assert results[1]["event_type"] == "embedding"

    async def test_handles_empty_results(self, service):
        svc, mock_client = service
        mock_client.search = AsyncMock(return_value={"hits": {"hits": []}})

        results = await svc.query(60)

        assert results == []

    async def test_default_minutes(self, service):
        svc, mock_client = service
        mock_client.search = AsyncMock(return_value={"hits": {"hits": []}})

        await svc.query()

        body = mock_client.search.call_args[1]["body"]
        assert body["query"]["range"]["timestamp"]["gte"] == "now-60m"


class TestExtractOllamaMetrics:
    def test_full_generate_response(self):
        result = {
            "response": "The answer is 42.",
            "done": True,
            "prompt_eval_count": 100,
            "eval_count": 50,
            "prompt_eval_duration": 500_000_000,  # 500ms
            "eval_duration": 1_000_000_000,  # 1000ms
            "load_duration": 200_000_000,  # 200ms
            "total_duration": 1_700_000_000,  # 1700ms
        }

        metrics = extract_ollama_metrics(result)

        assert metrics["prompt_tokens"] == 100
        assert metrics["completion_tokens"] == 50
        assert metrics["total_tokens"] == 150
        assert metrics["ollama_prompt_eval_ms"] == 500.0
        assert metrics["ollama_eval_ms"] == 1000.0
        assert metrics["ollama_load_ms"] == 200.0
        assert metrics["ollama_total_ms"] == 1700.0

    def test_embed_response_partial_fields(self):
        """Embed responses typically have prompt_eval_count but no eval_count."""
        result = {
            "embeddings": [[0.1] * 768],
            "prompt_eval_count": 25,
            "total_duration": 100_000_000,  # 100ms
        }

        metrics = extract_ollama_metrics(result)

        assert metrics["prompt_tokens"] == 25
        assert "completion_tokens" not in metrics
        assert metrics["total_tokens"] == 25
        assert metrics["ollama_total_ms"] == 100.0
        assert "ollama_eval_ms" not in metrics

    def test_empty_response(self):
        metrics = extract_ollama_metrics({})

        assert metrics == {}

    def test_no_token_counts_no_total(self):
        """If neither prompt nor completion tokens, total_tokens is not set."""
        result = {"load_duration": 50_000_000}

        metrics = extract_ollama_metrics(result)

        assert "total_tokens" not in metrics
        assert metrics["ollama_load_ms"] == 50.0
