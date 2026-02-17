"""Tests for app.services.embeddings — Ollama embedding client."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock, PropertyMock

import httpx

from app.services.embeddings import EmbeddingService


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient for EmbeddingService."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False
    return client


@pytest.fixture
def service(mock_httpx_client):
    """EmbeddingService with a mocked httpx client."""
    with patch("app.services.embeddings.metrics_service") as mock_metrics:
        mock_metrics.record_background = MagicMock()
        svc = EmbeddingService()
        svc._client = mock_httpx_client
        yield svc


class TestEmbedSingle:
    async def test_returns_vector(self, service, mock_httpx_client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1] * 768]}
        mock_resp.raise_for_status = MagicMock()
        mock_httpx_client.post.return_value = mock_resp

        result = await service.embed_single("hello")
        assert len(result) == 768
        mock_httpx_client.post.assert_called_once()

    async def test_calls_embed_endpoint(self, service, mock_httpx_client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.5] * 768]}
        mock_resp.raise_for_status = MagicMock()
        mock_httpx_client.post.return_value = mock_resp

        await service.embed_single("test text")
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "/api/embed"


class TestEmbed:
    async def test_returns_list_of_vectors(self, service, mock_httpx_client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1] * 768, [0.2] * 768]}
        mock_resp.raise_for_status = MagicMock()
        mock_httpx_client.post.return_value = mock_resp

        result = await service.embed(["text1", "text2"])
        assert len(result) == 2
        assert len(result[0]) == 768

    async def test_sends_all_texts(self, service, mock_httpx_client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1] * 768] * 3}
        mock_resp.raise_for_status = MagicMock()
        mock_httpx_client.post.return_value = mock_resp

        await service.embed(["a", "b", "c"])
        call_json = mock_httpx_client.post.call_args[1]["json"]
        assert call_json["input"] == ["a", "b", "c"]

    async def test_http_error_raises(self, service, mock_httpx_client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        mock_httpx_client.post.return_value = mock_resp

        with pytest.raises(httpx.HTTPStatusError):
            await service.embed(["text"])


class TestEnsureModel:
    async def test_skips_pull_when_model_exists(self, service, mock_httpx_client):
        mock_tags = MagicMock()
        mock_tags.json.return_value = {"models": [{"name": "nomic-embed-text:latest"}]}
        mock_tags.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = mock_tags

        await service.ensure_model()
        mock_httpx_client.post.assert_not_called()

    async def test_pulls_model_when_missing(self, service, mock_httpx_client):
        mock_tags = MagicMock()
        mock_tags.json.return_value = {"models": []}
        mock_tags.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = mock_tags

        mock_pull = MagicMock()
        mock_pull.raise_for_status = MagicMock()
        mock_httpx_client.post.return_value = mock_pull

        await service.ensure_model()
        mock_httpx_client.post.assert_called_once()
        assert "/api/pull" in str(mock_httpx_client.post.call_args)


class TestClose:
    async def test_closes_client(self, service, mock_httpx_client):
        await service.close()
        mock_httpx_client.aclose.assert_called_once()

    async def test_close_when_no_client(self):
        svc = EmbeddingService()
        await svc.close()  # Should not raise
