"""Tests for app.services.prompts — PromptsService CRUD operations."""

import pytest
from unittest.mock import AsyncMock, patch

from elasticsearch import NotFoundError

from app.services.prompts import PromptsService, DEFAULT_PROMPTS


@pytest.fixture
def mock_es_client():
    """Mock AsyncElasticsearch client."""
    client = AsyncMock()
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock()
    client.index = AsyncMock()
    client.get = AsyncMock()
    client.search = AsyncMock()
    client.update = AsyncMock()
    client.count = AsyncMock(return_value={"count": 0})
    return client


@pytest.fixture
def service(mock_es_client):
    """PromptsService with mocked ES client via es_service."""
    svc = PromptsService()
    mock_es_svc = AsyncMock()
    mock_es_svc.client = mock_es_client
    with patch("app.services.prompts.es_service", mock_es_svc):
        yield svc, mock_es_client


class TestInitIndex:
    async def test_creates_index_and_seeds_when_not_exists(self, service):
        svc, client = service
        client.indices.exists.return_value = False
        client.count.return_value = {"count": 0}
        await svc.init_index()
        client.indices.create.assert_called_once()
        # Should seed 4 default prompts
        assert client.index.call_count == 4

    async def test_seeds_when_index_exists_but_empty(self, service):
        svc, client = service
        client.indices.exists.return_value = True
        client.count.return_value = {"count": 0}
        await svc.init_index()
        client.indices.create.assert_not_called()
        assert client.index.call_count == 4

    async def test_skips_seeding_when_data_exists(self, service):
        svc, client = service
        client.indices.exists.return_value = True
        client.count.return_value = {"count": 4}
        await svc.init_index()
        client.indices.create.assert_not_called()
        client.index.assert_not_called()


class TestGetPrompt:
    async def test_found(self, service):
        svc, client = service
        client.get.return_value = {
            "_source": {
                "key": "rag_system",
                "name": "RAG System Prompt",
                "description": "desc",
                "content": "You are helpful",
                "variables": [],
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        }
        result = await svc.get_prompt("rag_system")
        assert result is not None
        assert result["key"] == "rag_system"

    async def test_not_found(self, service):
        svc, client = service
        client.get.side_effect = NotFoundError(404, "not found", {})
        result = await svc.get_prompt("nonexistent")
        assert result is None


class TestListPrompts:
    async def test_returns_all_prompts(self, service):
        svc, client = service
        client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"key": "rag_system", "name": "RAG System Prompt"}},
                    {"_source": {"key": "rag_user", "name": "RAG User Template"}},
                ]
            }
        }
        result = await svc.list_prompts()
        assert len(result) == 2
        assert result[0]["key"] == "rag_system"

    async def test_empty_list(self, service):
        svc, client = service
        client.search.return_value = {"hits": {"hits": []}}
        result = await svc.list_prompts()
        assert result == []


class TestUpdatePrompt:
    async def test_updates_existing(self, service):
        svc, client = service
        client.get.return_value = {
            "_source": {
                "key": "rag_system",
                "content": "old content",
            }
        }
        result = await svc.update_prompt("rag_system", "new content")
        assert result is not None
        assert result["key"] == "rag_system"
        assert result["content"] == "new content"
        assert "updated_at" in result
        client.update.assert_called_once()

    async def test_returns_none_when_not_found(self, service):
        svc, client = service
        client.get.side_effect = NotFoundError(404, "not found", {})
        result = await svc.update_prompt("nonexistent", "content")
        assert result is None


class TestResetPrompt:
    async def test_resets_to_default(self, service):
        svc, client = service
        result = await svc.reset_prompt("rag_system")
        assert result is not None
        assert result["key"] == "rag_system"
        assert result["content"] == DEFAULT_PROMPTS["rag_system"]["content"]
        assert "updated_at" in result
        client.update.assert_called_once()

    async def test_returns_none_for_unknown_key(self, service):
        svc, client = service
        result = await svc.reset_prompt("nonexistent")
        assert result is None
        client.update.assert_not_called()
