"""Tests for app.services.chat — ChatService CRUD operations."""

import pytest
from unittest.mock import AsyncMock, patch

from elasticsearch import NotFoundError

from app.services.chat import ChatService


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
    client.delete = AsyncMock()
    return client


@pytest.fixture
def service(mock_es_client):
    """ChatService with mocked ES client via es_service."""
    svc = ChatService()
    mock_es_svc = AsyncMock()
    mock_es_svc.client = mock_es_client
    with patch("app.services.chat.es_service", mock_es_svc):
        yield svc, mock_es_client


class TestInitIndex:
    async def test_creates_index_when_not_exists(self, service):
        svc, client = service
        client.indices.exists.return_value = False
        await svc.init_index()
        client.indices.create.assert_called_once()

    async def test_skips_when_index_exists(self, service):
        svc, client = service
        client.indices.exists.return_value = True
        await svc.init_index()
        client.indices.create.assert_not_called()


class TestCreateChat:
    async def test_creates_with_default_title(self, service):
        svc, client = service
        result = await svc.create_chat()
        assert result["title"] == "New Chat"
        assert result["messages"] == []
        assert result["message_count"] == 0
        assert "chat_id" in result
        client.index.assert_called_once()

    async def test_creates_with_custom_title(self, service):
        svc, client = service
        result = await svc.create_chat(title="My Chat")
        assert result["title"] == "My Chat"


class TestGetChat:
    async def test_found(self, service):
        svc, client = service
        client.get.return_value = {
            "_source": {
                "chat_id": "abc",
                "title": "Test",
                "messages": [],
                "message_count": 0,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        }
        result = await svc.get_chat("abc")
        assert result is not None
        assert result["chat_id"] == "abc"

    async def test_not_found(self, service):
        svc, client = service
        client.get.side_effect = NotFoundError(404, "not found", {})
        result = await svc.get_chat("nonexistent")
        assert result is None


class TestListChats:
    async def test_returns_chats_sorted(self, service):
        svc, client = service
        client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "chat_id": "c1",
                            "title": "Chat 1",
                            "message_count": 2,
                            "created_at": "2026-01-02T00:00:00+00:00",
                            "updated_at": "2026-01-02T00:00:00+00:00",
                        }
                    },
                    {
                        "_source": {
                            "chat_id": "c2",
                            "title": "Chat 2",
                            "message_count": 0,
                            "created_at": "2026-01-01T00:00:00+00:00",
                            "updated_at": "2026-01-01T00:00:00+00:00",
                        }
                    },
                ]
            }
        }
        result = await svc.list_chats()
        assert len(result) == 2
        assert result[0]["chat_id"] == "c1"

    async def test_empty_list(self, service):
        svc, client = service
        client.search.return_value = {"hits": {"hits": []}}
        result = await svc.list_chats()
        assert result == []


class TestAppendMessages:
    async def test_appends_and_returns_updated(self, service):
        svc, client = service
        existing = {
            "chat_id": "c1",
            "title": "Existing",
            "messages": [{"role": "user", "content": "Hi", "timestamp": "t1"}],
            "message_count": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        # get_chat called twice: once for append, once for return
        client.get.return_value = {"_source": existing}

        new_msg = {"role": "assistant", "content": "Hello", "timestamp": "t2"}
        result = await svc.append_messages("c1", [new_msg])
        assert result is not None
        client.index.assert_called_once()

    async def test_auto_titles_from_first_user_message(self, service):
        svc, client = service
        existing = {
            "chat_id": "c1",
            "title": "New Chat",
            "messages": [],
            "message_count": 0,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        updated = {**existing, "title": "What is RAG?", "message_count": 1}
        client.get.side_effect = [
            {"_source": existing},
            {"_source": updated},
        ]

        new_msg = {"role": "user", "content": "What is RAG?", "timestamp": "t1"}
        await svc.append_messages("c1", [new_msg])

        call_kwargs = client.index.call_args[1]
        assert call_kwargs["body"]["title"] == "What is RAG?"

    async def test_returns_none_when_chat_not_found(self, service):
        svc, client = service
        client.get.side_effect = NotFoundError(404, "not found", {})
        result = await svc.append_messages("nonexistent", [])
        assert result is None


class TestRenameChat:
    async def test_renames_existing(self, service):
        svc, client = service
        client.get.return_value = {
            "_source": {
                "chat_id": "c1",
                "title": "Old",
                "messages": [],
                "message_count": 0,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        }
        result = await svc.rename_chat("c1", "New Title")
        assert result is not None
        assert result["title"] == "New Title"
        client.update.assert_called_once()

    async def test_returns_none_when_not_found(self, service):
        svc, client = service
        client.get.side_effect = NotFoundError(404, "not found", {})
        result = await svc.rename_chat("nonexistent", "Title")
        assert result is None


class TestDeleteChat:
    async def test_deletes_existing(self, service):
        svc, client = service
        result = await svc.delete_chat("c1")
        assert result is True
        client.delete.assert_called_once()

    async def test_returns_false_when_not_found(self, service):
        svc, client = service
        client.delete.side_effect = NotFoundError(404, "not found", {})
        result = await svc.delete_chat("nonexistent")
        assert result is False
