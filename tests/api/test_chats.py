"""Tests for /chats CRUD endpoints."""

import pytest
from unittest.mock import AsyncMock


class TestCreateChat:
    async def test_create_returns_201(self, app_client):
        resp = await app_client.post("/chats", json={})
        assert resp.status_code == 201
        data = resp.json()
        assert "chat_id" in data
        assert "title" in data
        assert "created_at" in data
        assert "updated_at" in data

    async def test_create_with_custom_title(self, app_client):
        app_client._mock_chat.create_chat.return_value = {
            "chat_id": "chat-xyz",
            "title": "My Custom Chat",
            "messages": [],
            "message_count": 0,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        resp = await app_client.post("/chats", json={"title": "My Custom Chat"})
        assert resp.status_code == 201
        assert resp.json()["title"] == "My Custom Chat"
        app_client._mock_chat.create_chat.assert_called_with(title="My Custom Chat")


class TestListChats:
    async def test_list_returns_chats(self, app_client):
        resp = await app_client.get("/chats")
        assert resp.status_code == 200
        data = resp.json()
        assert "chats" in data
        assert "total" in data
        assert data["total"] == 1
        assert data["chats"][0]["chat_id"] == "chat-abc"

    async def test_list_empty(self, app_client):
        app_client._mock_chat.list_chats.return_value = []
        resp = await app_client.get("/chats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["chats"] == []


class TestGetChat:
    async def test_found(self, app_client):
        resp = await app_client.get("/chats/chat-abc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["chat_id"] == "chat-abc"
        assert data["title"] == "Test Chat"
        assert len(data["messages"]) == 2
        assert data["message_count"] == 2

    async def test_not_found(self, app_client):
        app_client._mock_chat.get_chat.return_value = None
        resp = await app_client.get("/chats/nonexistent")
        assert resp.status_code == 404


class TestRenameChat:
    async def test_rename_found(self, app_client):
        resp = await app_client.patch("/chats/chat-abc", json={"title": "Renamed"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["chat_id"] == "chat-abc"
        assert data["title"] == "Renamed"

    async def test_rename_not_found(self, app_client):
        app_client._mock_chat.rename_chat.return_value = None
        resp = await app_client.patch("/chats/nonexistent", json={"title": "X"})
        assert resp.status_code == 404


class TestDeleteChat:
    async def test_delete_found(self, app_client):
        resp = await app_client.delete("/chats/chat-abc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["chat_id"] == "chat-abc"
        assert data["status"] == "deleted"

    async def test_delete_not_found(self, app_client):
        app_client._mock_chat.delete_chat.return_value = False
        resp = await app_client.delete("/chats/nonexistent")
        assert resp.status_code == 404
