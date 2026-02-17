"""Tests for /prompts CRUD endpoints."""

import pytest


FAKE_PROMPT = {
    "key": "rag_system",
    "name": "RAG System Prompt",
    "description": "Instructions for how the LLM answers RAG queries",
    "content": "You are a helpful assistant...",
    "variables": [],
    "updated_at": "2026-01-01T00:00:00+00:00",
}


class TestListPrompts:
    async def test_list_returns_prompts(self, app_client):
        resp = await app_client.get("/prompts")
        assert resp.status_code == 200
        data = resp.json()
        assert "prompts" in data
        assert "total" in data
        assert data["total"] == 1
        assert data["prompts"][0]["key"] == "rag_system"

    async def test_list_empty(self, app_client):
        app_client._mock_prompts.list_prompts.return_value = []
        resp = await app_client.get("/prompts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["prompts"] == []


class TestGetPrompt:
    async def test_found(self, app_client):
        resp = await app_client.get("/prompts/rag_system")
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "rag_system"
        assert data["name"] == "RAG System Prompt"
        assert "content" in data

    async def test_not_found(self, app_client):
        app_client._mock_prompts.get_prompt.return_value = None
        resp = await app_client.get("/prompts/nonexistent")
        assert resp.status_code == 404


class TestUpdatePrompt:
    async def test_update_returns_updated(self, app_client):
        resp = await app_client.patch(
            "/prompts/rag_system",
            json={"content": "Updated content"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "rag_system"
        assert data["content"] == "Updated content"
        assert "updated_at" in data

    async def test_update_not_found(self, app_client):
        app_client._mock_prompts.update_prompt.return_value = None
        resp = await app_client.patch(
            "/prompts/nonexistent",
            json={"content": "New content"},
        )
        assert resp.status_code == 404


class TestResetPrompt:
    async def test_reset_returns_default(self, app_client):
        resp = await app_client.post("/prompts/rag_system/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "rag_system"
        assert data["content"] == "Default content"
        assert "updated_at" in data

    async def test_reset_not_found(self, app_client):
        app_client._mock_prompts.reset_prompt.return_value = None
        resp = await app_client.post("/prompts/nonexistent/reset")
        assert resp.status_code == 404
