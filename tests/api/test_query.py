"""Tests for query endpoints."""

import pytest
from unittest.mock import AsyncMock


class TestListModels:
    async def test_returns_models_and_default(self, app_client):
        resp = await app_client.get("/query/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "default" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) == 1
        assert "llama3.2" in data["models"]
        assert "nomic-embed-text" not in data["models"]
        assert data["default"] == "llama3.2"

    async def test_filters_embedding_models(self, app_client):
        resp = await app_client.get("/query/models")
        data = resp.json()
        for name in data["models"]:
            assert "embed" not in name.lower()

    async def test_models_sorted(self, app_client):
        resp = await app_client.get("/query/models")
        data = resp.json()
        assert data["models"] == sorted(data["models"])


class TestQuery:
    async def test_basic_query(self, app_client):
        resp = await app_client.post("/query", json={"question": "What is X?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "model" in data
        assert "duration_ms" in data
        assert "sources" in data

    async def test_return_sources_false(self, app_client):
        # Configure mock to return sources, but endpoint should strip them
        app_client._mock_rag.return_value = {
            "answer": "Answer",
            "sources": [{"content": "c", "score": 0.9, "metadata": {}}],
            "model": "llama3.2",
            "duration_ms": 100.0,
        }
        resp = await app_client.post(
            "/query", json={"question": "Q?", "return_sources": False}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["sources"] == []

    async def test_custom_top_k(self, app_client):
        resp = await app_client.post(
            "/query", json={"question": "Q?", "top_k": 10}
        )
        assert resp.status_code == 200
        call_kwargs = app_client._mock_rag.call_args[1]
        assert call_kwargs["top_k"] == 10

    async def test_history_passed_through(self, app_client):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        resp = await app_client.post(
            "/query", json={"question": "Follow up?", "history": history}
        )
        assert resp.status_code == 200
        call_kwargs = app_client._mock_rag.call_args[1]
        assert len(call_kwargs["history"]) == 2

    async def test_top_k_zero_rejected(self, app_client):
        resp = await app_client.post("/query", json={"question": "Q?", "top_k": 0})
        assert resp.status_code == 422

    async def test_top_k_21_rejected(self, app_client):
        resp = await app_client.post("/query", json={"question": "Q?", "top_k": 21})
        assert resp.status_code == 422

    async def test_missing_question(self, app_client):
        resp = await app_client.post("/query", json={})
        assert resp.status_code == 422

    async def test_custom_model(self, app_client):
        resp = await app_client.post(
            "/query", json={"question": "Q?", "model": "custom-model"}
        )
        assert resp.status_code == 200
        call_kwargs = app_client._mock_rag.call_args[1]
        assert call_kwargs["model"] == "custom-model"
