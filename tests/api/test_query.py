"""Tests for query endpoints."""

import json
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

    async def test_tags_passed_through(self, app_client):
        resp = await app_client.post(
            "/query", json={"question": "Q?", "tags": ["research", "ml"]}
        )
        assert resp.status_code == 200
        call_kwargs = app_client._mock_rag.call_args[1]
        assert call_kwargs["tags"] == ["research", "ml"]

    async def test_empty_tags_passes_none(self, app_client):
        resp = await app_client.post(
            "/query", json={"question": "Q?"}
        )
        assert resp.status_code == 200
        call_kwargs = app_client._mock_rag.call_args[1]
        assert call_kwargs["tags"] is None


class TestQueryStream:
    async def test_returns_event_stream_content_type(self, app_client):
        resp = await app_client.post(
            "/query/stream", json={"question": "What is X?"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    async def test_sse_body_contains_expected_events(self, app_client):
        resp = await app_client.post(
            "/query/stream", json={"question": "What is X?"}
        )
        body = resp.text

        assert "event: sources" in body
        assert "event: token" in body
        assert "event: done" in body

    async def test_sources_event_has_sources(self, app_client):
        resp = await app_client.post(
            "/query/stream", json={"question": "What is X?"}
        )
        body = resp.text

        # Parse the sources event
        for block in body.strip().split("\n\n"):
            lines = block.strip().split("\n")
            if len(lines) >= 2 and lines[0] == "event: sources":
                data = json.loads(lines[1].removeprefix("data: "))
                assert "sources" in data
                assert len(data["sources"]) > 0
                break
        else:
            pytest.fail("No sources event found")

    async def test_done_event_has_model_and_duration(self, app_client):
        resp = await app_client.post(
            "/query/stream", json={"question": "What is X?"}
        )
        body = resp.text

        for block in body.strip().split("\n\n"):
            lines = block.strip().split("\n")
            if len(lines) >= 2 and lines[0] == "event: done":
                data = json.loads(lines[1].removeprefix("data: "))
                assert data["model"] == "llama3.2"
                assert "duration_ms" in data
                break
        else:
            pytest.fail("No done event found")

    async def test_no_buffering_header(self, app_client):
        resp = await app_client.post(
            "/query/stream", json={"question": "What is X?"}
        )
        assert resp.headers.get("x-accel-buffering") == "no"

    async def test_error_event_on_failure(self, app_client):
        async def _failing_stream(**kwargs):
            raise RuntimeError("boom")
            yield  # make it a generator  # noqa: E501

        app_client._mock_rag_stream.side_effect = _failing_stream

        resp = await app_client.post(
            "/query/stream", json={"question": "What is X?"}
        )
        assert resp.status_code == 200
        body = resp.text
        assert "event: error" in body
        assert "boom" in body
