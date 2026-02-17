"""Tests for GET /metrics endpoint."""

import pytest


class TestGetMetrics:
    async def test_default_params(self, app_client):
        resp = await app_client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "events" in data
        assert "total" in data
        assert data["total"] == 0
        assert data["events"] == []

    async def test_custom_minutes(self, app_client):
        resp = await app_client.get("/metrics?minutes=30")
        assert resp.status_code == 200
        app_client._mock_metrics.query.assert_called_with(30)

    async def test_returns_events(self, app_client):
        app_client._mock_metrics.query.return_value = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "event_type": "query",
                "model": "llama3.2",
                "total_tokens": 150,
                "duration_ms": 500.0,
            },
        ]
        resp = await app_client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["events"][0]["event_type"] == "query"
        assert data["events"][0]["total_tokens"] == 150

    async def test_response_shape(self, app_client):
        app_client._mock_metrics.query.return_value = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "event_type": "embedding",
                "model": "nomic-embed-text",
                "prompt_tokens": 25,
                "total_tokens": 25,
            },
        ]
        resp = await app_client.get("/metrics")
        data = resp.json()
        event = data["events"][0]
        assert "timestamp" in event
        assert "event_type" in event
        assert "model" in event

    async def test_invalid_minutes_too_low(self, app_client):
        resp = await app_client.get("/metrics?minutes=0")
        assert resp.status_code == 422

    async def test_invalid_minutes_too_high(self, app_client):
        resp = await app_client.get("/metrics?minutes=2000")
        assert resp.status_code == 422
