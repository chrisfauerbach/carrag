"""Tests for GET /health endpoint."""

import pytest


class TestHealth:
    async def test_health_returns_ok(self, app_client):
        resp = await app_client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
