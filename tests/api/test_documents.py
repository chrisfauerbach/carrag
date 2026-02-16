"""Tests for GET/DELETE /documents endpoints."""

import pytest
from unittest.mock import AsyncMock


class TestListDocuments:
    async def test_list_returns_documents(self, app_client):
        resp = await app_client.get("/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data
        assert "total" in data
        assert data["total"] == 1
        assert data["documents"][0]["document_id"] == "doc-123"

    async def test_list_empty(self, app_client):
        app_client._mock_es.list_documents.return_value = []
        resp = await app_client.get("/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["documents"] == []


class TestGetDocument:
    async def test_found(self, app_client):
        resp = await app_client.get("/documents/doc-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_id"] == "doc-123"
        assert data["filename"] == "test.txt"
        assert "metadata" in data

    async def test_not_found(self, app_client):
        app_client._mock_es.get_document.return_value = None
        resp = await app_client.get("/documents/nonexistent")
        assert resp.status_code == 404

    async def test_response_shape(self, app_client):
        resp = await app_client.get("/documents/doc-123")
        data = resp.json()
        assert "document_id" in data
        assert "filename" in data
        assert "source_type" in data
        assert "chunk_count" in data
        assert "metadata" in data


class TestDeleteDocument:
    async def test_delete_found(self, app_client):
        resp = await app_client.delete("/documents/doc-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_id"] == "doc-123"
        assert data["chunks_deleted"] == 3
        assert data["status"] == "deleted"

    async def test_delete_not_found(self, app_client):
        app_client._mock_es.get_document.return_value = None
        resp = await app_client.delete("/documents/nonexistent")
        assert resp.status_code == 404

    async def test_delete_calls_es_service(self, app_client):
        await app_client.delete("/documents/doc-123")
        app_client._mock_es.delete_document.assert_called_with("doc-123")

    async def test_delete_verifies_existence_first(self, app_client):
        await app_client.delete("/documents/doc-123")
        app_client._mock_es.get_document.assert_called_with("doc-123")

    async def test_delete_returns_correct_count(self, app_client):
        app_client._mock_es.delete_document.return_value = 7
        resp = await app_client.delete("/documents/doc-123")
        assert resp.json()["chunks_deleted"] == 7
