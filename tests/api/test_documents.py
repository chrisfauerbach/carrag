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


class TestGetDocumentChunks:
    async def test_returns_chunks(self, app_client):
        resp = await app_client.get("/documents/doc-123/chunks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_id"] == "doc-123"
        assert data["filename"] == "test.txt"
        assert data["source_type"] == "text"
        assert data["chunk_count"] == 3
        assert len(data["chunks"]) == 3

    async def test_chunk_shape(self, app_client):
        resp = await app_client.get("/documents/doc-123/chunks")
        chunk = resp.json()["chunks"][0]
        assert "content" in chunk
        assert "chunk_index" in chunk
        assert "char_start" in chunk
        assert "char_end" in chunk

    async def test_not_found(self, app_client):
        app_client._mock_es.get_document.return_value = None
        resp = await app_client.get("/documents/nonexistent/chunks")
        assert resp.status_code == 404

    async def test_calls_both_service_methods(self, app_client):
        await app_client.get("/documents/doc-123/chunks")
        app_client._mock_es.get_document.assert_called_with("doc-123")
        app_client._mock_es.get_document_chunks.assert_called_with("doc-123")

    async def test_empty_chunks(self, app_client):
        app_client._mock_es.get_document_chunks.return_value = []
        resp = await app_client.get("/documents/doc-123/chunks")
        data = resp.json()
        assert data["chunk_count"] == 0
        assert data["chunks"] == []


class TestDocumentSimilarity:
    async def test_returns_nodes_and_edges(self, app_client):
        app_client._mock_sim.return_value = {
            "nodes": [
                {"document_id": "doc-1", "filename": "a.txt", "source_type": "text", "chunk_count": 3},
                {"document_id": "doc-2", "filename": "b.pdf", "source_type": "pdf", "chunk_count": 5},
            ],
            "edges": [
                {"source": "doc-1", "target": "doc-2", "similarity": 0.85},
            ],
            "threshold": 0.3,
        }
        resp = await app_client.get("/documents/similarity")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert data["edges"][0]["similarity"] == 0.85

    async def test_threshold_parameter(self, app_client):
        resp = await app_client.get("/documents/similarity?threshold=0.7")
        assert resp.status_code == 200
        app_client._mock_sim.assert_called_with(threshold=0.7)

    async def test_invalid_threshold(self, app_client):
        resp = await app_client.get("/documents/similarity?threshold=1.5")
        assert resp.status_code == 422

    async def test_negative_threshold(self, app_client):
        resp = await app_client.get("/documents/similarity?threshold=-0.1")
        assert resp.status_code == 422

    async def test_empty_index(self, app_client):
        app_client._mock_sim.return_value = {
            "nodes": [],
            "edges": [],
            "threshold": 0.3,
        }
        resp = await app_client.get("/documents/similarity")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []
        assert data["edges"] == []


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
