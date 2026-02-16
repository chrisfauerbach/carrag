"""Tests for app.services.elasticsearch — ES index management, bulk insert, kNN search."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.elasticsearch import ElasticsearchService


@pytest.fixture
def mock_es_client():
    """Mock AsyncElasticsearch client."""
    client = AsyncMock()
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock()
    client.indices.refresh = AsyncMock()
    client.search = AsyncMock()
    client.count = AsyncMock()
    client.delete_by_query = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def service(mock_es_client):
    """ElasticsearchService with mocked client."""
    svc = ElasticsearchService()
    svc._client = mock_es_client
    return svc


class TestInit:
    async def test_creates_index_when_not_exists(self, service, mock_es_client):
        mock_es_client.indices.exists.return_value = False
        await service.init()
        mock_es_client.indices.create.assert_called_once()

    async def test_skips_when_index_exists(self, service, mock_es_client):
        mock_es_client.indices.exists.return_value = True
        await service.init()
        mock_es_client.indices.create.assert_not_called()


class TestIndexChunks:
    async def test_returns_count(self, service, mock_es_client):
        chunks = [
            {"text": "chunk1", "document_id": "d1", "chunk_index": 0, "char_start": 0, "char_end": 6},
            {"text": "chunk2", "document_id": "d1", "chunk_index": 1, "char_start": 6, "char_end": 12},
        ]
        embeddings = [[0.1] * 768, [0.2] * 768]
        metadata = {"filename": "test.txt", "source_type": "text"}

        with patch("app.services.elasticsearch.async_bulk", new_callable=AsyncMock) as mock_bulk:
            mock_bulk.return_value = (2, [])
            result = await service.index_chunks(chunks, embeddings, metadata)

        assert result == 2

    async def test_calls_bulk_and_refresh(self, service, mock_es_client):
        chunks = [{"text": "c", "document_id": "d1", "chunk_index": 0, "char_start": 0, "char_end": 1}]
        embeddings = [[0.1] * 768]
        metadata = {"filename": "f.txt"}

        with patch("app.services.elasticsearch.async_bulk", new_callable=AsyncMock) as mock_bulk:
            mock_bulk.return_value = (1, [])
            await service.index_chunks(chunks, embeddings, metadata)

        mock_bulk.assert_called_once()
        mock_es_client.indices.refresh.assert_called_once()

    async def test_bulk_actions_have_correct_fields(self, service, mock_es_client):
        chunks = [{"text": "hello", "document_id": "d1", "chunk_index": 0, "char_start": 0, "char_end": 5}]
        embeddings = [[0.1] * 768]
        metadata = {"filename": "test.txt"}

        with patch("app.services.elasticsearch.async_bulk", new_callable=AsyncMock) as mock_bulk:
            mock_bulk.return_value = (1, [])
            await service.index_chunks(chunks, embeddings, metadata)

        # Get the generator from the call and consume it
        call_args = mock_bulk.call_args
        gen = call_args[0][1]  # second positional arg is the generator
        actions = list(gen)
        assert len(actions) == 1
        doc = actions[0]["_source"]
        assert doc["content"] == "hello"
        assert doc["document_id"] == "d1"
        assert doc["embedding"] == [0.1] * 768
        assert "created_at" in doc


class TestKnnSearch:
    async def test_returns_formatted_results(self, service, mock_es_client):
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_score": 0.95,
                        "_source": {
                            "content": "relevant text",
                            "document_id": "d1",
                            "chunk_index": 0,
                            "metadata": {"filename": "test.txt"},
                            "created_at": "2026-01-01T00:00:00",
                        },
                    }
                ]
            }
        }

        results = await service.knn_search([0.1] * 768, top_k=5)
        assert len(results) == 1
        assert results[0]["content"] == "relevant text"
        assert results[0]["score"] == 0.95
        assert results[0]["document_id"] == "d1"
        assert results[0]["metadata"]["filename"] == "test.txt"

    async def test_empty_results(self, service, mock_es_client):
        mock_es_client.search.return_value = {"hits": {"hits": []}}
        results = await service.knn_search([0.1] * 768)
        assert results == []

    async def test_passes_top_k(self, service, mock_es_client):
        mock_es_client.search.return_value = {"hits": {"hits": []}}
        await service.knn_search([0.1] * 768, top_k=10)
        call_kwargs = mock_es_client.search.call_args[1]
        assert call_kwargs["size"] == 10


class TestListDocuments:
    async def test_returns_document_list(self, service, mock_es_client):
        mock_es_client.search.return_value = {
            "aggregations": {
                "documents": {
                    "buckets": [
                        {
                            "key": "doc-1",
                            "doc_count": 5,
                            "doc_info": {
                                "hits": {
                                    "hits": [
                                        {
                                            "_source": {
                                                "metadata": {"filename": "file.txt", "source_type": "text"},
                                                "created_at": "2026-01-01T00:00:00",
                                            }
                                        }
                                    ]
                                }
                            },
                        }
                    ]
                }
            }
        }

        docs = await service.list_documents()
        assert len(docs) == 1
        assert docs[0]["document_id"] == "doc-1"
        assert docs[0]["filename"] == "file.txt"
        assert docs[0]["chunk_count"] == 5


class TestGetDocument:
    async def test_found(self, service, mock_es_client):
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "metadata": {"filename": "test.txt", "source_type": "text"},
                            "created_at": "2026-01-01T00:00:00",
                        }
                    }
                ]
            }
        }
        mock_es_client.count.return_value = {"count": 3}

        result = await service.get_document("doc-1")
        assert result is not None
        assert result["document_id"] == "doc-1"
        assert result["chunk_count"] == 3

    async def test_not_found(self, service, mock_es_client):
        mock_es_client.search.return_value = {"hits": {"hits": []}}
        result = await service.get_document("nonexistent")
        assert result is None


class TestGetDocumentChunks:
    async def test_returns_sorted_chunks(self, service, mock_es_client):
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"content": "first", "chunk_index": 0, "char_start": 0, "char_end": 5}},
                    {"_source": {"content": "second", "chunk_index": 1, "char_start": 5, "char_end": 11}},
                ]
            }
        }
        chunks = await service.get_document_chunks("doc-1")
        assert len(chunks) == 2
        assert chunks[0]["content"] == "first"
        assert chunks[0]["chunk_index"] == 0
        assert chunks[1]["content"] == "second"

    async def test_empty_results(self, service, mock_es_client):
        mock_es_client.search.return_value = {"hits": {"hits": []}}
        chunks = await service.get_document_chunks("nonexistent")
        assert chunks == []

    async def test_excludes_embedding(self, service, mock_es_client):
        mock_es_client.search.return_value = {"hits": {"hits": []}}
        await service.get_document_chunks("doc-1")
        call_kwargs = mock_es_client.search.call_args[1]
        source_fields = call_kwargs["body"]["_source"]
        assert "embedding" not in source_fields

    async def test_sorts_by_chunk_index(self, service, mock_es_client):
        mock_es_client.search.return_value = {"hits": {"hits": []}}
        await service.get_document_chunks("doc-1")
        call_kwargs = mock_es_client.search.call_args[1]
        assert call_kwargs["body"]["sort"] == [{"chunk_index": "asc"}]


class TestDeleteDocument:
    async def test_returns_deleted_count(self, service, mock_es_client):
        mock_es_client.delete_by_query.return_value = {"deleted": 5}
        count = await service.delete_document("doc-1")
        assert count == 5

    async def test_calls_refresh(self, service, mock_es_client):
        mock_es_client.delete_by_query.return_value = {"deleted": 1}
        await service.delete_document("doc-1")
        mock_es_client.indices.refresh.assert_called_once()


class TestClose:
    async def test_closes_client(self, service, mock_es_client):
        await service.close()
        mock_es_client.close.assert_called_once()

    async def test_close_when_no_client(self):
        svc = ElasticsearchService()
        svc._client = None
        await svc.close()  # Should not raise
