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


class TestHybridSearch:
    def _make_hit(self, _id, content="text", doc_id="d1", chunk_index=0, score=1.0):
        return {
            "_id": _id,
            "_score": score,
            "_source": {
                "content": content,
                "document_id": doc_id,
                "chunk_index": chunk_index,
                "metadata": {"filename": "test.txt"},
                "created_at": "2026-01-01T00:00:00",
            },
        }

    async def test_returns_formatted_results(self, service, mock_es_client):
        bm25_resp = {"hits": {"hits": [self._make_hit("id1", content="relevant text")]}}
        knn_resp = {"hits": {"hits": [self._make_hit("id1", content="relevant text")]}}
        mock_es_client.search.side_effect = [bm25_resp, knn_resp]

        results = await service.hybrid_search([0.1] * 768, "test query", top_k=5)
        assert len(results) == 1
        assert results[0]["content"] == "relevant text"
        assert results[0]["document_id"] == "d1"
        assert results[0]["metadata"]["filename"] == "test.txt"
        # Appears rank 1 in both lists: 1/(60+1) + 1/(60+1)
        expected_score = 2.0 / 61
        assert abs(results[0]["score"] - expected_score) < 1e-9

    async def test_empty_results(self, service, mock_es_client):
        mock_es_client.search.side_effect = [
            {"hits": {"hits": []}},
            {"hits": {"hits": []}},
        ]
        results = await service.hybrid_search([0.1] * 768, "test query")
        assert results == []

    async def test_passes_top_k_to_both_queries(self, service, mock_es_client):
        mock_es_client.search.side_effect = [
            {"hits": {"hits": []}},
            {"hits": {"hits": []}},
        ]
        await service.hybrid_search([0.1] * 768, "test query", top_k=10)
        assert mock_es_client.search.call_count == 2
        bm25_kwargs = mock_es_client.search.call_args_list[0][1]
        knn_kwargs = mock_es_client.search.call_args_list[1][1]
        assert bm25_kwargs["size"] == 10
        assert knn_kwargs["size"] == 10

    async def test_sends_separate_bm25_and_knn_queries(self, service, mock_es_client):
        mock_es_client.search.side_effect = [
            {"hits": {"hits": []}},
            {"hits": {"hits": []}},
        ]
        await service.hybrid_search([0.1] * 768, "test query", top_k=5)
        assert mock_es_client.search.call_count == 2

        bm25_body = mock_es_client.search.call_args_list[0][1]["body"]
        assert bm25_body["query"]["match"]["content"]["query"] == "test query"
        assert "knn" not in bm25_body
        assert "rank" not in bm25_body

        knn_body = mock_es_client.search.call_args_list[1][1]["body"]
        assert knn_body["knn"]["field"] == "embedding"
        assert "rank" not in knn_body

    async def test_rrf_fuses_overlapping_results(self, service, mock_es_client):
        """Docs appearing in both lists get higher RRF scores than docs in only one."""
        bm25_resp = {
            "hits": {
                "hits": [
                    self._make_hit("shared", content="shared doc", chunk_index=0),
                    self._make_hit("bm25_only", content="bm25 only", chunk_index=1),
                ]
            }
        }
        knn_resp = {
            "hits": {
                "hits": [
                    self._make_hit("shared", content="shared doc", chunk_index=0),
                    self._make_hit("knn_only", content="knn only", chunk_index=2),
                ]
            }
        }
        mock_es_client.search.side_effect = [bm25_resp, knn_resp]

        results = await service.hybrid_search([0.1] * 768, "test query", top_k=3)

        assert len(results) == 3
        # "shared" appears in both lists at rank 1 -> score = 2/(60+1)
        assert results[0]["content"] == "shared doc"
        assert abs(results[0]["score"] - 2.0 / 61) < 1e-9
        # The other two each appear once at rank 2 -> score = 1/(60+2)
        single_scores = {r["content"] for r in results[1:]}
        assert single_scores == {"bm25 only", "knn only"}
        for r in results[1:]:
            assert abs(r["score"] - 1.0 / 62) < 1e-9


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


class TestGetAllEmbeddingsByDocument:
    async def test_groups_by_document_id(self, service, mock_es_client):
        mock_es_client.search.return_value = {
            "_scroll_id": "scroll-1",
            "hits": {
                "hits": [
                    {"_source": {"document_id": "doc-1", "embedding": [0.1, 0.2]}},
                    {"_source": {"document_id": "doc-1", "embedding": [0.3, 0.4]}},
                    {"_source": {"document_id": "doc-2", "embedding": [0.5, 0.6]}},
                ]
            },
        }
        mock_es_client.scroll.return_value = {
            "_scroll_id": "scroll-2",
            "hits": {"hits": []},
        }
        mock_es_client.clear_scroll = AsyncMock()

        result = await service.get_all_embeddings_by_document()
        assert len(result["doc-1"]) == 2
        assert len(result["doc-2"]) == 1
        assert result["doc-1"][0] == [0.1, 0.2]

    async def test_empty_index(self, service, mock_es_client):
        mock_es_client.search.return_value = {
            "_scroll_id": "scroll-1",
            "hits": {"hits": []},
        }
        mock_es_client.clear_scroll = AsyncMock()

        result = await service.get_all_embeddings_by_document()
        assert result == {}

    async def test_clears_scroll(self, service, mock_es_client):
        mock_es_client.search.return_value = {
            "_scroll_id": "scroll-1",
            "hits": {"hits": []},
        }
        mock_es_client.clear_scroll = AsyncMock()

        await service.get_all_embeddings_by_document()
        mock_es_client.clear_scroll.assert_called_once_with(scroll_id="scroll-1")

    async def test_handles_multiple_pages(self, service, mock_es_client):
        mock_es_client.search.return_value = {
            "_scroll_id": "scroll-1",
            "hits": {
                "hits": [
                    {"_source": {"document_id": "doc-1", "embedding": [0.1]}},
                ]
            },
        }
        mock_es_client.scroll.side_effect = [
            {
                "_scroll_id": "scroll-2",
                "hits": {
                    "hits": [
                        {"_source": {"document_id": "doc-2", "embedding": [0.2]}},
                    ]
                },
            },
            {
                "_scroll_id": "scroll-3",
                "hits": {"hits": []},
            },
        ]
        mock_es_client.clear_scroll = AsyncMock()

        result = await service.get_all_embeddings_by_document()
        assert "doc-1" in result
        assert "doc-2" in result


class TestClose:
    async def test_closes_client(self, service, mock_es_client):
        await service.close()
        mock_es_client.close.assert_called_once()

    async def test_close_when_no_client(self):
        svc = ElasticsearchService()
        svc._client = None
        await svc.close()  # Should not raise
