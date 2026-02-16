"""Tests for app.services.similarity — centroid, cosine similarity, orchestration."""

import math

import pytest
from unittest.mock import AsyncMock, patch

from app.services.similarity import (
    compute_centroid,
    cosine_similarity,
    compute_document_similarity,
)


class TestComputeCentroid:
    def test_single_vector(self):
        result = compute_centroid([[1.0, 2.0, 3.0]])
        assert result == [1.0, 2.0, 3.0]

    def test_two_vectors_average(self):
        result = compute_centroid([[1.0, 0.0], [3.0, 4.0]])
        assert result == [2.0, 2.0]

    def test_empty(self):
        result = compute_centroid([])
        assert result == []


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert cosine_similarity(a, b) == 0.0


class TestComputeDocumentSimilarity:
    @pytest.fixture
    def mock_es(self):
        with patch("app.services.similarity.es_service") as mock:
            mock.get_all_embeddings_by_document = AsyncMock()
            mock.list_documents = AsyncMock()
            yield mock

    async def test_two_docs(self, mock_es):
        mock_es.get_all_embeddings_by_document.return_value = {
            "doc-1": [[1.0, 0.0, 0.0]],
            "doc-2": [[0.9, 0.1, 0.0]],
        }
        mock_es.list_documents.return_value = [
            {"document_id": "doc-1", "filename": "a.txt", "source_type": "text"},
            {"document_id": "doc-2", "filename": "b.pdf", "source_type": "pdf"},
        ]

        result = await compute_document_similarity(threshold=0.0)
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        assert result["edges"][0]["similarity"] > 0.9

    async def test_threshold_filtering(self, mock_es):
        mock_es.get_all_embeddings_by_document.return_value = {
            "doc-1": [[1.0, 0.0]],
            "doc-2": [[0.0, 1.0]],
        }
        mock_es.list_documents.return_value = [
            {"document_id": "doc-1", "filename": "a.txt", "source_type": "text"},
            {"document_id": "doc-2", "filename": "b.txt", "source_type": "text"},
        ]

        result = await compute_document_similarity(threshold=0.5)
        assert len(result["edges"]) == 0

    async def test_empty_index(self, mock_es):
        mock_es.get_all_embeddings_by_document.return_value = {}

        result = await compute_document_similarity(threshold=0.3)
        assert result["nodes"] == []
        assert result["edges"] == []

    async def test_single_doc(self, mock_es):
        mock_es.get_all_embeddings_by_document.return_value = {
            "doc-1": [[1.0, 0.0]],
        }
        mock_es.list_documents.return_value = [
            {"document_id": "doc-1", "filename": "a.txt", "source_type": "text"},
        ]

        result = await compute_document_similarity(threshold=0.0)
        assert len(result["nodes"]) == 1
        assert len(result["edges"]) == 0
