"""Tests for app.services.reranker — RerankerService."""

import pytest
from unittest.mock import patch, MagicMock


FAKE_PASSAGES = [
    {"content": "First passage.", "score": 0.9, "metadata": {"filename": "a.txt"}, "document_id": "d1", "chunk_index": 0},
    {"content": "Second passage.", "score": 0.8, "metadata": {"filename": "a.txt"}, "document_id": "d1", "chunk_index": 1},
    {"content": "Third passage.", "score": 0.7, "metadata": {"filename": "b.txt"}, "document_id": "d2", "chunk_index": 0},
    {"content": "Fourth passage.", "score": 0.6, "metadata": {"filename": "b.txt"}, "document_id": "d2", "chunk_index": 1},
]


class TestRerankerInit:
    def test_init_enabled(self):
        with patch("app.services.reranker.settings") as mock_settings:
            mock_settings.rerank_enabled = True
            mock_settings.rerank_model = "test-model"

            mock_ranker = MagicMock()
            with patch("app.services.reranker.Ranker", create=True) as MockRanker:
                # Patch the import inside init
                import app.services.reranker as mod
                with patch.dict("sys.modules", {"flashrank": MagicMock(Ranker=MockRanker)}):
                    from app.services.reranker import RerankerService
                    svc = RerankerService()
                    svc.init()
                    assert svc.enabled is True

    def test_init_disabled(self):
        with patch("app.services.reranker.settings") as mock_settings:
            mock_settings.rerank_enabled = False

            from app.services.reranker import RerankerService
            svc = RerankerService()
            svc.init()
            assert svc.enabled is False

    def test_init_import_failure(self):
        with patch("app.services.reranker.settings") as mock_settings:
            mock_settings.rerank_enabled = True
            mock_settings.rerank_model = "test-model"

            from app.services.reranker import RerankerService
            svc = RerankerService()
            # flashrank not installed — import will fail
            with patch.dict("sys.modules", {"flashrank": None}):
                svc.init()
            assert svc.enabled is False


class TestRerankerRerank:
    def test_returns_top_k_when_disabled(self):
        from app.services.reranker import RerankerService
        svc = RerankerService()
        # Not initialized — disabled by default
        result = svc.rerank("query", FAKE_PASSAGES, top_k=2)
        assert len(result) == 2
        assert result[0]["content"] == "First passage."
        assert result[1]["content"] == "Second passage."

    def test_empty_passages(self):
        from app.services.reranker import RerankerService
        svc = RerankerService()
        result = svc.rerank("query", [], top_k=5)
        assert result == []

    def test_rerank_with_mock_ranker(self):
        from app.services.reranker import RerankerService
        svc = RerankerService()
        svc._enabled = True

        # Mock ranker that reverses the order
        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": 2, "text": "Third passage.", "score": 0.99, "meta": FAKE_PASSAGES[2]},
            {"id": 0, "text": "First passage.", "score": 0.95, "meta": FAKE_PASSAGES[0]},
            {"id": 3, "text": "Fourth passage.", "score": 0.90, "meta": FAKE_PASSAGES[3]},
            {"id": 1, "text": "Second passage.", "score": 0.80, "meta": FAKE_PASSAGES[1]},
        ]
        svc._ranker = mock_ranker

        mock_rerank_request = MagicMock()
        with patch("app.services.reranker.RerankRequest", create=True) as MockRR:
            import sys
            mock_flashrank = MagicMock()
            mock_flashrank.RerankRequest = MockRR
            with patch.dict(sys.modules, {"flashrank": mock_flashrank}):
                result = svc.rerank("query", FAKE_PASSAGES, top_k=2)

        assert len(result) == 2
        assert result[0]["content"] == "Third passage."
        assert result[1]["content"] == "First passage."
        assert "rerank_score" in result[0]

    def test_fallback_on_error(self):
        from app.services.reranker import RerankerService
        svc = RerankerService()
        svc._enabled = True
        svc._ranker = MagicMock()

        # Make the import inside rerank raise an error
        import sys
        with patch.dict(sys.modules, {"flashrank": None}):
            result = svc.rerank("query", FAKE_PASSAGES, top_k=2)

        assert len(result) == 2
        assert result[0]["content"] == "First passage."
