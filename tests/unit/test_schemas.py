"""Tests for app.models.schemas — Pydantic validation."""

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    ChatMessage,
    QueryRequest,
    QueryResponse,
    IngestResponse,
    IngestURLRequest,
    SourceChunk,
    DocumentInfo,
    DocumentListResponse,
    DocumentDetailResponse,
    DocumentDeleteResponse,
)


class TestQueryRequest:
    def test_defaults(self):
        req = QueryRequest(question="What is X?")
        assert req.top_k == 5
        assert req.history == []
        assert req.return_sources is True
        assert req.model is None

    def test_custom_values(self):
        req = QueryRequest(question="Q?", top_k=10, model="custom", return_sources=False)
        assert req.top_k == 10
        assert req.model == "custom"
        assert req.return_sources is False

    def test_top_k_min_violation(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Q?", top_k=0)

    def test_top_k_max_violation(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Q?", top_k=21)

    def test_top_k_boundary_valid(self):
        req1 = QueryRequest(question="Q?", top_k=1)
        assert req1.top_k == 1
        req20 = QueryRequest(question="Q?", top_k=20)
        assert req20.top_k == 20

    def test_missing_question(self):
        with pytest.raises(ValidationError):
            QueryRequest()

    def test_history_with_messages(self):
        req = QueryRequest(
            question="Follow-up?",
            history=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        )
        assert len(req.history) == 2
        assert req.history[0].role == "user"


class TestChatMessage:
    def test_valid(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_missing_role(self):
        with pytest.raises(ValidationError):
            ChatMessage(content="Hello")

    def test_missing_content(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user")


class TestIngestResponse:
    def test_defaults(self):
        resp = IngestResponse(document_id="abc", filename="test.txt", chunk_count=5)
        assert resp.status == "ingested"

    def test_custom_status(self):
        resp = IngestResponse(document_id="abc", filename="test.txt", chunk_count=5, status="custom")
        assert resp.status == "custom"


class TestQueryResponse:
    def test_defaults(self):
        resp = QueryResponse(answer="Yes", model="llama3.2", duration_ms=100.0)
        assert resp.sources == []

    def test_with_sources(self):
        resp = QueryResponse(
            answer="Yes",
            model="llama3.2",
            duration_ms=100.0,
            sources=[SourceChunk(content="chunk", score=0.9, metadata={})],
        )
        assert len(resp.sources) == 1


class TestDocumentModels:
    def test_document_info_optional_datetime(self):
        info = DocumentInfo(
            document_id="d1", filename="f.txt", source_type="text", chunk_count=3
        )
        assert info.created_at is None

    def test_document_info_with_datetime(self):
        info = DocumentInfo(
            document_id="d1",
            filename="f.txt",
            source_type="text",
            chunk_count=3,
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert info.created_at is not None

    def test_document_delete_response_defaults(self):
        resp = DocumentDeleteResponse(document_id="d1", chunks_deleted=5)
        assert resp.status == "deleted"

    def test_document_detail_response_optional_datetime(self):
        resp = DocumentDetailResponse(
            document_id="d1",
            filename="f.txt",
            source_type="text",
            chunk_count=3,
            metadata={},
        )
        assert resp.created_at is None


class TestIngestURLRequest:
    def test_valid(self):
        req = IngestURLRequest(url="https://example.com")
        assert req.url == "https://example.com"

    def test_missing_url(self):
        with pytest.raises(ValidationError):
            IngestURLRequest()
