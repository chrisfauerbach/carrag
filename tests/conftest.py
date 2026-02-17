"""Shared fixtures for all tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from httpx import ASGITransport


EMBEDDING_DIM = 768
FAKE_VECTOR = [0.1] * EMBEDDING_DIM


@pytest.fixture
def mock_es_service():
    """AsyncMock of ElasticsearchService with all methods stubbed."""
    svc = AsyncMock()
    svc.init = AsyncMock()
    svc.index_chunks = AsyncMock(return_value=3)
    svc.hybrid_search = AsyncMock(return_value=[
        {
            "content": "chunk text 1",
            "score": 0.95,
            "metadata": {"filename": "test.txt", "source_type": "text"},
            "document_id": "doc-123",
            "chunk_index": 0,
        },
        {
            "content": "chunk text 2",
            "score": 0.88,
            "metadata": {"filename": "test.txt", "source_type": "text"},
            "document_id": "doc-123",
            "chunk_index": 1,
        },
    ])
    svc.update_document_tags = AsyncMock(return_value=3)
    svc.list_documents = AsyncMock(return_value=[
        {
            "document_id": "doc-123",
            "filename": "test.txt",
            "source_type": "text",
            "chunk_count": 3,
            "tags": [],
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ])
    svc.get_document = AsyncMock(return_value={
        "document_id": "doc-123",
        "filename": "test.txt",
        "source_type": "text",
        "chunk_count": 3,
        "tags": [],
        "metadata": {"filename": "test.txt", "source_type": "text"},
        "created_at": "2026-01-01T00:00:00+00:00",
    })
    svc.get_document_chunks = AsyncMock(return_value=[
        {"content": "chunk text 1", "chunk_index": 0, "char_start": 0, "char_end": 12},
        {"content": "chunk text 2", "chunk_index": 1, "char_start": 12, "char_end": 24},
        {"content": "chunk text 3", "chunk_index": 2, "char_start": 24, "char_end": 36},
    ])
    svc.delete_document = AsyncMock(return_value=3)
    svc.find_document_by_source = AsyncMock(return_value=None)
    svc.get_all_embeddings_by_document = AsyncMock(return_value={})
    svc.close = AsyncMock()
    return svc


@pytest.fixture
def mock_embedding_service():
    """AsyncMock of EmbeddingService — returns deterministic vectors."""
    svc = AsyncMock()

    async def _embed(texts, prefix=""):
        return [FAKE_VECTOR] * len(texts)

    async def _embed_single(text, prefix=""):
        return FAKE_VECTOR

    svc.embed = AsyncMock(side_effect=_embed)
    svc.embed_single = AsyncMock(side_effect=_embed_single)
    svc.ensure_model = AsyncMock()
    svc.close = AsyncMock()
    return svc


@pytest.fixture
def mock_metrics_service():
    """Mock MetricsService — record_background is a no-op MagicMock, record is AsyncMock."""
    svc = MagicMock()
    svc.record = AsyncMock()
    svc.record_background = MagicMock()
    svc.init_index = AsyncMock()
    svc.query = AsyncMock(return_value=[])
    return svc


@pytest.fixture
def mock_chat_service():
    """AsyncMock of ChatService with all methods stubbed."""
    svc = AsyncMock()
    svc.init_index = AsyncMock()
    svc.create_chat = AsyncMock(return_value={
        "chat_id": "chat-abc",
        "title": "New Chat",
        "messages": [],
        "message_count": 0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    })
    svc.get_chat = AsyncMock(return_value={
        "chat_id": "chat-abc",
        "title": "Test Chat",
        "messages": [
            {"role": "user", "content": "Hello", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"role": "assistant", "content": "Hi there", "timestamp": "2026-01-01T00:00:01+00:00", "model": "llama3.2", "duration_ms": 100.0, "sources": []},
        ],
        "message_count": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:01+00:00",
    })
    svc.list_chats = AsyncMock(return_value=[
        {
            "chat_id": "chat-abc",
            "title": "Test Chat",
            "message_count": 2,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:01+00:00",
        }
    ])
    svc.append_messages = AsyncMock(return_value={
        "chat_id": "chat-abc",
        "title": "Test Chat",
        "messages": [],
        "message_count": 4,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:02+00:00",
    })
    svc.rename_chat = AsyncMock(return_value={
        "chat_id": "chat-abc",
        "title": "Renamed",
        "updated_at": "2026-01-01T00:00:03+00:00",
    })
    svc.delete_chat = AsyncMock(return_value=True)
    return svc


@pytest.fixture
def mock_reranker_service():
    """Mock RerankerService — rerank passes through in original order."""
    svc = MagicMock()
    svc.enabled = False
    svc.init = MagicMock()
    svc.rerank = MagicMock(side_effect=lambda query, passages, top_k: passages[:top_k])
    return svc


@pytest.fixture
def mock_prompts_service():
    """AsyncMock of PromptsService with all methods stubbed."""
    svc = AsyncMock()
    svc.init_index = AsyncMock()
    svc.get_prompt = AsyncMock(return_value={
        "key": "rag_system",
        "name": "RAG System Prompt",
        "description": "Instructions for how the LLM answers RAG queries",
        "content": "You are a helpful assistant...",
        "variables": [],
        "default_content": "You are a helpful assistant...",
        "updated_at": "2026-01-01T00:00:00+00:00",
    })
    svc.list_prompts = AsyncMock(return_value=[
        {
            "key": "rag_system",
            "name": "RAG System Prompt",
            "description": "Instructions for how the LLM answers RAG queries",
            "content": "You are a helpful assistant...",
            "variables": [],
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
    ])
    svc.update_prompt = AsyncMock(return_value={
        "key": "rag_system",
        "content": "Updated content",
        "updated_at": "2026-01-01T00:00:01+00:00",
    })
    svc.reset_prompt = AsyncMock(return_value={
        "key": "rag_system",
        "content": "Default content",
        "updated_at": "2026-01-01T00:00:02+00:00",
    })
    return svc


@pytest.fixture
def fake_query_rag_result():
    """Dict matching query_rag() return shape."""
    return {
        "answer": "The answer is 42.",
        "sources": [
            {
                "content": "chunk text 1",
                "score": 0.95,
                "metadata": {"filename": "test.txt", "source_type": "text"},
            }
        ],
        "model": "llama3.2",
        "duration_ms": 123.4,
    }


@pytest.fixture
async def app_client(mock_es_service, mock_embedding_service, mock_chat_service, mock_metrics_service, mock_prompts_service, mock_reranker_service):
    """httpx.AsyncClient bound to the FastAPI app with patched services."""
    from contextlib import ExitStack, asynccontextmanager

    with ExitStack() as stack:
        # Patch service singletons
        for target, mock_obj in [
            ("app.services.elasticsearch.es_service", mock_es_service),
            ("app.services.embeddings.embedding_service", mock_embedding_service),
            ("app.services.metrics.metrics_service", mock_metrics_service),
            ("app.services.embeddings.metrics_service", mock_metrics_service),
            ("app.services.rag.metrics_service", mock_metrics_service),
            ("app.api.routes.ingest.metrics_service", mock_metrics_service),
            ("app.api.routes.metrics.metrics_service", mock_metrics_service),
            ("app.api.routes.ingest.es_service", mock_es_service),
            ("app.api.routes.ingest.embedding_service", mock_embedding_service),
            ("app.api.routes.documents.es_service", mock_es_service),
            ("app.services.similarity.es_service", mock_es_service),
            ("app.services.chat.chat_service", mock_chat_service),
            ("app.api.routes.chats.chat_service", mock_chat_service),
            ("app.api.routes.query.chat_service", mock_chat_service),
            ("app.services.prompts.prompts_service", mock_prompts_service),
            ("app.api.routes.prompts.prompts_service", mock_prompts_service),
            ("app.services.rag.prompts_service", mock_prompts_service),
            ("app.services.reranker.reranker_service", mock_reranker_service),
            ("app.services.rag.reranker_service", mock_reranker_service),
        ]:
            stack.enter_context(patch(target, mock_obj))

        mock_gen_tags = stack.enter_context(
            patch("app.api.routes.ingest.generate_tags", new_callable=AsyncMock)
        )
        mock_sim = stack.enter_context(
            patch("app.api.routes.documents.compute_document_similarity", new_callable=AsyncMock)
        )
        mock_rag = stack.enter_context(
            patch("app.api.routes.query.query_rag", new_callable=AsyncMock)
        )
        mock_rag_stream = stack.enter_context(
            patch("app.api.routes.query.query_rag_stream")
        )
        mock_httpx_cls = stack.enter_context(
            patch("app.api.routes.query.HttpxClient")
        )

        # Mock Ollama /api/tags response
        mock_httpx_resp = MagicMock()
        mock_httpx_resp.json.return_value = {
            "models": [
                {"name": "llama3.2", "details": {"families": ["llama"]}},
                {"name": "nomic-embed-text", "details": {"families": ["nomic-bert"]}},
            ]
        }
        mock_httpx_resp.raise_for_status = MagicMock()
        mock_httpx_instance = AsyncMock()
        mock_httpx_instance.get = AsyncMock(return_value=mock_httpx_resp)
        mock_httpx_instance.__aenter__ = AsyncMock(return_value=mock_httpx_instance)
        mock_httpx_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_httpx_instance

        mock_gen_tags.return_value = []

        mock_sim.return_value = {
            "nodes": [
                {"document_id": "doc-123", "filename": "test.txt", "source_type": "text", "chunk_count": 3},
            ],
            "edges": [],
            "threshold": 0.3,
        }

        mock_rag.return_value = {
            "answer": "The answer is 42.",
            "sources": [
                {
                    "content": "chunk text 1",
                    "score": 0.95,
                    "metadata": {"filename": "test.txt"},
                }
            ],
            "model": "llama3.2",
            "duration_ms": 123.4,
        }

        async def _fake_stream(**kwargs):
            yield {
                "type": "sources",
                "data": {"sources": [{"content": "chunk text 1", "score": 0.95, "metadata": {"filename": "test.txt"}}]},
            }
            yield {"type": "token", "data": {"token": "The"}}
            yield {"type": "token", "data": {"token": " answer"}}
            yield {"type": "done", "data": {"model": "llama3.2", "duration_ms": 123.4}}

        mock_rag_stream.side_effect = lambda **kwargs: _fake_stream(**kwargs)

        from app.main import app

        @asynccontextmanager
        async def _noop_lifespan(app):
            yield

        app.router.lifespan_context = _noop_lifespan

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            client._mock_rag = mock_rag
            client._mock_rag_stream = mock_rag_stream
            client._mock_es = mock_es_service
            client._mock_embed = mock_embedding_service
            client._mock_sim = mock_sim
            client._mock_httpx = mock_httpx_instance
            client._mock_chat = mock_chat_service
            client._mock_gen_tags = mock_gen_tags
            client._mock_metrics = mock_metrics_service
            client._mock_prompts = mock_prompts_service
            yield client


@pytest.fixture
def sample_text():
    """Multi-paragraph text for chunker tests."""
    return (
        "First paragraph with some content about testing.\n\n"
        "Second paragraph that discusses different topics. "
        "It has multiple sentences. Each one adds detail.\n\n"
        "Third paragraph wraps up the text with a conclusion."
    )


@pytest.fixture
def sample_pdf_bytes():
    """Minimal valid PDF bytes created programmatically via fitz."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello from test PDF")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes
