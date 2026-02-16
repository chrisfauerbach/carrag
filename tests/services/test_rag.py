"""Tests for app.services.rag — RAG pipeline orchestration."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.rag import query_rag, SYSTEM_PROMPT


FAKE_VECTOR = [0.1] * 768
FAKE_CHUNKS = [
    {
        "content": "First chunk content.",
        "score": 0.95,
        "metadata": {"filename": "doc.txt", "source_type": "text"},
        "document_id": "d1",
        "chunk_index": 0,
    },
    {
        "content": "Second chunk content.",
        "score": 0.88,
        "metadata": {"filename": "doc.txt", "source_type": "text"},
        "document_id": "d1",
        "chunk_index": 1,
    },
]


@pytest.fixture
def mock_services():
    """Patch embedding_service and es_service used by rag module."""
    with (
        patch("app.services.rag.embedding_service") as mock_embed,
        patch("app.services.rag.es_service") as mock_es,
    ):
        mock_embed.embed_single = AsyncMock(return_value=FAKE_VECTOR)
        mock_es.knn_search = AsyncMock(return_value=FAKE_CHUNKS)
        yield mock_embed, mock_es


@pytest.fixture
def mock_ollama_generate():
    """Patch httpx.AsyncClient used for Ollama generation."""
    with patch("app.services.rag.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Generated answer."}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        yield mock_client


class TestQueryRag:
    async def test_full_pipeline(self, mock_services, mock_ollama_generate):
        mock_embed, mock_es = mock_services
        result = await query_rag("What is X?")

        assert result["answer"] == "Generated answer."
        assert len(result["sources"]) == 2
        assert result["model"] == "llama3.2"
        assert "duration_ms" in result
        mock_embed.embed_single.assert_called_once_with("What is X?")
        mock_es.knn_search.assert_called_once()

    async def test_prompt_contains_context(self, mock_services, mock_ollama_generate):
        await query_rag("What is X?")
        call_json = mock_ollama_generate.post.call_args[1]["json"]
        prompt = call_json["prompt"]
        assert "[Source 1: doc.txt]" in prompt
        assert "First chunk content." in prompt
        assert "[Source 2: doc.txt]" in prompt
        assert call_json["system"] == SYSTEM_PROMPT

    async def test_history_included_in_prompt(self, mock_services, mock_ollama_generate):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        await query_rag("Follow up?", history=history)
        call_json = mock_ollama_generate.post.call_args[1]["json"]
        prompt = call_json["prompt"]
        assert "Conversation history:" in prompt
        assert "User: Hello" in prompt
        assert "Assistant: Hi there" in prompt

    async def test_empty_history_no_history_block(self, mock_services, mock_ollama_generate):
        await query_rag("Question?", history=[])
        call_json = mock_ollama_generate.post.call_args[1]["json"]
        prompt = call_json["prompt"]
        assert "Conversation history:" not in prompt

    async def test_none_history_no_history_block(self, mock_services, mock_ollama_generate):
        await query_rag("Question?", history=None)
        call_json = mock_ollama_generate.post.call_args[1]["json"]
        prompt = call_json["prompt"]
        assert "Conversation history:" not in prompt

    async def test_custom_model_passed_through(self, mock_services, mock_ollama_generate):
        result = await query_rag("Q?", model="custom-model")
        call_json = mock_ollama_generate.post.call_args[1]["json"]
        assert call_json["model"] == "custom-model"
        assert result["model"] == "custom-model"

    async def test_custom_top_k(self, mock_services, mock_ollama_generate):
        mock_embed, mock_es = mock_services
        await query_rag("Q?", top_k=10)
        mock_es.knn_search.assert_called_once_with(FAKE_VECTOR, top_k=10)

    async def test_sources_format(self, mock_services, mock_ollama_generate):
        result = await query_rag("Q?")
        for source in result["sources"]:
            assert "content" in source
            assert "score" in source
            assert "metadata" in source

    async def test_ollama_error_propagates(self, mock_services):
        with patch("app.services.rag.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = Exception("Ollama down")
            mock_client.post.return_value = mock_resp

            with pytest.raises(Exception, match="Ollama down"):
                await query_rag("Q?")

    async def test_history_with_pydantic_objects(self, mock_services, mock_ollama_generate):
        """History items can be Pydantic models with .role/.content attrs."""
        from app.models.schemas import ChatMessage
        history = [
            ChatMessage(role="user", content="First message"),
            ChatMessage(role="assistant", content="Response"),
        ]
        await query_rag("Follow up?", history=history)
        call_json = mock_ollama_generate.post.call_args[1]["json"]
        prompt = call_json["prompt"]
        assert "User: First message" in prompt

    async def test_duration_ms_is_positive(self, mock_services, mock_ollama_generate):
        result = await query_rag("Q?")
        assert result["duration_ms"] >= 0

    async def test_default_model_from_settings(self, mock_services, mock_ollama_generate):
        result = await query_rag("Q?")
        assert result["model"] == "llama3.2"
