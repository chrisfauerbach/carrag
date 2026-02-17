"""Tests for app.services.rag — RAG pipeline orchestration."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.rag import query_rag, query_rag_stream, _prepare_rag_context, generate_tags, SYSTEM_PROMPT


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
        mock_es.hybrid_search = AsyncMock(return_value=FAKE_CHUNKS)
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


class TestGenerateTags:
    @pytest.fixture
    def mock_ollama_tags(self):
        """Patch httpx.AsyncClient for generate_tags Ollama call."""
        with patch("app.services.rag.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"response": "research, machine learning, python"}
            mock_client.post.return_value = mock_resp

            yield mock_client

    async def test_successful_generation(self, mock_ollama_tags):
        tags = await generate_tags("Some document content about ML research in Python.")
        assert tags == ["research", "machine learning", "python"]

    async def test_truncation(self, mock_ollama_tags):
        long_content = "x" * 5000
        await generate_tags(long_content)
        call_json = mock_ollama_tags.post.call_args[1]["json"]
        # The prompt should contain at most ~2000 chars of content plus the preamble
        assert "x" * 2000 in call_json["prompt"]
        assert "x" * 2001 not in call_json["prompt"]

    async def test_max_tags_limit(self, mock_ollama_tags):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": "a, b, c, d, e, f, g"}
        mock_ollama_tags.post.return_value = mock_resp

        tags = await generate_tags("content", max_tags=5)
        assert len(tags) == 5

    async def test_error_returns_empty_list(self):
        with patch("app.services.rag.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.post.side_effect = Exception("Ollama down")

            tags = await generate_tags("some content")
            assert tags == []

    async def test_normalization(self, mock_ollama_tags):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": "  Research ,  ML , , Python  "}
        mock_ollama_tags.post.return_value = mock_resp

        tags = await generate_tags("content")
        assert tags == ["research", "ml", "python"]


class TestPrepareRagContext:
    async def test_returns_prompt_sources_model(self, mock_services):
        prompt, system_prompt, sources, llm_model = await _prepare_rag_context("What is X?")

        assert "What is X?" in prompt
        assert "[Source 1: doc.txt]" in prompt
        assert system_prompt == SYSTEM_PROMPT
        assert len(sources) == 2
        assert sources[0]["content"] == "First chunk content."
        assert llm_model == "llama3.2"

    async def test_custom_model(self, mock_services):
        _, _, _, llm_model = await _prepare_rag_context("Q?", model="custom-model")
        assert llm_model == "custom-model"

    async def test_history_in_prompt(self, mock_services):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        prompt, _, _, _ = await _prepare_rag_context("Follow up?", history=history)
        assert "Conversation history:" in prompt
        assert "User: Hello" in prompt

    async def test_custom_top_k(self, mock_services):
        mock_embed, mock_es = mock_services
        await _prepare_rag_context("Q?", top_k=10)
        mock_es.hybrid_search.assert_called_once_with(FAKE_VECTOR, "Q?", top_k=10, tags=None)

    async def test_tags_forwarded_to_hybrid_search(self, mock_services):
        mock_embed, mock_es = mock_services
        await _prepare_rag_context("Q?", tags=["research", "ml"])
        mock_es.hybrid_search.assert_called_once_with(FAKE_VECTOR, "Q?", top_k=5, tags=["research", "ml"])

    async def test_no_tags_passes_none(self, mock_services):
        mock_embed, mock_es = mock_services
        await _prepare_rag_context("Q?")
        mock_es.hybrid_search.assert_called_once_with(FAKE_VECTOR, "Q?", top_k=5, tags=None)


class TestQueryRag:
    async def test_full_pipeline(self, mock_services, mock_ollama_generate):
        mock_embed, mock_es = mock_services
        result = await query_rag("What is X?")

        assert result["answer"] == "Generated answer."
        assert len(result["sources"]) == 2
        assert result["model"] == "llama3.2"
        assert "duration_ms" in result
        mock_embed.embed_single.assert_called_once_with("What is X?", prefix="search_query: ")
        mock_es.hybrid_search.assert_called_once()

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
        mock_es.hybrid_search.assert_called_once_with(FAKE_VECTOR, "Q?", top_k=10, tags=None)

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


class AsyncIteratorMock:
    """Helper to mock async line iterator for Ollama streaming."""

    def __init__(self, lines):
        self._lines = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._lines)
        except StopIteration:
            raise StopAsyncIteration


class TestQueryRagStream:
    @pytest.fixture
    def mock_ollama_stream(self):
        """Patch httpx.AsyncClient for streaming Ollama responses."""
        with patch("app.services.rag.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()

            # Default NDJSON lines simulating Ollama streaming
            mock_resp.aiter_lines.return_value = AsyncIteratorMock([
                json.dumps({"response": "The", "done": False}),
                json.dumps({"response": " answer", "done": False}),
                json.dumps({"response": "", "done": True}),
            ])

            mock_stream_ctx = MagicMock()
            mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
            # stream() is a regular method returning an async context manager
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)

            yield mock_resp

    async def test_full_event_sequence(self, mock_services, mock_ollama_stream):
        events = []
        async for event in query_rag_stream("What is X?"):
            events.append(event)

        assert events[0]["type"] == "sources"
        assert len(events[0]["data"]["sources"]) == 2

        tokens = [e for e in events if e["type"] == "token"]
        assert len(tokens) == 2
        assert tokens[0]["data"]["token"] == "The"
        assert tokens[1]["data"]["token"] == " answer"

        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["data"]["model"] == "llama3.2"
        assert "duration_ms" in done_events[0]["data"]

    async def test_empty_tokens_skipped(self, mock_services, mock_ollama_stream):
        mock_ollama_stream.aiter_lines.return_value = AsyncIteratorMock([
            json.dumps({"response": "Hello", "done": False}),
            json.dumps({"response": "", "done": False}),
            json.dumps({"response": " world", "done": False}),
            json.dumps({"response": "", "done": True}),
        ])

        events = []
        async for event in query_rag_stream("Q?"):
            events.append(event)

        tokens = [e for e in events if e["type"] == "token"]
        assert len(tokens) == 2
        assert tokens[0]["data"]["token"] == "Hello"
        assert tokens[1]["data"]["token"] == " world"

    async def test_blank_lines_skipped(self, mock_services, mock_ollama_stream):
        mock_ollama_stream.aiter_lines.return_value = AsyncIteratorMock([
            "",
            json.dumps({"response": "Hi", "done": False}),
            "   ",
            json.dumps({"response": "", "done": True}),
        ])

        events = []
        async for event in query_rag_stream("Q?"):
            events.append(event)

        tokens = [e for e in events if e["type"] == "token"]
        assert len(tokens) == 1
        assert tokens[0]["data"]["token"] == "Hi"

    async def test_error_propagates(self, mock_services):
        with patch("app.services.rag.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = Exception("Ollama stream error")
            mock_stream_ctx = MagicMock()
            mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)

            events = []
            with pytest.raises(Exception, match="Ollama stream error"):
                async for event in query_rag_stream("Q?"):
                    events.append(event)

            # Sources should have been yielded before the error
            assert len(events) == 1
            assert events[0]["type"] == "sources"
