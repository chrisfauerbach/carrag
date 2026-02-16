"""Tests for app.services.chunker — recursive text splitting with overlap."""

import pytest

from app.services.chunker import chunk_text, _recursive_split


class TestChunkTextEmpty:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("", "doc-1", chunk_size=100, chunk_overlap=10) == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n\n  ", "doc-1", chunk_size=100, chunk_overlap=10) == []


class TestChunkTextSingleChunk:
    def test_short_text_returns_one_chunk(self):
        text = "Short text."
        chunks = chunk_text(text, "doc-1", chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0]["text"] == text

    def test_text_exactly_chunk_size(self):
        text = "x" * 50
        chunks = chunk_text(text, "doc-1", chunk_size=50, chunk_overlap=10)
        assert len(chunks) == 1


class TestChunkTextSplitting:
    def test_paragraph_splitting(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_text(text, "doc-1", chunk_size=30, chunk_overlap=0)
        assert len(chunks) >= 2
        texts = [c["text"] for c in chunks]
        assert any("Paragraph one" in t for t in texts)
        assert any("Paragraph three" in t for t in texts)

    def test_sentence_splitting(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, "doc-1", chunk_size=35, chunk_overlap=0)
        assert len(chunks) >= 2

    def test_word_splitting(self):
        text = "one two three four five six seven eight nine ten"
        chunks = chunk_text(text, "doc-1", chunk_size=20, chunk_overlap=0)
        assert len(chunks) >= 2

    def test_hard_character_split(self):
        text = "a" * 100
        chunks = chunk_text(text, "doc-1", chunk_size=30, chunk_overlap=0)
        assert len(chunks) >= 3
        for c in chunks:
            assert len(c["text"]) <= 30


class TestChunkTextOverlap:
    def test_overlap_produces_shared_content(self):
        text = "Word " * 50  # 250 chars
        chunks = chunk_text(text, "doc-1", chunk_size=60, chunk_overlap=15)
        assert len(chunks) >= 2
        # With overlap, char_start of chunk N+1 < char_end of chunk N
        for i in range(len(chunks) - 1):
            assert chunks[i + 1]["char_start"] < chunks[i]["char_end"]

    def test_zero_overlap(self):
        text = "Paragraph one content.\n\nParagraph two content.\n\nParagraph three content."
        chunks = chunk_text(text, "doc-1", chunk_size=30, chunk_overlap=0)
        assert len(chunks) >= 2


class TestChunkTextMetadata:
    def test_chunk_index_sequential(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_text(text, "doc-1", chunk_size=20, chunk_overlap=0)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_document_id_propagated(self):
        text = "Some text.\n\nMore text."
        chunks = chunk_text(text, "my-doc-id", chunk_size=15, chunk_overlap=0)
        for chunk in chunks:
            assert chunk["document_id"] == "my-doc-id"

    def test_char_start_end_correct_for_single_chunk(self):
        text = "Hello world"
        chunks = chunk_text(text, "doc-1", chunk_size=100, chunk_overlap=0)
        assert len(chunks) == 1
        assert chunks[0]["char_start"] == 0
        assert chunks[0]["char_end"] == len(text)

    def test_char_offsets_present(self):
        text = "First.\n\nSecond.\n\nThird."
        chunks = chunk_text(text, "doc-1", chunk_size=10, chunk_overlap=0)
        for chunk in chunks:
            assert "char_start" in chunk
            assert "char_end" in chunk
            assert chunk["char_end"] > chunk["char_start"]


class TestChunkTextUnicode:
    def test_unicode_text(self):
        text = "Привет мир. " * 20
        chunks = chunk_text(text, "doc-1", chunk_size=50, chunk_overlap=5)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk["text"], str)

    def test_emoji_text(self):
        text = "Hello 🌍! " * 20
        chunks = chunk_text(text, "doc-1", chunk_size=40, chunk_overlap=5)
        assert len(chunks) >= 1


class TestChunkTextParametrized:
    @pytest.mark.parametrize("size,expected_min", [
        (10, 3),
        (50, 1),
        (200, 1),
    ])
    def test_various_sizes(self, size, expected_min):
        text = "Hello world. " * 10  # ~130 chars
        chunks = chunk_text(text, "doc-1", chunk_size=size, chunk_overlap=0)
        assert len(chunks) >= expected_min

    def test_defaults_from_settings(self, sample_text):
        """chunk_text without explicit size/overlap uses settings defaults."""
        chunks = chunk_text(sample_text, "doc-1")
        assert len(chunks) >= 1
        assert chunks[0]["document_id"] == "doc-1"


class TestRecursiveSplit:
    def test_text_within_limit(self):
        result = _recursive_split("short", ["\n\n", "\n", ". ", " "], 100)
        assert result == ["short"]

    def test_empty_text(self):
        result = _recursive_split("", ["\n\n"], 10)
        assert result == []
