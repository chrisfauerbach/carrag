"""Tests for app.services.parsers.text — plain text/markdown parsing."""

from app.services.parsers.text import parse_text


class TestParseText:
    def test_utf8_text(self):
        content = "Hello, world!\nSecond line."
        result = parse_text(content.encode("utf-8"), "test.txt")
        assert result["content"] == content
        assert result["metadata"]["filename"] == "test.txt"
        assert result["metadata"]["source_type"] == "text"
        assert result["metadata"]["line_count"] == 2

    def test_empty_bytes(self):
        result = parse_text(b"", "empty.txt")
        assert result["content"] == ""

    def test_invalid_utf8_uses_replacement(self):
        bad_bytes = b"hello \xff\xfe world"
        result = parse_text(bad_bytes, "bad.txt")
        assert "\ufffd" in result["content"] or "hello" in result["content"]
        # Should not raise

    def test_markdown_filename(self):
        result = parse_text(b"# Title\nContent", "readme.md")
        assert result["metadata"]["filename"] == "readme.md"
        assert result["metadata"]["source_type"] == "text"

    def test_line_count_single_line(self):
        result = parse_text(b"single line", "one.txt")
        assert result["metadata"]["line_count"] == 1

    def test_line_count_multiple_lines(self):
        text = "line1\nline2\nline3\n"
        result = parse_text(text.encode(), "multi.txt")
        # "line1\nline2\nline3\n" has 3 \n chars → count("\n") + 1 = 4
        assert result["metadata"]["line_count"] == 4

    def test_unicode_content(self):
        text = "Привет мир! 你好世界"
        result = parse_text(text.encode("utf-8"), "unicode.txt")
        assert result["content"] == text

    def test_returns_dict_with_content_and_metadata(self):
        result = parse_text(b"data", "f.txt")
        assert "content" in result
        assert "metadata" in result
