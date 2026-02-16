"""Tests for app.services.parsers.web — web page fetching and extraction."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.parsers.web import parse_url


SAMPLE_HTML = """
<html>
<head><title>Test Page</title></head>
<body>
<p>Main content paragraph.</p>
<p>Second paragraph with details.</p>
</body>
</html>
"""

HTML_WITH_SCRIPTS = """
<html>
<head><title>Scripted</title></head>
<body>
<script>alert('x')</script>
<style>.hide{display:none}</style>
<nav>Nav content</nav>
<p>Real content here.</p>
<footer>Footer stuff</footer>
</body>
</html>
"""


def _mock_response(text, status_code=200):
    resp = MagicMock()
    resp.text = text
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


@pytest.fixture
def mock_httpx_get():
    """Patch httpx.AsyncClient to return controlled responses."""
    with patch("app.services.parsers.web.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        yield instance


class TestParseUrl:
    async def test_successful_trafilatura_extraction(self, mock_httpx_get):
        mock_httpx_get.get.return_value = _mock_response(SAMPLE_HTML)

        with patch("app.services.parsers.web.trafilatura.extract", return_value="Extracted content"):
            result = await parse_url("https://example.com/page")

        assert result["content"] == "Extracted content"
        assert result["metadata"]["url"] == "https://example.com/page"
        assert result["metadata"]["source_type"] == "web"
        assert result["metadata"]["title"] == "Test Page"

    async def test_trafilatura_none_falls_back_to_beautifulsoup(self, mock_httpx_get):
        mock_httpx_get.get.return_value = _mock_response(SAMPLE_HTML)

        with patch("app.services.parsers.web.trafilatura.extract", return_value=None):
            result = await parse_url("https://example.com/page")

        assert "Main content paragraph" in result["content"]
        assert result["metadata"]["title"] == "Test Page"

    async def test_script_style_stripped_in_fallback(self, mock_httpx_get):
        mock_httpx_get.get.return_value = _mock_response(HTML_WITH_SCRIPTS)

        with patch("app.services.parsers.web.trafilatura.extract", return_value=None):
            result = await parse_url("https://example.com/scripted")

        assert "alert" not in result["content"]
        assert "display:none" not in result["content"]
        assert "Real content here" in result["content"]

    async def test_http_error_propagated(self, mock_httpx_get):
        mock_httpx_get.get.return_value = _mock_response("", status_code=500)

        with pytest.raises(Exception, match="HTTP 500"):
            await parse_url("https://example.com/error")

    async def test_timeout_propagated(self, mock_httpx_get):
        import httpx as httpx_mod
        mock_httpx_get.get.side_effect = httpx_mod.TimeoutException("timed out")

        with pytest.raises(httpx_mod.TimeoutException):
            await parse_url("https://example.com/slow")

    async def test_metadata_has_extracted_at(self, mock_httpx_get):
        mock_httpx_get.get.return_value = _mock_response(SAMPLE_HTML)

        with patch("app.services.parsers.web.trafilatura.extract", return_value="Content"):
            result = await parse_url("https://example.com")

        assert "extracted_at" in result["metadata"]

    async def test_filename_is_url(self, mock_httpx_get):
        mock_httpx_get.get.return_value = _mock_response(SAMPLE_HTML)

        with patch("app.services.parsers.web.trafilatura.extract", return_value="Content"):
            result = await parse_url("https://example.com/my-page")

        assert result["metadata"]["filename"] == "https://example.com/my-page"

    async def test_empty_content_returns_empty_string(self, mock_httpx_get):
        empty_html = "<html><head><title>Empty</title></head><body></body></html>"
        mock_httpx_get.get.return_value = _mock_response(empty_html)

        with patch("app.services.parsers.web.trafilatura.extract", return_value=None):
            result = await parse_url("https://example.com/empty")

        # Content could be empty string or whitespace-only
        assert isinstance(result["content"], str)
