"""Tests for app.services.parsers.pdf — PDF text extraction via PyMuPDF."""

import pytest
import fitz

from app.services.parsers.pdf import parse_pdf


def _make_pdf(pages_text: list[str]) -> bytes:
    """Create a minimal PDF with the given text on each page."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestParsePdf:
    def test_single_page_pdf(self):
        pdf_bytes = _make_pdf(["Hello from PDF"])
        result = parse_pdf(pdf_bytes, "test.pdf")
        assert "Hello from PDF" in result["content"]
        assert result["metadata"]["filename"] == "test.pdf"
        assert result["metadata"]["source_type"] == "pdf"
        assert result["metadata"]["total_pages"] == 1
        assert result["metadata"]["pages_with_text"] == 1

    def test_multi_page_pdf(self):
        pdf_bytes = _make_pdf(["Page one text", "Page two text"])
        result = parse_pdf(pdf_bytes, "multi.pdf")
        assert "Page one text" in result["content"]
        assert "Page two text" in result["content"]
        assert result["metadata"]["total_pages"] == 2
        assert result["metadata"]["pages_with_text"] == 2

    def test_empty_page_pdf(self):
        pdf_bytes = _make_pdf(["Has text", ""])
        result = parse_pdf(pdf_bytes, "partial.pdf")
        assert result["metadata"]["total_pages"] == 2
        assert result["metadata"]["pages_with_text"] == 1

    def test_all_empty_pages(self):
        pdf_bytes = _make_pdf(["", ""])
        result = parse_pdf(pdf_bytes, "blank.pdf")
        assert result["metadata"]["pages_with_text"] == 0
        assert result["content"] == ""

    def test_invalid_bytes_raises(self):
        with pytest.raises(Exception):
            parse_pdf(b"not a pdf", "bad.pdf")

    def test_sample_pdf_fixture(self, sample_pdf_bytes):
        result = parse_pdf(sample_pdf_bytes, "fixture.pdf")
        assert "Hello from test PDF" in result["content"]
        assert result["metadata"]["source_type"] == "pdf"
