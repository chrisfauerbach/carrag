import fitz


def parse_pdf(file_bytes: bytes, filename: str) -> dict:
    """Extract text from a PDF file.

    Returns dict with 'content' (full text) and 'metadata'.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)

    content = "\n\n".join(pages)
    metadata = {
        "filename": filename,
        "source_type": "pdf",
        "total_pages": len(doc),
        "pages_with_text": len(pages),
    }
    doc.close()
    return {"content": content, "metadata": metadata}
