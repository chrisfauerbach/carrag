def parse_text(file_bytes: bytes, filename: str) -> dict:
    """Extract text from a plain text or markdown file.

    Returns dict with 'content' and 'metadata'.
    """
    content = file_bytes.decode("utf-8", errors="replace")
    line_count = content.count("\n") + 1
    metadata = {
        "filename": filename,
        "source_type": "text",
        "line_count": line_count,
    }
    return {"content": content, "metadata": metadata}
