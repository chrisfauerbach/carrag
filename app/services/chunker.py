from app.config import settings


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """Split text into overlapping chunks using recursive character splitting.

    Splits on paragraphs first, then sentences, then words.
    Returns a list of dicts with 'text', 'document_id', 'chunk_index', 'char_start', 'char_end'.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    separators = ["\n\n", "\n", ". ", " "]

    pieces = _recursive_split(text, separators, chunk_size)

    chunks = []
    char_offset = 0
    for i, piece in enumerate(pieces):
        chunks.append({
            "text": piece,
            "document_id": document_id,
            "chunk_index": i,
            "char_start": char_offset,
            "char_end": char_offset + len(piece),
        })
        char_offset += len(piece) - chunk_overlap
        if char_offset < 0:
            char_offset = 0

    return chunks


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively split text, trying each separator in order."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if len(separators) > 1 else []

    if sep:
        parts = text.split(sep)
    else:
        # Last resort: hard split by character
        result = []
        for i in range(0, len(text), chunk_size):
            segment = text[i:i + chunk_size]
            if segment.strip():
                result.append(segment)
        return result

    chunks = []
    current = ""

    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            if len(part) > chunk_size:
                # This part is too big even alone — split further
                chunks.extend(_recursive_split(part, remaining_seps, chunk_size))
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current)

    return chunks
