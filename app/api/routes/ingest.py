import uuid
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.schemas import IngestURLRequest, IngestResponse
from app.services.parsers.pdf import parse_pdf
from app.services.parsers.text import parse_text
from app.services.parsers.web import parse_url
from app.services.chunker import chunk_text
from app.services.embeddings import embedding_service
from app.services.elasticsearch import es_service

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".text", ".markdown"}


@router.post("/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a PDF or text file."""
    filename = file.filename or "unknown"
    ext = _get_extension(filename)

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "Empty file")

    # Parse
    if ext == ".pdf":
        parsed = parse_pdf(file_bytes, filename)
    else:
        parsed = parse_text(file_bytes, filename)

    if not parsed["content"].strip():
        raise HTTPException(400, "No text content could be extracted from the file")

    # Chunk, embed, store
    document_id = str(uuid.uuid4())
    return await _ingest_content(document_id, parsed["content"], parsed["metadata"])


@router.post("/url", response_model=IngestResponse)
async def ingest_url(request: IngestURLRequest):
    """Ingest content from a web URL."""
    try:
        parsed = await parse_url(request.url)
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch URL: {e}")

    if not parsed["content"].strip():
        raise HTTPException(400, "No text content could be extracted from the URL")

    document_id = str(uuid.uuid4())
    return await _ingest_content(document_id, parsed["content"], parsed["metadata"])


async def _ingest_content(document_id: str, content: str, metadata: dict) -> IngestResponse:
    """Shared logic: chunk -> embed -> index."""
    chunks = chunk_text(content, document_id)
    logger.info(f"Document {document_id}: {len(chunks)} chunks created")

    # Embed in batches of 32
    # Use 'search_document:' prefix (required by nomic-embed-text) and
    # prepend source filename so the embedding captures document context.
    source_label = metadata.get("filename", "unknown")
    doc_prefix = f"search_document: {source_label}\n\n"
    all_embeddings = []
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch_texts = [c["text"] for c in chunks[i:i + batch_size]]
        batch_embeddings = await embedding_service.embed(batch_texts, prefix=doc_prefix)
        all_embeddings.extend(batch_embeddings)

    # Store in Elasticsearch
    indexed = await es_service.index_chunks(chunks, all_embeddings, metadata)
    logger.info(f"Document {document_id}: {indexed} chunks indexed")

    return IngestResponse(
        document_id=document_id,
        filename=metadata.get("filename", "unknown"),
        chunk_count=len(chunks),
    )


def _get_extension(filename: str) -> str:
    """Extract lowercase file extension."""
    dot_idx = filename.rfind(".")
    if dot_idx == -1:
        return ""
    return filename[dot_idx:].lower()
