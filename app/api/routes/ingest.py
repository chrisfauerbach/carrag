import asyncio
import uuid
import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.models.schemas import IngestURLRequest, JobResponse
from app.services.parsers.pdf import parse_pdf
from app.services.parsers.text import parse_text
from app.services.elasticsearch import es_service
from app.services.jobs import job_service
from app.services.ingest_pipeline import run_ingest_pipeline, run_url_ingest_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".text", ".markdown"}


@router.post("/file", response_model=JobResponse)
async def ingest_file(file: UploadFile = File(...), tags: str = Form("")):
    """Upload and ingest a PDF or text file.

    Parses the file eagerly for validation, then runs the rest
    (tagging, embedding, indexing) in the background. Returns immediately
    with a job ID that can be polled for progress.
    """
    filename = file.filename or "unknown"
    ext = _get_extension(filename)

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "Empty file")

    # Parse eagerly so validation errors return immediately
    if ext == ".pdf":
        parsed = parse_pdf(file_bytes, filename)
    else:
        parsed = parse_text(file_bytes, filename)

    if not parsed["content"].strip():
        raise HTTPException(400, "No text content could be extracted from the file")

    parsed_tags = [t.strip() for t in tags.split(",") if t.strip()]
    source_type = parsed["metadata"].get("source_type", "unknown")

    # Check for existing document with same filename + source_type
    existing_id = None
    if filename != "unknown":
        existing_id = await es_service.find_document_by_source(filename, source_type)

    if existing_id:
        document_id = existing_id
        await es_service.delete_document(document_id)
        logger.info(f"Replacing existing document {document_id} ({filename})")
    else:
        document_id = str(uuid.uuid4())

    # Create job and launch background pipeline
    job = job_service.create_job(filename=filename, source_type=source_type)
    job.document_id = document_id
    asyncio.create_task(
        run_ingest_pipeline(job, parsed["content"], parsed["metadata"], parsed_tags, document_id)
    )

    return JobResponse(job_id=job.job_id, filename=filename, status="queued")


@router.post("/url", response_model=JobResponse)
async def ingest_url(request: IngestURLRequest):
    """Ingest content from a web URL.

    Returns immediately with a job ID. Fetching, parsing, tagging,
    embedding, and indexing all run in the background.
    """
    job = job_service.create_job(filename=request.url, source_type="web")
    asyncio.create_task(
        run_url_ingest_pipeline(job, request.url, request.tags or [])
    )

    return JobResponse(job_id=job.job_id, filename=request.url, status="queued")


def _get_extension(filename: str) -> str:
    """Extract lowercase file extension."""
    dot_idx = filename.rfind(".")
    if dot_idx == -1:
        return ""
    return filename[dot_idx:].lower()
