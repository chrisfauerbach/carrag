"""Background ingestion pipeline that runs as an asyncio task.

Updates job status at each stage, checks for cancellation between stages,
and routes Ollama calls through the priority semaphore.
"""

import asyncio
import logging
import time
import uuid

from app.services.jobs import Job, job_service
from app.services.chunker import chunk_text
from app.services.embeddings import embedding_service
from app.services.elasticsearch import es_service
from app.services.ollama_semaphore import ollama_semaphore, Priority
from app.services.rag import generate_tags
from app.services.parsers.web import parse_url
from app.services.metrics import metrics_service

logger = logging.getLogger(__name__)


async def run_ingest_pipeline(
    job: Job,
    content: str,
    metadata: dict,
    tags: list[str],
    document_id: str,
):
    """Run the full ingestion pipeline in the background.

    Stages: tagging → embedding → indexing → completed.
    Parsing already happened in the route handler for immediate validation.
    """
    try:
        start = time.time()
        resolved_tags = list(tags)

        # --- Tagging ---
        job.set_stage("tagging")
        job.check_cancelled()
        auto_tags = await generate_tags(content, filename=metadata.get("filename", ""))
        resolved_tags = list(set(resolved_tags + auto_tags))
        metadata["tags"] = resolved_tags

        # --- Chunking (CPU-only, fast) ---
        job.check_cancelled()
        chunks = chunk_text(content, document_id)
        job.total_chunks = len(chunks)
        logger.info(f"Job {job.job_id}: {len(chunks)} chunks created")

        # --- Embedding ---
        job.set_stage("embedding")
        source_label = metadata.get("filename", "unknown")
        doc_prefix = f"search_document: {source_label}\n\n"
        all_embeddings = []
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            job.check_cancelled()
            batch_texts = [c["text"] for c in chunks[i:i + batch_size]]
            batch_embeddings = await ollama_semaphore.execute(
                Priority.EMBEDDING, embedding_service.embed, batch_texts, prefix=doc_prefix
            )
            all_embeddings.extend(batch_embeddings)
            job.embedded_chunks = len(all_embeddings)

        # --- Indexing ---
        job.set_stage("indexing")
        job.check_cancelled()
        indexed = await es_service.index_chunks(chunks, all_embeddings, metadata, tags=resolved_tags)
        logger.info(f"Job {job.job_id}: {indexed} chunks indexed")

        duration_ms = (time.time() - start) * 1000
        metrics_service.record_background(
            "ingest",
            "",
            duration_ms=round(duration_ms, 1),
            document_id=document_id,
            metadata={
                "filename": metadata.get("filename", "unknown"),
                "chunk_count": len(chunks),
                "content_length": len(content),
                "tags": resolved_tags,
            },
        )

        job.complete(document_id=document_id, chunk_count=len(chunks), tags=resolved_tags)
        await job_service.finish_job(job)
        logger.info(f"Job {job.job_id} completed: {len(chunks)} chunks")

    except asyncio.CancelledError:
        if job.status != "cancelled":
            job.cancel()
        await job_service.finish_job(job)
        logger.info(f"Job {job.job_id} cancelled")
    except Exception as e:
        job.fail(str(e))
        await job_service.finish_job(job)
        logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)


async def run_url_ingest_pipeline(
    job: Job,
    url: str,
    tags: list[str],
):
    """Run URL ingestion pipeline — includes fetch/parse stage."""
    try:
        # --- Parsing ---
        job.set_stage("parsing")
        parsed = await parse_url(url)

        if not parsed["content"].strip():
            job.fail("No text content could be extracted from the URL")
            await job_service.finish_job(job)
            return

        content = parsed["content"]
        metadata = parsed["metadata"]

        # Check for existing document with same URL
        source_url = metadata.get("filename", url)
        existing_id = await es_service.find_document_by_source(source_url, "web")

        if existing_id:
            document_id = existing_id
            await es_service.delete_document(document_id)
            logger.info(f"Replacing existing document {document_id} ({source_url})")
        else:
            document_id = str(uuid.uuid4())

        await run_ingest_pipeline(job, content, metadata, tags, document_id)

    except asyncio.CancelledError:
        if job.status != "cancelled":
            job.cancel()
            await job_service.finish_job(job)
        logger.info(f"Job {job.job_id} cancelled")
    except Exception as e:
        job.fail(str(e))
        await job_service.finish_job(job)
        logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
