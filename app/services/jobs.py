"""Job tracking with in-memory active jobs and ES persistence for terminal states.

Active jobs (queued/parsing/tagging/embedding/indexing) live in memory for
real-time progress and cancellation support. When a job reaches a terminal
state (completed/failed/cancelled), it's persisted to ES and removed from
the in-memory dict. Listing merges both sources.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.config import settings
from app.services.elasticsearch import es_service

logger = logging.getLogger(__name__)

JOBS_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "job_id": {"type": "keyword"},
            "filename": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "source_type": {"type": "keyword"},
            "status": {"type": "keyword"},
            "created_at": {"type": "date"},
            "started_at": {"type": "date"},
            "completed_at": {"type": "date"},
            "total_chunks": {"type": "integer"},
            "embedded_chunks": {"type": "integer"},
            "chunk_count": {"type": "integer"},
            "current_stage": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "tags": {"type": "text"},
            "error": {"type": "text"},
        }
    }
}


@dataclass
class Job:
    job_id: str
    filename: str
    source_type: str  # "pdf", "text", "web"
    status: str = "queued"  # queued → parsing → tagging → embedding → indexing → completed | failed | cancelled
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    total_chunks: int = 0
    embedded_chunks: int = 0
    current_stage: str | None = None
    # Result fields
    document_id: str | None = None
    chunk_count: int | None = None
    tags: list[str] | None = None
    error: str | None = None
    # Internal
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def set_stage(self, stage: str):
        self.status = stage
        self.current_stage = stage
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat()

    def complete(self, document_id: str, chunk_count: int, tags: list[str]):
        self.status = "completed"
        self.current_stage = None
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.document_id = document_id
        self.chunk_count = chunk_count
        self.tags = tags

    def fail(self, error: str):
        self.status = "failed"
        self.current_stage = None
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.error = error

    def cancel(self):
        self.status = "cancelled"
        self.current_stage = None
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self._cancel_event.set()

    def check_cancelled(self):
        """Raise CancelledError if cancellation was requested."""
        if self._cancel_event.is_set():
            raise asyncio.CancelledError("Job cancelled by user")

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "source_type": self.source_type,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_chunks": self.total_chunks,
            "embedded_chunks": self.embedded_chunks,
            "current_stage": self.current_stage,
            "document_id": self.document_id,
            "chunk_count": self.chunk_count,
            "tags": self.tags,
            "error": self.error,
        }


class JobService:
    def __init__(self):
        self._jobs: dict[str, Job] = {}

    @property
    def client(self):
        return es_service.client

    async def init_index(self):
        """Create the jobs ES index if it doesn't exist."""
        exists = await self.client.indices.exists(index=settings.es_jobs_index)
        if not exists:
            await self.client.indices.create(
                index=settings.es_jobs_index, body=JOBS_INDEX_MAPPING
            )
            logger.info(f"Created jobs index: {settings.es_jobs_index}")
        else:
            logger.info(f"Jobs index {settings.es_jobs_index} already exists.")

    async def _persist(self, job: Job):
        """Persist a job document to ES using job_id as the doc ID."""
        try:
            await self.client.index(
                index=settings.es_jobs_index,
                id=job.job_id,
                document=job.to_dict(),
            )
        except Exception:
            logger.error(f"Failed to persist job {job.job_id} to ES", exc_info=True)

    async def finish_job(self, job: Job):
        """Persist a terminal job to ES and remove from in-memory dict."""
        await self._persist(job)
        self._jobs.pop(job.job_id, None)

    def create_job(self, filename: str, source_type: str) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, filename=filename, source_type=source_type)
        self._jobs[job_id] = job
        logger.info(f"Job {job_id} created for {filename}")
        return job

    async def get_job(self, job_id: str) -> Job | dict | None:
        """Get a job — check in-memory first, then ES."""
        job = self._jobs.get(job_id)
        if job:
            return job
        try:
            resp = await self.client.get(index=settings.es_jobs_index, id=job_id)
            return resp["_source"]
        except Exception:
            return None

    async def list_jobs(self) -> list[dict]:
        """Return in-memory active jobs + ES historical jobs, sorted by created_at desc."""
        # In-memory active jobs
        active = [j.to_dict() for j in self._jobs.values()]

        # Historical jobs from ES
        historical = []
        try:
            resp = await self.client.search(
                index=settings.es_jobs_index,
                body={
                    "query": {"match_all": {}},
                    "sort": [{"created_at": {"order": "desc"}}],
                    "size": 100,
                },
            )
            historical = [hit["_source"] for hit in resp["hits"]["hits"]]
        except Exception:
            logger.error("Failed to fetch historical jobs from ES", exc_info=True)

        # Merge: active jobs first (most relevant), then historical
        # Deduplicate by job_id in case of race conditions
        seen = set()
        merged = []
        for j in active:
            seen.add(j["job_id"])
            merged.append(j)
        for j in historical:
            if j["job_id"] not in seen:
                merged.append(j)

        # Sort all by created_at descending
        merged.sort(key=lambda j: j["created_at"], reverse=True)
        return merged

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an active in-memory job. Historical jobs cannot be cancelled."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status in ("completed", "failed", "cancelled"):
            return False
        job.cancel()
        await self.finish_job(job)
        logger.info(f"Job {job_id} cancelled")
        return True


job_service = JobService()
