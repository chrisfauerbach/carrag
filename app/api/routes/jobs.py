import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import JobDetailResponse, JobListResponse
from app.services.jobs import job_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("", response_model=JobListResponse)
async def list_jobs():
    """List all ingestion jobs, most recent first."""
    jobs = await job_service.list_jobs()
    return JobListResponse(
        jobs=[JobDetailResponse(**j) for j in jobs],
        total=len(jobs),
    )


@router.get("/{job_id}", response_model=JobDetailResponse)
async def get_job(job_id: str):
    """Get detailed status of a specific job."""
    job = await job_service.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    data = job.to_dict() if hasattr(job, "to_dict") else job
    return JobDetailResponse(**data)


@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    """Cancel an active ingestion job."""
    job = await job_service.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if not await job_service.cancel_job(job_id):
        status = job.status if hasattr(job, "status") else job.get("status")
        raise HTTPException(400, f"Job cannot be cancelled (status: {status})")

    return {"job_id": job_id, "status": "cancelled"}
