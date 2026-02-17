from fastapi import APIRouter, Query

from app.models.schemas import MetricEvent, MetricsResponse
from app.services.metrics import metrics_service

router = APIRouter()


@router.get("", response_model=MetricsResponse)
async def get_metrics(minutes: int = Query(default=60, ge=1, le=1440)):
    """Retrieve usage metrics for the last N minutes."""
    events = await metrics_service.query(minutes)
    return MetricsResponse(
        events=[MetricEvent(**e) for e in events],
        total=len(events),
    )
