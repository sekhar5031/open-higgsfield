"""Submit a generation. Returns a job id; the work happens on the queue."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..container import Container
from ..schemas.generation import GenerationRequest
from ..schemas.jobs import JobAccepted
from .deps import container

router = APIRouter(prefix="/v1", tags=["generation"])


@router.post("/generate", response_model=JobAccepted, status_code=202)
def generate(request: GenerationRequest, app: Container = Depends(container)) -> JobAccepted:
    state = app.jobs.submit(request)
    return JobAccepted(
        job_id=state.job_id,
        status=state.status,
        queue_position=app.jobs.queue_position(state.job_id),
    )
