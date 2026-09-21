"""Job status and cancellation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..container import Container
from ..schemas.jobs import Job, JobList
from .deps import container

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.get("", response_model=JobList)
def list_jobs(
    app: Container = Depends(container),
    limit: int = Query(default=50, ge=1, le=200),
) -> JobList:
    return JobList(jobs=[app.jobs.to_schema(state) for state in app.jobs.list(limit=limit)])


@router.get("/{job_id}", response_model=Job)
def get_job(job_id: str, app: Container = Depends(container)) -> Job:
    return app.jobs.to_schema(app.jobs.get(job_id))


@router.post("/{job_id}/cancel", response_model=Job)
def cancel_job(job_id: str, app: Container = Depends(container)) -> Job:
    return app.jobs.to_schema(app.jobs.cancel(job_id))
