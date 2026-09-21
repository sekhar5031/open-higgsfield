"""Job lifecycle as the API reports it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .generation import GeneratedAsset

JobStatus = Literal[
    "queued",
    "loading_model",
    "generating",
    "encoding",
    "completed",
    "failed",
    "cancelled",
]

#: Statuses the job manager never moves off again.
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})


class JobProgress(BaseModel):
    #: 0..1 across the whole job, not just the sampling loop.
    fraction: float = 0.0
    step: int = 0
    total_steps: int = 0
    stage: str = "queued"


class Job(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    job_id: str
    status: JobStatus
    model: str
    task: str
    prompt: str
    progress: JobProgress = JobProgress()
    assets: list[GeneratedAsset] = []
    error: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    #: Position in the queue while status is "queued"; 0 once it is running.
    queue_position: int = 0


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatus = "queued"
    queue_position: int = 0


class JobList(BaseModel):
    jobs: list[Job]
