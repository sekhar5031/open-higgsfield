"""The asynchronous generation queue.

One GPU, one worker: jobs run strictly one at a time. Two diffusion jobs
sharing 32 GB finish later than the same two in sequence and risk OOMing each
other, so the queue is the feature, not a limitation. Adding a second worker
per additional GPU is the extension point.

Submitting returns a job id immediately; the studio polls `/v1/jobs/{id}`.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field

from engines.base import EngineResult

from ..errors import Cancelled, JobNotFound, LocalAIError
from ..schemas.generation import GeneratedAsset, GenerationRequest
from ..schemas.jobs import Job, JobProgress, JobStatus
from .inference_router import InferenceRouter
from .storage import Storage

log = logging.getLogger(__name__)

#: Finished jobs kept in memory for the studio to poll after the fact.
MAX_RETAINED = 200

#: Sampling is the long pole but not the whole job; leave room so the bar does
#: not sit at 100% through VAE decode and the file write.
SAMPLING_SHARE = 0.95


@dataclass
class JobState:
    job_id: str
    request: GenerationRequest
    status: JobStatus = "queued"
    stage: str = "queued"
    step: int = 0
    total_steps: int = 0
    assets: list[GeneratedAsset] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancel_requested: bool = False
    meta: dict = field(default_factory=dict)

    def fraction(self) -> float:
        if self.status == "completed":
            return 1.0
        if self.status in {"failed", "cancelled"}:
            return 0.0
        if self.status == "encoding":
            return SAMPLING_SHARE
        if self.total_steps <= 0:
            return 0.0
        return round(min(1.0, self.step / self.total_steps) * SAMPLING_SHARE, 4)


class JobManager:
    def __init__(self, router: InferenceRouter, storage: Storage) -> None:
        self.router = router
        self.storage = storage
        self._jobs: dict[str, JobState] = {}
        self._order: list[str] = []
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._current: str | None = None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="generation-worker", daemon=True)
        self._worker.start()
        log.info("generation worker started")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._queue.put("")  # wake the worker out of its blocking get
        if self._worker is not None:
            self._worker.join(timeout=timeout)
        self._worker = None

    # -- public API -------------------------------------------------------

    def submit(self, request: GenerationRequest) -> JobState:
        # Routing errors belong to the caller of POST /v1/generate, not to a
        # job that fails four seconds later with nobody watching.
        self.router.route(request)
        state = JobState(job_id=f"job_{uuid.uuid4().hex[:20]}", request=request)
        with self._lock:
            self._jobs[state.job_id] = state
            self._order.append(state.job_id)
            self._trim()
        self._queue.put(state.job_id)
        self.start()
        log.info("queued %s for %s", state.job_id, request.model)
        return state

    def get(self, job_id: str) -> JobState:
        with self._lock:
            state = self._jobs.get(job_id)
        if state is None:
            raise JobNotFound(f"Unknown job: {job_id}")
        return state

    def list(self, limit: int = 50) -> list[JobState]:
        with self._lock:
            return [self._jobs[job_id] for job_id in reversed(self._order[-limit:])]

    def cancel(self, job_id: str) -> JobState:
        state = self.get(job_id)
        with self._lock:
            if state.status in {"completed", "failed", "cancelled"}:
                return state
            state.cancel_requested = True
            running = self._current == job_id
        if running:
            # Cooperative: the engine stops at its next step boundary.
            spec = self.router.registry.find(state.request.model)
            if spec is not None:
                engine = self.router.pool._engines.get(spec.id)  # noqa: SLF001 - same package
                if engine is not None:
                    engine.cancel(job_id)
        else:
            self._finish(state, "cancelled", error="Cancelled before it started")
        return state

    def queue_position(self, job_id: str) -> int:
        with self._lock:
            if self._current == job_id:
                return 0
            waiting = [
                other
                for other in self._order
                if self._jobs[other].status == "queued" and other != self._current
            ]
            return waiting.index(job_id) + 1 if job_id in waiting else 0

    def to_schema(self, state: JobState) -> Job:
        return Job(
            job_id=state.job_id,
            status=state.status,
            model=state.request.model,
            task=state.request.task,
            prompt=state.request.prompt,
            progress=JobProgress(
                fraction=state.fraction(),
                step=state.step,
                total_steps=state.total_steps,
                stage=state.stage,
            ),
            assets=state.assets,
            error=state.error,
            created_at=state.created_at,
            started_at=state.started_at,
            finished_at=state.finished_at,
            queue_position=self.queue_position(state.job_id),
        )

    # -- worker -----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            job_id = self._queue.get()
            if not job_id or self._stop.is_set():
                continue
            try:
                state = self.get(job_id)
            except JobNotFound:
                continue  # trimmed out from under us
            if state.cancel_requested:
                self._finish(state, "cancelled", error="Cancelled before it started")
                continue
            with self._lock:
                self._current = job_id
            try:
                self._execute(state)
            except Cancelled:
                self._finish(state, "cancelled", error="Cancelled")
            except LocalAIError as exc:
                log.warning("%s failed: %s", job_id, exc.detail)
                self._finish(state, "failed", error=exc.detail)
            except Exception as exc:  # noqa: BLE001 - the worker must survive anything
                log.exception("%s crashed", job_id)
                self._finish(state, "failed", error=_readable(exc))
            finally:
                with self._lock:
                    self._current = None

    def _execute(self, state: JobState) -> None:
        state.started_at = time.time()
        spec = self.router.route(state.request)

        self._set(state, "loading_model", stage="loading_model")
        engine = self.router.engine_for(spec)
        if state.cancel_requested:
            raise Cancelled(state.job_id)

        request = self.router.prepare(spec, state.request, state.job_id)
        state.total_steps = request.steps
        self._set(state, "generating", stage="generating")

        def report(stage: str, step: int, total_steps: int) -> None:
            state.stage = stage
            state.step = step
            state.total_steps = total_steps or state.total_steps
            if stage == "encoding" and state.status == "generating":
                state.status = "encoding"

        result: EngineResult = engine.generate(request, report=report)

        self._set(state, "encoding", stage="encoding")
        state.assets = [
            GeneratedAsset(
                id=output.path.stem,
                url=self.storage.url_for(output.path),
                kind=output.kind,  # type: ignore[arg-type]
                width=output.width,
                height=output.height,
                seed=output.seed,
                content_type=output.content_type,
            )
            for output in result.outputs
        ]
        state.meta = result.meta
        for output in result.outputs:
            self.storage.write_sidecar(
                output.path,
                {
                    "job_id": state.job_id,
                    "model": spec.id,
                    "model_repository": spec.repository,
                    "task": state.request.task,
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "settings": {
                        "width": request.width,
                        "height": request.height,
                        "steps": request.steps,
                        "guidance": request.guidance,
                        "seed": output.seed,
                        "num_images": request.num_images,
                    },
                    "engine": result.meta,
                },
            )
        self._finish(state, "completed")
        log.info("%s completed in %.1fs", state.job_id, (state.finished_at or 0) - (state.started_at or 0))

    # -- internals --------------------------------------------------------

    def _set(self, state: JobState, status: JobStatus, stage: str | None = None) -> None:
        with self._lock:
            state.status = status
            if stage is not None:
                state.stage = stage

    def _finish(self, state: JobState, status: JobStatus, error: str | None = None) -> None:
        with self._lock:
            state.status = status
            state.stage = status
            state.error = error
            state.finished_at = time.time()

    def _trim(self) -> None:
        """Drop the oldest finished jobs. Running and queued ones are never
        trimmed — the studio is still waiting on those."""
        while len(self._order) > MAX_RETAINED:
            for index, job_id in enumerate(self._order):
                if self._jobs[job_id].status in {"completed", "failed", "cancelled"}:
                    self._order.pop(index)
                    self._jobs.pop(job_id, None)
                    break
            else:
                return


def _readable(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    if "out of memory" in text.lower():
        return (
            "The GPU ran out of memory during generation. Try a smaller resolution, "
            "fewer images per run, or unload the other resident model."
        )
    return text
