"""The contract every model family implements.

An engine owns one loaded model at a time. It is deliberately synchronous and
blocking: diffusion is CPU/GPU-bound work, and the job manager already runs it
on a worker thread. Making these coroutines would only hide that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from server.schemas.models import ModelSpec


@dataclass
class EngineRequest:
    """One unit of work, already resolved against the model's capabilities.

    The engine receives concrete numbers — never a `"16:9"` to interpret and
    never a `None` to substitute a default for. Resolution happens once, in the
    router, so two engines cannot disagree about what a setting means.
    """

    job_id: str
    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    steps: int
    guidance: float
    seed: int
    num_images: int
    output_dir: Path
    extra: dict = field(default_factory=dict)


@dataclass
class EngineOutput:
    path: Path
    width: int
    height: int
    seed: int
    content_type: str = "image/png"
    kind: str = "image"


@dataclass
class EngineResult:
    outputs: list[EngineOutput]
    meta: dict = field(default_factory=dict)


@dataclass
class Progress:
    stage: str = "idle"
    step: int = 0
    total_steps: int = 0

    @property
    def fraction(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return min(1.0, self.step / self.total_steps)


class ProgressFn(Protocol):
    def __call__(self, stage: str, step: int, total_steps: int) -> None: ...


class InferenceEngine(ABC):
    """One model family's native implementation, behind a shared interface."""

    #: Registry key, matched against `ModelSpec.engine`.
    key: str = "base"

    def __init__(self, device: str = "cuda", dtype: str = "bfloat16") -> None:
        self.device = device
        self.dtype = dtype
        self.spec: ModelSpec | None = None
        self._progress = Progress()
        self._cancelled: set[str] = set()

    @abstractmethod
    def load_model(self, spec: ModelSpec, path: Path) -> None:
        """Bring the weights at `path` into memory. Idempotent for the same spec."""

    @abstractmethod
    def unload_model(self) -> None:
        """Release the pipeline and the VRAM it holds."""

    @abstractmethod
    def generate(self, request: EngineRequest, report: ProgressFn | None = None) -> EngineResult:
        """Run inference. Raises `Cancelled` if `cancel()` was called for this job."""

    def get_progress(self) -> Progress:
        return self._progress

    def cancel(self, job_id: str) -> None:
        """Ask a running job to stop at its next step boundary. Diffusion steps
        are not interruptible mid-kernel, so cancellation is cooperative."""
        self._cancelled.add(job_id)

    # -- helpers for subclasses ------------------------------------------

    def _is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancelled

    def _clear_cancel(self, job_id: str) -> None:
        self._cancelled.discard(job_id)

    def _set_progress(self, stage: str, step: int, total_steps: int, report: ProgressFn | None) -> None:
        self._progress = Progress(stage=stage, step=step, total_steps=total_steps)
        if report is not None:
            report(stage=stage, step=step, total_steps=total_steps)


EngineFactory = Callable[[str, str], InferenceEngine]
