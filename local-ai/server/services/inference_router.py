"""Maps a studio request onto an installed local model and a loaded engine.

Selection is explicit: the request names a model and the router validates it.
Automatic selection by task and VRAM is deliberately not implemented yet — the
data it would need (`by_task`, `fits_device`) is in place, but guessing which
model the user meant is not a feature until choosing one by hand works.
"""

from __future__ import annotations

import logging
import math
import re
import secrets
import threading
from collections import OrderedDict
from pathlib import Path

from engines.base import EngineRequest, InferenceEngine
from engines.loader import create_engine

from ..config import Settings
from ..errors import ModelDisabled, ModelNotInstalled, UnsupportedTask
from ..schemas.generation import GenerationRequest, GenerationSettings
from ..schemas.models import Capabilities, LoadState, ModelSpec
from .gpu_manager import GpuManager
from .model_manager import ModelManager
from .model_registry import ModelRegistry

log = logging.getLogger(__name__)

RATIO = re.compile(r"^(\d{1,2}):(\d{1,2})$")
DEFAULT_BASE_EDGE = 1024
MIN_EDGE = 256


def resolve_dimensions(aspect_ratio: str, base: int, caps: Capabilities) -> tuple[int, int]:
    """Turn `"16:9"` plus a base edge into pixel dimensions.

    The area lands near `base**2` whatever the ratio, so switching from 1:1 to
    16:9 changes the shape of the picture and not how long it takes. Edges are
    snapped to the VAE's stride and clamped to what the model declares.
    """
    match = RATIO.match(aspect_ratio or "")
    if not match or aspect_ratio == "auto":
        width_units = height_units = 1.0
    else:
        width_units, height_units = float(match.group(1)), float(match.group(2))
        if width_units <= 0 or height_units <= 0:
            width_units = height_units = 1.0

    scale = base / math.sqrt(width_units * height_units)
    step = max(1, caps.size_multiple)

    def snap(value: float, limit: int) -> int:
        pixels = int(round(value / step) * step)
        return max(MIN_EDGE, min(limit - limit % step, pixels))

    return snap(width_units * scale, caps.max_width), snap(height_units * scale, caps.max_height)


def _base_edge(settings: GenerationSettings) -> int:
    raw = (settings.resolution or "").strip().lower()
    if not raw:
        return DEFAULT_BASE_EDGE
    if raw.endswith("k"):
        try:
            return int(float(raw[:-1]) * 1024)
        except ValueError:
            return DEFAULT_BASE_EDGE
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_BASE_EDGE


class EnginePool:
    """Loaded engines, newest use last, evicted LRU beyond `max_resident`.

    Keeping a model resident is the difference between a 4-second generation
    and a 90-second one, so the pool holds on to what it can. It never holds on
    at the cost of the job in front of it: eviction happens before admission.
    """

    def __init__(self, settings: Settings, gpu: GpuManager, max_resident: int | None = None) -> None:
        self.settings = settings
        self.gpu = gpu
        self.max_resident = max_resident or settings.resident_models
        self._engines: "OrderedDict[str, InferenceEngine]" = OrderedDict()
        self._loading: set[str] = set()
        self._lock = threading.RLock()

    def load_state(self, model_id: str) -> LoadState:
        with self._lock:
            if model_id in self._loading:
                return "loading"
            return "ready" if model_id in self._engines else "not_loaded"

    def states(self) -> dict[str, LoadState]:
        with self._lock:
            states: dict[str, LoadState] = {model_id: "ready" for model_id in self._engines}
            states.update({model_id: "loading" for model_id in self._loading})
            return states

    def acquire(self, spec: ModelSpec, path: Path) -> InferenceEngine:
        with self._lock:
            existing = self._engines.get(spec.id)
            if existing is not None:
                self._engines.move_to_end(spec.id)
                return existing
            self._loading.add(spec.id)

        try:
            # Make room first, then ask whether what is left is enough. Asking
            # in the other order rejects loads that would have fit perfectly
            # well once the previous model was released.
            self._evict_to(self.max_resident - 1)
            self.gpu.admit(spec.id, spec.vram_gb)
            engine = create_engine(spec.engine, device=self.gpu.torch_device(), dtype=self.settings.dtype)
            engine.load_model(spec, path)
        except Exception:
            with self._lock:
                self._loading.discard(spec.id)
            raise

        with self._lock:
            self._loading.discard(spec.id)
            self._engines[spec.id] = engine
            self._engines.move_to_end(spec.id)
        return engine

    def _evict_to(self, keep: int) -> None:
        while True:
            with self._lock:
                if len(self._engines) <= max(0, keep):
                    return
                model_id, engine = self._engines.popitem(last=False)
            log.info("evicting %s to make room", model_id)
            try:
                engine.unload_model()
            except Exception:  # pragma: no cover - engine dependent
                log.exception("failed to unload %s cleanly", model_id)
            self.gpu.empty_cache()

    def unload(self, model_id: str) -> bool:
        with self._lock:
            engine = self._engines.pop(model_id, None)
        if engine is None:
            return False
        engine.unload_model()
        self.gpu.empty_cache()
        return True

    def unload_all(self) -> None:
        self._evict_to(0)


class InferenceRouter:
    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        manager: ModelManager,
        gpu: GpuManager,
        pool: EnginePool | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.manager = manager
        self.gpu = gpu
        self.pool = pool or EnginePool(settings, gpu)

    def route(self, request: GenerationRequest) -> ModelSpec:
        """The model this request will run on, or a 4xx explaining why not."""
        spec = self.registry.get(request.model)
        if not spec.enabled:
            raise ModelDisabled(f"{spec.id} is registered but disabled.")
        if request.task not in spec.tasks:
            raise UnsupportedTask(
                f"{spec.id} does not do {request.task}. It supports: {', '.join(spec.tasks)}."
            )
        if not self.manager.is_installed(spec):
            raise ModelNotInstalled(
                f"{spec.name} is not installed. Install it first: "
                f"POST /v1/models/{spec.id}/install"
            )
        return spec

    def prepare(self, spec: ModelSpec, request: GenerationRequest, job_id: str) -> EngineRequest:
        """Resolve the studio's settings into the concrete numbers an engine takes."""
        settings = request.settings
        caps = spec.capabilities
        if settings.width and settings.height:
            step = max(1, caps.size_multiple)
            width = min(caps.max_width, settings.width - settings.width % step)
            height = min(caps.max_height, settings.height - settings.height % step)
        else:
            width, height = resolve_dimensions(settings.aspectRatio, _base_edge(settings), caps)

        seed = settings.seed if settings.seed is not None else secrets.randbelow(2**31)
        return EngineRequest(
            job_id=job_id,
            prompt=request.prompt.strip(),
            negative_prompt=settings.negativePrompt,
            width=width,
            height=height,
            steps=int(caps.steps.clamp(settings.steps)),
            guidance=float(caps.guidance.clamp(settings.guidance)),
            seed=int(seed),
            num_images=max(1, min(caps.max_images, settings.numImages or 1)),
            output_dir=self.settings.output_dir,
        )

    def engine_for(self, spec: ModelSpec) -> InferenceEngine:
        return self.pool.acquire(spec, self.manager.path_for(spec))

    def load_states(self) -> dict[str, LoadState]:
        return self.pool.states()
