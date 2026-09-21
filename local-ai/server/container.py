"""Wires the services together once, so routes can stay thin.

Held on `app.state.container`. Tests build one against a tmp path and a fake
engine without touching the process-wide settings cache.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, load_settings
from .services.gpu_manager import GpuManager
from .services.inference_router import EnginePool, InferenceRouter
from .services.job_manager import JobManager
from .services.model_manager import ModelManager
from .services.model_registry import ModelRegistry
from .services.storage import Storage


@dataclass
class Container:
    settings: Settings
    registry: ModelRegistry
    gpu: GpuManager
    models: ModelManager
    storage: Storage
    router: InferenceRouter
    jobs: JobManager

    @classmethod
    def build(cls, settings: Settings | None = None) -> "Container":
        settings = settings or load_settings()
        settings.ensure_dirs()
        registry = ModelRegistry.load(extra=settings.extra_registry)
        gpu = GpuManager(preferred=settings.device)
        models = ModelManager(settings, registry, gpu)
        storage = Storage(settings.output_dir)
        router = InferenceRouter(settings, registry, models, gpu, EnginePool(settings, gpu))
        return cls(
            settings=settings,
            registry=registry,
            gpu=gpu,
            models=models,
            storage=storage,
            router=router,
            jobs=JobManager(router, storage),
        )

    def shutdown(self) -> None:
        self.jobs.stop()
        self.router.pool.unload_all()
