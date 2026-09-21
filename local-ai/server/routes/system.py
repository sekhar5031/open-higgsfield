"""Liveness, the GPU report, and where everything lives on disk."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..container import Container
from ..schemas.system import GpuReport, Health, SystemInfo
from ..version import VERSION
from .deps import container

router = APIRouter(prefix="/v1/system", tags=["system"])


@router.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", version=VERSION)


@router.get("/gpu", response_model=GpuReport)
def gpu(app: Container = Depends(container)) -> GpuReport:
    """Read from torch and NVML — nothing about the device is assumed."""
    return app.gpu.report()


@router.get("/info", response_model=SystemInfo)
def info(app: Container = Depends(container)) -> SystemInfo:
    settings = app.settings
    return SystemInfo(
        version=VERSION,
        device=app.gpu.torch_device(),
        dtype=settings.dtype,
        offline=settings.offline,
        resident_models=settings.resident_models,
        paths={
            "root": str(settings.root),
            "models": str(settings.model_dir),
            "outputs": str(settings.output_dir),
            "assets": str(settings.asset_dir),
            "cache": str(settings.cache_dir),
            "logs": str(settings.log_dir),
        },
        disk_free_bytes=app.storage.free_bytes(),
        installed_models=app.models.installed_count(),
        registered_models=len(app.registry),
    )
