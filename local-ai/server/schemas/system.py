"""System and GPU reporting shapes."""

from __future__ import annotations

from pydantic import BaseModel


class GpuInfo(BaseModel):
    available: bool
    name: str | None = None
    index: int | None = None
    vram_total: int = 0
    vram_used: int = 0
    vram_free: int = 0
    utilization: float | None = None
    temperature: float | None = None
    driver: str | None = None
    cuda: str | None = None
    torch: str | None = None
    #: Present when CUDA is unusable, so the caller learns why rather than
    #: reading `available: false` and guessing.
    detail: str | None = None


class GpuReport(BaseModel):
    devices: list[GpuInfo]
    device_in_use: str


class Health(BaseModel):
    status: str = "ok"
    version: str


class SystemInfo(BaseModel):
    version: str
    device: str
    dtype: str
    offline: bool
    resident_models: int
    paths: dict[str, str]
    disk_free_bytes: int
    installed_models: int
    registered_models: int
