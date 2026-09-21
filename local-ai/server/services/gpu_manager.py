"""What the platform knows about the GPU.

Everything is read from torch and, where available, NVML — nothing about the
device is hardcoded, so the same code reports an RTX 5090, a pair of them, or
no CUDA device at all. torch is imported lazily: the API, the registry and the
job queue all work on a machine that has no CUDA build installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..errors import InsufficientVram
from ..schemas.system import GpuInfo, GpuReport

log = logging.getLogger(__name__)

GB = 1024**3
#: Headroom kept free for activations, the CUDA context and allocator
#: fragmentation. A pipeline sized to the last byte OOMs during sampling.
VRAM_HEADROOM_BYTES = 2 * GB


def _torch():
    try:
        import torch  # noqa: PLC0415 - deliberate lazy import
    except Exception as exc:  # pragma: no cover - depends on the host
        log.debug("torch unavailable: %s", exc)
        return None
    return torch


@dataclass(frozen=True)
class DeviceMemory:
    total: int
    free: int
    used: int


class GpuManager:
    def __init__(self, preferred: str = "auto") -> None:
        self.preferred = preferred

    # -- device selection -------------------------------------------------

    def torch_device(self) -> str:
        if self.preferred != "auto":
            return self.preferred
        torch = _torch()
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def is_cuda(self) -> bool:
        return self.torch_device().startswith("cuda")

    # -- reporting --------------------------------------------------------

    def devices(self) -> list[GpuInfo]:
        torch = _torch()
        if torch is None:
            return [GpuInfo(available=False, detail="PyTorch is not installed in this environment")]
        if not torch.cuda.is_available():
            return [
                GpuInfo(
                    available=False,
                    torch=torch.__version__,
                    detail="PyTorch is installed but reports no usable CUDA device",
                )
            ]

        infos: list[GpuInfo] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            try:
                free, total = torch.cuda.mem_get_info(index)
            except Exception:  # pragma: no cover - driver dependent
                total = int(props.total_memory)
                free = max(0, total - int(torch.cuda.memory_reserved(index)))
            extra = self._nvml(index)
            infos.append(
                GpuInfo(
                    available=True,
                    name=props.name,
                    index=index,
                    vram_total=int(total),
                    vram_used=int(total - free),
                    vram_free=int(free),
                    utilization=extra.get("utilization"),
                    temperature=extra.get("temperature"),
                    driver=extra.get("driver"),
                    cuda=torch.version.cuda,
                    torch=torch.__version__,
                )
            )
        return infos

    def report(self) -> GpuReport:
        return GpuReport(devices=self.devices(), device_in_use=self.torch_device())

    @staticmethod
    def _nvml(index: int) -> dict:
        """Utilisation and temperature, which torch does not expose. Optional:
        a machine without pynvml still gets memory and device name."""
        try:
            import pynvml  # noqa: PLC0415 - optional dependency
        except Exception:
            return {}
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            driver = pynvml.nvmlSystemGetDriverVersion()
            return {
                "utilization": float(rates.gpu),
                "temperature": float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
                "driver": driver.decode() if isinstance(driver, bytes) else str(driver),
            }
        except Exception as exc:  # pragma: no cover - driver dependent
            log.debug("NVML unavailable: %s", exc)
            return {}

    # -- admission --------------------------------------------------------

    def memory(self, index: int = 0) -> DeviceMemory | None:
        torch = _torch()
        if torch is None or not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(index)
        return DeviceMemory(total=int(total), free=int(free), used=int(total - free))

    def fits_device(self, vram_gb: float) -> bool | None:
        """Whether the card is big enough at all, ignoring what is resident.
        None when there is no CUDA device to compare against."""
        memory = self.memory()
        if memory is None:
            return None
        return vram_gb * GB + VRAM_HEADROOM_BYTES <= memory.total

    def admit(self, model_id: str, vram_gb: float) -> None:
        """Refuse a load that clearly cannot fit, naming both numbers.

        A CUDA OOM twelve seconds into sampling tells the user nothing they can
        act on; this tells them the model needs more card than they have, or
        that something else is holding the memory.
        """
        memory = self.memory()
        if memory is None:
            return  # CPU execution is slow, not impossible — let it through.
        required = vram_gb * GB + VRAM_HEADROOM_BYTES
        if required > memory.total:
            raise InsufficientVram(
                f"{model_id} needs about {vram_gb:.1f} GB of VRAM; this device has "
                f"{memory.total / GB:.1f} GB in total."
            )
        if required > memory.free:
            raise InsufficientVram(
                f"{model_id} needs about {vram_gb:.1f} GB of VRAM and only "
                f"{memory.free / GB:.1f} GB of {memory.total / GB:.1f} GB is free. "
                "Unload another model or stop whatever else is using the GPU."
            )

    def empty_cache(self) -> None:
        torch = _torch()
        if torch is None or not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
