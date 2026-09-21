"""Text-to-image through Diffusers, directly.

This is the whole inference path: `from_pretrained` on a local snapshot, then
call the pipeline. No graph is built, no workflow is compiled, no external
process is launched. The model's own official implementation — the one its
authors publish on the model card — is what runs.

torch and diffusers are imported inside `load_model` so that importing this
module costs nothing on a machine without a CUDA build.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from server.errors import Cancelled, LocalAIError
from server.schemas.models import ModelSpec

from ..base import EngineOutput, EngineRequest, EngineResult, InferenceEngine, ProgressFn

log = logging.getLogger(__name__)

DTYPES = ("bfloat16", "float16", "float32")


class DiffusersTextToImageEngine(InferenceEngine):
    key = "diffusers-t2i"

    def __init__(self, device: str = "cuda", dtype: str = "bfloat16") -> None:
        super().__init__(device=device, dtype=dtype)
        self._pipe = None
        self._path: Path | None = None

    # -- lifecycle --------------------------------------------------------

    def load_model(self, spec: ModelSpec, path: Path) -> None:
        if self._pipe is not None and self.spec is not None and self.spec.id == spec.id:
            return  # already resident
        if self._pipe is not None:
            self.unload_model()

        torch = _import_torch()
        diffusers = _import_diffusers()

        pipeline_cls = self._pipeline_class(diffusers, spec)
        torch_dtype = self._torch_dtype(torch)
        started = time.monotonic()
        log.info("loading %s (%s) from %s as %s", spec.id, pipeline_cls.__name__, path, self.dtype)

        pipe = pipeline_cls.from_pretrained(
            str(path),
            torch_dtype=torch_dtype,
            local_files_only=True,
            **dict(spec.load_kwargs),
        )
        pipe.set_progress_bar_config(disable=True)

        # Offloading is what lets a 34 GB pipeline run in 32 GB of VRAM: only
        # the component currently executing is resident, the rest waits in
        # pinned host memory. "none" is faster when the whole thing fits.
        if not self.device.startswith("cuda"):
            pipe.to(self.device)
        elif spec.offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif spec.offload == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)

        for enable in ("enable_vae_slicing", "enable_vae_tiling"):
            method = getattr(pipe, enable, None)
            if callable(method):
                try:
                    method()
                except Exception as exc:  # pragma: no cover - pipeline dependent
                    log.debug("%s unsupported on %s: %s", enable, spec.id, exc)

        self._pipe = pipe
        self._path = path
        self.spec = spec
        log.info("loaded %s in %.1fs", spec.id, time.monotonic() - started)

    def unload_model(self) -> None:
        if self._pipe is None:
            return
        model_id = self.spec.id if self.spec else "?"
        del self._pipe
        self._pipe = None
        self._path = None
        self.spec = None
        try:
            import gc

            gc.collect()
            torch = _import_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:  # pragma: no cover - depends on the host
            log.debug("cache release after unloading %s: %s", model_id, exc)
        log.info("unloaded %s", model_id)

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    # -- inference --------------------------------------------------------

    def generate(self, request: EngineRequest, report: ProgressFn | None = None) -> EngineResult:
        if self._pipe is None or self.spec is None:
            raise LocalAIError("generate() called before load_model()")

        torch = _import_torch()
        spec = self.spec
        total = request.steps
        self._set_progress("generating", 0, total, report)

        def on_step_end(pipe, step_index: int, timestep, callback_kwargs):  # noqa: ANN001
            if self._is_cancelled(request.job_id):
                raise Cancelled(request.job_id)
            self._set_progress("generating", step_index + 1, total, report)
            return callback_kwargs

        generator_device = "cuda" if self.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(request.seed)

        call: dict = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance,
            "num_images_per_prompt": request.num_images,
            "generator": generator,
            "callback_on_step_end": on_step_end,
            **dict(spec.call_kwargs),
        }
        if spec.capabilities.supports_negative_prompt and request.negative_prompt:
            call["negative_prompt"] = request.negative_prompt

        started = time.monotonic()
        try:
            result = self._pipe(**call)
        except Cancelled:
            self._clear_cancel(request.job_id)
            raise
        elapsed = time.monotonic() - started

        self._set_progress("encoding", total, total, report)
        outputs: list[EngineOutput] = []
        request.output_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(result.images):
            path = request.output_dir / f"{request.job_id}-{index}.png"
            image.save(path, format="PNG")
            outputs.append(
                EngineOutput(
                    path=path,
                    width=image.width,
                    height=image.height,
                    # Every image of a batch is a different sample of the same
                    # seeded generator, so record which offset produced it.
                    seed=request.seed + index,
                )
            )

        self._clear_cancel(request.job_id)
        self._set_progress("idle", total, total, report)
        return EngineResult(
            outputs=outputs,
            meta={
                "model": spec.id,
                "pipeline": type(self._pipe).__name__,
                "device": self.device,
                "dtype": self.dtype,
                "offload": spec.offload,
                "seconds": round(elapsed, 2),
                "seconds_per_step": round(elapsed / max(1, request.steps), 3),
            },
        )

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _pipeline_class(diffusers, spec: ModelSpec):  # noqa: ANN001
        name = spec.pipeline or "DiffusionPipeline"
        cls = getattr(diffusers, name, None)
        if cls is None:
            raise LocalAIError(
                f"diffusers {getattr(diffusers, '__version__', '?')} has no pipeline named "
                f"{name!r} — upgrade diffusers or correct the registry entry for {spec.id}."
            )
        return cls

    def _torch_dtype(self, torch):  # noqa: ANN001
        if self.dtype not in DTYPES:
            raise LocalAIError(f"AI_DTYPE must be one of {', '.join(DTYPES)}, got {self.dtype!r}")
        # bf16 is the right default on Blackwell and Ada; fp16 overflows some
        # of these transformers and fp32 doubles an already tight budget.
        return getattr(torch, self.dtype)


def _import_torch():
    try:
        import torch  # noqa: PLC0415 - deliberate lazy import
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise LocalAIError(
            "PyTorch is not installed. On an RTX 5090 install a CUDA 12.8+ build: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu128"
        ) from exc
    return torch


def _import_diffusers():
    try:
        import diffusers  # noqa: PLC0415 - deliberate lazy import
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise LocalAIError(
            "diffusers is not installed. Install the runtime extras: pip install -e '.[gpu]'"
        ) from exc
    return diffusers
