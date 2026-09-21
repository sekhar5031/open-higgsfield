"""The model registry's data shapes.

A model is data, not code: adding one means adding a row to
`server/data/models.json`, not writing a class. The engine field names which
adapter executes it, and the licence fields are recorded so install can refuse
to fetch weights whose terms the user has not accepted.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ModelType = Literal["image", "video", "audio", "3d", "llm"]
ModelSource = Literal["huggingface", "modelscope", "local"]

#: Lifecycle of the weights on disk, not of a loaded pipeline.
InstallState = Literal["not_installed", "downloading", "installed", "corrupt"]

#: Lifecycle of the pipeline in VRAM.
LoadState = Literal["not_loaded", "loading", "ready", "running", "unloaded"]


class Bound(BaseModel):
    """An inclusive numeric range with the value used when the caller omits it."""

    model_config = ConfigDict(frozen=True)

    min: float
    max: float
    default: float

    @field_validator("max")
    @classmethod
    def _ordered(cls, value: float, info) -> float:
        low = info.data.get("min")
        if low is not None and value < low:
            raise ValueError("max must be >= min")
        return value

    def clamp(self, value: float | None) -> float:
        if value is None:
            return self.default
        return min(self.max, max(self.min, value))


class Capabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_width: int = 1536
    max_height: int = 1536
    #: Output edges are rounded down to a multiple of this; most latent VAEs
    #: need 8 or 16 and fail late and cryptically otherwise.
    size_multiple: int = 16
    steps: Bound = Bound(min=1, max=50, default=28)
    guidance: Bound = Bound(min=0, max=20, default=3.5)
    max_images: int = 4
    supports_negative_prompt: bool = False
    supports_seed: bool = True


class ModelSpec(BaseModel):
    """One open-weight model the platform knows how to fetch and execute."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    id: str
    name: str
    type: ModelType
    tasks: tuple[str, ...]
    source: ModelSource = "huggingface"
    repository: str
    revision: str = "main"
    #: Restricts the snapshot download; `None` fetches the whole repository.
    allow_patterns: tuple[str, ...] | None = None
    #: Formats a repository may also ship that this engine never reads. Skipping
    #: them halves the download for repositories that publish both.
    ignore_patterns: tuple[str, ...] = (
        "*.bin", "*.pt", "*.pth", "*.ckpt", "*.msgpack", "*.onnx", "*.onnx_data",
    )
    engine: str
    #: Diffusers pipeline class, when the engine needs a specific one.
    pipeline: str | None = None
    #: Extra kwargs for `from_pretrained`, straight from the model card.
    load_kwargs: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    #: Extra kwargs for the pipeline call — `max_sequence_length` and friends,
    #: which differ per release. Data, so a new model needs no code branch.
    call_kwargs: dict[str, float | int | str | bool] = Field(default_factory=dict)
    size_gb: float = Field(gt=0)
    vram_gb: float = Field(gt=0)
    ram_gb: float = Field(gt=0)
    license: str
    license_url: str = ""
    #: Hugging Face repositories behind a terms click-through. Install refuses
    #: these without a token rather than failing with a bare 401 mid-download.
    gated: bool = False
    #: How much of the pipeline may live on the CPU between steps. "model"
    #: keeps one component at a time on the GPU, which is what lets a 34 GB
    #: pipeline run in 32 GB of VRAM; "sequential" goes further and slower.
    offload: Literal["none", "model", "sequential"] = "model"
    description: str = ""
    capabilities: Capabilities = Capabilities()
    default_settings: dict[str, float | int | str | bool] = Field(default_factory=dict)
    #: A model can be shipped in the registry but withheld from routing, e.g.
    #: while its licence is still being verified.
    enabled: bool = True

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        if not value or not all(char.isalnum() or char in "._-" for char in value):
            raise ValueError("model id must be alphanumeric with . _ - only")
        if ".." in value:
            raise ValueError("model id must not contain ..")
        return value


class ModelStatus(BaseModel):
    """A spec plus what is true of it on this machine right now."""

    model_config = ConfigDict(protected_namespaces=())

    spec: ModelSpec
    install_state: InstallState
    load_state: LoadState
    path: str | None = None
    disk_bytes: int = 0
    installed_revision: str | None = None
    #: False when the declared VRAM exceeds what the device physically has.
    fits_gpu: bool | None = None
    message: str | None = None


class ModelList(BaseModel):
    models: list[ModelStatus]


class InstallRequest(BaseModel):
    revision: str | None = None
    #: Re-download even when the marker says the snapshot is complete.
    force: bool = False
