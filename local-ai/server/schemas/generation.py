"""Request and result shapes for a generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Task = Literal[
    "text-to-image",
    "image-to-image",
    "text-to-video",
    "image-to-video",
    "text-to-3d",
    "image-to-3d",
    "speech-to-text",
    "text-to-speech",
    "text-generation",
]


class GenerationSettings(BaseModel):
    """The dials the studio sends. Everything is optional: the model's own
    capabilities supply the defaults, so the frontend never has to know them."""

    model_config = ConfigDict(extra="allow")

    aspectRatio: str = "1:1"
    #: Target for the longer calculation: output area lands near base**2.
    resolution: str | None = None
    width: int | None = Field(default=None, ge=64, le=4096)
    height: int | None = Field(default=None, ge=64, le=4096)
    steps: int | None = Field(default=None, ge=1, le=200)
    guidance: float | None = Field(default=None, ge=0, le=50)
    seed: int | None = Field(default=None, ge=0, le=2**53)
    numImages: int | None = Field(default=None, ge=1, le=8)
    negativePrompt: str | None = None


class GenerationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    task: Task = "text-to-image"
    prompt: str = Field(min_length=1, max_length=8000)
    settings: GenerationSettings = GenerationSettings()
    #: URLs or asset ids for input media. Unused by text-to-image; present so
    #: the contract does not change when img2img lands.
    media: dict[str, list[str]] = Field(default_factory=dict)


class GeneratedAsset(BaseModel):
    id: str
    #: Server-relative; the studio prefixes it with the API's public origin.
    url: str
    kind: Literal["image", "video", "audio", "model3d"]
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    content_type: str = "image/png"
