"""Runtime configuration for the local inference platform.

Every path is configurable and none of them lives inside the source tree —
weights and outputs must never end up in git. The defaults land under the
user's data directory so a fresh clone runs before `/AI` exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

DEFAULT_ROOT = Path.home() / ".local" / "share" / "open-higgsfield-local-ai"

#: Subdirectories of the model root, one per model type.
MODEL_TYPES = ("image", "video", "audio", "3d", "llm")


@dataclass(frozen=True)
class Settings:
    """Resolved paths and knobs. Frozen so a request cannot reconfigure the process."""

    root: Path
    model_dir: Path
    output_dir: Path
    asset_dir: Path
    cache_dir: Path
    log_dir: Path
    extra_registry: Path | None
    host: str
    port: int
    cors_origins: tuple[str, ...]
    #: How many models may stay resident in VRAM at once (LRU beyond this).
    resident_models: int
    device: str
    dtype: str
    hf_token: str | None
    #: Refuse any network call; only already-downloaded weights can be used.
    offline: bool

    def dir_for(self, model_type: str) -> Path:
        return self.model_dir / model_type

    def ensure_dirs(self) -> None:
        for path in (self.model_dir, self.output_dir, self.asset_dir, self.cache_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)
        for model_type in MODEL_TYPES:
            (self.model_dir / model_type).mkdir(parents=True, exist_ok=True)


def _path(env: Mapping[str, str], key: str, fallback: Path) -> Path:
    raw = env.get(key, "").strip()
    return Path(raw).expanduser().resolve() if raw else fallback


def _int(env: Mapping[str, str], key: str, fallback: int) -> int:
    raw = env.get(key, "").strip()
    if not raw:
        return fallback
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


def _bool(env: Mapping[str, str], key: str, fallback: bool) -> bool:
    raw = env.get(key, "").strip().lower()
    if not raw:
        return fallback
    return raw in {"1", "true", "yes", "on"}


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    root = _path(env, "AI_ROOT", DEFAULT_ROOT)
    origins = tuple(
        origin.strip()
        for origin in env.get("AI_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if origin.strip()
    )
    extra = env.get("AI_MODEL_REGISTRY", "").strip()
    return Settings(
        root=root,
        model_dir=_path(env, "AI_MODEL_DIR", root / "models"),
        output_dir=_path(env, "AI_OUTPUT_DIR", root / "outputs"),
        asset_dir=_path(env, "AI_ASSET_DIR", root / "assets"),
        cache_dir=_path(env, "AI_CACHE_DIR", root / "cache"),
        log_dir=_path(env, "AI_LOG_DIR", root / "logs"),
        extra_registry=Path(extra).expanduser().resolve() if extra else None,
        host=env.get("AI_HOST", "127.0.0.1").strip() or "127.0.0.1",
        port=_int(env, "AI_PORT", 8000),
        cors_origins=origins,
        resident_models=max(1, _int(env, "AI_RESIDENT_MODELS", 1)),
        device=env.get("AI_DEVICE", "auto").strip() or "auto",
        dtype=env.get("AI_DTYPE", "bfloat16").strip() or "bfloat16",
        hf_token=(env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None,
        offline=_bool(env, "AI_OFFLINE", False),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
