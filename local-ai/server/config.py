"""Runtime configuration for the local inference platform.

Every path is configurable and none of them lives inside the source tree —
weights and outputs must never end up in git. The defaults land under the
user's data directory so a fresh clone runs before `/AI` exists.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

log = logging.getLogger(__name__)

DEFAULT_ROOT = Path.home() / ".local" / "share" / "open-higgsfield-local-ai"

#: Read at startup when it exists, so `cp .env.example .env` does what everyone
#: expects it to. Under Docker the file is absent and compose supplies the
#: environment instead.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

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


def read_env_file(path: Path) -> dict[str, str]:
    """A deliberately small .env parser: `KEY=value`, `#` comments, an optional
    `export ` prefix, and optional surrounding quotes.

    Hand-rolled rather than pulled in as a dependency, and it strips `\r`
    itself: a file edited on Windows, or checked out with CRLF endings, would
    otherwise set AI_ROOT to a path with a carriage return on the end and fail
    somewhere far away from the cause.
    """
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return values

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            log.warning("%s:%d ignored, no '=' in %r", path.name, number, raw.strip())
            continue
        key = key.strip()
        value = value.strip().strip("\r")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


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


#: Distinguishes "caller said no file" from "caller said nothing".
_UNSET = object()


def load_settings(
    env: Mapping[str, str] | None = None,
    env_file: Path | None | object = _UNSET,
) -> Settings:
    """Resolve configuration. The real environment wins over the file, so a
    container's compose settings are never overridden by a stray .env that
    found its way into the image.

    Passing an explicit `env` means "use exactly this" and reads no file —
    otherwise a developer's own .env would leak into every test.
    """
    if env_file is _UNSET:
        env_file = ENV_FILE if env is None else None
    env = os.environ if env is None else env
    if isinstance(env_file, Path) and env_file.is_file():
        from_file = read_env_file(env_file)
        if from_file:
            log.info("read %d settings from %s", len(from_file), env_file)
        env = {**from_file, **{key: value for key, value in env.items() if value != ""}}
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
