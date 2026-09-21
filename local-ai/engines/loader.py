"""Engine key -> implementation.

Imports are deferred until a key is actually requested, so the API, the
registry and the tests run in an environment with no torch and no diffusers.
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable

from server.errors import EngineNotFound

from .base import EngineFactory, InferenceEngine

log = logging.getLogger(__name__)

#: key -> "module:ClassName", resolved on first use.
BUILTIN: dict[str, str] = {
    "diffusers-t2i": "engines.image.diffusers_t2i:DiffusersTextToImageEngine",
}

_overrides: dict[str, EngineFactory] = {}


def register_engine(key: str, factory: EngineFactory) -> None:
    """Install a factory for a key, replacing any built-in of the same name.

    Used by tests to stand a fake engine behind a real model spec, and by
    anyone extending the platform without editing this file.
    """
    _overrides[key] = factory


def unregister_engine(key: str) -> None:
    _overrides.pop(key, None)


def available_engines() -> list[str]:
    return sorted({*BUILTIN, *_overrides})


def create_engine(key: str, device: str, dtype: str) -> InferenceEngine:
    factory = _overrides.get(key)
    if factory is not None:
        return factory(device, dtype)
    target = BUILTIN.get(key)
    if target is None:
        raise EngineNotFound(
            f"No engine named {key!r}. Known engines: {', '.join(available_engines()) or 'none'}."
        )
    module_name, _, class_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise EngineNotFound(
            f"Engine {key!r} needs runtime dependencies that are not installed "
            f"({exc}). Install them with: pip install -e '.[gpu]'"
        ) from exc
    cls: Callable[..., InferenceEngine] = getattr(module, class_name)
    return cls(device=device, dtype=dtype)
