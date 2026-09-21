"""Loads the model registry from JSON and answers questions about it.

The registry is data. A built-in file ships with the platform; `AI_MODEL_REGISTRY`
points at an optional second file whose rows are merged over it by id, so a user
can add or override a model without editing the repository.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from ..errors import ModelNotFound
from ..schemas.models import ModelSpec

log = logging.getLogger(__name__)

BUILTIN_REGISTRY = Path(__file__).resolve().parent.parent / "data" / "models.json"


def _read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError(f"{path}: expected a list, or an object with a 'models' list")
    return models


class ModelRegistry:
    def __init__(self, specs: Iterable[ModelSpec]) -> None:
        self._by_id: dict[str, ModelSpec] = {}
        for spec in specs:
            self._by_id[spec.id] = spec

    @classmethod
    def load(cls, builtin: Path = BUILTIN_REGISTRY, extra: Path | None = None) -> "ModelRegistry":
        rows = _read(builtin)
        if extra is not None:
            if extra.is_file():
                rows = rows + _read(extra)
            else:
                log.warning("AI_MODEL_REGISTRY points at %s, which does not exist — ignoring", extra)
        registry = cls(ModelSpec.model_validate(row) for row in rows)
        log.info("registry loaded: %d models", len(registry))
        return registry

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._by_id

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._by_id[model_id]
        except KeyError:
            raise ModelNotFound(f"Unknown model: {model_id}") from None

    def find(self, model_id: str) -> ModelSpec | None:
        return self._by_id.get(model_id)

    def all(self) -> list[ModelSpec]:
        return sorted(self._by_id.values(), key=lambda spec: (spec.type, spec.name))

    def by_type(self, model_type: str) -> list[ModelSpec]:
        return [spec for spec in self.all() if spec.type == model_type]

    def by_task(self, task: str) -> list[ModelSpec]:
        """Every enabled model that declares this task. The router picks from
        here; a disabled row is registered but never routed to."""
        return [spec for spec in self.all() if spec.enabled and task in spec.tasks]
