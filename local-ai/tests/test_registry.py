import json
from pathlib import Path

import pytest

from server.errors import ModelNotFound
from server.services.model_registry import BUILTIN_REGISTRY, ModelRegistry


def test_builtin_registry_parses():
    registry = ModelRegistry.load()
    assert len(registry) >= 3
    assert "flux1-schnell" in registry


def test_every_builtin_row_declares_its_licence_and_engine():
    for spec in ModelRegistry.load().all():
        assert spec.license, f"{spec.id} has no licence"
        assert spec.engine, f"{spec.id} has no engine"
        assert spec.tasks, f"{spec.id} declares no task"
        assert spec.vram_gb > 0 and spec.size_gb > 0


def test_gated_models_are_flagged_so_install_can_refuse_early():
    registry = ModelRegistry.load()
    assert registry.get("flux1-dev").gated is True
    assert registry.get("flux1-schnell").gated is False


def test_unknown_model_raises_a_404_shaped_error():
    with pytest.raises(ModelNotFound):
        ModelRegistry.load().get("no-such-model")
    assert ModelRegistry.load().find("no-such-model") is None


def test_by_task_skips_disabled_rows(tmp_path: Path):
    extra = tmp_path / "extra.json"
    extra.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "disabled-one",
                        "name": "Disabled",
                        "type": "image",
                        "tasks": ["text-to-image"],
                        "repository": "org/disabled",
                        "engine": "diffusers-t2i",
                        "size_gb": 1,
                        "vram_gb": 1,
                        "ram_gb": 1,
                        "license": "Apache-2.0",
                        "enabled": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = ModelRegistry.load(extra=extra)
    assert "disabled-one" in registry
    assert "disabled-one" not in {spec.id for spec in registry.by_task("text-to-image")}


def test_extra_registry_overrides_a_builtin_row_by_id(tmp_path: Path):
    extra = tmp_path / "extra.json"
    builtin = json.loads(BUILTIN_REGISTRY.read_text())["models"][0]
    extra.write_text(json.dumps({"models": [{**builtin, "name": "Renamed"}]}), encoding="utf-8")
    registry = ModelRegistry.load(extra=extra)
    assert registry.get(builtin["id"]).name == "Renamed"


def test_a_missing_extra_registry_is_a_warning_not_a_crash(tmp_path: Path):
    registry = ModelRegistry.load(extra=tmp_path / "absent.json")
    assert len(registry) >= 3


def test_model_ids_are_path_safe():
    from server.schemas.models import ModelSpec

    row = dict(
        id="../escape",
        name="x",
        type="image",
        tasks=["text-to-image"],
        repository="org/x",
        engine="diffusers-t2i",
        size_gb=1,
        vram_gb=1,
        ram_gb=1,
        license="Apache-2.0",
    )
    with pytest.raises(ValueError):
        ModelSpec.model_validate(row)
