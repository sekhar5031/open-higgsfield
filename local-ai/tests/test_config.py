from pathlib import Path

import pytest

from server.config import DEFAULT_ROOT, load_settings


def test_defaults_land_outside_the_repo_and_outside_slash_ai():
    settings = load_settings({})
    assert settings.root == DEFAULT_ROOT
    assert settings.model_dir == DEFAULT_ROOT / "models"
    # The brief's /AI is an example, not an assumption.
    assert not str(settings.root).startswith("/AI")
    assert "open-higgsfield/local-ai" not in str(settings.model_dir)


def test_every_path_is_overridable(tmp_path: Path):
    settings = load_settings(
        {
            "AI_ROOT": str(tmp_path),
            "AI_MODEL_DIR": str(tmp_path / "weights"),
            "AI_OUTPUT_DIR": str(tmp_path / "out"),
        }
    )
    assert settings.model_dir == tmp_path / "weights"
    assert settings.output_dir == tmp_path / "out"
    # Unset members still hang off the overridden root.
    assert settings.cache_dir == tmp_path / "cache"


def test_ensure_dirs_creates_a_folder_per_model_type(tmp_path: Path):
    settings = load_settings({"AI_ROOT": str(tmp_path)})
    settings.ensure_dirs()
    for model_type in ("image", "video", "audio", "3d", "llm"):
        assert (settings.model_dir / model_type).is_dir()


def test_booleans_and_ints_are_parsed():
    settings = load_settings({"AI_OFFLINE": "yes", "AI_PORT": "9001", "AI_RESIDENT_MODELS": "3"})
    assert settings.offline is True
    assert settings.port == 9001
    assert settings.resident_models == 3


def test_a_bad_int_is_rejected_at_startup_not_at_request_time():
    with pytest.raises(ValueError, match="AI_PORT"):
        load_settings({"AI_PORT": "eight thousand"})


def test_hf_token_accepts_either_spelling():
    assert load_settings({"HF_TOKEN": "hf_x"}).hf_token == "hf_x"
    assert load_settings({"HUGGING_FACE_HUB_TOKEN": "hf_y"}).hf_token == "hf_y"
    assert load_settings({}).hf_token is None
