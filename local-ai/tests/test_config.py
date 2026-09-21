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


def test_an_env_file_is_read_so_cp_env_example_actually_does_something(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("AI_ROOT=/srv/ai\nAI_PORT=9100\n", encoding="utf-8")
    settings = load_settings({}, env_file=env_file)
    assert settings.root == Path("/srv/ai")
    assert settings.port == 9100


def test_the_real_environment_wins_over_the_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("AI_PORT=9100\nAI_DTYPE=float16\n", encoding="utf-8")
    settings = load_settings({"AI_PORT": "8123"}, env_file=env_file)
    # Compose's environment must not be overridden by a stray file in the image.
    assert settings.port == 8123
    assert settings.dtype == "float16"


def test_an_empty_environment_variable_does_not_mask_the_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=hf_from_file\n", encoding="utf-8")
    # compose writes HF_TOKEN: "" for an unset variable; that is absence.
    assert load_settings({"HF_TOKEN": ""}, env_file=env_file).hf_token == "hf_from_file"


def test_a_file_written_on_windows_does_not_smuggle_a_carriage_return(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"AI_ROOT=/srv/ai\r\nAI_DTYPE=float32\r\n")
    settings = load_settings({}, env_file=env_file)
    assert settings.root == Path("/srv/ai")
    assert settings.dtype == "float32"
    assert "\r" not in settings.dtype


def test_the_parser_handles_comments_quotes_blank_lines_and_export(tmp_path):
    from server.config import read_env_file

    env_file = tmp_path / ".env"
    env_file.write_text(
        '\n# a comment\nexport AI_DEVICE=cuda:1\nAI_CORS_ORIGINS="http://a,http://b"\n'
        "  AI_DTYPE = bfloat16  \nnonsense-line\n",
        encoding="utf-8",
    )
    values = read_env_file(env_file)
    assert values["AI_DEVICE"] == "cuda:1"
    assert values["AI_CORS_ORIGINS"] == "http://a,http://b"
    assert values["AI_DTYPE"] == "bfloat16"
    assert "nonsense-line" not in values


def test_a_missing_env_file_is_not_an_error(tmp_path):
    assert load_settings({}, env_file=tmp_path / "absent").port == 8000


def test_an_explicit_environment_reads_no_file_so_tests_stay_hermetic(tmp_path, monkeypatch):
    import server.config as config

    stray = tmp_path / ".env"
    stray.write_text("AI_PORT=9999\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", stray)
    assert config.load_settings({}).port == 8000
