import json

import pytest

from server.services.storage import Storage, dir_size


@pytest.fixture
def storage(tmp_path):
    (tmp_path / "outputs").mkdir()
    return Storage(tmp_path / "outputs")


def test_url_is_server_relative_so_the_studio_can_choose_the_origin(storage):
    path = storage.path_for("abc123")
    assert storage.url_for(path) == "/v1/assets/abc123.png"


def test_resolve_returns_a_real_file(storage):
    path = storage.path_for("abc123")
    path.write_bytes(b"x")
    assert storage.resolve("abc123.png") == path.resolve()


def test_resolve_refuses_to_escape_the_output_directory(storage, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("hunter2")
    for attempt in ("../secret.txt", "..%2Fsecret.txt", "/etc/passwd", "sub/../../secret.txt", ".hidden"):
        assert storage.resolve(attempt) is None


def test_resolve_returns_none_for_a_missing_file(storage):
    assert storage.resolve("nothing.png") is None


def test_content_type_is_read_from_the_extension(storage):
    assert storage.content_type(storage.path_for("a", ".png")) == "image/png"
    assert storage.content_type(storage.path_for("a", ".mp4")) == "video/mp4"
    assert storage.content_type(storage.path_for("a", ".xyz")) == "application/octet-stream"


def test_the_sidecar_sits_next_to_the_asset_and_carries_the_settings(storage):
    path = storage.path_for("abc123")
    path.write_bytes(b"x")
    sidecar = storage.write_sidecar(path, {"model": "flux1-schnell", "settings": {"seed": 3}})
    assert sidecar.name == "abc123.png.json"
    payload = json.loads(sidecar.read_text())
    assert payload["model"] == "flux1-schnell"
    assert payload["settings"]["seed"] == 3
    assert payload["created_at"] > 0


def test_dir_size_counts_nested_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one").write_bytes(b"0" * 100)
    (tmp_path / "two").write_bytes(b"0" * 50)
    assert dir_size(tmp_path) == 150
    assert dir_size(tmp_path / "absent") == 0
