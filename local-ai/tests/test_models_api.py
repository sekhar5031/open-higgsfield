from server.container import Container
from server.main import create_app
from fastapi.testclient import TestClient


def test_list_reports_every_registered_model_with_its_install_state(client):
    body = client.get("/v1/models").json()
    ids = {row["spec"]["id"] for row in body["models"]}
    assert {"flux1-schnell", "flux1-dev", "qwen-image"} <= ids
    assert all(row["install_state"] == "not_installed" for row in body["models"])


def test_list_filters_by_type_task_and_install_state(client, install):
    install("flux1-schnell")
    assert client.get("/v1/models", params={"type": "video"}).json()["models"] == []
    tasks = client.get("/v1/models", params={"task": "text-to-image"}).json()["models"]
    assert len(tasks) >= 3
    installed = client.get("/v1/models", params={"installed": True}).json()["models"]
    assert [row["spec"]["id"] for row in installed] == ["flux1-schnell"]


def test_get_reports_disk_usage_and_revision_once_installed(client, install):
    install("flux1-schnell")
    body = client.get("/v1/models/flux1-schnell").json()
    assert body["install_state"] == "installed"
    assert body["installed_revision"] == "deadbeef"
    assert body["disk_bytes"] > 0
    assert body["path"]


def test_an_unknown_model_is_a_404_with_a_readable_detail(client):
    response = client.get("/v1/models/nope")
    assert response.status_code == 404
    assert "Unknown model" in response.json()["detail"]


def test_installing_a_gated_repository_without_a_token_is_refused_before_any_download(client):
    response = client.post("/v1/models/flux1-dev/install")
    assert response.status_code == 403
    detail = response.json()["detail"]
    assert "gated" in detail and "HF_TOKEN" in detail


def test_offline_mode_refuses_to_download(settings, fake_engine):
    offline = Container.build(settings)
    offline.settings = settings
    from dataclasses import replace

    offline.models.settings = replace(settings, offline=True)
    with TestClient(create_app(container=offline)) as client:
        response = client.post("/v1/models/flux1-schnell/install")
    assert response.status_code == 503
    assert "AI_OFFLINE" in response.json()["detail"]


def test_installing_an_already_installed_model_is_a_no_op(client, install):
    install("flux1-schnell")
    response = client.post("/v1/models/flux1-schnell/install")
    assert response.status_code == 202
    assert response.json()["install_state"] == "installed"


def test_delete_reclaims_the_directory(client, install, container):
    path = install("flux1-schnell")
    assert path.exists()
    body = client.delete("/v1/models/flux1-schnell").json()
    assert body["install_state"] == "not_installed"
    assert not path.exists()


def test_files_without_a_marker_read_as_corrupt_not_installed(client, container):
    spec = container.registry.get("flux1-schnell")
    path = container.models.path_for(spec)
    path.mkdir(parents=True, exist_ok=True)
    (path / "half-a-file.safetensors").write_bytes(b"0" * 16)
    body = client.get("/v1/models/flux1-schnell").json()
    assert body["install_state"] == "corrupt"
    assert "Interrupted" in body["message"]


def test_unload_frees_vram_without_touching_the_weights(client, install, container, fake_engine):
    path = install("flux1-schnell")
    container.router.pool.acquire(container.registry.get("flux1-schnell"), path)
    assert client.get("/v1/models/flux1-schnell").json()["load_state"] == "ready"
    body = client.post("/v1/models/flux1-schnell/unload").json()
    assert body["load_state"] == "not_loaded"
    assert body["install_state"] == "installed"
    assert path.exists()
