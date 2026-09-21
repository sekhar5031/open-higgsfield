def test_health_reports_the_version(client):
    body = client.get("/v1/system/health").json()
    assert body["status"] == "ok"
    assert body["version"]


def test_gpu_report_degrades_honestly_without_cuda(client):
    body = client.get("/v1/system/gpu").json()
    assert body["devices"]
    device = body["devices"][0]
    if not device["available"]:
        # A machine with no CUDA says why, rather than reporting a fake card.
        assert device["detail"]
    else:
        assert device["name"] and device["vram_total"] > 0


def test_info_lists_the_configured_paths_and_model_counts(client, install, container):
    install("flux1-schnell")
    body = client.get("/v1/system/info").json()
    assert body["paths"]["models"] == str(container.settings.model_dir)
    assert body["paths"]["outputs"] == str(container.settings.output_dir)
    assert body["installed_models"] == 1
    assert body["registered_models"] >= 3
    assert body["device"] in {"cpu", "cuda"} or body["device"].startswith("cuda:")


def test_the_root_document_points_at_the_docs(client):
    assert client.get("/").json()["docs"] == "/docs"


def test_cors_allows_the_studio_origin(client):
    response = client.get("/v1/system/health", headers={"Origin": "http://localhost:3000"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
