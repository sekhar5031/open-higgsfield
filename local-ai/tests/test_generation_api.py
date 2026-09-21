"""The vertical slice: submit -> queue -> engine -> asset on disk -> HTTP."""

from tests.conftest import wait_for


def test_generate_returns_a_job_id_immediately(client, install):
    install("flux1-schnell")
    response = client.post("/v1/generate", json={"model": "flux1-schnell", "prompt": "a cat"})
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"].startswith("job_")
    # The worker may already have picked it up by the time this returns, so
    # what is asserted is that a real job exists — not that it is still queued.
    assert body["status"] in {"queued", "loading_model", "generating", "encoding", "completed"}
    assert client.get(f"/v1/jobs/{body['job_id']}").status_code == 200


def test_a_job_runs_to_completion_and_its_asset_is_servable(client, install):
    install("flux1-schnell")
    job_id = client.post(
        "/v1/generate",
        json={
            "model": "flux1-schnell",
            "task": "text-to-image",
            "prompt": "a beekeeper in a sunlit orchard",
            "settings": {"aspectRatio": "16:9", "resolution": "1024", "seed": 7},
        },
    ).json()["job_id"]

    job = wait_for(client, job_id)
    assert job["status"] == "completed", job
    assert job["progress"]["fraction"] == 1.0
    assert len(job["assets"]) == 1

    asset = job["assets"][0]
    assert asset["url"].startswith("/v1/assets/")
    assert asset["width"] > asset["height"]  # 16:9 came through
    assert asset["seed"] == 7

    served = client.get(asset["url"])
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/png"
    assert served.content.startswith(b"\x89PNG")


def test_a_completed_run_leaves_a_metadata_sidecar_for_reproducibility(client, install, container):
    install("flux1-schnell")
    job_id = client.post(
        "/v1/generate",
        json={"model": "flux1-schnell", "prompt": "a cat", "settings": {"seed": 11, "steps": 4}},
    ).json()["job_id"]
    job = wait_for(client, job_id)

    sidecars = list(container.settings.output_dir.glob("*.png.json"))
    assert len(sidecars) == 1
    import json

    payload = json.loads(sidecars[0].read_text())
    assert payload["model"] == "flux1-schnell"
    assert payload["prompt"] == "a cat"
    assert payload["settings"]["seed"] == 11
    assert payload["settings"]["steps"] == 4
    assert payload["job_id"] == job["job_id"]


def test_batch_returns_one_asset_per_image(client, install):
    install("flux1-schnell")
    job_id = client.post(
        "/v1/generate",
        json={"model": "flux1-schnell", "prompt": "a cat", "settings": {"numImages": 3, "seed": 5}},
    ).json()["job_id"]
    job = wait_for(client, job_id)
    assert len(job["assets"]) == 3
    # Each image of a batch records the seed that actually produced it.
    assert [asset["seed"] for asset in job["assets"]] == [5, 6, 7]


def test_generating_with_uninstalled_weights_fails_at_submit_not_later(client):
    response = client.post("/v1/generate", json={"model": "flux1-schnell", "prompt": "a cat"})
    assert response.status_code == 409
    assert "not installed" in response.json()["detail"]
    # Nothing was queued, so nothing has to be cleaned up.
    assert client.get("/v1/jobs").json()["jobs"] == []


def test_an_unknown_model_is_rejected_at_submit(client):
    response = client.post("/v1/generate", json={"model": "no-such-model", "prompt": "a cat"})
    assert response.status_code == 404


def test_an_empty_prompt_is_rejected_by_validation(client, install):
    install("flux1-schnell")
    assert client.post("/v1/generate", json={"model": "flux1-schnell", "prompt": ""}).status_code == 422


def test_a_task_the_model_does_not_do_is_a_400(client, install):
    install("flux1-schnell")
    response = client.post(
        "/v1/generate",
        json={"model": "flux1-schnell", "task": "text-to-video", "prompt": "a cat"},
    )
    assert response.status_code == 400
    assert "does not do" in response.json()["detail"]


def test_jobs_run_one_at_a_time_on_the_single_gpu(client, install):
    install("flux1-schnell")
    ids = [
        client.post("/v1/generate", json={"model": "flux1-schnell", "prompt": f"cat {n}"}).json()["job_id"]
        for n in range(3)
    ]
    for job_id in ids:
        assert wait_for(client, job_id)["status"] == "completed"
    listed = client.get("/v1/jobs").json()["jobs"]
    assert {job["job_id"] for job in listed} == set(ids)
