import time

from tests.conftest import wait_for


def test_an_unknown_job_is_a_404(client):
    response = client.get("/v1/jobs/job_nope")
    assert response.status_code == 404
    assert "Unknown job" in response.json()["detail"]


def test_progress_is_reported_while_a_job_runs(client, install, fake_engine):
    install("flux1-schnell")
    fake_engine.step_delay = 0.05  # 4 steps ~ 200ms, long enough to observe
    job_id = client.post(
        "/v1/generate", json={"model": "flux1-schnell", "prompt": "a cat", "settings": {"steps": 4}}
    ).json()["job_id"]

    seen = set()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        seen.add(body["status"])
        if body["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)

    assert "completed" in seen
    # The studio needs something between "queued" and "done" to render.
    assert seen & {"loading_model", "generating", "encoding"}


def test_cancelling_a_running_job_stops_it(client, install, fake_engine):
    install("flux1-schnell")
    fake_engine.step_delay = 0.1
    job_id = client.post(
        "/v1/generate", json={"model": "flux1-schnell", "prompt": "a cat", "settings": {"steps": 12}}
    ).json()["job_id"]

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if client.get(f"/v1/jobs/{job_id}").json()["status"] == "generating":
            break
        time.sleep(0.01)

    assert client.post(f"/v1/jobs/{job_id}/cancel").status_code == 200
    job = wait_for(client, job_id)
    assert job["status"] == "cancelled"
    assert job["assets"] == []


def test_cancelling_a_queued_job_never_runs_it(client, install, fake_engine, container):
    install("flux1-schnell")
    fake_engine.step_delay = 0.1
    first = client.post(
        "/v1/generate", json={"model": "flux1-schnell", "prompt": "one", "settings": {"steps": 8}}
    ).json()["job_id"]
    second = client.post(
        "/v1/generate", json={"model": "flux1-schnell", "prompt": "two", "settings": {"steps": 8}}
    ).json()["job_id"]

    client.post(f"/v1/jobs/{second}/cancel")
    assert wait_for(client, second, timeout=15)["status"] == "cancelled"
    assert wait_for(client, first, timeout=15)["status"] == "completed"
    # Nothing was rendered for the cancelled job.
    assert not list(container.settings.output_dir.glob(f"{second}*.png"))


def test_cancelling_a_finished_job_is_a_no_op(client, install):
    install("flux1-schnell")
    job_id = client.post("/v1/generate", json={"model": "flux1-schnell", "prompt": "a cat"}).json()["job_id"]
    wait_for(client, job_id)
    assert client.post(f"/v1/jobs/{job_id}/cancel").json()["status"] == "completed"


def test_an_engine_failure_becomes_a_failed_job_with_a_reason(client, install, container):
    from engines.loader import register_engine

    class Broken:
        def __init__(self, device, dtype):
            pass

        def load_model(self, spec, path):
            raise RuntimeError("CUDA out of memory while loading the transformer")

        def unload_model(self):
            pass

    register_engine("diffusers-t2i", Broken)
    install("flux1-schnell")
    job_id = client.post("/v1/generate", json={"model": "flux1-schnell", "prompt": "a cat"}).json()["job_id"]
    job = wait_for(client, job_id)
    assert job["status"] == "failed"
    # OOM is translated into something the user can act on.
    assert "smaller resolution" in job["error"]


def test_the_worker_survives_a_failure_and_runs_the_next_job(client, install, fake_engine):
    install("flux1-schnell")
    bad = client.post(
        "/v1/generate", json={"model": "flux1-schnell", "prompt": "a cat", "settings": {"numImages": 1}}
    ).json()["job_id"]
    wait_for(client, bad)
    good = client.post("/v1/generate", json={"model": "flux1-schnell", "prompt": "a dog"}).json()["job_id"]
    assert wait_for(client, good)["status"] == "completed"
