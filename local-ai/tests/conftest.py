"""Test fixtures.

The suite runs on any machine: no GPU, no torch, no weights. A fake engine is
registered behind the real engine key, so everything above the engine boundary
— routing, admission, the queue, progress, storage, the HTTP surface — is
exercised for real.
"""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engines.base import EngineOutput, EngineRequest, EngineResult, InferenceEngine
from engines.loader import register_engine, unregister_engine
from server.config import load_settings
from server.container import Container
from server.errors import Cancelled
from server.main import create_app
from server.schemas.models import ModelSpec

# A 1x1 opaque PNG — enough for the asset route to have real bytes to serve.
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class FakeEngine(InferenceEngine):
    """Stands in for Diffusers. Honours steps, seeds, cancellation and progress."""

    key = "diffusers-t2i"
    #: Seconds per simulated step; raised by the cancellation test.
    step_delay = 0.0
    loads: list[str] = []
    unloads: list[str] = []

    def __init__(self, device: str = "cpu", dtype: str = "float32") -> None:
        super().__init__(device=device, dtype=dtype)
        self._path: Path | None = None

    def load_model(self, spec: ModelSpec, path: Path) -> None:
        self.spec = spec
        self._path = path
        FakeEngine.loads.append(spec.id)

    def unload_model(self) -> None:
        if self.spec is not None:
            FakeEngine.unloads.append(self.spec.id)
        self.spec = None
        self._path = None

    def generate(self, request: EngineRequest, report=None) -> EngineResult:
        for step in range(request.steps):
            if self._is_cancelled(request.job_id):
                self._clear_cancel(request.job_id)
                raise Cancelled(request.job_id)
            if FakeEngine.step_delay:
                time.sleep(FakeEngine.step_delay)
            self._set_progress("generating", step + 1, request.steps, report)

        self._set_progress("encoding", request.steps, request.steps, report)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for index in range(request.num_images):
            path = request.output_dir / f"{request.job_id}-{index}.png"
            path.write_bytes(PNG_1X1)
            outputs.append(
                EngineOutput(
                    path=path,
                    width=request.width,
                    height=request.height,
                    seed=request.seed + index,
                )
            )
        return EngineResult(outputs=outputs, meta={"engine": "fake", "seconds": 0.0})


@pytest.fixture
def fake_engine():
    FakeEngine.loads = []
    FakeEngine.unloads = []
    FakeEngine.step_delay = 0.0
    register_engine("diffusers-t2i", lambda device, dtype: FakeEngine(device, dtype))
    yield FakeEngine
    unregister_engine("diffusers-t2i")


@pytest.fixture
def settings(tmp_path: Path):
    return load_settings(
        {
            "AI_ROOT": str(tmp_path),
            "AI_DEVICE": "cpu",
            "AI_DTYPE": "float32",
            "AI_RESIDENT_MODELS": "1",
        }
    )


@pytest.fixture
def container(settings) -> Container:
    return Container.build(settings)


@pytest.fixture
def install(container: Container):
    """Mark a model installed without downloading 34 GB."""

    def _install(model_id: str) -> Path:
        spec = container.registry.get(model_id)
        path = container.models.path_for(spec)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model_index.json").write_text("{}", encoding="utf-8")
        container.models.marker_for(spec).write_text(
            '{"complete": true, "repository": "%s", "revision": "main", "commit": "deadbeef"}'
            % spec.repository,
            encoding="utf-8",
        )
        return path

    return _install


@pytest.fixture
def client(container: Container, fake_engine):
    app = create_app(container=container)
    with TestClient(app) as test_client:
        yield test_client


def wait_for(client: TestClient, job_id: str, *, timeout: float = 10.0) -> dict:
    """Poll a job the way the studio does, until it reaches a terminal state."""
    deadline = time.monotonic() + timeout
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not settle in {timeout}s: {body}")


@pytest.fixture
def barrier() -> threading.Event:
    return threading.Event()
