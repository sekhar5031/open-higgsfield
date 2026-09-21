"""The model library: what exists, what is on disk, install and delete."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..container import Container
from ..schemas.models import InstallRequest, ModelList, ModelStatus
from .deps import container

router = APIRouter(prefix="/v1/models", tags=["models"])


def _status(app: Container, model_id: str) -> ModelStatus:
    spec = app.registry.get(model_id)
    return app.models.status(spec, load_state=app.router.load_states().get(spec.id, "not_loaded"))


@router.get("", response_model=ModelList)
def list_models(
    app: Container = Depends(container),
    type: str | None = Query(default=None, description="Filter by model type"),
    task: str | None = Query(default=None, description="Filter by supported task"),
    installed: bool | None = Query(default=None, description="Filter by install state"),
) -> ModelList:
    load_states = app.router.load_states()
    rows = [
        app.models.status(spec, load_state=load_states.get(spec.id, "not_loaded"))
        for spec in app.registry.all()
        if (type is None or spec.type == type) and (task is None or task in spec.tasks)
    ]
    if installed is not None:
        rows = [row for row in rows if (row.install_state == "installed") is installed]
    return ModelList(models=rows)


@router.get("/{model_id}", response_model=ModelStatus)
def get_model(model_id: str, app: Container = Depends(container)) -> ModelStatus:
    return _status(app, model_id)


@router.post("/{model_id}/install", response_model=ModelStatus, status_code=202)
def install_model(
    model_id: str,
    body: InstallRequest = InstallRequest(),
    app: Container = Depends(container),
) -> ModelStatus:
    """Start a resumable download in the background and answer immediately.

    A 34 GB snapshot takes far longer than any sensible HTTP timeout, so this
    returns 202 and the caller watches `install_state` on GET /v1/models/{id}.
    """
    spec = app.registry.get(model_id)
    app.models.start_install(spec, revision=body.revision, force=body.force)
    return _status(app, model_id)


@router.delete("/{model_id}", response_model=ModelStatus)
def delete_model(model_id: str, app: Container = Depends(container)) -> ModelStatus:
    spec = app.registry.get(model_id)
    app.router.pool.unload(spec.id)
    app.models.uninstall(spec)
    return _status(app, model_id)


@router.post("/{model_id}/unload", response_model=ModelStatus)
def unload_model(model_id: str, app: Container = Depends(container)) -> ModelStatus:
    """Free the VRAM without deleting the weights."""
    spec = app.registry.get(model_id)
    app.router.pool.unload(spec.id)
    return _status(app, model_id)
