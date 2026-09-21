"""Serves generated media off disk.

The studio's <img> points straight here, so this is a plain file response with
a long cache lifetime — an asset name is a uuid and its bytes never change.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..container import Container
from .deps import container

router = APIRouter(prefix="/v1/assets", tags=["assets"])


@router.get("/{filename}")
def get_asset(filename: str, app: Container = Depends(container)) -> FileResponse:
    path = app.storage.resolve(filename)
    if path is None:
        raise HTTPException(status_code=404, detail=f"No such asset: {filename}")
    return FileResponse(
        path,
        media_type=app.storage.content_type(path),
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
