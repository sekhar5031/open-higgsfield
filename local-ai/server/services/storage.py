"""Where generated media and its metadata live on disk.

Every output is written next to a JSON sidecar naming the model, the resolved
settings and the seed. Reproducibility is a file on disk, not a log line that
rotates away.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Extensions the asset route is allowed to serve, mapped to their media type.
CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".wav": "audio/wav",
    ".glb": "model/gltf-binary",
}


@dataclass(frozen=True)
class StoredAsset:
    id: str
    path: Path
    url: str
    content_type: str


class Storage:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def new_asset_id(self) -> str:
        return uuid.uuid4().hex

    def path_for(self, asset_id: str, suffix: str = ".png") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / f"{asset_id}{suffix}"

    def url_for(self, path: Path) -> str:
        """Server-relative on purpose: the studio prefixes it with whatever
        origin the browser can actually reach the API on, which is not
        necessarily the origin the Next server used."""
        return f"/v1/assets/{path.name}"

    def resolve(self, filename: str) -> Path | None:
        """Resolve a requested asset name inside the output directory, or None.

        Rejects anything that escapes the directory — the name arrives from a
        URL path, so `..` and absolute paths have to die here.
        """
        if not filename or "/" in filename or "\\" in filename or filename.startswith("."):
            return None
        candidate = (self.output_dir / filename).resolve()
        root = self.output_dir.resolve()
        if root != candidate.parent:
            return None
        return candidate if candidate.is_file() else None

    def content_type(self, path: Path) -> str:
        return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")

    def write_sidecar(self, path: Path, metadata: dict) -> Path:
        sidecar = path.with_suffix(path.suffix + ".json")
        payload = {"created_at": time.time(), **metadata}
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return sidecar

    def free_bytes(self) -> int:
        target = self.output_dir if self.output_dir.exists() else self.output_dir.parent
        while not target.exists() and target != target.parent:
            target = target.parent
        try:
            return shutil.disk_usage(target).free
        except OSError as exc:  # pragma: no cover - platform dependent
            log.warning("could not read free disk space at %s: %s", target, exc)
            return 0


def dir_size(path: Path) -> int:
    """Bytes a model directory occupies, following the symlinks the Hugging
    Face cache uses so the reported size is the space actually consumed."""
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total
