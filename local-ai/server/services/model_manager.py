"""Weights on disk: what is installed, how big it is, how it got there.

Downloads go through the repository's own documented mechanism — Hugging Face
`snapshot_download`, which is resumable and verifies each file's hash against
the repository metadata. No URLs are hardcoded anywhere.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..errors import DownloadError, GatedRepository, LocalAIError, OfflineError
from ..schemas.models import InstallState, ModelSpec, ModelStatus
from .gpu_manager import GpuManager
from .model_registry import ModelRegistry
from .storage import dir_size

log = logging.getLogger(__name__)

#: Written into a model directory once its snapshot is complete. Its presence
#: is what "installed" means — a half-finished download leaves no marker, so an
#: interrupted install is never mistaken for a usable one.
MARKER = ".openhiggsfield-install.json"


@dataclass
class InstallTask:
    model_id: str
    state: InstallState = "downloading"
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None


class ModelManager:
    def __init__(self, settings: Settings, registry: ModelRegistry, gpu: GpuManager) -> None:
        self.settings = settings
        self.registry = registry
        self.gpu = gpu
        self._installs: dict[str, InstallTask] = {}
        self._lock = threading.Lock()

    # -- paths ------------------------------------------------------------

    def path_for(self, spec: ModelSpec) -> Path:
        return self.settings.dir_for(spec.type) / spec.id

    def marker_for(self, spec: ModelSpec) -> Path:
        return self.path_for(spec) / MARKER

    def read_marker(self, spec: ModelSpec) -> dict | None:
        marker = self.marker_for(spec)
        if not marker.is_file():
            return None
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    # -- state ------------------------------------------------------------

    def is_installed(self, spec: ModelSpec) -> bool:
        marker = self.read_marker(spec)
        return bool(marker and marker.get("complete") is True)

    def install_state(self, spec: ModelSpec) -> InstallState:
        with self._lock:
            task = self._installs.get(spec.id)
        if task is not None and task.state == "downloading":
            return "downloading"
        if self.is_installed(spec):
            return "installed"
        # Files with no marker are the wreckage of an interrupted download.
        path = self.path_for(spec)
        if path.exists() and any(path.iterdir()):
            return "corrupt"
        return "not_installed"

    def status(self, spec: ModelSpec, load_state: str = "not_loaded") -> ModelStatus:
        state = self.install_state(spec)
        marker = self.read_marker(spec)
        with self._lock:
            task = self._installs.get(spec.id)
        return ModelStatus(
            spec=spec,
            install_state=state,
            load_state=load_state,  # type: ignore[arg-type]
            path=str(self.path_for(spec)) if state != "not_installed" else None,
            disk_bytes=dir_size(self.path_for(spec)),
            installed_revision=(marker or {}).get("commit") or (marker or {}).get("revision"),
            fits_gpu=self.gpu.fits_device(spec.vram_gb),
            message=(task.error if task and task.error else None)
            or ("Interrupted download — install again to resume." if state == "corrupt" else None),
        )

    # -- install ----------------------------------------------------------

    def start_install(self, spec: ModelSpec, revision: str | None = None, force: bool = False) -> InstallTask:
        """Kick a download off in the background and return immediately.

        Downloads take tens of minutes for a 34 GB repository; the HTTP request
        that asked for one must not be the thing holding it open.
        """
        if self.settings.offline:
            raise OfflineError("AI_OFFLINE is set — refusing to download weights.")
        if spec.gated and not self.settings.hf_token:
            raise GatedRepository(
                f"{spec.repository} is a gated repository. Accept its licence at "
                f"{spec.license_url or 'its model card'} and set HF_TOKEN before installing."
            )
        with self._lock:
            running = self._installs.get(spec.id)
            if running is not None and running.state == "downloading":
                return running
            if self.is_installed(spec) and not force:
                done = InstallTask(model_id=spec.id, state="installed", finished_at=time.time())
                self._installs[spec.id] = done
                return done
            task = InstallTask(model_id=spec.id)
            self._installs[spec.id] = task

        thread = threading.Thread(
            target=self._run_install,
            args=(spec, revision or spec.revision, force, task),
            name=f"install-{spec.id}",
            daemon=True,
        )
        thread.start()
        return task

    def install_task(self, model_id: str) -> InstallTask | None:
        with self._lock:
            return self._installs.get(model_id)

    def _run_install(self, spec: ModelSpec, revision: str, force: bool, task: InstallTask) -> None:
        target = self.path_for(spec)
        try:
            if force and target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            commit = self._download(spec, revision, target)
            self.marker_for(spec).write_text(
                json.dumps(
                    {
                        "complete": True,
                        "repository": spec.repository,
                        "revision": revision,
                        "commit": commit,
                        "source": spec.source,
                        "installed_at": time.time(),
                        "bytes": dir_size(target),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            task.state = "installed"
            log.info("installed %s into %s", spec.id, target)
        except Exception as exc:
            task.state = "not_installed"
            task.error = str(exc)
            log.exception("install failed for %s", spec.id)
        finally:
            task.finished_at = time.time()

    def _download(self, spec: ModelSpec, revision: str, target: Path) -> str | None:
        if spec.source == "huggingface":
            return self._download_hf(spec, revision, target)
        if spec.source == "modelscope":
            return self._download_modelscope(spec, revision, target)
        raise DownloadError(f"{spec.source} downloads are not implemented yet")

    def _download_hf(self, spec: ModelSpec, revision: str, target: Path) -> str | None:
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415 - lazy
        except ImportError as exc:
            raise DownloadError(
                "huggingface_hub is not installed. Install the runtime extras: "
                "pip install -e '.[gpu]'"
            ) from exc
        log.info("downloading %s@%s -> %s", spec.repository, revision, target)
        path = snapshot_download(
            repo_id=spec.repository,
            revision=revision,
            local_dir=str(target),
            cache_dir=str(self.settings.cache_dir),
            token=self.settings.hf_token,
            allow_patterns=list(spec.allow_patterns) if spec.allow_patterns else None,
            ignore_patterns=list(spec.ignore_patterns) or None,
            max_workers=4,
        )
        return self._commit_of(Path(path))

    def _download_modelscope(self, spec: ModelSpec, revision: str, target: Path) -> str | None:
        try:
            from modelscope import snapshot_download  # noqa: PLC0415 - lazy, optional
        except ImportError as exc:
            raise DownloadError(
                "modelscope is not installed; add it to use a ModelScope-hosted model."
            ) from exc
        snapshot_download(spec.repository, revision=revision, cache_dir=str(target))
        return revision

    @staticmethod
    def _commit_of(path: Path) -> str | None:
        """The resolved commit, so a reinstall can be told from an update."""
        head = path / ".cache" / "huggingface" / "download"
        if not head.exists():
            return None
        for candidate in head.rglob("*.metadata"):
            try:
                first = candidate.read_text(encoding="utf-8").splitlines()[0].strip()
            except (OSError, IndexError):
                continue
            if first:
                return first
        return None

    # -- delete -----------------------------------------------------------

    def uninstall(self, spec: ModelSpec) -> int:
        """Remove a model's weights. Returns the bytes reclaimed."""
        target = self.path_for(spec)
        if not target.exists():
            return 0
        with self._lock:
            task = self._installs.get(spec.id)
            if task is not None and task.state == "downloading":
                raise LocalAIError(f"{spec.id} is still downloading — cancel or wait before deleting.")
        size = dir_size(target)
        shutil.rmtree(target)
        with self._lock:
            self._installs.pop(spec.id, None)
        log.info("uninstalled %s, reclaimed %.1f GB", spec.id, size / 1024**3)
        return size

    def installed_count(self) -> int:
        return sum(1 for spec in self.registry.all() if self.is_installed(spec))
