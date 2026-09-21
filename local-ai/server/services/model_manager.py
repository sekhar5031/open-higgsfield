"""Weights on disk: what is installed, how big it is, how it got there.

Downloads go through the repository's own documented mechanism — Hugging Face
`snapshot_download`, which is resumable and verifies each file's hash against
the repository metadata. No URLs are hardcoded anywhere.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

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
class Verification:
    """What the repository actually says, against what the registry claims.

    Registry rows are written by hand from model cards, and model cards move:
    repositories are renamed, relicensed, gated, or reorganised. Checking costs
    one API call and catches a wrong row in seconds rather than at the end of a
    34 GB download — or, worse, after it succeeds against the wrong weights.
    """

    model_id: str
    repository: str
    exists: bool
    ok: bool
    gated: bool = False
    license: str | None = None
    download_bytes: int = 0
    file_count: int = 0
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def wanted_files(
    names: Iterable[str],
    allow_patterns: tuple[str, ...] | None,
    ignore_patterns: tuple[str, ...],
) -> list[str]:
    """The files snapshot_download would actually fetch, by the same rules."""
    kept = []
    for name in names:
        if allow_patterns and not any(fnmatch.fnmatch(name, p) for p in allow_patterns):
            continue
        if any(fnmatch.fnmatch(name, p) for p in ignore_patterns):
            continue
        kept.append(name)
    return kept


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

    # -- verify -----------------------------------------------------------

    def verify(self, spec: ModelSpec, model_info: Callable | None = None) -> Verification:
        """Check a registry row against the live repository. Never downloads."""
        if spec.source != "huggingface":
            return Verification(
                model_id=spec.id,
                repository=spec.repository,
                exists=False,
                ok=False,
                problems=[f"verification is only implemented for huggingface, not {spec.source}"],
            )
        if model_info is None:
            try:
                from huggingface_hub import HfApi  # noqa: PLC0415 - lazy
            except ImportError as exc:
                raise DownloadError(f"huggingface_hub is not installed ({exc})") from exc
            model_info = HfApi().model_info

        try:
            info = model_info(
                spec.repository,
                revision=spec.revision,
                files_metadata=True,
                token=self.settings.hf_token,
            )
        except Exception as exc:
            return Verification(
                model_id=spec.id,
                repository=spec.repository,
                exists=False,
                ok=False,
                problems=[f"could not read the repository: {_short(exc)}"],
            )

        problems: list[str] = []
        notes: list[str] = []

        gated = bool(getattr(info, "gated", False) or False)
        if gated and not spec.gated:
            problems.append("repository is gated but the registry says it is not — set gated: true")
        elif spec.gated and not gated:
            notes.append("registry marks this gated; the repository no longer is")
        if gated and not self.settings.hf_token:
            problems.append("gated: accept the licence on the model card and set HF_TOKEN")

        card = getattr(info, "card_data", None) or getattr(info, "cardData", None) or {}
        license_id = card.get("license") if hasattr(card, "get") else None
        if license_id and license_id.lower().replace(" ", "-") not in spec.license.lower().replace(" ", "-"):
            notes.append(f"repository licence is {license_id!r}; registry says {spec.license!r}")

        siblings = list(getattr(info, "siblings", None) or [])
        names = [getattr(sib, "rfilename", "") for sib in siblings]
        keep = set(wanted_files(names, spec.allow_patterns, spec.ignore_patterns))
        total = sum(
            int(getattr(sib, "size", 0) or 0)
            for sib in siblings
            if getattr(sib, "rfilename", "") in keep
        )

        if not keep:
            problems.append("no files match this row's allow/ignore patterns")
        if not any(name.endswith((".safetensors", ".gguf")) for name in keep):
            problems.append("no .safetensors in the download set — check the patterns")

        actual_gb = total / 1024**3
        if total and abs(actual_gb - spec.size_gb) > max(2.0, spec.size_gb * 0.25):
            notes.append(f"download is {actual_gb:.1f} GB; registry says {spec.size_gb:.1f} GB")

        fits = self.gpu.fits_device(spec.vram_gb)
        if fits is False:
            problems.append(f"needs ~{spec.vram_gb:.0f} GB of VRAM, which exceeds this device")

        return Verification(
            model_id=spec.id,
            repository=spec.repository,
            exists=True,
            ok=not problems,
            gated=gated,
            license=license_id,
            download_bytes=total,
            file_count=len(keep),
            problems=problems,
            notes=notes,
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


def _short(exc: Exception) -> str:
    text = str(exc).strip().splitlines()
    return text[0][:200] if text else exc.__class__.__name__
