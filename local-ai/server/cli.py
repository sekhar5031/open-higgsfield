"""A small command line over the same services the API uses.

Weights have to be on disk before the studio can generate, and asking someone
to curl a POST for that is poor hospitality.

    python -m server.cli models
    python -m server.cli install flux1-schnell
    python -m server.cli remove flux1-schnell
    python -m server.cli gpu
"""

from __future__ import annotations

import argparse
import sys
import time

from .container import Container
from .errors import LocalAIError
from .logging_setup import configure

GB = 1024**3


def _models(app: Container) -> int:
    states = app.router.load_states()
    print(f"{'ID':<16} {'TYPE':<6} {'STATE':<14} {'ON DISK':>9}  LICENCE")
    for spec in app.registry.all():
        status = app.models.status(spec, load_state=states.get(spec.id, "not_loaded"))
        size = f"{status.disk_bytes / GB:.1f} GB" if status.disk_bytes else "-"
        print(f"{spec.id:<16} {spec.type:<6} {status.install_state:<14} {size:>9}  {spec.license}")
    return 0


def _install(app: Container, model_id: str, force: bool) -> int:
    spec = app.registry.get(model_id)
    if spec.gated:
        print(f"note: {spec.repository} is gated — its licence must be accepted on the model card.")
    print(f"downloading {spec.repository} (~{spec.size_gb:.0f} GB) into {app.models.path_for(spec)}")
    task = app.models.start_install(spec, force=force)
    while task.state == "downloading":
        time.sleep(2)
    if task.error:
        print(f"failed: {task.error}", file=sys.stderr)
        return 1
    print(f"installed {spec.id}")
    return 0


def _remove(app: Container, model_id: str) -> int:
    spec = app.registry.get(model_id)
    reclaimed = app.models.uninstall(spec)
    print(f"removed {spec.id}, reclaimed {reclaimed / GB:.1f} GB")
    return 0


def _gpu(app: Container) -> int:
    report = app.gpu.report()
    print(f"device in use: {report.device_in_use}")
    for device in report.devices:
        if not device.available:
            print(f"  no CUDA device — {device.detail}")
            continue
        print(
            f"  [{device.index}] {device.name}: "
            f"{device.vram_free / GB:.1f} GB free of {device.vram_total / GB:.1f} GB"
            + (f", {device.utilization:.0f}% busy" if device.utilization is not None else "")
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="local-ai", description="Manage local models.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="list registered models and their install state")
    install = sub.add_parser("install", help="download a model's weights")
    install.add_argument("model_id")
    install.add_argument("--force", action="store_true", help="re-download even if complete")
    remove = sub.add_parser("remove", help="delete a model's weights")
    remove.add_argument("model_id")
    sub.add_parser("gpu", help="report the GPU")

    args = parser.parse_args(argv)
    configure(None)
    app = Container.build()
    try:
        if args.command == "models":
            return _models(app)
        if args.command == "install":
            return _install(app, args.model_id, args.force)
        if args.command == "remove":
            return _remove(app, args.model_id)
        return _gpu(app)
    except LocalAIError as exc:
        print(f"error: {exc.detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
