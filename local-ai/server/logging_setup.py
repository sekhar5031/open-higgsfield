"""Logging that is useful at 2am: one line per state change, to stderr and to
a file under AI_LOG_DIR."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure(log_dir: Path | None = None, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(getattr(handler, "_local_ai", False) for handler in root.handlers):
        return
    root.setLevel(level)

    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter(FORMAT))
    stream._local_ai = True  # type: ignore[attr-defined]
    root.addHandler(stream)

    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            rotating = logging.handlers.RotatingFileHandler(
                log_dir / "local-ai.log", maxBytes=8 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            rotating.setFormatter(logging.Formatter(FORMAT))
            rotating._local_ai = True  # type: ignore[attr-defined]
            root.addHandler(rotating)
        except OSError:
            logging.getLogger(__name__).warning("could not open a log file in %s", log_dir)

    # The hub's per-file progress bars are noise in a server log.
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
