"""Errors the API turns into a status code and a sentence a user can act on."""

from __future__ import annotations


class LocalAIError(Exception):
    status_code = 500

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class ModelNotFound(LocalAIError):
    status_code = 404


class ModelNotInstalled(LocalAIError):
    status_code = 409


class ModelDisabled(LocalAIError):
    status_code = 409


class UnsupportedTask(LocalAIError):
    status_code = 400


class InsufficientVram(LocalAIError):
    status_code = 507


class EngineNotFound(LocalAIError):
    status_code = 501


class JobNotFound(LocalAIError):
    status_code = 404


class DownloadError(LocalAIError):
    status_code = 502


class GatedRepository(LocalAIError):
    status_code = 403


class OfflineError(LocalAIError):
    status_code = 503


class Cancelled(Exception):
    """Raised inside an engine when a job was cancelled mid-step."""
