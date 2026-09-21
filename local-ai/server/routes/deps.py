"""Shared route dependency: reach the container hung off the app."""

from __future__ import annotations

from fastapi import Request

from ..container import Container


def container(request: Request) -> Container:
    return request.app.state.container
