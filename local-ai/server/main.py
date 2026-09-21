"""The Local AI API.

    OpenHiggsfield Studio -> this service -> model router -> job/GPU manager
    -> native engine (Diffusers/PyTorch) -> CUDA -> the GPU

ComfyUI is not involved at any layer. Weights are loaded and executed by the
model authors' own implementations, in this process.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings, load_settings
from .container import Container
from .errors import LocalAIError
from .logging_setup import configure
from .routes import assets, generation, jobs, models, system
from .version import VERSION

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or (container.settings if container else load_settings())
    configure(settings.log_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = container or Container.build(settings)
        app.state.container.jobs.start()
        log.info(
            "local-ai %s ready on %s — %d models registered, %d installed",
            VERSION,
            app.state.container.gpu.torch_device(),
            len(app.state.container.registry),
            app.state.container.models.installed_count(),
        )
        try:
            yield
        finally:
            app.state.container.shutdown()

    app = FastAPI(
        title="OpenHiggsfield Local AI",
        version=VERSION,
        summary="Local, native inference for the OpenHiggsfield studio. No ComfyUI.",
        lifespan=lifespan,
    )

    # The studio runs on a different port, so the browser needs this to fetch
    # generated assets and to download them.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins) or ["http://localhost:3000"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(LocalAIError)
    async def _domain_error(_: Request, exc: LocalAIError) -> JSONResponse:
        # One shape for every failure, so the studio can render `detail`
        # verbatim rather than guessing at the body.
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    app.include_router(system.router)
    app.include_router(models.router)
    app.include_router(generation.router)
    app.include_router(jobs.router)
    app.include_router(assets.router)

    @app.get("/", tags=["system"])
    def root() -> dict:
        return {"name": "openhiggsfield-local-ai", "version": VERSION, "docs": "/docs"}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = load_settings()
    uvicorn.run("server.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
