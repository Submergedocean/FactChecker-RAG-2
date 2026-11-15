"""The FastAPI app: `/api/v1` JSON API, health checks at the root and, when
`api.static_dir` holds a build, the SPA on the same origin.

Run with one Uvicorn worker (`claimcheck api serve`): one copy of the model and index in RAM,
and the in-memory rate limiter is only correct in a single process.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.routing import Match, Mount
from starlette.types import ASGIApp, Scope

from claimcheck.api import errors
from claimcheck.api.deps import AppState
from claimcheck.api.routes import (
    evalsummary,
    health,
    meta,
    pages,
    passages,
    verifications,
    verify,
)
from claimcheck.api.security import (
    BodySizeLimitMiddleware,
    SecurityHeadersMiddleware,
    check_startup,
)
from claimcheck.config import Settings, get_settings

log = logging.getLogger(__name__)


class SPAStaticFiles(StaticFiles):
    """The built SPA; unknown paths get `index.html` so client routes (`/v/:id`) survive a
    reload. Only reached through `SPAMount`, so never for an API path."""

    def __init__(self, *, directory: Path) -> None:
        super().__init__(directory=directory, html=True)

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            return await super().get_response("index.html", scope)


class SPAMount(Mount):
    """A mount at "/" that matches only GET and HEAD outside the API prefix.

    A plain `Mount("/")` answers every path and method with a full match, which beats an API
    route's partial (method-only) match: `GET /api/v1/verify` would get the SPA's 404 instead of
    405, and `POST /api/v1/nope` the static files' 405 instead of 404. Declining here leaves
    those to the router, which answers with the error envelope."""

    def __init__(self, app: ASGIApp, *, api_prefix: str) -> None:
        super().__init__("/", app=app, name="spa")
        self.api_prefix = api_prefix.rstrip("/")

    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        if scope["type"] != "http" or scope.get("method") not in ("GET", "HEAD"):
            return Match.NONE, {}
        path = scope.get("path", "")
        if path == self.api_prefix or path.startswith(self.api_prefix + "/"):
            return Match.NONE, {}
        return super().matches(scope)


def create_app(settings: Settings | None = None, *, state: AppState | None = None) -> FastAPI:
    """The app. `state` replaces loading from settings (tests); startup refuses unsafe
    settings either way."""
    settings = settings or (state.settings if state is not None else get_settings())
    check_startup(settings)
    prefix = settings.api.prefix.rstrip("/")
    dev = settings.env != "prod"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.claimcheck = state if state is not None else AppState.load(settings)
        log.info(
            "ready: index %s, pipeline %s",
            app.state.claimcheck.service.index_version,
            app.state.claimcheck.service.pipeline_version,
        )
        yield

    docs_url = f"{prefix}/docs" if dev else None
    app = FastAPI(
        title="Claim Checker API",
        version="1.0",
        lifespan=lifespan,
        openapi_url=f"{prefix}/openapi.json",
        docs_url=docs_url,
        redoc_url=None,
    )
    errors.install(app)

    for router in (
        verify.router,
        verifications.router,
        passages.router,
        pages.router,
        meta.router,
        evalsummary.router,
        health.router,
    ):
        app.include_router(router, prefix=prefix)
    app.include_router(health.router, include_in_schema=False)

    static_dir = settings.api.static_dir
    if static_dir is not None and (static_dir / "index.html").is_file():
        app.router.routes.append(SPAMount(SPAStaticFiles(directory=static_dir), api_prefix=prefix))

    if settings.api.cors_origins:  # same origin in production; none needed
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(
        SecurityHeadersMiddleware,
        frame_ancestors=settings.api.frame_ancestors,
        docs_paths={docs_url} if docs_url else set(),
    )
    return app
