"""Application state, created once in the FastAPI lifespan and injected.

Immutable singletons: settings and `ClaimCheckService` (which holds the index, the embedder,
the LLM client, the source index and the verification store). Mutable state: SQLite and the
in-memory rate-limit buckets.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import Engine, text

from claimcheck.api.render import PageRenderer
from claimcheck.api.security import RateLimiter, client_id
from claimcheck.config import Settings
from claimcheck.pipeline.service import ClaimCheckService
from claimcheck.pipeline.sources import SourceIndex
from claimcheck.schemas import ChunkRecord

log = logging.getLogger(__name__)


@dataclass
class AppState:
    settings: Settings
    service: ClaimCheckService
    renderer: PageRenderer
    verify_limiter: RateLimiter
    feedback_limiter: RateLimiter
    pages_limiter: RateLimiter
    llm_slots: threading.BoundedSemaphore = field(
        default_factory=lambda: threading.BoundedSemaphore(4)
    )

    # ---------------------------------------------------------------- shortcuts

    @property
    def sources(self) -> SourceIndex:
        return self.service.sources

    @property
    def chunks(self) -> tuple[ChunkRecord, ...]:
        return self.service.retriever.store.chunks

    @property
    def row_of(self) -> dict[str, int]:
        return self.service.retriever.store.row_of

    @property
    def engine(self) -> Engine | None:
        return None if self.service.store is None else self.service.store.engine

    def checks(self) -> dict[str, bool]:
        """Readiness: index loaded, database open."""
        index_ok = len(self.chunks) > 0
        db_ok = False
        if self.engine is not None:
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                db_ok = True
            except Exception as exc:  # noqa: BLE001 — readiness reports, never raises
                log.warning("database check failed: %s", exc)
        return {"index": index_ok, "database": db_ok}

    # ---------------------------------------------------------------- construction

    @classmethod
    def build(cls, settings: Settings, service: ClaimCheckService) -> AppState:
        lim = settings.limits
        return cls(
            settings=settings,
            service=service,
            renderer=PageRenderer(
                settings.paths.raw_dir,
                settings.paths.page_cache,
                zoom=settings.api.page_zoom,
                cap_mb=settings.api.page_cache_mb,
            ),
            verify_limiter=RateLimiter(
                lim.per_client_requests_per_minute, lim.per_client_requests_per_day
            ),
            feedback_limiter=RateLimiter(
                lim.per_client_feedback_per_minute, lim.per_client_feedback_per_day
            ),
            pages_limiter=RateLimiter(
                lim.per_client_pages_per_minute, lim.per_client_pages_per_day
            ),
            llm_slots=threading.BoundedSemaphore(lim.max_concurrent_llm),
        )

    @classmethod
    def load(cls, settings: Settings) -> AppState:
        """Load the index, embedder and LLM client (the slow part of startup)."""
        llm = None
        if settings.fake_llm:
            from claimcheck.api.fake_llm import CannedLLM

            log.warning("CLAIMCHECK_FAKE_LLM is set: canned LLM answers, no provider calls")
            llm = CannedLLM()
        service = ClaimCheckService.from_settings(settings, llm=llm)
        return cls.build(settings, service)


# ------------------------------------------------------------------------ dependencies


def get_state(request: Request) -> AppState:
    return request.app.state.claimcheck


def _client(request: Request, state: AppState) -> str:
    return client_id(request, state.settings.api.trusted_proxy_hops)


def verify_limit(request: Request) -> AppState:
    state = get_state(request)
    state.verify_limiter.check(_client(request, state))
    return state


def feedback_limit(request: Request) -> AppState:
    state = get_state(request)
    state.feedback_limiter.check(_client(request, state))
    return state


def pages_limit(request: Request) -> AppState:
    state = get_state(request)
    state.pages_limiter.check(_client(request, state))
    return state


# Annotated aliases for route signatures.
State = Annotated[AppState, Depends(get_state)]
VerifyState = Annotated[AppState, Depends(verify_limit)]
FeedbackState = Annotated[AppState, Depends(feedback_limit)]
PagesState = Annotated[AppState, Depends(pages_limit)]
