"""Security controls that live in the API: client identity, per-client rate limits, response
headers and the startup check on test mode.

Client identity is used only as an in-memory rate-limit key; it is never stored or logged.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from claimcheck.api.errors import APIError
from claimcheck.config import Settings
from claimcheck.schemas import ErrorCode


class UnsafeConfigError(RuntimeError):
    """The settings must not be served (e.g. the fake LLM in production)."""


def check_startup(settings: Settings) -> None:
    """Refuse to start with canned LLM answers in production."""
    if settings.env == "prod" and settings.fake_llm:
        raise UnsafeConfigError(
            "CLAIMCHECK_FAKE_LLM is set with CLAIMCHECK_ENV=prod: refusing to start"
        )


# --------------------------------------------------------------------- client identity


def client_id(request: Request, trusted_proxy_hops: int) -> str:
    """The client address: the Nth entry from the right of `X-Forwarded-For` with N trusted
    proxy hops, else the socket peer. The leftmost entry is client-controlled and never used.

    Each trusted proxy appends the address it received the request from, so with N hops the
    entry N from the right was written by the outermost trusted proxy. A header with fewer
    entries than N did not pass through all the proxies: the socket peer is used instead.
    """
    peer = request.client.host if request.client else "unknown"
    if trusted_proxy_hops <= 0:
        return peer
    header = ",".join(request.headers.getlist("x-forwarded-for"))
    entries = [e.strip() for e in header.split(",") if e.strip()]
    if len(entries) < trusted_proxy_hops:
        return peer
    return entries[-trusted_proxy_hops]


# ------------------------------------------------------------------------ rate limits


@dataclass
class _Bucket:
    minute: deque[float] = field(default_factory=deque)
    day: str = ""
    day_count: int = 0
    last_seen: float = 0.0


class RateLimiter:
    """Per-client sliding minute window plus a UTC-day counter, in memory (correct with
    one worker only). Every request counts, valid or not."""

    MAX_CLIENTS = 50_000  # stale buckets are pruned past this many

    def __init__(
        self,
        per_minute: int,
        per_day: int,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.per_minute = per_minute
        self.per_day = per_day
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def _today(self, now: float) -> str:
        return datetime.fromtimestamp(now, UTC).date().isoformat()

    def _prune(self, now: float) -> None:
        stale = [k for k, b in self._buckets.items() if now - b.last_seen > 86_400]
        for k in stale:
            del self._buckets[k]
        if len(self._buckets) > self.MAX_CLIENTS:  # still too many: drop the oldest half
            by_age = sorted(self._buckets, key=lambda k: self._buckets[k].last_seen)
            for k in by_age[: len(by_age) // 2]:
                del self._buckets[k]

    def hit(self, key: str) -> float | None:
        """Count one request; None if allowed, else the seconds to wait."""
        now = self._clock()
        today = self._today(now)
        with self._lock:
            if len(self._buckets) > self.MAX_CLIENTS:
                self._prune(now)
            b = self._buckets.setdefault(key, _Bucket())
            b.last_seen = now
            if b.day != today:
                b.day, b.day_count = today, 0
            while b.minute and now - b.minute[0] >= 60:
                b.minute.popleft()
            if b.day_count >= self.per_day:
                midnight = datetime.fromisoformat(today).replace(tzinfo=UTC).timestamp() + 86_400
                return max(1.0, midnight - now)
            if len(b.minute) >= self.per_minute:
                return max(1.0, 60 - (now - b.minute[0]))
            b.minute.append(now)
            b.day_count += 1
            return None

    def check(self, key: str) -> None:
        wait = self.hit(key)
        if wait is not None:
            raise APIError(ErrorCode.RATE_LIMITED, retry_after_s=wait)


# ------------------------------------------------------------------------- body size


MAX_BODY_BYTES = 16 * 1024  # 500 characters of UTF-8 plus JSON, with room to spare


class BodySizeLimitMiddleware:
    """Refuse request bodies over `max_bytes` before the app reads them (FastAPI parses the
    body before dependencies such as the rate limiter run). The body is buffered up to the
    cap and replayed; past it the answer is 422 `INPUT_INVALID`, the contract's "too long"."""

    def __init__(self, app: ASGIApp, *, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(self, send: Send) -> None:
        from claimcheck.api.errors import error_response

        resp = error_response(ErrorCode.INPUT_INVALID, "The request body is too large.")
        await send(
            {"type": "http.response.start", "status": resp.status_code, "headers": resp.raw_headers}
        )
        await send({"type": "http.response.body", "body": resp.body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    too_big = int(value) > self.max_bytes
                except ValueError:
                    too_big = True
                if too_big:
                    await self._reject(send)
                    return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


# ---------------------------------------------------------------------------- headers


class SecurityHeadersMiddleware:
    """CSP, nosniff and Referrer-Policy on every response. The interactive docs
    (dev only) load assets from a CDN and get no CSP."""

    def __init__(self, app: ASGIApp, *, frame_ancestors: list[str], docs_paths: set[str]):
        self.app = app
        ancestors = " ".join(["'self'", *frame_ancestors])
        self.csp = f"default-src 'self'; frame-ancestors {ancestors}; base-uri 'self'"
        self.docs_paths = docs_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_docs = scope.get("path", "") in self.docs_paths

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-content-type-options", b"nosniff"))
                headers.append((b"referrer-policy", b"strict-origin-when-cross-origin"))
                if not is_docs:
                    headers.append((b"content-security-policy", self.csp.encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)
