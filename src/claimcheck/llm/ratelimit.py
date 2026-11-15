"""Rate limits and retry delays for the LLM client.

- `backoff_delay`: exponential backoff with full jitter; a `Retry-After` from the provider is a
  floor, never shortened.
- `parse_retry_after`: seconds from a `Retry-After` header (seconds or HTTP date) or from the
  `retryDelay` ("17s") Gemini puts in a 429 body.
- `SlidingWindow`: at most N events in any window of W seconds, thread-safe. Used per provider
  for requests per minute.
- `UsageLedger`: requests and tokens per UTC day, provider and model (`llm_usage`), which the
  client checks against each provider's daily limit and the global daily budget.
"""

from __future__ import annotations

import random
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.sqlite import insert

from claimcheck.db.models import LLMUsage, utc_day
from claimcheck.db.session import make_session_factory, session_scope


def backoff_delay(
    attempt: int,
    *,
    base_s: float,
    max_s: float,
    retry_after_s: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Seconds to wait before retry number `attempt` (0 for the first retry).

    Without `retry_after_s`: uniform in [0, min(max_s, base_s * 2**attempt)] ("full jitter").
    With it: `retry_after_s` plus up to `base_s` of jitter, so clients that were told the same
    instant do not all return at once.
    """
    rng = rng or random
    if retry_after_s is not None:
        return max(0.0, retry_after_s) + rng.uniform(0.0, base_s)
    return rng.uniform(0.0, min(max_s, base_s * (2**attempt)))


_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')


def parse_retry_after(header: str | None, body: str | None = None) -> float | None:
    """Seconds to wait as told by the provider, or None when it said nothing usable."""
    if header:
        header = header.strip()
        try:
            return max(0.0, float(header))
        except ValueError:
            pass
        try:
            when = parsedate_to_datetime(header)
        except (TypeError, ValueError):
            when = None
        if when is not None:
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            return max(0.0, (when - datetime.now(UTC)).total_seconds())
    if body:
        match = _RETRY_DELAY_RE.search(body)
        if match:
            return float(match.group(1))
    return None


class SlidingWindow:
    """Allows at most `limit` events in any `window_s` seconds."""

    def __init__(
        self, limit: int, window_s: float, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if limit < 1 or window_s <= 0:
            raise ValueError("limit must be >= 1 and window_s > 0")
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0] >= self.window_s:
            self._events.popleft()

    def wait_time(self) -> float:
        """Seconds until one more event is allowed (0 when allowed now)."""
        with self._lock:
            now = self._clock()
            self._prune(now)
            if len(self._events) < self.limit:
                return 0.0
            return self._events[0] + self.window_s - now

    def try_acquire(self) -> bool:
        """Record an event if allowed now; False (nothing recorded) otherwise."""
        with self._lock:
            now = self._clock()
            self._prune(now)
            if len(self._events) >= self.limit:
                return False
            self._events.append(now)
            return True

    def acquire(
        self,
        *,
        max_wait_s: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> bool:
        """Record an event, waiting up to `max_wait_s` for room. False if it would wait longer."""
        waited = 0.0
        while True:
            if self.try_acquire():
                return True
            delay = self.wait_time()
            if waited + delay > max_wait_s:
                return False
            sleep(delay)
            waited += delay


class UsageLedger:
    """Daily request and token counts per provider and model, on the `llm_usage` table.

    Every request sent to a provider is counted, failed ones included, because providers count
    them against the same quota. Days are UTC dates.
    """

    def __init__(self, engine: Engine) -> None:
        self._sessions = make_session_factory(engine)

    def record(
        self,
        provider: str,
        model: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        day: str | None = None,
    ) -> None:
        stmt = insert(LLMUsage).values(
            day=day or utc_day(),
            provider=provider,
            model=model,
            requests=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["day", "provider", "model"],
            set_={
                "requests": LLMUsage.requests + 1,
                "prompt_tokens": LLMUsage.prompt_tokens + stmt.excluded.prompt_tokens,
                "completion_tokens": LLMUsage.completion_tokens + stmt.excluded.completion_tokens,
            },
        )
        with session_scope(self._sessions) as s:
            s.execute(stmt)

    def requests_today(
        self, provider: str | None = None, model: str | None = None, *, day: str | None = None
    ) -> int:
        """Requests sent today: all, one provider's, or one provider model's."""
        return self._sum(LLMUsage.requests, provider, model, day)

    def tokens_today(
        self, provider: str | None = None, model: str | None = None, *, day: str | None = None
    ) -> int:
        """Prompt plus completion tokens used today (same filters as `requests_today`)."""
        return self._sum(LLMUsage.prompt_tokens + LLMUsage.completion_tokens, provider, model, day)

    def _sum(self, column: Any, provider: str | None, model: str | None, day: str | None) -> int:
        query = select(func.coalesce(func.sum(column), 0)).where(LLMUsage.day == (day or utc_day()))
        if provider is not None:
            query = query.where(LLMUsage.provider == provider)
        if model is not None:
            query = query.where(LLMUsage.model == model)
        with session_scope(self._sessions) as s:
            return int(s.scalar(query) or 0)
