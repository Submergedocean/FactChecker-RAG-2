"""LLM client: JSON answers validated against a Pydantic model.

One code path for every provider: the `openai` SDK pointed at an OpenAI-compatible endpoint
(Groq, Gemini). `LLMClient.complete_json` does, in order:

1. Cache: the first provider (in order) with a cached, valid answer for this exact request
   returns it, with no provider call and no quota spent.
2. Global daily budget (`limits.daily_llm_request_budget`): when spent, `BudgetExhausted`.
3. Per provider: daily limit and requests-per-minute window, then the request (30 s timeout).
   429, 408, 5xx, timeouts and connection errors are retried `max_retries` times with
   full-jitter backoff; a `Retry-After` is honoured as a floor, and one longer than
   `max_retry_after_s` exhausts the provider at once.
4. The answer must be a JSON object valid for the schema. If not, one repair request shows the
   model its answer and the validation error; a second invalid answer exhausts the provider.
5. An exhausted provider falls through to the next one, unless fallback is off (always off in
   evaluation mode, so a run never mixes models). All exhausted: `LLMUnavailable`.

Valid answers are cached under the original request's key. Every request sent is counted in
`llm_usage`. API keys and message text are never logged.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx
import openai
from pydantic import BaseModel, ValidationError

from claimcheck.config import ProviderSettings, Settings, get_settings
from claimcheck.db.session import make_engine
from claimcheck.llm.cache import LLMCache, cache_key
from claimcheck.llm.ratelimit import (
    SlidingWindow,
    UsageLedger,
    backoff_delay,
    parse_retry_after,
)

log = logging.getLogger(__name__)

Role = Literal["extractor", "verifier"]
Messages = list[dict[str, str]]

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
REPAIR_PROMPT = (
    "Your previous reply could not be used: {error}\n"
    "Reply again with only one JSON object that matches this JSON Schema, and no other text:\n"
    "{schema}"
)


# ------------------------------------------------------------------------------ errors


class LLMError(Exception):
    """Base class of the client's errors."""


class TransportError(LLMError):
    """A request that did not return an answer. `status` is None for timeouts and network."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after_s: float | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after_s = retry_after_s
        if retryable is None:
            retryable = status is None or status in RETRYABLE_STATUS
        self.retryable = retryable


@dataclass(frozen=True)
class ProviderFailure:
    provider: str
    model: str
    reason: str  # not_configured | daily_limit | rate_limited | exhausted | error | invalid_output
    detail: str = ""


class LLMUnavailable(LLMError):
    """Every provider tried failed (API error code LLM_UNAVAILABLE)."""

    def __init__(self, failures: list[ProviderFailure]) -> None:
        self.failures = failures
        summary = "; ".join(f"{f.provider}/{f.model}: {f.reason}" for f in failures) or "none"
        super().__init__(f"no LLM provider answered ({summary})")


class BudgetExhausted(LLMError):
    """The global daily request budget is spent (API error code DAILY_BUDGET_EXHAUSTED)."""


# --------------------------------------------------------------------------- transport


@dataclass(frozen=True)
class ChatReply:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    headers: Mapping[str, str] = field(default_factory=dict)


class ChatTransport(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: Messages,
        temperature: float,
        json_mode: bool,
        extra_body: Mapping[str, Any],
    ) -> ChatReply: ...


class OpenAITransport:
    """Chat completions on an OpenAI-compatible endpoint. The SDK's own retries are off."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_s: float,
        http_client: httpx.Client | None = None,  # tests pass an httpx.MockTransport client
    ) -> None:
        self._client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_s,
            max_retries=0,
            http_client=http_client,
        )

    def chat(
        self,
        *,
        model: str,
        messages: Messages,
        temperature: float,
        json_mode: bool,
        extra_body: Mapping[str, Any],
    ) -> ChatReply:
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if extra_body:
            kwargs["extra_body"] = dict(extra_body)
        try:
            raw = self._client.chat.completions.with_raw_response.create(**kwargs)
        except openai.APIStatusError as exc:
            body = exc.response.text if exc.response is not None else ""
            retry_after = parse_retry_after(exc.response.headers.get("retry-after"), body)
            raise TransportError(
                f"HTTP {exc.status_code}: {_short(body)}",
                status=exc.status_code,
                retry_after_s=retry_after,
            ) from exc
        except openai.APITimeoutError as exc:
            raise TransportError("timeout") from exc
        except openai.APIConnectionError as exc:
            raise TransportError(f"connection error: {type(exc.__cause__).__name__}") from exc
        except openai.APIError as exc:  # any other SDK error, e.g. response validation
            raise TransportError(f"API error: {type(exc).__name__}") from exc
        # A 200 whose body is not a chat completion (a proxy's HTML page, broken JSON) must
        # become a retryable TransportError, not escape: the SDK raises JSONDecodeError or
        # returns a bare string for these.
        try:
            completion = raw.parse()
            if not completion.choices:
                raise TransportError("empty reply (no choices)", retryable=True)
            usage = completion.usage
            return ChatReply(
                text=completion.choices[0].message.content or "",
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                headers=dict(raw.headers),
            )
        except TransportError:
            raise
        except (openai.APIError, ValueError, AttributeError, TypeError, IndexError) as exc:
            raise TransportError(
                f"malformed reply (HTTP {raw.status_code}): {type(exc).__name__}", retryable=True
            ) from exc


def _short(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ----------------------------------------------------------------------------- client


@dataclass
class Provider:
    """A configured endpoint at run time. `transport` is None when its API key is missing."""

    name: str
    transport: ChatTransport | None
    extractor_model: str
    verifier_model: str
    requests_per_minute: int = 30  # per model
    requests_per_day: int = 1000  # per model
    tokens_per_day: int | None = None  # per model
    json_mode: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self._windows: dict[str, SlidingWindow] = {}

    def window(self, model: str) -> SlidingWindow:
        """The model's requests-per-minute window (created on first use)."""
        if model not in self._windows:
            self._windows[model] = SlidingWindow(self.requests_per_minute, 60.0, clock=self.clock)
        return self._windows[model]

    def model_for(self, role: Role) -> str:
        return self.extractor_model if role == "extractor" else self.verifier_model

    @classmethod
    def from_settings(cls, settings: ProviderSettings, transport: ChatTransport | None) -> Provider:
        return cls(
            name=settings.name,
            transport=transport,
            extractor_model=settings.extractor_model,
            verifier_model=settings.verifier_model,
            requests_per_minute=settings.requests_per_minute,
            requests_per_day=settings.requests_per_day,
            tokens_per_day=settings.tokens_per_day,
            json_mode=settings.json_mode,
            extra_body=dict(settings.extra_body),
        )


@dataclass(frozen=True)
class LLMResult[T: BaseModel]:
    value: T
    text: str  # the validated JSON text
    provider: str
    model: str
    cached: bool
    repaired: bool = False
    latency_ms: int | None = None  # provider time of the answer used (cached: when stored)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    requests: int = 0  # provider requests this call sent (0 on a cache hit)
    failures: tuple[ProviderFailure, ...] = ()  # providers exhausted before this answer

    @property
    def fallback_used(self) -> bool:
        return bool(self.failures)


class _Exhausted(Exception):
    def __init__(self, failure: ProviderFailure) -> None:
        self.failure = failure


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def parse_json[T: BaseModel](text: str, schema: type[T]) -> T:
    """Validate `text` as one JSON object of `schema`; a single ```json fence is tolerated."""
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1)
    return schema.model_validate_json(text)


def _validation_summary(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {err.get('msg', '')}")
    return _short("; ".join(parts), 500)


class LLMClient:
    def __init__(
        self,
        providers: list[Provider],
        *,
        cache: LLMCache | None = None,
        ledger: UsageLedger | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 8.0,
        max_retry_after_s: float = 20.0,
        max_rate_wait_s: float = 5.0,
        daily_budget: int | None = None,
        fallback: bool = True,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = providers
        self.cache = cache
        self.ledger = ledger
        self.temperature = temperature
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.max_retry_after_s = max_retry_after_s
        self.max_rate_wait_s = max_rate_wait_s
        self.daily_budget = daily_budget
        self.fallback = fallback
        self._sleep = sleep
        self._rng = rng or random.Random()

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        evaluation: bool = False,
        transports: Mapping[str, ChatTransport] | None = None,
    ) -> LLMClient:
        """The app client (`data/app.db`, global budget, fallback per settings) or, with
        `evaluation`, the evaluation client (`eval_llm_cache.db`, no fallback, no app budget).

        `transports` replaces the real transport of the named providers (tests, smoke checks).
        """
        settings = settings or get_settings()
        cfg = settings.llm
        if not cfg.providers:
            raise ValueError("no LLM providers configured (llm.providers)")
        providers = []
        for p in cfg.providers:
            transport = (transports or {}).get(p.name)
            if transport is None:
                key = cfg.api_key(p)
                if key:
                    transport = OpenAITransport(p.base_url, key, timeout_s=cfg.timeout_s)
                else:
                    log.warning("LLM provider %s has no API key (%s)", p.name, p.api_key_env)
            providers.append(Provider.from_settings(p, transport))
        db_path = settings.paths.eval_cache_db if evaluation else settings.paths.app_db
        engine = make_engine(db_path)
        return cls(
            providers,
            cache=LLMCache(engine),
            ledger=UsageLedger(engine),
            temperature=cfg.temperature,
            max_retries=cfg.max_retries,
            backoff_base_s=cfg.backoff_base_s,
            backoff_max_s=cfg.backoff_max_s,
            max_retry_after_s=cfg.max_retry_after_s,
            max_rate_wait_s=cfg.max_rate_wait_s,
            daily_budget=None if evaluation else settings.limits.daily_llm_request_budget,
            fallback=cfg.fallback and not evaluation,
        )

    @property
    def active_providers(self) -> list[Provider]:
        """Providers this client may use, in order (only the first without fallback)."""
        return self.providers if self.fallback else self.providers[:1]

    # ---------------------------------------------------------------------- requests

    def complete_json[T: BaseModel](
        self,
        messages: Messages,
        schema: type[T],
        *,
        role: Role,
        prompt_version: str,
        use_cache: bool = True,
    ) -> LLMResult[T]:
        schema_json = schema.model_json_schema()
        providers = self.active_providers

        if use_cache and self.cache is not None:
            for provider in providers:
                model = provider.model_for(role)
                key = cache_key(provider.name, model, self.temperature, messages, schema_json)
                hit = self.cache.get(key)
                if hit is None:
                    continue
                try:
                    value = parse_json(hit.response_text, schema)
                except ValidationError:
                    log.warning("ignoring invalid cache entry %s", key[:12])
                    continue
                return LLMResult(
                    value=value,
                    text=hit.response_text,
                    provider=provider.name,
                    model=model,
                    cached=True,
                    latency_ms=hit.latency_ms,
                    prompt_tokens=hit.prompt_tokens,
                    completion_tokens=hit.completion_tokens,
                )

        failures: list[ProviderFailure] = []
        for provider in providers:
            model = provider.model_for(role)
            try:
                result = self._ask(provider, model, messages, schema, schema_json)
            except _Exhausted as exc:
                failures.append(exc.failure)
                log.warning(
                    "LLM provider %s/%s exhausted: %s %s",
                    exc.failure.provider,
                    exc.failure.model,
                    exc.failure.reason,
                    exc.failure.detail,
                )
                continue
            if use_cache and self.cache is not None:
                key = cache_key(provider.name, model, self.temperature, messages, schema_json)
                self.cache.put(
                    key,
                    provider=provider.name,
                    model=model,
                    prompt_version=prompt_version,
                    response_text=result.text,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    latency_ms=result.latency_ms,
                )
            return LLMResult(
                value=result.value,
                text=result.text,
                provider=result.provider,
                model=result.model,
                cached=False,
                repaired=result.repaired,
                latency_ms=result.latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                requests=result.requests,
                failures=tuple(failures),
            )
        raise LLMUnavailable(failures)

    def _ask[T: BaseModel](
        self,
        provider: Provider,
        model: str,
        messages: Messages,
        schema: type[T],
        schema_json: dict[str, Any],
    ) -> LLMResult[T]:
        """One provider: answer, then one repair if the answer is invalid."""
        if provider.transport is None:
            raise _Exhausted(ProviderFailure(provider.name, model, "not_configured", "no API key"))
        counter = [0]
        reply, latency_ms = self._send(provider, model, messages, counter)
        prompt_tokens, completion_tokens = reply.prompt_tokens, reply.completion_tokens
        try:
            value = parse_json(reply.text, schema)
            repaired = False
            text = reply.text
        except ValidationError as exc:
            error = _validation_summary(exc)
            repair = [
                *messages,
                {"role": "assistant", "content": reply.text},
                {
                    "role": "user",
                    "content": REPAIR_PROMPT.format(
                        error=error, schema=json.dumps(schema_json, separators=(",", ":"))
                    ),
                },
            ]
            reply, repair_ms = self._send(provider, model, repair, counter)
            latency_ms += repair_ms
            prompt_tokens = _add(prompt_tokens, reply.prompt_tokens)
            completion_tokens = _add(completion_tokens, reply.completion_tokens)
            try:
                value = parse_json(reply.text, schema)
            except ValidationError as exc2:
                raise _Exhausted(
                    ProviderFailure(
                        provider.name, model, "invalid_output", _validation_summary(exc2)
                    )
                ) from exc2
            repaired = True
            text = reply.text
        match = _FENCE_RE.match(text)
        if match:
            text = match.group(1)
        return LLMResult(
            value=value,
            text=text,
            provider=provider.name,
            model=model,
            cached=False,
            repaired=repaired,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            requests=counter[0],
        )

    def _send(
        self, provider: Provider, model: str, messages: Messages, counter: list[int]
    ) -> tuple[ChatReply, int]:
        """One request with retries. Returns the reply and its latency in ms."""
        assert provider.transport is not None
        attempt = 0
        while True:
            self._check_budget()
            self._check_daily_limits(provider, model)
            if not provider.window(model).acquire(
                max_wait_s=self.max_rate_wait_s, sleep=self._sleep
            ):
                raise _Exhausted(
                    ProviderFailure(provider.name, model, "rate_limited", "requests per minute")
                )
            start = time.perf_counter()
            counter[0] += 1
            try:
                reply = provider.transport.chat(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    json_mode=provider.json_mode,
                    extra_body=provider.extra_body,
                )
            except TransportError as exc:
                self._record(provider, model, None)
                if not exc.retryable:
                    raise _Exhausted(
                        ProviderFailure(provider.name, model, "error", str(exc))
                    ) from exc
                if exc.retry_after_s is not None and exc.retry_after_s > self.max_retry_after_s:
                    raise _Exhausted(
                        ProviderFailure(
                            provider.name,
                            model,
                            "rate_limited",
                            f"Retry-After {exc.retry_after_s:.0f} s",
                        )
                    ) from exc
                if attempt >= self.max_retries:
                    raise _Exhausted(
                        ProviderFailure(provider.name, model, "exhausted", str(exc))
                    ) from exc
                delay = backoff_delay(
                    attempt,
                    base_s=self.backoff_base_s,
                    max_s=self.backoff_max_s,
                    retry_after_s=exc.retry_after_s,
                    rng=self._rng,
                )
                log.info(
                    "LLM %s/%s: %s, retry %d in %.2f s",
                    provider.name,
                    model,
                    exc,
                    attempt + 1,
                    delay,
                )
                self._sleep(delay)
                attempt += 1
                continue
            latency_ms = int((time.perf_counter() - start) * 1000)
            self._record(provider, model, reply)
            return reply, latency_ms

    def _check_daily_limits(self, provider: Provider, model: str) -> None:
        if self.ledger is None:
            return
        if self.ledger.requests_today(provider.name, model) >= provider.requests_per_day:
            raise _Exhausted(ProviderFailure(provider.name, model, "daily_limit", "requests"))
        if (
            provider.tokens_per_day is not None
            and self.ledger.tokens_today(provider.name, model) >= provider.tokens_per_day
        ):
            raise _Exhausted(ProviderFailure(provider.name, model, "daily_limit", "tokens"))

    def _check_budget(self) -> None:
        if self.daily_budget is None or self.ledger is None:
            return
        if self.ledger.requests_today() >= self.daily_budget:
            raise BudgetExhausted(f"daily LLM request budget of {self.daily_budget} is spent")

    def _record(self, provider: Provider, model: str, reply: ChatReply | None) -> None:
        if self.ledger is None:
            return
        self.ledger.record(
            provider.name,
            model,
            prompt_tokens=(reply.prompt_tokens or 0) if reply else 0,
            completion_tokens=(reply.completion_tokens or 0) if reply else 0,
        )


def _add(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)
