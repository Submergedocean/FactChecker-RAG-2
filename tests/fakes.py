"""Test doubles. Unit tests use these instead of real LLM providers.

- `FakeTransport`: a scripted provider endpoint behind the real `LLMClient`, for testing the
  client's cache, retries, repair and fallback.
- `FakeLLM`: has the real client's interface (`complete_json`) and runs the real client over a
  `FakeTransport` with no cache, no ledger and no sleeping, so pipeline tests queue answers
  and get the same validation and repair behaviour as production.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from claimcheck.llm.client import (
    ChatReply,
    LLMClient,
    LLMResult,
    Messages,
    Provider,
    Role,
    TransportError,
)

Scripted = str | Mapping[str, Any] | BaseModel | ChatReply | Exception


def rate_limited(retry_after_s: float | None = 1.0) -> TransportError:
    return TransportError("HTTP 429: rate limited", status=429, retry_after_s=retry_after_s)


def server_error(status: int = 503) -> TransportError:
    return TransportError(f"HTTP {status}: unavailable", status=status)


def bad_request(status: int = 400) -> TransportError:
    return TransportError(f"HTTP {status}: bad request", status=status)


def _to_reply(item: Scripted) -> ChatReply:
    if isinstance(item, ChatReply):
        return item
    if isinstance(item, BaseModel):
        text = item.model_dump_json()
    elif isinstance(item, Mapping):
        text = json.dumps(item)
    else:
        text = item
    return ChatReply(text=text, prompt_tokens=10, completion_tokens=5)


@dataclass
class FakeTransport:
    """Returns (or raises) the scripted items in order and records every request."""

    script: list[Scripted] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat(
        self,
        *,
        model: str,
        messages: Messages,
        temperature: float,
        json_mode: bool,
        extra_body: Mapping[str, Any],
    ) -> ChatReply:
        self.calls.append(
            {
                "model": model,
                "messages": [dict(m) for m in messages],
                "temperature": temperature,
                "json_mode": json_mode,
                "extra_body": dict(extra_body),
            }
        )
        if not self.script:
            raise AssertionError("FakeTransport has no scripted response left")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _to_reply(item)


def fake_provider(
    name: str = "fake",
    script: list[Scripted] | None = None,
    *,
    requests_per_minute: int = 1000,
    requests_per_day: int = 100_000,
    **kwargs: Any,
) -> tuple[Provider, FakeTransport]:
    transport = FakeTransport(list(script or []))
    provider = Provider(
        name=name,
        transport=transport,
        extractor_model=f"{name}-small",
        verifier_model=f"{name}-large",
        requests_per_minute=requests_per_minute,
        requests_per_day=requests_per_day,
        **kwargs,
    )
    return provider, transport


@dataclass
class FakeLLM:
    """Queue answers (JSON text, dicts, models or exceptions); call like `LLMClient`."""

    responses: list[Scripted] = field(default_factory=list)

    def __post_init__(self) -> None:
        provider, self.transport = fake_provider("fake", self.responses)
        self.responses = self.transport.script  # one queue, shared with the transport
        self.client = LLMClient([provider], sleep=lambda _s: None, max_retries=0)

    @property
    def calls(self) -> list[Messages]:
        return [c["messages"] for c in self.transport.calls]

    def complete_json[T: BaseModel](
        self,
        messages: Messages,
        schema: type[T],
        *,
        role: Role,
        prompt_version: str,
        use_cache: bool = True,
    ) -> LLMResult[T]:
        return self.client.complete_json(
            messages, schema, role=role, prompt_version=prompt_version, use_cache=use_cache
        )
