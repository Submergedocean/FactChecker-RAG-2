"""LLM response cache on the `llm_cache` table.

The key is sha256 over (provider, model, temperature, messages, schema), serialised as canonical
JSON, so any change to the prompt, the output schema, the model or the temperature misses.
Only responses that passed schema validation are stored. The app writes `data/app.db`;
evaluation writes `data/cache/eval_llm_cache.db`, so evaluation runs replay exactly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.sqlite import insert

from claimcheck.db.models import LLMCacheEntry, utc_now_iso
from claimcheck.db.session import make_session_factory, session_scope


def cache_key(
    provider: str,
    model: str,
    temperature: float,
    messages: list[dict[str, str]],
    schema: dict[str, Any] | None,
) -> str:
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "temperature": float(temperature),
            "messages": messages,
            "schema": schema,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedResponse:
    provider: str
    model: str
    prompt_version: str
    response_text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int | None
    created_at: str


class LLMCache:
    def __init__(self, engine: Engine) -> None:
        self._sessions = make_session_factory(engine)

    def get(self, key: str) -> CachedResponse | None:
        with session_scope(self._sessions) as s:
            row = s.scalar(select(LLMCacheEntry).where(LLMCacheEntry.key == key))
            if row is None:
                return None
            return CachedResponse(
                provider=row.provider,
                model=row.model,
                prompt_version=row.prompt_version,
                response_text=row.response_text,
                prompt_tokens=row.prompt_tokens,
                completion_tokens=row.completion_tokens,
                latency_ms=row.latency_ms,
                created_at=row.created_at,
            )

    def put(
        self,
        key: str,
        *,
        provider: str,
        model: str,
        prompt_version: str,
        response_text: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        latency_ms: int | None,
    ) -> None:
        values = {
            "key": key,
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "response_text": response_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "created_at": utc_now_iso(),
        }
        stmt = insert(LLMCacheEntry).values(**values)
        # Two concurrent misses for the same key: keep the first answer.
        stmt = stmt.on_conflict_do_nothing(index_elements=["key"])
        with session_scope(self._sessions) as s:
            s.execute(stmt)

    def __len__(self) -> int:
        with session_scope(self._sessions) as s:
            return s.scalar(select(func.count()).select_from(LLMCacheEntry)) or 0
