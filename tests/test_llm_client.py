"""LLMClient with scripted providers (no network, no quota): cache, retries, repair, fallback."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

import claimcheck.config as config
from claimcheck.config import LLMSettings, PathsSettings, ProviderSettings, Settings
from claimcheck.db.session import make_engine
from claimcheck.llm.cache import LLMCache, cache_key
from claimcheck.llm.client import (
    BudgetExhausted,
    ChatReply,
    LLMClient,
    LLMUnavailable,
    OpenAITransport,
    Provider,
    TransportError,
    parse_json,
)
from claimcheck.llm.ratelimit import (
    SlidingWindow,
    UsageLedger,
    backoff_delay,
    parse_retry_after,
)
from tests.fakes import FakeLLM, bad_request, fake_provider, rate_limited, server_error


class Answer(BaseModel):
    verdict: str
    n: int


GOOD = {"verdict": "SUPPORTED", "n": 3}
MSGS = [{"role": "system", "content": "Reply in JSON."}, {"role": "user", "content": "claim"}]


@pytest.fixture
def engine(tmp_path: Path):
    return make_engine(tmp_path / "app.db")


def make_client(providers: list[Provider], engine, sleeps: list[float] | None = None, **kw):
    sleeps = sleeps if sleeps is not None else []
    return LLMClient(
        providers,
        cache=LLMCache(engine),
        ledger=UsageLedger(engine),
        sleep=sleeps.append,
        rng=random.Random(0),
        **kw,
    )


def ask(client: LLMClient, role: str = "verifier", messages=MSGS):
    return client.complete_json(messages, Answer, role=role, prompt_version="v1")


# ----------------------------------------------------------------------------- cache


def test_cache_hit_makes_zero_provider_calls(engine) -> None:
    provider, transport = fake_provider("groq", [GOOD])
    client = make_client([provider], engine)
    first = ask(client)
    assert not first.cached and first.requests == 1 and first.value == Answer(**GOOD)

    second = ask(client)
    assert second.cached and second.requests == 0 and second.value == first.value
    assert len(transport.calls) == 1

    # A new client on the same database (e.g. after a restart) also hits.
    provider2, transport2 = fake_provider("groq", [])
    third = ask(make_client([provider2], engine))
    assert third.cached and transport2.calls == []
    assert UsageLedger(engine).requests_today() == 1


def test_cache_key_covers_provider_model_temperature_messages_and_schema() -> None:
    schema = Answer.model_json_schema()
    base = cache_key("groq", "m", 0.0, MSGS, schema)
    assert base == cache_key("groq", "m", 0, [dict(m) for m in MSGS], dict(schema))
    others = [
        cache_key("gemini", "m", 0.0, MSGS, schema),
        cache_key("groq", "m2", 0.0, MSGS, schema),
        cache_key("groq", "m", 0.2, MSGS, schema),
        cache_key("groq", "m", 0.0, MSGS[:1], schema),
        cache_key("groq", "m", 0.0, MSGS, None),
    ]
    assert base not in others and len(set(others)) == len(others)


def test_roles_use_their_own_model_and_cache_entry(engine) -> None:
    provider, transport = fake_provider("groq", [GOOD, GOOD])
    client = make_client([provider], engine)
    assert ask(client, "extractor").model == "groq-small"
    assert ask(client, "verifier").model == "groq-large"
    assert [c["model"] for c in transport.calls] == ["groq-small", "groq-large"]
    assert len(LLMCache(engine)) == 2


def test_use_cache_false_neither_reads_nor_writes(engine) -> None:
    provider, transport = fake_provider("groq", [GOOD, GOOD])
    client = make_client([provider], engine)
    for _ in range(2):
        client.complete_json(MSGS, Answer, role="verifier", prompt_version="v1", use_cache=False)
    assert len(transport.calls) == 2 and len(LLMCache(engine)) == 0


def test_cached_answer_served_after_budget_is_spent(engine) -> None:
    provider, _ = fake_provider("groq", [GOOD])
    ask(make_client([provider], engine))
    provider2, transport2 = fake_provider("groq", [])
    res = ask(make_client([provider2], engine, daily_budget=1))
    assert res.cached and transport2.calls == []


# --------------------------------------------------------------------------- retries


def test_429_then_success_honours_retry_after(engine) -> None:
    provider, transport = fake_provider("groq", [rate_limited(2.0), GOOD])
    sleeps: list[float] = []
    res = ask(make_client([provider], engine, sleeps, backoff_base_s=0.5))
    assert res.value == Answer(**GOOD) and res.provider == "groq" and not res.fallback_used
    assert res.requests == 2 and len(transport.calls) == 2
    assert len(sleeps) == 1 and 2.0 <= sleeps[0] <= 2.5  # Retry-After is a floor, plus jitter
    assert UsageLedger(engine).requests_today("groq") == 2  # the 429 counts against quota


def test_5xx_retries_with_full_jitter_backoff(engine) -> None:
    provider, _ = fake_provider("groq", [server_error(503), server_error(500), GOOD])
    sleeps: list[float] = []
    res = ask(make_client([provider], engine, sleeps, backoff_base_s=1.0, backoff_max_s=8.0))
    assert res.requests == 3
    assert 0 <= sleeps[0] <= 1.0 and 0 <= sleeps[1] <= 2.0


def test_retries_exhausted_then_unavailable(engine) -> None:
    provider, transport = fake_provider("groq", [server_error()] * 3)
    with pytest.raises(LLMUnavailable) as err:
        ask(make_client([provider], engine, max_retries=2))
    assert len(transport.calls) == 3
    assert [(f.provider, f.reason) for f in err.value.failures] == [("groq", "exhausted")]


def test_long_retry_after_exhausts_provider_without_waiting(engine) -> None:
    groq, groq_t = fake_provider("groq", [rate_limited(3600.0)])
    gemini, _ = fake_provider("gemini", [GOOD])
    sleeps: list[float] = []
    res = ask(make_client([groq, gemini], engine, sleeps, max_retry_after_s=20.0))
    assert res.provider == "gemini" and sleeps == [] and len(groq_t.calls) == 1
    assert res.failures[0].reason == "rate_limited"


def test_client_error_is_not_retried(engine) -> None:
    groq, groq_t = fake_provider("groq", [bad_request(404)])
    gemini, _ = fake_provider("gemini", [GOOD])
    sleeps: list[float] = []
    res = ask(make_client([groq, gemini], engine, sleeps))
    assert len(groq_t.calls) == 1 and sleeps == []
    assert res.provider == "gemini" and res.failures[0].reason == "error"


# ---------------------------------------------------------------------------- repair


def test_invalid_json_then_repair(engine) -> None:
    provider, transport = fake_provider("groq", ["Sure! The verdict is SUPPORTED.", GOOD])
    res = ask(make_client([provider], engine))
    assert res.repaired and res.value == Answer(**GOOD) and res.requests == 2
    repair = transport.calls[1]["messages"]
    assert repair[: len(MSGS)] == MSGS
    assert repair[-2] == {"role": "assistant", "content": "Sure! The verdict is SUPPORTED."}
    assert repair[-1]["role"] == "user" and '"verdict"' in repair[-1]["content"]
    # The repaired answer is cached under the original request.
    provider2, transport2 = fake_provider("groq", [])
    assert ask(make_client([provider2], engine)).cached and transport2.calls == []


def test_schema_violation_is_repaired_with_the_error(engine) -> None:
    provider, transport = fake_provider("groq", [{"verdict": "NEI", "n": "many"}, GOOD])
    assert ask(make_client([provider], engine)).repaired
    assert "n:" in transport.calls[1]["messages"][-1]["content"]


def test_second_invalid_answer_exhausts_provider(engine) -> None:
    groq, groq_t = fake_provider("groq", ["{", "still not json"])
    gemini, _ = fake_provider("gemini", [GOOD])
    res = ask(make_client([groq, gemini], engine))
    assert len(groq_t.calls) == 2 and res.provider == "gemini"
    assert res.failures[0].reason == "invalid_output"
    assert len(LLMCache(engine)) == 1  # only the valid answer is cached


def test_fenced_json_is_accepted() -> None:
    assert parse_json('```json\n{"verdict": "NEI", "n": 1}\n```', Answer).n == 1


# -------------------------------------------------------------------------- fallback


def test_fallback_order(engine) -> None:
    a, a_t = fake_provider("groq", [server_error()] * 3)
    b, b_t = fake_provider("gemini", [bad_request(401)])
    c, c_t = fake_provider("third", [GOOD])
    res = ask(make_client([a, b, c], engine))
    assert res.provider == "third" and res.model == "third-large" and res.fallback_used
    assert [f.provider for f in res.failures] == ["groq", "gemini"]
    assert (len(a_t.calls), len(b_t.calls), len(c_t.calls)) == (3, 1, 1)


def test_missing_key_provider_is_skipped(engine) -> None:
    missing = Provider("groq", None, "s", "l")
    gemini, _ = fake_provider("gemini", [GOOD])
    res = ask(make_client([missing, gemini], engine))
    assert res.provider == "gemini" and res.failures[0].reason == "not_configured"


def test_fallback_off_never_tries_second_provider(engine) -> None:
    groq, _ = fake_provider("groq", [server_error()] * 3)
    gemini, gemini_t = fake_provider("gemini", [GOOD])
    with pytest.raises(LLMUnavailable):
        ask(make_client([groq, gemini], engine, fallback=False))
    assert gemini_t.calls == []


def test_cached_fallback_answer_is_reused_before_calling_primary(engine) -> None:
    groq, _ = fake_provider("groq", [server_error()] * 3)
    gemini, _ = fake_provider("gemini", [GOOD])
    ask(make_client([groq, gemini], engine))
    groq2, groq2_t = fake_provider("groq", [])
    gemini2, _ = fake_provider("gemini", [])
    res = ask(make_client([groq2, gemini2], engine))
    assert res.cached and res.provider == "gemini" and groq2_t.calls == []


def _settings(tmp_path: Path) -> Settings:
    provider = {"base_url": "http://unused", "api_key_env": "UNUSED_KEY_P7"}
    return Settings(
        paths=PathsSettings(
            app_db=tmp_path / "app.db", eval_cache_db=tmp_path / "cache" / "eval.db"
        ),
        llm=LLMSettings(
            providers=[
                ProviderSettings(
                    name="groq", extractor_model="gs", verifier_model="gl", **provider
                ),
                ProviderSettings(
                    name="gemini", extractor_model="f", verifier_model="f", **provider
                ),
            ],
            fallback=True,
        ),
    )


def test_no_fallback_in_evaluation_mode(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    groq, gemini = fake_provider("groq", [server_error()] * 3)[1], fake_provider("gemini")[1]
    gemini.script.append(GOOD)
    transports = {"groq": groq, "gemini": gemini}

    client = LLMClient.from_settings(settings, evaluation=True, transports=transports)
    client._sleep = lambda _s: None
    assert client.fallback is False and client.daily_budget is None
    assert [p.name for p in client.active_providers] == ["groq"]
    with pytest.raises(LLMUnavailable):
        ask(client)
    assert gemini.calls == []
    assert (tmp_path / "cache" / "eval.db").exists() and not (tmp_path / "app.db").exists()

    # The app client on the same settings does fall back, and writes data/app.db.
    groq.script.extend([server_error()] * 3)
    app = LLMClient.from_settings(settings, transports=transports)
    app._sleep = lambda _s: None
    assert app.fallback is True and app.daily_budget == settings.limits.daily_llm_request_budget
    assert ask(app).provider == "gemini" and (tmp_path / "app.db").exists()


# ---------------------------------------------------------------------------- limits


def test_global_budget_blocks_before_any_call(engine) -> None:
    ledger = UsageLedger(engine)
    for _ in range(3):
        ledger.record("groq", "groq-large", prompt_tokens=1, completion_tokens=1)
    provider, transport = fake_provider("groq", [GOOD])
    with pytest.raises(BudgetExhausted):
        ask(make_client([provider], engine, daily_budget=3))
    assert transport.calls == []


def test_provider_daily_limit_falls_back(engine) -> None:
    UsageLedger(engine).record("groq", "groq-large")
    groq, groq_t = fake_provider("groq", [GOOD], requests_per_day=1)
    gemini, _ = fake_provider("gemini", [GOOD])
    res = ask(make_client([groq, gemini], engine))
    assert groq_t.calls == [] and res.provider == "gemini"
    assert (res.failures[0].reason, res.failures[0].detail) == ("daily_limit", "requests")


def test_daily_limits_are_per_model(engine) -> None:
    # The verifier model's day is spent; the extractor model still answers.
    UsageLedger(engine).record("groq", "groq-large", prompt_tokens=900, completion_tokens=100)
    groq, groq_t = fake_provider("groq", [GOOD], requests_per_day=5, tokens_per_day=1000)
    gemini, _ = fake_provider("gemini", [GOOD])
    client = make_client([groq, gemini], engine)
    res = ask(client, "verifier")
    assert res.provider == "gemini" and res.failures[0].detail == "tokens"
    assert ask(client, "extractor").provider == "groq" and len(groq_t.calls) == 1


def test_per_minute_window_falls_back_when_wait_too_long(engine) -> None:
    groq, groq_t = fake_provider("groq", [GOOD, GOOD], requests_per_minute=1)
    gemini, _ = fake_provider("gemini", [GOOD])
    client = make_client([groq, gemini], engine, max_rate_wait_s=0.0)
    ask(client, messages=[{"role": "user", "content": "a"}])
    res = ask(client, messages=[{"role": "user", "content": "b"}])
    assert len(groq_t.calls) == 1 and res.provider == "gemini"
    assert res.failures[0].reason == "rate_limited"
    # The window is per model: the extractor model is not held up by the verifier's.
    assert ask(client, "extractor", [{"role": "user", "content": "c"}]).provider == "groq"


def test_usage_ledger_sums_tokens(engine) -> None:
    provider, _ = fake_provider(
        "groq", [ChatReply('{"verdict": "x", "n": 1}', prompt_tokens=7, completion_tokens=2)]
    )
    ask(make_client([provider], engine))
    from sqlalchemy import text

    with engine.connect() as conn:
        row = conn.execute(text("SELECT requests, prompt_tokens, completion_tokens FROM llm_usage"))
        assert row.one() == (1, 7, 2)


def test_sliding_window() -> None:
    now = [0.0]
    window = SlidingWindow(2, 60.0, clock=lambda: now[0])
    assert window.try_acquire() and window.try_acquire() and not window.try_acquire()
    assert window.wait_time() == 60.0
    now[0] = 30.0
    assert window.wait_time() == 30.0
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    assert not window.acquire(max_wait_s=10.0, sleep=sleep) and slept == []
    assert window.acquire(max_wait_s=30.0, sleep=sleep) and slept == [30.0]


def test_backoff_delay_bounds() -> None:
    rng = random.Random(1)
    for attempt in range(6):
        cap = min(8.0, 0.5 * 2**attempt)
        for _ in range(50):
            assert 0.0 <= backoff_delay(attempt, base_s=0.5, max_s=8.0, rng=rng) <= cap
    for _ in range(50):
        d = backoff_delay(0, base_s=0.5, max_s=8.0, retry_after_s=3.0, rng=rng)
        assert 3.0 <= d <= 3.5


def test_parse_retry_after() -> None:
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after(" 1.5 ") == 1.5
    when = datetime.now(UTC) + timedelta(seconds=30)
    assert 25 <= parse_retry_after(format_datetime(when, usegmt=True)) <= 31
    body = '{"error": {"details": [{"@type": "RetryInfo", "retryDelay": "17s"}]}}'
    assert parse_retry_after(None, body) == 17.0
    assert parse_retry_after(None, "no hint") is None
    assert parse_retry_after("garbage") is None


# ---------------------------------------------------------------------------- fakes


def test_fake_llm_matches_client_interface() -> None:
    fake = FakeLLM([Answer(verdict="NEI", n=0), "not json", GOOD])
    first = fake.complete_json(MSGS, Answer, role="extractor", prompt_version="v1")
    second = fake.complete_json(MSGS, Answer, role="verifier", prompt_version="v1")
    assert first.value.n == 0 and second.repaired and second.value.n == 3
    assert len(fake.calls) == 3 and fake.responses == []


# ------------------------------------------------- real transport over a mock HTTP layer

COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": '{"verdict": "NEI", "n": 2}'},
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
}


def _transport(status: int, body: bytes | dict, headers: dict | None = None):
    content = json.dumps(body).encode() if isinstance(body, dict) else body
    hdrs = {"content-type": "application/json", **(headers or {})}
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, headers=hdrs, content=content)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAITransport("https://llm.test/v1", "k", timeout_s=5, http_client=client), seen


def _chat(transport):
    return transport.chat(
        model="m", messages=MSGS, temperature=0.0, json_mode=True, extra_body={"x": 1}
    )


def test_openai_transport_success_returns_text_tokens_and_headers() -> None:
    transport, seen = _transport(200, COMPLETION, {"x-ratelimit-limit-requests": "1000"})
    reply = _chat(transport)
    assert reply.text == '{"verdict": "NEI", "n": 2}'
    assert (reply.prompt_tokens, reply.completion_tokens) == (11, 4)
    assert reply.headers["x-ratelimit-limit-requests"] == "1000"
    sent = json.loads(seen[0].content)
    assert sent["response_format"] == {"type": "json_object"} and sent["x"] == 1
    assert sent["temperature"] == 0.0 and len(seen) == 1  # SDK retries are off


@pytest.mark.parametrize(
    ("status", "headers", "body", "retryable", "retry_after"),
    [
        (429, {"retry-after": "7"}, {"error": {"message": "slow down"}}, True, 7.0),
        (
            429,
            {},
            b'[{"error": {"details": [{"retryDelay": "17s"}]}}]',
            True,
            17.0,
        ),
        (503, {}, {"error": {"message": "high demand"}}, True, None),
        (401, {}, {"error": {"message": "bad key"}}, False, None),
    ],
)
def test_openai_transport_http_errors(status, headers, body, retryable, retry_after) -> None:
    transport, _ = _transport(status, body, headers)
    with pytest.raises(TransportError) as err:
        _chat(transport)
    assert err.value.status == status and err.value.retryable is retryable
    assert err.value.retry_after_s == retry_after


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b"<html>proxy error</html>", "text/html"),  # SDK returned a bare str
        (b"{not json", "application/json"),  # SDK raised JSONDecodeError
        (b'{"foo": 1}', "application/json"),  # no choices
    ],
)
def test_openai_transport_malformed_200_is_retryable(body, content_type) -> None:
    transport, _ = _transport(200, body, {"content-type": content_type})
    with pytest.raises(TransportError) as err:
        _chat(transport)
    assert err.value.retryable and err.value.status is None


def test_malformed_200_retries_then_falls_back(engine) -> None:
    groq_t, seen = _transport(200, b"<html>proxy error</html>", {"content-type": "text/html"})
    groq = Provider("groq", groq_t, "gs", "gl")
    gemini, _ = fake_provider("gemini", [GOOD])
    res = ask(make_client([groq, gemini], engine, max_retries=2))
    assert len(seen) == 3 and res.provider == "gemini"
    assert res.failures[0].reason == "exhausted"


# ----------------------------------------------------------------------------- api keys


def test_api_key_from_environment_wins_and_empty_means_none(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("P7_TEST_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    settings = LLMSettings()
    provider = ProviderSettings(
        name="p",
        base_url="http://unused",
        api_key_env="P7_TEST_KEY",
        extractor_model="a",
        verifier_model="b",
    )
    monkeypatch.delenv("P7_TEST_KEY", raising=False)
    assert settings.api_key(provider) == "from-dotenv"
    monkeypatch.setenv("P7_TEST_KEY", "from-env")
    assert settings.api_key(provider) == "from-env"
    monkeypatch.setenv("P7_TEST_KEY", "")
    assert settings.api_key(provider) is None
