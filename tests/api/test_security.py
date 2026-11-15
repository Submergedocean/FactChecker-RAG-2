"""Security controls — headers, client identity, rate limits, test-mode startup check,
and the canned LLM used for UI work."""

from __future__ import annotations

import pytest
from starlette.requests import Request
from typer.testing import CliRunner

from claimcheck.api import cli as api_cli
from claimcheck.api.fake_llm import CannedLLM
from claimcheck.api.main import create_app
from claimcheck.api.security import RateLimiter, UnsafeConfigError, client_id
from claimcheck.config import Settings, get_settings
from tests.api.conftest import Harness, build_harness


def test_security_headers(h: Harness) -> None:
    resp = h.client.get(h.url("/meta"))
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["referrer-policy"]
    csp = resp.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'self' https://huggingface.co" in csp
    assert "set-cookie" not in resp.headers
    assert "access-control-allow-origin" not in resp.headers  # no CORS


def test_startup_fails_with_fake_llm_in_prod() -> None:
    with pytest.raises(UnsafeConfigError):
        create_app(Settings(env="prod", fake_llm=True))
    create_app(Settings(env="dev", fake_llm=True))  # allowed outside prod


def test_serve_refuses_fake_llm_in_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIMCHECK_ENV", "prod")
    monkeypatch.setenv("CLAIMCHECK_FAKE_LLM", "1")
    get_settings.cache_clear()
    try:
        result = CliRunner().invoke(api_cli.app, ["serve"])
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 2
    assert "refusing to start" in result.output


def _request(peer: str, xff: list[str] | None = None) -> Request:
    headers = [(b"x-forwarded-for", v.encode()) for v in (xff or [])]
    return Request({"type": "http", "headers": headers, "client": (peer, 1234)})


def test_client_id_nth_from_right() -> None:
    # Client 9.9.9.9 spoofs a leftmost entry; two trusted proxies append theirs.
    req = _request("10.0.0.2", ["6.6.6.6, 9.9.9.9, 10.0.0.1"])
    assert client_id(req, 0) == "10.0.0.2"  # no proxy: the socket peer
    assert client_id(req, 1) == "10.0.0.1"
    assert client_id(req, 2) == "9.9.9.9"
    assert client_id(_request("10.0.0.2", ["9.9.9.9"]), 2) == "10.0.0.2"  # too few entries
    assert client_id(_request("10.0.0.2", ["1.1.1.1", "2.2.2.2"]), 1) == "2.2.2.2"  # two headers


def test_spoofed_leftmost_entry_does_not_escape_the_limit(tmp_path) -> None:  # noqa: ANN001
    h = build_harness(tmp_path, per_client_requests_per_minute=2)
    h.state.settings.api.trusted_proxy_hops = 1
    with h.client:
        codes = [
            h.client.post(
                h.url("/verify"),
                json={"text": "x"},
                headers={"X-Forwarded-For": f"1.2.3.{i}, 5.5.5.5"},
            ).status_code
            for i in range(3)
        ]
        assert codes == [422, 422, 429]
        other = h.client.post(
            h.url("/verify"), json={"text": "x"}, headers={"X-Forwarded-For": "6.6.6.6"}
        )
        assert other.status_code == 422  # a different client has its own bucket


def test_rate_limiter_minute_and_day() -> None:
    now = [1_000_000.0]
    lim = RateLimiter(2, 3, clock=lambda: now[0])
    assert lim.hit("a") is None and lim.hit("a") is None
    wait = lim.hit("a")
    assert wait is not None and 0 < wait <= 60
    assert lim.hit("b") is None  # per client
    now[0] += 61
    assert lim.hit("a") is None  # window slid; day count now 3
    now[0] += 61
    day_wait = lim.hit("a")
    assert day_wait is not None and day_wait > 60  # daily cap: wait until UTC midnight


def test_rate_limiter_prunes_clients() -> None:
    now = [0.0]
    lim = RateLimiter(5, 5, clock=lambda: now[0])
    lim.MAX_CLIENTS = 10
    for i in range(30):
        lim.hit(str(i))
        now[0] += 1
    assert len(lim._buckets) <= 11


def test_canned_llm_through_the_pipeline(tmp_path) -> None:  # noqa: ANN001
    h = build_harness(tmp_path, llm=CannedLLM(), per_client_requests_per_minute=100)
    with h.client:
        seen = set()
        for i in range(12):
            body = h.client.post(
                h.url("/verify"), json={"text": f"Pakistan's GDP grew {i}.5 percent in FY2026."}
            ).json()
            claim = body["claims"][0]
            seen.add(claim["verdict"])
            if claim["verdict"] != "NEI":
                assert claim["citations"], claim  # the canned quote passes the real check
                assert "downgraded_no_citation" not in claim["flags"]
        assert seen == {"SUPPORTED", "REFUTED", "NEI"}
        q = h.client.post(h.url("/verify"), json={"text": "What was GDP growth in FY2026?"})
        assert q.json()["reject_reason"] == "question"
        meta = h.client.get(h.url("/meta")).json()
        assert meta["models"]["extractor"] == "fake/canned-extractor"


def test_oversized_body_refused_before_parsing(h: Harness) -> None:
    big = '{"text": "' + "x" * 20_000 + '"}'
    resp = h.client.post(
        h.url("/verify"), content=big, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INPUT_INVALID"
    assert "too large" in resp.json()["error"]["message"]
    assert resp.headers["content-security-policy"]  # still behind the header middleware

    def chunks():  # noqa: ANN202 — no Content-Length: chunked transfer
        for _ in range(20):
            yield b"x" * 1024

    resp = h.client.post(h.url("/verify"), content=chunks())
    assert resp.status_code == 422 and "too large" in resp.json()["error"]["message"]


def test_small_body_replayed_intact(h: Harness) -> None:
    h.llm.responses.append({"checkable": False, "reject_reason": "off_topic", "claims": []})
    resp = h.client.post(h.url("/verify"), json={"text": "Cricket is a lovely game."})
    assert resp.status_code == 200 and resp.json()["input"]["text"] == "Cricket is a lovely game."


def test_serve_runs_one_worker_without_access_log(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    seen: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    monkeypatch.setattr("claimcheck.api.main.create_app", lambda settings: object())
    result = CliRunner().invoke(api_cli.app, ["serve", "--port", "8123"])
    assert result.exit_code == 0, result.output
    assert seen["workers"] == 1 and seen["access_log"] is False and seen["port"] == 8123
    assert seen["proxy_headers"] is False
