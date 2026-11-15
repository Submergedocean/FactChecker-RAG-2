"""`POST /verify` contract and its error codes."""

from __future__ import annotations

import time

import pytest

from claimcheck.llm.client import BudgetExhausted, LLMUnavailable, ProviderFailure
from claimcheck.schemas import VerificationResponse
from tests.api.conftest import Harness, build_harness
from tests.fakes import FakeLLM
from tests.test_service import GDP_QUOTE, extraction, rejection, verdict

CLAIM = "Pakistan's real GDP grew by 3.7 percent in FY2026."


def script_ok(h: Harness) -> None:
    h.llm.responses.extend([extraction(CLAIM), verdict("SUPPORTED", [("P1", GDP_QUOTE)])])


def assert_error(resp, status: int, code: str) -> dict:  # noqa: ANN001
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    return body["error"]


def test_verify_ok_matches_contract(h: Harness) -> None:
    script_ok(h)
    resp = h.client.post(h.url("/verify"), json={"text": CLAIM})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    VerificationResponse.model_validate(body)  # exact schema, extra fields forbidden
    assert body["status"] == "ok" and body["reject_reason"] is None and body["message"] is None
    claim = body["claims"][0]
    assert claim["verdict"] == "SUPPORTED"
    cit = claim["citations"][0]
    assert cit["page"] == 3 and cit["source_url"].endswith("#page=3")
    assert [e["rank"] for e in claim["evidence"]] == [1, 2, 3]
    assert claim["evidence"][0]["cited"] is True
    assert body["meta"]["cached"] is False
    assert set(body["meta"]["latency_ms"]) == {"extract", "retrieve", "verify", "total"}


def test_verify_rejected_has_message_and_no_claims(h: Harness) -> None:
    h.llm.responses.append(rejection("off_topic"))
    body = h.client.post(h.url("/verify"), json={"text": "Cricket is a lovely game."}).json()
    assert body["status"] == "rejected"
    assert body["reject_reason"] == "off_topic"
    assert body["claims"] == [] and body["message"]


def test_repeat_is_cached_and_fast(h: Harness) -> None:
    script_ok(h)
    first = h.client.post(h.url("/verify"), json={"text": CLAIM}).json()
    calls = len(h.llm.calls)
    t0 = time.perf_counter()
    resp = h.client.post(h.url("/verify"), json={"text": CLAIM})
    elapsed_ms = (time.perf_counter() - t0) * 1000
    body = resp.json()
    assert body["meta"]["cached"] is True
    assert body["id"] == first["id"]
    assert len(h.llm.calls) == calls  # no LLM call
    assert elapsed_ms < 100


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "x" * 9},  # 9 characters
        {"text": "x" * 501},  # 501 characters
        {"text": ""},
        {"text": 12345678901},  # wrong type
        {"text": None},
        {},
        {"text": CLAIM, "extra": 1},
        [CLAIM],
    ],
    ids=["9-chars", "501-chars", "empty", "number", "null", "missing", "extra-field", "list"],
)
def test_input_invalid(h: Harness, payload: object) -> None:
    resp = h.client.post(h.url("/verify"), json=payload)
    assert_error(resp, 422, "INPUT_INVALID")
    assert h.llm.calls == []


def test_blank_input_fails_the_guard(h: Harness) -> None:
    # 12 characters by length, empty after the input guard trims it.
    assert_error(h.client.post(h.url("/verify"), json={"text": " " * 12}), 422, "INPUT_INVALID")


def test_malformed_json(h: Harness) -> None:
    resp = h.client.post(
        h.url("/verify"), content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert_error(resp, 422, "INPUT_INVALID")


def test_boundaries_10_and_500_accepted(h: Harness) -> None:
    for text in ("x" * 10, "y" * 500):
        h.llm.responses.append(rejection("not_a_claim"))
        assert h.client.post(h.url("/verify"), json={"text": text}).status_code == 200


def test_burst_hits_429_with_retry_after(h: Harness) -> None:
    h.llm.responses.append(rejection("off_topic"))
    limit = h.settings.limits.per_client_requests_per_minute
    for _ in range(limit):
        assert (
            h.client.post(h.url("/verify"), json={"text": "Cricket is lovely."}).status_code == 200
        )
    resp = h.client.post(h.url("/verify"), json={"text": "Cricket is lovely."})
    err = assert_error(resp, 429, "RATE_LIMITED")
    assert int(resp.headers["Retry-After"]) >= 1
    assert err["retry_after_s"] == int(resp.headers["Retry-After"])
    assert err["retry_after_s"] <= 60


def test_invalid_requests_count_against_the_limit(h: Harness) -> None:
    limit = h.settings.limits.per_client_requests_per_minute
    for _ in range(limit):
        h.client.post(h.url("/verify"), json={"text": "short"})
    assert h.client.post(h.url("/verify"), json={"text": "short"}).status_code == 429


class RaisingLLM(FakeLLM):
    """Answers from the script until `error` is set, then raises it."""

    error: Exception | None = None

    def complete_json(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        if self.error is not None:
            raise self.error
        return super().complete_json(*args, **kwargs)


def test_budget_exhausted_503_but_cached_still_served(tmp_path) -> None:  # noqa: ANN001
    llm = RaisingLLM()
    h = build_harness(tmp_path, llm=llm)
    with h.client:
        script_ok(h)
        assert h.client.post(h.url("/verify"), json={"text": CLAIM}).status_code == 200
        llm.error = BudgetExhausted("spent")
        cached = h.client.post(h.url("/verify"), json={"text": CLAIM})
        assert cached.status_code == 200 and cached.json()["meta"]["cached"] is True
        resp = h.client.post(h.url("/verify"), json={"text": "Inflation was 4.5% in FY2025."})
        assert_error(resp, 503, "DAILY_BUDGET_EXHAUSTED")


def test_llm_unavailable_503(tmp_path) -> None:  # noqa: ANN001
    llm = RaisingLLM()
    llm.error = LLMUnavailable([ProviderFailure("groq", "m", "rate_limited", "429")])
    h = build_harness(tmp_path, llm=llm)
    with h.client:
        err = assert_error(
            h.client.post(h.url("/verify"), json={"text": CLAIM}), 503, "LLM_UNAVAILABLE"
        )
        assert "groq" not in err["message"]  # provider details stay in the log


def test_internal_error_hides_details(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_text: str) -> None:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(h.state.service, "verify", boom)
    resp = h.client.post(h.url("/verify"), json={"text": CLAIM})
    assert_error(resp, 500, "INTERNAL")
    assert "secret" not in resp.text and "Traceback" not in resp.text


def test_nothing_stored_on_input_error(h: Harness) -> None:
    h.client.post(h.url("/verify"), json={"text": " " * 12})
    with h.state.engine.connect() as conn:
        from sqlalchemy import text

        assert conn.execute(text("SELECT count(*) FROM verification")).scalar() == 0


def _hold_all_slots(h: Harness) -> int:
    n = 0
    while h.state.llm_slots.acquire(blocking=False):
        n += 1
    return n


def test_cached_and_invalid_requests_do_not_wait_for_llm_slots(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from claimcheck.api.routes import verify as verify_route

    script_ok(h)
    assert h.client.post(h.url("/verify"), json={"text": CLAIM}).status_code == 200
    held = _hold_all_slots(h)  # 4 uncached verifications in flight
    assert held == h.settings.limits.max_concurrent_llm
    monkeypatch.setattr(verify_route, "SLOT_TIMEOUT_S", 0.05)
    try:
        t0 = time.perf_counter()
        cached = h.client.post(h.url("/verify"), json={"text": CLAIM})
        assert cached.status_code == 200 and cached.json()["meta"]["cached"] is True
        assert (time.perf_counter() - t0) * 1000 < 100
        assert_error(h.client.post(h.url("/verify"), json={"text": " " * 12}), 422, "INPUT_INVALID")
        # An uncached claim does wait, and gives up with 503 when no slot frees.
        busy = h.client.post(h.url("/verify"), json={"text": "Inflation was 4.5% in FY2025."})
        err = assert_error(busy, 503, "LLM_UNAVAILABLE")
        assert "busy" in err["message"]
    finally:
        for _ in range(held):
            h.state.llm_slots.release()
