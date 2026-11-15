"""Permalinks and feedback."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from tests.api.conftest import Harness
from tests.api.test_verify import CLAIM, assert_error, script_ok

UNKNOWN_ID = "ver_01JCXYZ0000000000000000000"


def make_result(h: Harness) -> dict:
    script_ok(h)
    return h.client.post(h.url("/verify"), json={"text": CLAIM}).json()


def test_permalink_replays_the_stored_payload(h: Harness) -> None:
    first = make_result(h)
    resp = h.client.get(h.url(f"/verifications/{first['id']}"))
    assert resp.status_code == 200
    assert resp.json() == first


@pytest.mark.parametrize("vid", [UNKNOWN_ID, "ver_bad", "x" * 40, "..%2Fapp.db"])
def test_permalink_unknown_404(h: Harness, vid: str) -> None:
    assert_error(h.client.get(h.url(f"/verifications/{vid}")), 404, "NOT_FOUND")


def test_feedback_204_and_stored(h: Harness) -> None:
    vid = make_result(h)["id"]
    resp = h.client.post(
        h.url(f"/verifications/{vid}/feedback"),
        json={"claim_index": 0, "rating": "down", "comment": "  wrong page  "},
    )
    assert resp.status_code == 204 and resp.content == b""
    with h.state.engine.connect() as conn:
        row = conn.execute(text("SELECT rating, comment FROM feedback")).one()
    assert tuple(row) == ("down", "wrong page")


def test_feedback_unknown_verification_or_claim_404(h: Harness) -> None:
    body = {"claim_index": 0, "rating": "up"}
    assert_error(
        h.client.post(h.url(f"/verifications/{UNKNOWN_ID}/feedback"), json=body), 404, "NOT_FOUND"
    )
    vid = make_result(h)["id"]
    resp = h.client.post(
        h.url(f"/verifications/{vid}/feedback"), json={"claim_index": 3, "rating": "up"}
    )
    assert_error(resp, 404, "NOT_FOUND")


@pytest.mark.parametrize(
    "body",
    [
        {"claim_index": 0, "rating": "meh"},
        {"claim_index": -1, "rating": "up"},
        {"claim_index": 0, "rating": "up", "comment": "c" * 501},
        {"rating": "up"},
    ],
    ids=["rating", "negative-index", "comment-501", "missing-index"],
)
def test_feedback_invalid_422(h: Harness, body: dict) -> None:
    vid = make_result(h)["id"]
    assert_error(
        h.client.post(h.url(f"/verifications/{vid}/feedback"), json=body), 422, "INPUT_INVALID"
    )


def test_feedback_rate_limited(h: Harness) -> None:
    vid = make_result(h)["id"]
    limit = h.settings.limits.per_client_feedback_per_minute
    body = {"claim_index": 0, "rating": "up"}
    for _ in range(limit):
        assert h.client.post(h.url(f"/verifications/{vid}/feedback"), json=body).status_code == 204
    resp = h.client.post(h.url(f"/verifications/{vid}/feedback"), json=body)
    assert_error(resp, 429, "RATE_LIMITED")
    assert "Retry-After" in resp.headers
