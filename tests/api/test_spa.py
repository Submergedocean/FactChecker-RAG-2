"""With a built SPA on disk, the "/" mount serves the SPA and never takes
an API path or a non-GET method from the router (404/405 keep the error envelope)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.api.conftest import Harness, build_harness
from tests.api.test_verify import assert_error

INDEX = "<!doctype html><title>Claim Checker</title>"


@pytest.fixture
def spa(tmp_path: Path) -> Iterator[Harness]:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(INDEX, encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside the build", encoding="utf-8")
    h = build_harness(tmp_path, static_dir=dist)
    with h.client:
        yield h


def test_spa_files_and_client_routes(spa: Harness) -> None:
    for path in ("/", "/v/ver_01JCXYZ0000000000000000000", "/evaluation", "/about"):
        resp = spa.client.get(path)
        assert resp.status_code == 200 and resp.text == INDEX, path
        assert "default-src 'self'" in resp.headers["content-security-policy"]
    js = spa.client.get("/assets/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert spa.client.head("/").status_code == 200


def test_spa_does_not_serve_outside_the_build(spa: Harness) -> None:
    for path in ("/../secret.txt", "/%2E%2E/secret.txt", "/assets/..%2F..%2Fsecret.txt"):
        assert "outside the build" not in spa.client.get(path).text


def test_api_errors_unchanged_with_spa(spa: Harness) -> None:
    resp = spa.client.get(spa.url("/verify"))
    assert resp.status_code == 405 and resp.json()["error"]["code"] == "METHOD_NOT_ALLOWED"
    assert_error(spa.client.post(spa.url("/nope")), 404, "NOT_FOUND")
    assert_error(spa.client.get(spa.url("/nope")), 404, "NOT_FOUND")
    assert_error(spa.client.get("/api/v1"), 404, "NOT_FOUND")
    assert_error(spa.client.post("/v/whatever"), 404, "NOT_FOUND")  # SPA is GET/HEAD only


def test_api_and_health_still_routed_with_spa(spa: Harness) -> None:
    assert spa.client.get(spa.url("/meta")).json()["index_version"] == "test-index-v1"
    assert spa.client.get("/readyz").json()["status"] == "ok"
    assert spa.client.get("/healthz").json()["status"] == "ok"
