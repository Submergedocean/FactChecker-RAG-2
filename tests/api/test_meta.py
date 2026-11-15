"""`/meta`, `/documents`, `/eval/summary`, `/healthz`, `/readyz`, unknown routes."""

from __future__ import annotations

import json

from claimcheck.schemas import Manifest, MetaResponse
from tests.api.conftest import Harness
from tests.api.test_verify import assert_error


def test_meta(h: Harness) -> None:
    body = h.client.get(h.url("/meta")).json()
    MetaResponse.model_validate(body)
    assert body["index_version"] == "test-index-v1"
    assert body["pipeline_version"] == h.state.service.pipeline_version
    assert body["edition"] == "2025-26"
    assert body["models"]["extractor"] == "fake/fake-small"
    assert body["limits"]["input_max_chars"] == 500
    assert body["limits"]["per_client_requests_per_minute"] == 6


def test_documents(h: Harness) -> None:
    body = h.client.get(h.url("/documents")).json()
    manifest = Manifest.model_validate(body)
    assert {d.doc_id for d in manifest.documents} >= {"pes2526_ch01", "pes2526_ch07"}


def test_eval_summary_404_until_written(h: Harness) -> None:
    assert_error(h.client.get(h.url("/eval/summary")), 404, "NOT_FOUND")
    path = h.settings.paths.eval_summary
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"accuracy": 0.8}), encoding="utf-8")
    assert h.client.get(h.url("/eval/summary")).json() == {"accuracy": 0.8}


def test_health_at_root_and_prefix(h: Harness) -> None:
    for base in ("", "/api/v1"):
        assert h.client.get(f"{base}/healthz").json() == {"status": "ok", "checks": {}}
        ready = h.client.get(f"{base}/readyz")
        assert ready.status_code == 200
        assert ready.json() == {"status": "ok", "checks": {"index": True, "database": True}}


def test_readyz_503_when_database_missing(h: Harness, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(h.state.service, "store", None)
    resp = h.client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json()["checks"] == {"index": True, "database": False}


def test_unknown_route_404_envelope(h: Harness) -> None:
    assert_error(h.client.get(h.url("/nope")), 404, "NOT_FOUND")
    assert_error(h.client.get("/nope"), 404, "NOT_FOUND")


def test_wrong_method_keeps_envelope(h: Harness) -> None:
    resp = h.client.get(h.url("/verify"))
    assert resp.status_code == 405
    assert resp.json()["error"]["code"] == "METHOD_NOT_ALLOWED"


def test_openapi_documents_error_envelope(h: Harness) -> None:
    spec = h.client.get(h.url("/openapi.json")).json()
    responses = spec["paths"]["/api/v1/verify"]["post"]["responses"]
    assert {"200", "404", "422", "429", "500", "503"} <= set(responses)
    assert "ErrorEnvelope" in json.dumps(responses["429"])
