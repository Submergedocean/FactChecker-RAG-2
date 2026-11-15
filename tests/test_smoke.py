"""Packages import, settings load, schemas validate, CLI mounts every sub-app."""

import importlib

import pytest
from typer.testing import CliRunner

from claimcheck import schemas
from claimcheck.cli import SUB_APPS, app
from claimcheck.config import Settings

PACKAGES = [
    "claimcheck",
    "claimcheck.config",
    "claimcheck.schemas",
    "claimcheck.cli",
    "claimcheck.ingest.cli",
    "claimcheck.index.cli",
    "claimcheck.retrieval.cli",
    "claimcheck.llm",
    "claimcheck.llm.prompts",
    "claimcheck.pipeline.cli",
    "claimcheck.claims.cli",
    "claimcheck.eval.cli",
    "claimcheck.db",
    "claimcheck.api.cli",
    "claimcheck.api.routes",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_imports(name: str) -> None:
    importlib.import_module(name)


def test_settings_load_from_yaml() -> None:
    s = Settings()
    assert s.limits.input_min_chars == 10
    assert s.limits.input_max_chars == 500
    assert s.chunking.variant == "table_aware"
    assert s.api.prefix == "/api/v1"


def test_defaults_match_proposal() -> None:
    """Proposal sections 6.1-6.3. Change only with the proposal."""
    s = Settings()
    assert s.editions == ["2024-25", "2025-26"]
    assert s.latest_edition == "2025-26"
    assert (s.chunking.window_words, s.chunking.overlap_words) == (150, 30)
    assert s.chunking.max_tokens == 512
    assert s.embedding.model == "BAAI/bge-small-en-v1.5"
    assert s.retrieval.top_k == 5
    assert s.retrieval.rrf_k == 60
    assert s.llm.temperature == 0.0


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIMCHECK_RETRIEVAL__TOP_K", "11")
    assert Settings().retrieval.top_k == 11


def test_cli_lists_every_sub_app() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in SUB_APPS:
        assert name in result.output
    assert set(SUB_APPS) == {
        "ingest", "index", "retrieval", "pipeline", "claims", "eval", "api",
    }


EXAMPLE_RESPONSE = {
    "id": "ver_01JCXYZABCDEFGHJKMNPQRSTVW",
    "created_at": "2026-11-12T10:00:00Z",
    "input": {"text": "Pakistan ki GDP FY26 mein 3% barhi", "language": "roman_ur"},
    "status": "ok",
    "reject_reason": None,
    "message": None,
    "claims": [{
        "claim_en": "Pakistan's GDP grew by 3% in FY2026.",
        "verdict": "SUPPORTED",
        "reasoning": "...",
        "flags": [],
        "citations": [{
            "chunk_id": "9f2c0000aaaabbbb", "quote": "...", "doc_id": "pes2526_ch01",
            "edition": "2025-26", "chapter": "1 Growth and Investment", "page": 3,
            "printed_page": "3", "table_id": None,
            "source_url": "https://www.finance.gov.pk/x.pdf#page=3",
        }],
        "evidence": [{
            "chunk_id": "9f2c0000aaaabbbb", "rank": 1, "kind": "text", "text": "...",
            "doc_id": "pes2526_ch01", "edition": "2025-26",
            "chapter": "1 Growth and Investment", "page_start": 3, "page_end": 3,
            "table_id": None, "ranks": {"bm25": 2, "dense": 1}, "rrf": 0.0325, "cited": True,
        }],
    }],
    "meta": {
        "index_version": "v1", "pipeline_version": "p1",
        "models": {"extractor": "a", "verifier": "b"}, "cached": False,
        "latency_ms": {"extract": 600, "retrieve": 45, "verify": 1800, "total": 2500},
    },
}


def test_verification_response_roundtrip() -> None:
    r = schemas.VerificationResponse.model_validate(EXAMPLE_RESPONSE)
    assert r.claims[0].verdict is schemas.Verdict.SUPPORTED
    again = schemas.VerificationResponse.model_validate_json(r.model_dump_json())
    assert again == r


def test_verify_request_bounds() -> None:
    with pytest.raises(ValueError):
        schemas.VerifyRequest(text="short")
    with pytest.raises(ValueError):
        schemas.VerifyRequest(text="x" * 501)
    schemas.VerifyRequest(text="GDP grew by 3 percent")


def test_error_envelope() -> None:
    e = schemas.ErrorEnvelope.model_validate(
        {"error": {"code": "RATE_LIMITED", "message": "slow down", "retry_after_s": 30}}
    )
    assert e.error.code is schemas.ErrorCode.RATE_LIMITED


def test_claim_record() -> None:
    c = schemas.Claim.model_validate({
        "claim_id": "C0001", "fact_id": "F0001", "claim_text": "Inflation fell.",
        "label": "SUPPORTED", "language": "en", "claim_type": "textual",
        "reasoning_type": "trend", "topic": "inflation",
        "evidence": [{"doc_id": "pes2526_ch07", "page": 2, "quote": "Inflation fell"}],
    })
    assert c.evidence[0].page == 2
