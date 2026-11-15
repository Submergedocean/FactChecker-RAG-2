"""Database: five tables, columns, constraints, index, PRAGMAs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from claimcheck.db.models import (
    TABLES,
    Feedback,
    Verification,
    VerificationClaim,
    utc_day,
    utc_now_iso,
)
from claimcheck.db.session import make_engine, make_session_factory, session_scope

# The schema, column by column: (name, type, not null, primary-key position).
EXPECTED_COLUMNS = {
    "verification": [
        # NOT NULL on TEXT primary keys is stricter than the DDL (SQLite would allow NULL there).
        ("id", "TEXT", True, 1),
        ("created_at", "TEXT", True, 0),
        ("input_text", "TEXT", True, 0),
        ("input_hash", "TEXT", True, 0),
        ("language", "TEXT", False, 0),
        ("status", "TEXT", True, 0),
        ("reject_reason", "TEXT", False, 0),
        ("response_json", "TEXT", True, 0),
        ("index_version", "TEXT", True, 0),
        ("pipeline_version", "TEXT", True, 0),
        ("latency_ms", "INTEGER", False, 0),
    ],
    "verification_claim": [
        ("verification_id", "TEXT", True, 1),
        ("claim_index", "INTEGER", True, 2),
        ("claim_en", "TEXT", True, 0),
        ("verdict", "TEXT", True, 0),
        ("flags", "TEXT", True, 0),
        ("n_citations", "INTEGER", True, 0),
    ],
    "feedback": [
        ("id", "INTEGER", True, 1),  # INTEGER PRIMARY KEY AUTOINCREMENT (rowid)
        ("verification_id", "TEXT", True, 0),
        ("claim_index", "INTEGER", True, 0),
        ("rating", "TEXT", True, 0),
        ("comment", "TEXT", False, 0),
        ("created_at", "TEXT", True, 0),
    ],
    "llm_cache": [
        ("key", "TEXT", True, 1),
        ("provider", "TEXT", True, 0),
        ("model", "TEXT", True, 0),
        ("prompt_version", "TEXT", True, 0),
        ("response_text", "TEXT", True, 0),
        ("prompt_tokens", "INTEGER", False, 0),
        ("completion_tokens", "INTEGER", False, 0),
        ("latency_ms", "INTEGER", False, 0),
        ("created_at", "TEXT", True, 0),
    ],
    "llm_usage": [
        ("day", "TEXT", True, 1),
        ("provider", "TEXT", True, 2),
        ("model", "TEXT", True, 3),
        ("requests", "INTEGER", True, 0),
        ("prompt_tokens", "INTEGER", True, 0),
        ("completion_tokens", "INTEGER", True, 0),
    ],
}


@pytest.fixture
def engine(tmp_path: Path):
    return make_engine(tmp_path / "data" / "app.db")


@pytest.fixture
def sessions(engine):
    return make_session_factory(engine)


def _verification(vid: str = "ver_01TEST", **kw) -> Verification:
    values = {
        "id": vid,
        "created_at": utc_now_iso(),
        "input_text": "Inflation fell to 4.5 percent in FY2025.",
        "input_hash": "0" * 64,
        "language": "en",
        "status": "ok",
        "response_json": json.dumps({"id": vid}),
        "index_version": "idx",
        "pipeline_version": "pipe",
        "latency_ms": 1234,
    }
    values.update(kw)
    return Verification(**values)


def test_tables_from_section_8_exist(engine) -> None:
    assert sorted(inspect(engine).get_table_names()) == sorted(TABLES)
    with engine.connect() as conn:
        for table, expected in EXPECTED_COLUMNS.items():
            rows = conn.execute(text(f"PRAGMA table_info({table})")).all()
            got = [(r.name, r.type, bool(r.notnull), r.pk) for r in rows]
            assert got == expected, table


def test_feedback_id_autoincrements(engine) -> None:
    with engine.connect() as conn:
        ddl = conn.execute(text("SELECT sql FROM sqlite_master WHERE name = 'feedback'")).scalar()
    assert "PRIMARY KEY AUTOINCREMENT" in ddl.replace("\n", " ")


def test_wal_foreign_keys_and_busy_timeout_are_on(engine) -> None:
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
        assert conn.execute(text("PRAGMA busy_timeout")).scalar() == 5000


def test_lookup_index_and_foreign_keys(engine) -> None:
    with engine.connect() as conn:
        cols = conn.execute(text("PRAGMA index_xinfo(ix_verification_lookup)")).all()
        keyed = [(c.name, c.desc) for c in cols if c.key]
        assert keyed == [
            ("input_hash", 0),
            ("index_version", 0),
            ("pipeline_version", 0),
            ("created_at", 1),
        ]
        fks = conn.execute(text("PRAGMA foreign_key_list(feedback)")).all()
        assert {(f.table, f.on_delete) for f in fks} == {("verification_claim", "CASCADE")}
        assert sorted((f.__getattribute__("from"), f.to) for f in fks) == [
            ("claim_index", "claim_index"),
            ("verification_id", "verification_id"),
        ]
        fks = conn.execute(text("PRAGMA foreign_key_list(verification_claim)")).all()
        assert [(f.table, f.on_delete) for f in fks] == [("verification", "CASCADE")]


def test_defaults_and_cascade(sessions) -> None:
    with session_scope(sessions) as s:
        s.add(_verification())
        s.flush()
        s.add(
            VerificationClaim(
                verification_id="ver_01TEST", claim_index=0, claim_en="x", verdict="NEI"
            )
        )
        s.flush()
        s.add(
            Feedback(
                verification_id="ver_01TEST", claim_index=0, rating="up", created_at=utc_now_iso()
            )
        )
    with session_scope(sessions) as s:
        claim = s.get(VerificationClaim, ("ver_01TEST", 0))
        assert claim.flags == "[]" and claim.n_citations == 0
        s.delete(s.get(Verification, "ver_01TEST"))
    with session_scope(sessions) as s:
        assert s.query(VerificationClaim).count() == 0 and s.query(Feedback).count() == 0


@pytest.mark.parametrize(
    "make",
    [
        lambda: _verification(input_text="too short"),
        lambda: _verification(input_text="x" * 501),
        lambda: _verification(status="pending"),
        lambda: VerificationClaim(
            verification_id="ver_01TEST", claim_index=1, claim_en="x", verdict="TRUE"
        ),
        lambda: Feedback(verification_id="ver_01TEST", claim_index=0, rating="meh", created_at="t"),
        lambda: Feedback(
            verification_id="ver_01TEST",
            claim_index=0,
            rating="up",
            comment="c" * 501,
            created_at="t",
        ),
        # foreign keys: no such verification / claim
        lambda: VerificationClaim(
            verification_id="ver_MISSING", claim_index=0, claim_en="x", verdict="NEI"
        ),
        lambda: Feedback(verification_id="ver_01TEST", claim_index=9, rating="up", created_at="t"),
    ],
)
def test_constraints_reject_bad_rows(sessions, make) -> None:
    with session_scope(sessions) as s:
        s.add(_verification())
        s.flush()
        s.add(
            VerificationClaim(
                verification_id="ver_01TEST", claim_index=0, claim_en="x", verdict="NEI"
            )
        )
    with pytest.raises(IntegrityError), session_scope(sessions) as s:
        s.add(make())
        s.flush()


def test_make_engine_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "app.db"
    make_engine(path)
    engine = make_engine(path)
    assert sorted(inspect(engine).get_table_names()) == sorted(TABLES)


def test_time_helpers() -> None:
    assert utc_now_iso().endswith("+00:00")
    assert len(utc_day()) == 10 and utc_day()[4] == "-"
