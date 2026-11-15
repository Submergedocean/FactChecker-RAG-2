"""SQLAlchemy models of the five tables.

The column names, types, CHECK constraints, keys and the lookup index match the schema's DDL;
`tests/test_db.py` compares the created schema with it. Timestamps are UTC ISO-8601 text
(`utc_now_iso`), JSON columns are text.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Text,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now_iso() -> str:
    """Current UTC time as ISO-8601 with an explicit offset (`2026-11-15T09:30:00.123456+00:00`)."""
    return datetime.now(UTC).isoformat()


def utc_day() -> str:
    """Current UTC date (`2026-11-15`), the `llm_usage.day` key."""
    return datetime.now(UTC).date().isoformat()


class Base(DeclarativeBase):
    pass


class Verification(Base):
    __tablename__ = "verification"
    __table_args__ = (
        CheckConstraint("length(input_text) BETWEEN 10 AND 500", name="ck_verification_input"),
        CheckConstraint("status IN ('ok','rejected')", name="ck_verification_status"),
        Index(
            "ix_verification_lookup",
            "input_hash",
            "index_version",
            "pipeline_version",
            text("created_at DESC"),
        ),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)  # "ver_" + ULID
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    input_hash: Mapped[str] = mapped_column(Text, nullable=False)  # sha256 of normalised input
    language: Mapped[str | None] = mapped_column(Text)  # en | ur | roman_ur | other
    status: Mapped[str] = mapped_column(Text, nullable=False)
    reject_reason: Mapped[str | None] = mapped_column(Text)
    response_json: Mapped[str] = mapped_column(Text, nullable=False)  # exact API payload
    index_version: Mapped[str] = mapped_column(Text, nullable=False)
    pipeline_version: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)


class VerificationClaim(Base):
    __tablename__ = "verification_claim"
    __table_args__ = (
        CheckConstraint(
            "verdict IN ('SUPPORTED','REFUTED','NEI')", name="ck_verification_claim_verdict"
        ),
    )

    verification_id: Mapped[str] = mapped_column(
        Text, ForeignKey("verification.id", ondelete="CASCADE"), primary_key=True
    )
    claim_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    claim_en: Mapped[str] = mapped_column(Text, nullable=False)
    verdict: Mapped[str] = mapped_column(Text, nullable=False)
    flags: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'[]'"))
    n_citations: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint("rating IN ('up','down')", name="ck_feedback_rating"),
        CheckConstraint("comment IS NULL OR length(comment) <= 500", name="ck_feedback_comment"),
        ForeignKeyConstraint(
            ["verification_id", "claim_index"],
            ["verification_claim.verification_id", "verification_claim.claim_index"],
            ondelete="CASCADE",
        ),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    verification_id: Mapped[str] = mapped_column(Text, nullable=False)
    claim_index: Mapped[int] = mapped_column(Integer, nullable=False)
    rating: Mapped[str] = mapped_column(Text, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)


class LLMCacheEntry(Base):
    __tablename__ = "llm_cache"

    # sha256(provider, model, temperature, messages, schema): see claimcheck.llm.cache.cache_key
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_version: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)


class LLMUsage(Base):
    __tablename__ = "llm_usage"

    day: Mapped[str] = mapped_column(Text, primary_key=True)
    provider: Mapped[str] = mapped_column(Text, primary_key=True)
    model: Mapped[str] = mapped_column(Text, primary_key=True)
    requests: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    completion_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )


TABLES = ("verification", "verification_claim", "feedback", "llm_cache", "llm_usage")

__all__ = [
    "TABLES",
    "Base",
    "Feedback",
    "LLMCacheEntry",
    "LLMUsage",
    "Verification",
    "VerificationClaim",
    "utc_day",
    "utc_now_iso",
]
