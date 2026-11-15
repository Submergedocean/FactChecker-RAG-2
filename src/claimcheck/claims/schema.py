"""Claims dataset records and the drafting records around them.

`claimcheck.schemas.Claim` is the contract for one line of `data/claims/claims.jsonl`; this
module adds what the tools need around it: ids, the topic list, the de-duplication key, draft
records (`data/claims/drafts.jsonl`) and the proposal's column view.

Evidence conventions:

- Text evidence: `quote` is a verbatim span of the page's text (`table_id`, `row_label` and
  `col_label` empty).
- Table evidence: `table_id` names the table, `row_label` is the row's label and `col_label` the
  column header as printed in `tables.parquet`, and `quote` is that cell as printed ("1.44").
  The matching text used to find the evidence in a chunk is the last segment of the row label
  followed by the cell ("1. Crops 1.44"), see `evidence_match_text`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from claimcheck.schemas import (
    Claim,
    ClaimLanguage,
    ClaimType,
    DocKind,
    EvidenceRef,
    ManifestEntry,
    ReasoningType,
    Verdict,
)
from claimcheck.textnorm import canonicalize_fiscal_years, normalize_for_match

__all__ = [
    "CLAIM_ID_RE",
    "FACT_ID_RE",
    "PROPOSAL_COLUMNS",
    "PROPOSAL_LABELS",
    "TOPICS",
    "Claim",
    "ClaimLanguage",
    "ClaimType",
    "DraftRecord",
    "DraftStatus",
    "EvidenceRef",
    "Mutation",
    "ReasoningType",
    "Verdict",
    "dedup_key",
    "evidence_match_text",
    "format_id",
    "next_number",
    "row_leaf",
    "topic_for",
]

CLAIM_ID_RE = re.compile(r"^C(\d{4,})$")
FACT_ID_RE = re.compile(r"^F(\d{4,})$")


def format_id(prefix: str, n: int) -> str:
    """format_id('C', 7) -> 'C0007'."""
    return f"{prefix}{n:04d}"


def next_number(ids: Iterable[str | None], pattern: re.Pattern[str]) -> int:
    """1 + the largest number among `ids` that match `pattern` (1 when none do)."""
    numbers = [int(m.group(1)) for i in ids if i and (m := pattern.match(i))]
    return max(numbers, default=0) + 1


# --------------------------------------------------------------------------- topics

# One slug per PES chapter (proposal 7.2: "Chapter or sector, e.g. inflation, trade, debt").
CHAPTER_TOPICS: dict[int, str] = {
    1: "growth",
    2: "agriculture",
    3: "manufacturing",
    4: "fiscal",
    5: "money_credit",
    6: "capital_market",
    7: "inflation",
    8: "trade",
    9: "debt",
    10: "education",
    11: "health",
    12: "population",
    13: "transport",
    14: "energy",
    15: "it",
    16: "social_protection",
    17: "climate",
}
TOPICS: tuple[str, ...] = tuple(CHAPTER_TOPICS.values())

_ANNEX_TOPICS = {"contingent": "debt", "tax": "fiscal", "qna": "growth"}


def topic_for(entry: ManifestEntry) -> str | None:
    """The topic of a chapter or annex; None for documents that span topics (Overview,
    Highlights, Indicators), where the drafter picks one from `TOPICS`."""
    if entry.kind is DocKind.CHAPTER and entry.chapter_no in CHAPTER_TOPICS:
        return CHAPTER_TOPICS[entry.chapter_no]
    if entry.kind is DocKind.ANNEX:
        return _ANNEX_TOPICS.get(entry.doc_id.rsplit("_", 1)[-1])
    return None


# --------------------------------------------------------------------------- text keys

_EDGE = " .;:,!?\"'"


def dedup_key(text: str) -> str:
    """Key for duplicate claims (proposal 8.1: claim de-duplication with number and unit
    normalisation): fiscal years canonical (FY26 = 2025-26), then `normalize_for_match`
    (case, digits, thousands separators, % = per cent = percent, whitespace), edge
    punctuation dropped."""
    return normalize_for_match(canonicalize_fiscal_years(text)).strip(_EDGE)


def row_leaf(row_label: str) -> str:
    """Last segment of a heading-path row label: 'A AGRICULTURE / 1. Crops' -> '1. Crops'."""
    return row_label.rsplit(" / ", 1)[-1].strip()


def evidence_match_text(ev: EvidenceRef) -> str:
    """Text whose tokens locate the evidence in a chunk: the quote, and
    for a table cell the row label's last segment before it."""
    if ev.table_id and ev.row_label:
        return f"{row_leaf(ev.row_label)} {ev.quote}"
    return ev.quote


# --------------------------------------------------------------------------- drafts


class Mutation(StrEnum):
    """How a REFUTED twin differs from its SUPPORTED original."""

    NUMBER = "number"
    DIRECTION = "direction"
    YEAR = "year"
    ENTITY = "entity"


class DraftStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"  # imported into claims.jsonl (as drafted or edited)
    REJECTED = "rejected"


class DraftRecord(BaseModel):
    """One line of `data/claims/drafts.jsonl`: an LLM-proposed claim awaiting review."""

    model_config = ConfigDict(extra="forbid")

    claim: Claim
    status: DraftStatus = DraftStatus.PENDING
    source_chunk_id: str
    source_kind: str  # "text" | "table"
    mutation: Mutation | None = None  # REFUTED twins only
    checks: list[str] = Field(default_factory=list)  # code checks for the reviewer
    model: str | None = None
    prompt_version: str | None = None
    drafted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --------------------------------------------------------------------------- proposal view

# The proposal's claims table (7.2), in its order: eight columns.
PROPOSAL_COLUMNS: tuple[str, ...] = (
    "claim_id",
    "claim_text",
    "label",
    "evidence_ids",
    "source",
    "page",
    "topic",
    "claim_type",
)
PROPOSAL_LABELS: dict[Verdict, str] = {
    Verdict.SUPPORTED: "Supported",
    Verdict.REFUTED: "Contradicted",
    Verdict.NEI: "Not Enough Information",
}
