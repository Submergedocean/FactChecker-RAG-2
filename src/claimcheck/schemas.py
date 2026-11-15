"""Shared data models: corpus artifacts and claims and the API.

This is the shared contract: the API and the frontend's generated TypeScript types depend on
it. Change it only deliberately, together with every consumer.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- enums


class DocKind(StrEnum):
    CHAPTER = "chapter"
    OVERVIEW = "overview"
    HIGHLIGHTS = "highlights"
    ANNEX = "annex"
    INDICATORS = "indicators"


class ChunkKind(StrEnum):
    TEXT = "text"
    TABLE = "table"


class Variant(StrEnum):
    TABLE_AWARE = "table_aware"
    PLAIN_TEXT = "plain_text"


class Verdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NEI = "NEI"


class ClaimLanguage(StrEnum):
    EN = "en"
    UR = "ur"
    ROMAN_UR = "roman_ur"


class InputLanguage(StrEnum):
    EN = "en"
    UR = "ur"
    ROMAN_UR = "roman_ur"
    OTHER = "other"


class ClaimType(StrEnum):
    TEXTUAL = "textual"
    TABULAR = "tabular"


class ReasoningType(StrEnum):
    NUMERIC = "numeric"
    TREND = "trend"
    COMPARISON = "comparison"
    DESCRIPTIVE = "descriptive"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class VerificationStatus(StrEnum):
    OK = "ok"
    REJECTED = "rejected"


class Rating(StrEnum):
    UP = "up"
    DOWN = "down"


class ErrorCode(StrEnum):
    INPUT_INVALID = "INPUT_INVALID"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    DAILY_BUDGET_EXHAUSTED = "DAILY_BUDGET_EXHAUSTED"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    INTERNAL = "INTERNAL"


# ------------------------------------------------------------- 7.1 corpus artifacts


class ManifestEntry(_Model):
    doc_id: str  # e.g. "pes2526_ch07"
    edition: str
    kind: DocKind
    chapter_no: int | None = None
    title: str
    source_url: str
    filename: str
    sha256: str | None = None
    n_pages: int | None = None


class Manifest(_Model):
    documents: list[ManifestEntry]


class PageRecord(_Model):
    """One row of pages.parquet."""

    doc_id: str
    page: int = Field(ge=1)  # 1-based PDF page
    printed_page: str | None = None
    text_body: str  # tables removed
    text_raw: str  # pdfplumber plain extraction
    flags: list[str] = Field(default_factory=list)


class TableRecord(_Model):
    """One row of tables.parquet."""

    table_id: str  # "{doc_id}:p{page}:t{n}"
    doc_id: str
    page: int = Field(ge=1)
    bbox: tuple[float, float, float, float]
    caption: str | None = None
    units: str | None = None
    header: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)  # stored as JSON
    markdown: str
    continued_from: str | None = None
    flags: list[str] = Field(default_factory=list)


class ChunkRecord(_Model):
    """One row of chunks.parquet. Row i matches row i of faiss.index."""

    chunk_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    doc_id: str
    edition: str
    chapter_no: int | None = None
    chapter_title: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    kind: ChunkKind
    table_id: str | None = None
    text: str  # shown to LLM and user
    embed_text: str  # context header + content
    bm25_tokens: list[str]
    n_words: int
    n_tokens: int


class IndexMeta(_Model):
    """index_meta.json."""

    index_version: str
    variant: Variant
    embedding_model: str
    embedding_revision: str | None = None
    dim: int
    n_chunks: int
    chunking: dict[str, Any]
    built_at: datetime
    git_commit: str | None = None


# ----------------------------------------------------------------- 7.2 claims


class EvidenceRef(_Model):
    doc_id: str
    page: int = Field(ge=1)
    table_id: str | None = None
    quote: str
    row_label: str | None = None
    col_label: str | None = None


class Claim(_Model):
    """One line of data/claims/claims.jsonl."""

    claim_id: str = Field(pattern=r"^C\d{4,}$")
    fact_id: str
    claim_text: str
    label: Verdict
    language: ClaimLanguage
    parallel_of: str | None = None
    claim_type: ClaimType
    reasoning_type: ReasoningType
    topic: str
    evidence: list[EvidenceRef] = Field(default_factory=list)  # empty for NEI
    origin: str | None = None
    annotator: str | None = None
    second_label: Verdict | None = None
    notes: str | None = None


# data/claims/splits.json: {fact_id: split}
Splits = dict[str, Split]


# ------------------------------------------------------------------- 9.2 API


class VerifyRequest(_Model):
    text: str = Field(min_length=10, max_length=500)


class FeedbackRequest(_Model):
    claim_index: int = Field(ge=0)
    rating: Rating
    comment: str | None = Field(default=None, max_length=500)


class Citation(_Model):
    chunk_id: str
    quote: str
    doc_id: str
    edition: str
    chapter: str
    page: int
    printed_page: str | None = None
    table_id: str | None = None
    source_url: str


class EvidencePassage(_Model):
    chunk_id: str
    rank: int = Field(ge=1)
    kind: ChunkKind
    text: str
    doc_id: str
    edition: str
    chapter: str
    page_start: int
    page_end: int
    table_id: str | None = None
    ranks: dict[str, int | None]  # {"bm25": 2, "dense": 1}
    rrf: float
    cited: bool


class ClaimResult(_Model):
    claim_en: str
    verdict: Verdict
    reasoning: str
    flags: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[EvidencePassage] = Field(default_factory=list)


class InputInfo(_Model):
    text: str
    language: InputLanguage | None = None


class LatencyMs(_Model):
    extract: int | None = None
    retrieve: int | None = None
    verify: int | None = None
    total: int | None = None


class ResponseMeta(_Model):
    index_version: str
    pipeline_version: str
    models: dict[str, str]  # {"extractor": ..., "verifier": ...}
    cached: bool = False
    latency_ms: LatencyMs = Field(default_factory=LatencyMs)


class VerificationResponse(_Model):
    """Body of POST /verify and GET /verifications/{id}."""

    id: str = Field(pattern=r"^ver_[0-9A-HJKMNP-TV-Z]{26}$")
    created_at: datetime
    input: InputInfo
    status: VerificationStatus
    reject_reason: str | None = None
    message: str | None = None  # user-facing text when rejected
    claims: list[ClaimResult] = Field(default_factory=list)
    meta: ResponseMeta


class PassageResponse(_Model):
    """GET /passages/{chunk_id}."""

    chunk_id: str
    doc_id: str
    edition: str
    chapter: str
    page_start: int
    page_end: int
    kind: ChunkKind
    table_id: str | None = None
    text: str
    source_url: str
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None


class MetaResponse(_Model):
    """GET /meta."""

    index_version: str
    pipeline_version: str
    edition: str
    models: dict[str, str]
    limits: dict[str, int]
    example_claims: list[str] = Field(default_factory=list)


class HealthResponse(_Model):
    status: str  # "ok" | "not_ready"
    checks: dict[str, bool] = Field(default_factory=dict)


class ErrorBody(_Model):
    code: ErrorCode
    message: str
    retry_after_s: int | None = None


class ErrorEnvelope(_Model):
    """Every non-2xx response."""

    error: ErrorBody
