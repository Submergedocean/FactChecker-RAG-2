"""Claim extraction: one guarded message → at most 3 English claims, or a rejection.

The small extractor model handles English. Non-English input escalates to the verifier model:
Urdu script or common Roman Urdu words are detected in code and go straight to it;
when the small model itself reports a non-English language, the extraction is re-run on it.
Periods are normalised in code (FY26 / 2025-26 / FY2026 → FY2026).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel

from claimcheck.llm.client import LLMResult, Messages, Role
from claimcheck.llm.prompts import extractor as prompt
from claimcheck.llm.prompts.extractor import ExtractedClaim, ExtractorOutput, RejectReason
from claimcheck.schemas import InputLanguage
from claimcheck.textnorm import find_fiscal_years

MAX_CLAIMS = 3

_URDU_SCRIPT_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
# Function words that are frequent in Roman Urdu and rare in English text.
_ROMAN_URDU_WORDS = frozenset(
    [
        "ki",
        "ka",
        "ke",
        "mein",
        "main",
        "hai",
        "hain",
        "tha",
        "thi",
        "thay",
        "hua",
        "hui",
        "huay",
        "se",
        "ko",
        "ne",
        "aur",
        "bhi",
        "nahi",
        "nahin",
        "kya",
        "kiya",
        "gaya",
        "gayi",
        "saal",
        "mulk",
        "rupay",
        "rupaye",
        "arab",
        "kharab",
        "crore",
        "lakh",
        "barhi",
        "barha",
        "barh",
        "kam",
        "ziada",
        "zyada",
        "hogai",
        "hogaya",
        "ho",
        "gayi",
    ]
)


class JSONCompleter(Protocol):
    """`LLMClient` or `tests.fakes.FakeLLM`."""

    def complete_json[T: BaseModel](
        self,
        messages: Messages,
        schema: type[T],
        *,
        role: Role,
        prompt_version: str,
        use_cache: bool = True,
    ) -> LLMResult[T]: ...


def looks_non_english(text: str) -> bool:
    """Urdu script, or at least two distinct Roman Urdu function words."""
    if _URDU_SCRIPT_RE.search(text):
        return True
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & _ROMAN_URDU_WORDS) >= 2


def normalize_period(period: str | None) -> str | None:
    """Rewrite each fiscal year as FY<end year> ('2025-26' → 'FY2026', 'FY26' → 'FY2026');
    month ranges and other words are kept."""
    if not period:
        return None
    out, pos = [], 0
    for fy in find_fiscal_years(period):
        out.append(period[pos : fy.start])
        out.append(f"FY{fy.end_year}")
        pos = fy.end
    out.append(period[pos:])
    text = " ".join("".join(out).split())
    return text or None


@dataclass
class Extraction:
    language: InputLanguage
    claims: list[ExtractedClaim] = field(default_factory=list)
    reject_reason: RejectReason | None = None
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    calls: list[LLMResult[ExtractorOutput]] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return self.reject_reason is not None

    @property
    def cached(self) -> bool:
        return all(c.cached for c in self.calls)


class Extractor:
    def __init__(self, llm: JSONCompleter, *, max_claims: int = MAX_CLAIMS) -> None:
        self.llm = llm
        self.max_claims = max_claims

    def _call(self, text: str, role: Role, use_cache: bool) -> LLMResult[ExtractorOutput]:
        return self.llm.complete_json(
            prompt.build_messages(text),
            ExtractorOutput,
            role=role,
            prompt_version=prompt.PROMPT_VERSION,
            use_cache=use_cache,
        )

    def extract(self, text: str, *, use_cache: bool = True) -> Extraction:
        """`text` must already have passed the input guard. `use_cache=False` skips the LLM
        cache (no read, no write)."""
        calls: list[LLMResult[ExtractorOutput]] = []
        role: Role = "verifier" if looks_non_english(text) else "extractor"
        result = self._call(text, role, use_cache)
        calls.append(result)
        if role == "extractor" and result.value.language is not InputLanguage.EN:
            result = self._call(text, "verifier", use_cache)
            calls.append(result)
        return self._interpret(result, calls)

    def _interpret(
        self, result: LLMResult[ExtractorOutput], calls: list[LLMResult[ExtractorOutput]]
    ) -> Extraction:
        out = result.value
        latency = sum(c.latency_ms or 0 for c in calls if not c.cached)
        base = {
            "language": out.language,
            "provider": result.provider,
            "model": result.model,
            "latency_ms": latency,
            "calls": calls,
        }
        claims = [
            c.model_copy(
                update={
                    "claim_en": " ".join(c.claim_en.split()),
                    "period": normalize_period(c.period),
                }
            )
            for c in out.claims
            if c.claim_en.strip()
        ]
        if out.reject_reason is not None and not out.checkable:
            return Extraction(reject_reason=out.reject_reason, **base)
        if not out.checkable or not claims:
            # Inconsistent answers (checkable with no claims, or not checkable without a
            # reason) are treated as "no claim to check".
            return Extraction(reject_reason=out.reject_reason or RejectReason.NOT_A_CLAIM, **base)
        return Extraction(claims=claims[: self.max_claims], **base)
