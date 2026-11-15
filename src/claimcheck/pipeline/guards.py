"""Code guards around the LLM. The model is never trusted to cite.

- Input guard: trim, NFC, Urdu digits to ASCII, 10-500 characters (`textnorm.guard_input`).
- Citations name passage labels only; document, page and table id come from chunk metadata.
- A quote must appear in its cited passage after whitespace and number normalisation
  (`quote_in_passage`); invalid citations are dropped.
- SUPPORTED or REFUTED with no valid citation becomes NEI, flag `downgraded_no_citation`.
- `numeric_warning` (flag only, the verdict stays) when the number the verdict rests on is
  absent from every valid quote: the claim's key number for SUPPORTED, the evidence value for
  REFUTED (for REFUTED the claim's own number is by definition not in the source).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from claimcheck.llm.prompts.extractor import ExtractedClaim, RejectReason
from claimcheck.llm.prompts.verifier import REASONING_MAX_WORDS, VerifierOutput, passage_label
from claimcheck.schemas import ChunkRecord, Verdict
from claimcheck.textnorm import (
    InputGuardError,
    Quantity,
    find_quantities,
    guard_input,
    matches_at_precision,
    normalize_for_match,
    scale_factor,
)

GUARDS_VERSION = "guards-v1"

FLAG_DOWNGRADED = "downgraded_no_citation"
FLAG_NUMERIC = "numeric_warning"

MIN_QUOTE_CHARS = 8
MIN_QUOTE_TOKENS = 2

REJECT_MESSAGES: dict[RejectReason, str] = {
    RejectReason.OFF_TOPIC: (
        "This doesn't look like a claim about Pakistan's economy. The checker covers the "
        "topics of the Pakistan Economic Survey: growth, prices, trade, public finance, debt, "
        "energy, and the social sectors."
    ),
    RejectReason.NOT_A_CLAIM: (
        "There is no factual claim to check here. Enter a statement, for example "
        '"Inflation was 4.5% in FY2025."'
    ),
    RejectReason.QUESTION: (
        "This is a question. Rephrase it as a statement to check, for example "
        '"Pakistan\'s GDP grew 3% in FY2026."'
    ),
    RejectReason.TOO_VAGUE: (
        "The claim is too vague to check. Name the indicator and, if you can, the period "
        "and the value."
    ),
}

__all__ = [
    "FLAG_DOWNGRADED",
    "FLAG_NUMERIC",
    "GUARDS_VERSION",
    "REJECT_MESSAGES",
    "CheckedVerdict",
    "InputGuardError",
    "ValidCitation",
    "apply_guards",
    "check_citations",
    "guard_input",
    "key_quantity",
    "number_in_quotes",
    "quote_in_passage",
    "trim_words",
]


# ------------------------------------------------------------------------- citations


_TABLE_RULE_RE = re.compile(r"(?:\|\s*)?(?::?-{3,}:?\s*\|?\s*)+")
_EDGE_PUNCT = " \t\n\"'`“”‘’.,;:"


def match_key(text: str) -> str:
    """Comparison key for quotes and passages: `normalize_for_match` (Unicode, digits,
    quotes, dashes, case, thousands separators, percent, whitespace, hyphenation) with
    Markdown table syntax (cell bars, header rules) treated as whitespace."""
    text = _TABLE_RULE_RE.sub(" ", text.replace("|", " | "))
    text = normalize_for_match(text).replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()


def _clean_quote(quote: str) -> str:
    return quote.strip().strip(_EDGE_PUNCT).strip()


def quote_in_passage(quote: str, passage: str) -> bool:
    """True if `quote` is a contiguous span of `passage` after normalisation. Quotes shorter
    than 8 characters or 2 tokens are refused: they would match almost any passage."""
    q = match_key(_clean_quote(quote))
    if len(q) < MIN_QUOTE_CHARS or len(q.split()) < MIN_QUOTE_TOKENS:
        return False
    # Whole words and whole numbers only: "7 percent" must not match inside "3.7 percent".
    pattern = rf"(?<![\w.]){re.escape(q)}(?!\w|\.\d)"
    return re.search(pattern, match_key(passage)) is not None


@dataclass(frozen=True)
class ValidCitation:
    index: int  # 0-based position of the passage in the list given to the verifier
    chunk: ChunkRecord
    quote: str  # as the model wrote it, edge quotes and spaces trimmed


def check_citations(
    output: VerifierOutput, chunks: Sequence[ChunkRecord]
) -> tuple[list[ValidCitation], int]:
    """Valid citations in the model's order (duplicates once) and the number dropped."""
    labels = {passage_label(i): i - 1 for i in range(1, len(chunks) + 1)}
    valid: list[ValidCitation] = []
    seen: set[tuple[int, str]] = set()
    dropped = 0
    for cit in output.citations:
        idx = labels.get(cit.passage.strip().upper())
        quote = _clean_quote(cit.quote)
        if idx is None or not quote_in_passage(quote, chunks[idx].text):
            dropped += 1
            continue
        key = (idx, match_key(quote))
        if key in seen:
            continue
        seen.add(key)
        valid.append(ValidCitation(index=idx, chunk=chunks[idx], quote=quote))
    return valid, dropped


# --------------------------------------------------------------------------- numbers


def key_quantity(claim: ExtractedClaim) -> Quantity | None:
    """The claim's key number: the quantity in `claim_en` whose number is the extracted
    `value`, else the first quantity in `claim_en`, else the extracted value itself."""
    found = find_quantities(claim.claim_en)
    if claim.value:
        wanted = find_quantities(claim.value)
        if wanted:
            for q in found:
                if abs(q.number) == abs(wanted[0].number):
                    return q
            return wanted[0]
    return found[0] if found else None


def _same_number(target: Quantity, source: Quantity) -> bool:
    """`source` matches `target` at the target's precision, signs ignored (direction is the
    verifier's job), either as written or after scaling. The bare numbers are compared only
    when the scale words do not conflict: a table cell has no scale word (it is in the header),
    but "1.2 billion" is not "1.2 trillion"."""
    raw = target.raw.strip("()").lstrip("+-−–")
    t_factor = scale_factor(target.scale) if target.scale else None
    s_factor = scale_factor(source.scale) if source.scale else None
    scales_agree = t_factor is None or s_factor is None or t_factor == s_factor
    if scales_agree and matches_at_precision(raw, abs(source.number)):
        return True
    if target.scale or source.scale:
        factor = scale_factor(target.scale) if target.scale else 1.0
        if factor and matches_at_precision(raw, abs(source.value) / factor):
            return True
    return False


def number_in_quotes(target: Quantity, quotes: Sequence[str]) -> bool:
    return any(_same_number(target, q) for quote in quotes for q in find_quantities(quote))


# --------------------------------------------------------------------------- verdict


@dataclass
class CheckedVerdict:
    verdict: Verdict
    reasoning: str
    citations: list[ValidCitation]
    flags: list[str] = field(default_factory=list)
    dropped: int = 0
    model_verdict: Verdict | None = None


def trim_words(text: str, limit: int = REASONING_MAX_WORDS) -> str:
    words = text.split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(",;:") + " …"


def apply_guards(
    output: VerifierOutput, chunks: Sequence[ChunkRecord], claim: ExtractedClaim
) -> CheckedVerdict:
    """Apply the code guards to one verifier answer."""
    valid, dropped = check_citations(output, chunks)
    verdict = output.verdict
    flags: list[str] = []
    if verdict is not Verdict.NEI and not valid:
        verdict = Verdict.NEI
        flags.append(FLAG_DOWNGRADED)

    quotes = [c.quote for c in valid]
    if verdict is Verdict.SUPPORTED:
        target = key_quantity(claim)
        if target is not None and not number_in_quotes(target, quotes):
            flags.append(FLAG_NUMERIC)
    elif verdict is Verdict.REFUTED and key_quantity(claim) is not None:
        evidence = find_quantities(output.evidence_value or "")
        if not evidence or not number_in_quotes(evidence[0], quotes):
            flags.append(FLAG_NUMERIC)

    return CheckedVerdict(
        verdict=verdict,
        reasoning=trim_words(output.reasoning),
        citations=valid,
        flags=flags,
        dropped=dropped,
        model_verdict=output.verdict,
    )
