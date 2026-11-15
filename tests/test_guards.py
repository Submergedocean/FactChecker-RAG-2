"""Code guards around the LLM: input guard, citation check, verdict downgrade,
numeric warning. No model, no data."""

from __future__ import annotations

import pytest

from claimcheck.llm.prompts.extractor import ExtractedClaim
from claimcheck.llm.prompts.verifier import VerifierOutput
from claimcheck.pipeline.guards import (
    FLAG_DOWNGRADED,
    FLAG_NUMERIC,
    apply_guards,
    check_citations,
    guard_input,
    key_quantity,
    quote_in_passage,
    trim_words,
)
from claimcheck.schemas import ChunkKind, ChunkRecord, Verdict
from claimcheck.textnorm import InputGuardError

GDP_TEXT = (
    "On the back of these positive developments, real GDP growth is recorded at 3.7 percent "
    "in FY 2026. Remittances reached US$ 38,312 million.\nAgricul-\nture grew 2.89 percent."
)
TABLE_TEXT = (
    "Table 7.3: Core Inflation – Urban\nUnits: Percent\n"
    "| Months | Mar | Apr | Jul-Apr |\n| --- | --- | --- | --- |\n"
    "| 2024-25 | 8.2 | 7.4 | 8.8 |\n| 2025-26 | 7.4 | 8.0 | 7.2 |"
)


def chunk(text: str, *, n: int = 0, kind: ChunkKind = ChunkKind.TEXT, **kw: object) -> ChunkRecord:
    base: dict[str, object] = {
        "chunk_id": f"{n:016x}",
        "doc_id": "pes2526_ch01",
        "edition": "2025-26",
        "chapter_no": 1,
        "chapter_title": "Growth and Investment",
        "page_start": 2,
        "page_end": 2,
        "kind": kind,
        "table_id": None,
        "text": text,
        "embed_text": text,
        "bm25_tokens": [],
        "n_words": len(text.split()),
        "n_tokens": len(text.split()),
    }
    base.update(kw)
    return ChunkRecord.model_validate(base)


def claim(text: str, value: str | None = None) -> ExtractedClaim:
    return ExtractedClaim(claim_en=text, value=value)


def output(verdict: str, cits: list[tuple[str, str]], **kw: object) -> VerifierOutput:
    return VerifierOutput.model_validate(
        {
            "verdict": verdict,
            "reasoning": "r",
            "citations": [{"passage": p, "quote": q} for p, q in cits],
            **kw,
        }
    )


# ------------------------------------------------------------------------ input guard


@pytest.mark.parametrize(
    ("text", "code"),
    [("", "empty"), ("   \n\t ", "empty"), ("GDP 3%", "too_short"), ("x" * 501, "too_long")],
)
def test_input_guard_rejects_with_reason(text: str, code: str) -> None:
    with pytest.raises(InputGuardError) as err:
        guard_input(text)
    assert err.value.code == code


def test_input_guard_trims_and_maps_urdu_digits() -> None:
    assert guard_input("  مہنگائی ۷٫۳ فیصد رہی  ") == "مہنگائی 7.3 فیصد رہی"
    assert guard_input("  GDP grew 3.7% in FY26 \n") == "GDP grew 3.7% in FY26"


# ---------------------------------------------------------------------------- quotes


@pytest.mark.parametrize(
    "quote",
    [
        "real GDP growth is recorded at 3.7 percent in FY 2026",
        "Real GDP growth is recorded at 3.7 percent in FY 2026.",  # case, end punctuation
        '"real  GDP growth is recorded\nat 3.7%  in FY 2026"',  # quotes, spaces, % sign
        "Remittances reached US$ 38312 million",  # thousands separator
        "Agriculture grew 2.89 percent",  # hyphenation across a line break
    ],
)
def test_quote_found_after_normalisation(quote: str) -> None:
    assert quote_in_passage(quote, GDP_TEXT)


@pytest.mark.parametrize(
    "quote",
    [
        "real GDP growth is recorded at 3.8 percent in FY 2026",  # invented number
        "real GDP growth was recorded at 3.7 percent in FY 2026",  # added word
        "7 percent in FY 2026",  # part of a number ("3.7")
        "growth is recorded at 3",  # cut inside a number
        "3.7",  # too short: would match almost anything
        "",
    ],
)
def test_quote_not_found(quote: str) -> None:
    assert not quote_in_passage(quote, GDP_TEXT)


def test_table_row_quotes_match_with_or_without_bars() -> None:
    assert quote_in_passage("| 2025-26 | 7.4 | 8.0 | 7.2 |", TABLE_TEXT)
    assert quote_in_passage("2025-26 | 7.4 | 8.0 | 7.2", TABLE_TEXT)
    assert quote_in_passage("2025-26 7.4 8.0 7.2", TABLE_TEXT)
    assert not quote_in_passage("2025-26 | 7.4 | 8.1 | 7.2", TABLE_TEXT)


def test_check_citations_drops_bad_labels_and_invented_quotes() -> None:
    chunks = [chunk(GDP_TEXT, n=1), chunk(TABLE_TEXT, n=2, kind=ChunkKind.TABLE)]
    out = output(
        "SUPPORTED",
        [
            ("P1", "real GDP growth is recorded at 3.7 percent in FY 2026"),
            ("p1", "real GDP growth is recorded at 3.7 percent in FY 2026."),  # duplicate
            ("P2", "real GDP growth is recorded at 3.7 percent in FY 2026"),  # wrong passage
            ("P3", "2025-26 | 7.4 | 8.0 | 7.2"),  # no such passage
            ("P2", "| 2025-26 | 7.4 | 8.0 | 7.2 |"),
        ],
    )
    valid, dropped = check_citations(out, chunks)
    assert [(c.index, c.chunk.chunk_id) for c in valid] == [
        (0, chunks[0].chunk_id),
        (1, chunks[1].chunk_id),
    ]
    assert dropped == 2


# ------------------------------------------------------------------------- verdicts


def test_invented_quote_is_dropped_and_verdict_becomes_nei() -> None:
    out = output("SUPPORTED", [("P1", "real GDP growth is recorded at 4.5 percent in FY 2026")])
    checked = apply_guards(out, [chunk(GDP_TEXT)], claim("GDP grew 4.5% in FY2026", "4.5"))
    assert checked.verdict is Verdict.NEI
    assert checked.model_verdict is Verdict.SUPPORTED
    assert checked.flags == [FLAG_DOWNGRADED]
    assert checked.citations == [] and checked.dropped == 1


def test_refuted_without_citation_becomes_nei_but_nei_is_untouched() -> None:
    refuted = apply_guards(output("REFUTED", []), [chunk(GDP_TEXT)], claim("GDP grew 5%"))
    assert refuted.verdict is Verdict.NEI and refuted.flags == [FLAG_DOWNGRADED]
    nei = apply_guards(output("NEI", []), [chunk(GDP_TEXT)], claim("GDP grew 5%"))
    assert nei.verdict is Verdict.NEI and nei.flags == []


def test_supported_with_its_number_in_the_quote_has_no_flag() -> None:
    out = output("SUPPORTED", [("P1", "real GDP growth is recorded at 3.7 percent in FY 2026")])
    checked = apply_guards(out, [chunk(GDP_TEXT)], claim("Real GDP grew 3.7% in FY2026.", "3.7"))
    assert checked.verdict is Verdict.SUPPORTED and checked.flags == []


def test_numeric_warning_when_the_key_number_is_not_quoted() -> None:
    out = output("SUPPORTED", [("P1", "Agriculture grew 2.89 percent")])
    checked = apply_guards(out, [chunk(GDP_TEXT)], claim("Real GDP grew 3.7% in FY2026.", "3.7"))
    assert checked.verdict is Verdict.SUPPORTED  # flag only
    assert checked.flags == [FLAG_NUMERIC]


@pytest.mark.parametrize(
    ("claim_text", "value", "passage", "quote"),
    [
        # "about 3 percent" matches 2.68 at the claim's precision
        (
            "Inflation was about 3 percent",
            "3",
            "Inflation was 2.68 percent.",
            "Inflation was 2.68 percent",
        ),
        # scale words
        (
            "Revenue was Rs 1.2 trillion",
            "1.2",
            "Revenue was Rs 1,200 billion.",
            "Revenue was Rs 1,200 billion",
        ),
        # table cell: the unit lives in the header
        (
            "Urban core inflation was 8.0 percent in April 2026",
            "8.0",
            TABLE_TEXT,
            "2025-26 | 7.4 | 8.0 | 7.2",
        ),
        # direction words do not matter to the number check
        (
            "Output fell by 1.3 percent",
            "1.3",
            "Output changed by -1.3 percent.",
            "Output changed by -1.3 percent",
        ),
    ],
)
def test_numeric_match_rules(claim_text: str, value: str, passage: str, quote: str) -> None:
    checked = apply_guards(
        output("SUPPORTED", [("P1", quote)]), [chunk(passage)], claim(claim_text, value)
    )
    assert checked.flags == []


@pytest.mark.parametrize(
    ("claim_text", "passage"),
    [
        ("Revenue was Rs 1.2 trillion", "Revenue was Rs 1.2 billion."),
        ("Remittances were US$ 38 billion", "Remittances were US$ 38 million."),
        ("Spending was Rs 5 crore", "Spending was Rs 5 lakh."),
    ],
)
def test_numeric_warning_when_scale_words_conflict(claim_text: str, passage: str) -> None:
    quote = passage.rstrip(".")
    checked = apply_guards(
        output("SUPPORTED", [("P1", quote)]), [chunk(passage)], claim(claim_text)
    )
    assert checked.flags == [FLAG_NUMERIC]


def test_equivalent_scale_words_still_match() -> None:
    # 1 arab = 1 billion; "bn" is "billion".
    for claim_text, passage in [
        ("Exports were Rs 50 arab", "Exports were Rs 50 billion."),
        ("Exports were Rs 50 bn", "Exports were Rs 50 billion."),
    ]:
        quote = passage.rstrip(".")
        checked = apply_guards(
            output("SUPPORTED", [("P1", quote)]), [chunk(passage)], claim(claim_text)
        )
        assert checked.flags == [], claim_text


def test_numeric_warning_rounding_the_other_way() -> None:
    # "2.5 percent" does not match 2.68.
    passage = "Inflation was 2.68 percent."
    out = output("SUPPORTED", [("P1", "Inflation was 2.68 percent")])
    checked = apply_guards(out, [chunk(passage)], claim("Inflation was 2.5 percent", "2.5"))
    assert checked.flags == [FLAG_NUMERIC]


def test_refuted_checks_the_evidence_value() -> None:
    quote = "real GDP growth is recorded at 3.7 percent in FY 2026"
    ok = apply_guards(
        output("REFUTED", [("P1", quote)], evidence_value="3.7 percent"),
        [chunk(GDP_TEXT)],
        claim("Real GDP grew 4.5% in FY2026", "4.5"),
    )
    assert ok.verdict is Verdict.REFUTED and ok.flags == []
    bad = apply_guards(
        output("REFUTED", [("P1", quote)], evidence_value="2.9 percent"),
        [chunk(GDP_TEXT)],
        claim("Real GDP grew 4.5% in FY2026", "4.5"),
    )
    assert bad.verdict is Verdict.REFUTED and bad.flags == [FLAG_NUMERIC]


def test_claim_without_a_number_never_gets_numeric_warning() -> None:
    out = output("SUPPORTED", [("P1", "Agriculture grew 2.89 percent")])
    checked = apply_guards(out, [chunk(GDP_TEXT)], claim("Agriculture grew in FY2026"))
    assert checked.flags == []


def test_key_quantity_prefers_the_extracted_value() -> None:
    q = key_quantity(claim("In FY2026 wheat output rose 4.3% to 29.6 million tonnes", "29.6"))
    assert q is not None and q.number == 29.6
    q = key_quantity(claim("Wheat output rose 4.3% to 29.6 million tonnes"))
    assert q is not None and q.number == 4.3
    assert key_quantity(claim("Exports improved in FY2026")) is None


def test_reasoning_is_capped_at_120_words() -> None:
    assert trim_words("a b c", 120) == "a b c"
    long = " ".join(f"w{i}" for i in range(200))
    trimmed = trim_words(long)
    assert len(trimmed.split()) == 121 and trimmed.endswith("…")
