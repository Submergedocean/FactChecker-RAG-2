"""Claims validator (row rules, corpus rules, evidence resolution, gates)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claimcheck.claims.io import JsonLine
from claimcheck.claims.validate import (
    LoadedClaim,
    check_claims,
    gate_issues,
    parse_lines,
    resolve_evidence,
    validate_file,
)
from claimcheck.config import ClaimsSettings
from claimcheck.schemas import EvidenceRef, Split
from tests.claims_fixtures import TABLE_EV, TEXT_EV, claim, make_corpus


def codes(issues, severity: str = "error") -> list[str]:
    return [i.code for i in issues if i.severity == severity]


def test_good_claims_pass_with_corpus() -> None:
    claims = [
        claim(1),
        claim(2, label="REFUTED", claim_text="Average CPI inflation was 9.9 percent in FY2026."),
        claim(
            3,
            claim_type="tabular",
            topic="agriculture",
            evidence=[TABLE_EV],
            claim_text="Wheat production was 28,975 thousand tonnes in 2025-26.",
        ),
        claim(4, label="NEI", claim_text="Pakistan's tea imports doubled in FY2026."),
        claim(
            5,
            fact_id="F0001",
            language="ur",
            parallel_of="C0001",
            claim_text="مالی سال 2026 میں اوسط مہنگائی 1.5 فیصد رہی۔",
        ),
    ]
    issues = check_claims(claims, corpus=make_corpus())
    assert codes(issues) == []


def test_bad_label_is_rejected_with_its_line() -> None:
    good = claim(1).model_dump(mode="json")
    bad = dict(good, claim_id="C0002", label="TRUE", claim_text="Another claim of ten chars.")
    loaded, issues = parse_lines([JsonLine(1, good), JsonLine(2, bad)])
    assert len(loaded) == 1
    assert [(i.code, i.line, i.claim_id) for i in issues] == [("bad_label", 2, "C0002")]


def test_non_json_line_reported() -> None:
    _, issues = parse_lines([JsonLine(3, None, "not JSON: Expecting value")])
    assert codes(issues) == ["not_json"] and issues[0].line == 3


@pytest.mark.parametrize("label", ["SUPPORTED", "REFUTED"])
def test_missing_evidence_on_non_nei(label: str) -> None:
    assert "missing_evidence" in codes(check_claims([claim(1, label=label, evidence=[])]))


def test_nei_with_evidence() -> None:
    assert "nei_with_evidence" in codes(check_claims([claim(1, label="NEI", evidence=[TEXT_EV])]))


@pytest.mark.parametrize(
    "variant",
    [
        "Average CPI inflation was 1.5 percent in FY2026 in Pakistan.",  # identical
        "average cpi inflation was 1.5 % in 2025-26 in pakistan",  # %, FY form, case
        "Average CPI inflation was 1.5 per cent in FY26 in Pakistan!",  # per cent, FY26
    ],
)
def test_duplicate_text_after_normalisation(variant: str) -> None:
    issues = check_claims([claim(1), claim(2, claim_text=variant)])
    dup = [i for i in issues if i.code == "duplicate_text"]
    assert len(dup) == 1 and dup[0].claim_id == "C0002" and "C0001" in dup[0].message


def test_different_numbers_are_not_duplicates() -> None:
    other = "Average CPI inflation was 1.6 percent in FY2026 in Pakistan."
    assert "duplicate_text" not in codes(check_claims([claim(1), claim(2, claim_text=other)]))


def test_duplicate_id() -> None:
    other = claim(1, claim_text="Something else entirely about wheat output.")
    assert "duplicate_id" in codes(check_claims([claim(1), other]))


def test_orphan_parallel_of() -> None:
    ur = claim(
        2,
        fact_id="F0001",
        language="ur",
        parallel_of="C0099",
        claim_text="مالی سال 2026 میں مہنگائی 1.5 فیصد رہی۔",
    )
    assert "orphan_parallel" in codes(check_claims([claim(1), ur]))


def test_parallel_rules() -> None:
    base = claim(1)
    roman = claim(
        2,
        fact_id="F0001",
        language="roman_ur",
        parallel_of="C0001",
        claim_text="FY2026 mein mehngai 1.5 percent rahi.",
    )
    ur_of_roman = claim(
        3,
        fact_id="F0001",
        language="ur",
        parallel_of="C0002",
        claim_text="مالی سال 2026 میں مہنگائی 1.5 فیصد رہی۔",
    )
    other_fact = claim(
        4,
        fact_id="F0009",
        language="ur",
        parallel_of="C0001",
        claim_text="مہنگائی مالی سال 2026 میں 1.5 فیصد تھی۔",
    )
    wrong_label = claim(
        5,
        fact_id="F0001",
        language="roman_ur",
        parallel_of="C0001",
        label="REFUTED",
        claim_text="FY2026 mein mehngai 9 percent rahi.",
    )
    no_parallel = claim(6, language="ur", claim_text="مہنگائی کم ہو گئی، مالی سال 2026۔")
    en_with_parallel = claim(7, parallel_of="C0001", claim_text="Inflation eased in FY2026 a lot.")
    got = codes(
        check_claims(
            [base, roman, ur_of_roman, other_fact, wrong_label, no_parallel, en_with_parallel]
        )
    )
    for code in (
        "parallel_not_english",
        "parallel_fact_mismatch",
        "parallel_label_mismatch",
        "missing_parallel",
        "parallel_on_english",
    ):
        assert code in got, code


def test_tabular_claim_needs_a_table_cell() -> None:
    assert "tabular_without_table" in codes(check_claims([claim(1, claim_type="tabular")]))


def test_table_evidence_needs_row_and_column() -> None:
    ev = dict(TABLE_EV, col_label=None)
    issues = check_claims([claim(1, claim_type="tabular", evidence=[ev])])
    assert "table_evidence_incomplete" in codes(issues)


def test_short_text_quote_and_unknown_topic_and_length() -> None:
    ev = dict(TEXT_EV, quote="6.2")
    issues = check_claims([claim(1, evidence=[ev], topic="weather"), claim(2, claim_text="Short.")])
    got = codes(issues)
    assert {"short_quote", "unknown_topic", "claim_length"} <= set(got)


# --------------------------------------------------------------------------- corpus


def test_quote_must_be_on_the_cited_page() -> None:
    ev = dict(TEXT_EV, page=3)
    assert "quote_not_found" in codes(check_claims([claim(1, evidence=[ev])], corpus=make_corpus()))


def test_invented_quote_is_not_found() -> None:
    ev = dict(TEXT_EV, quote="average CPI inflation stood at 5.2 percent")
    assert "quote_not_found" in codes(check_claims([claim(1, evidence=[ev])], corpus=make_corpus()))


def test_unknown_doc_and_page() -> None:
    corpus = make_corpus()
    a = claim(1, evidence=[dict(TEXT_EV, doc_id="pes9999_ch01")])
    b = claim(
        2, claim_text="Average CPI inflation was 7 percent.", evidence=[dict(TEXT_EV, page=99)]
    )
    got = codes(check_claims([a, b], corpus=corpus))
    assert "unknown_doc" in got and "bad_page" in got


def test_table_cell_lookup() -> None:
    corpus = make_corpus()
    ok = claim(1, claim_type="tabular", topic="agriculture", evidence=[TABLE_EV])
    by_leaf = claim(
        2,
        claim_type="tabular",
        topic="agriculture",
        claim_text="Cotton rose 27.6 %.",
        evidence=[dict(TABLE_EV, row_label="A MAJOR / Cotton", col_label="% Change", quote="27.6")],
    )
    assert codes(check_claims([ok, by_leaf], corpus=corpus)) == []

    wrong_cell = claim(
        3, claim_type="tabular", topic="agriculture", evidence=[dict(TABLE_EV, quote="31,806")]
    )
    ambiguous = claim(
        4,
        claim_type="tabular",
        topic="agriculture",
        claim_text="Cotton was 130.",
        evidence=[dict(TABLE_EV, row_label="Cotton", quote="130")],
    )
    no_col = claim(
        5,
        claim_type="tabular",
        topic="agriculture",
        claim_text="Rice was 9,323 t.",
        evidence=[dict(TABLE_EV, row_label="Rice", col_label="2026")],
    )
    wrong_page = claim(
        6,
        claim_type="tabular",
        topic="agriculture",
        claim_text="Rice: 9,323.",
        evidence=[dict(TABLE_EV, page=6, row_label="Rice", quote="9,323")],
    )
    issues = check_claims([wrong_cell, ambiguous, no_col, wrong_page], corpus=corpus)
    by_claim = {i.claim_id: i.code for i in issues if i.severity == "error"}
    assert by_claim == {
        "C0003": "cell_mismatch",
        "C0004": "cell_not_found",
        "C0005": "cell_not_found",
        "C0006": "table_mismatch",
    }


def test_cell_number_formats_match() -> None:
    corpus = make_corpus()
    ev = dict(TABLE_EV, quote="28975")  # no thousands separator
    assert (
        codes(
            check_claims(
                [claim(1, claim_type="tabular", topic="agriculture", evidence=[ev])], corpus=corpus
            )
        )
        == []
    )


def test_evidence_resolves_in_both_variants() -> None:
    corpus = make_corpus()
    text = EvidenceRef.model_validate(TEXT_EV)
    table = EvidenceRef.model_validate(TABLE_EV)
    for variant in ("table_aware", "plain_text"):
        assert resolve_evidence(text, corpus, variant), variant
        assert resolve_evidence(table, corpus, variant), variant
    # the text quote is on page 2: the page-3 chunk of table_aware does not hold it
    assert resolve_evidence(text, corpus, "table_aware") == [
        corpus.chunks("table_aware")[0].chunk_id
    ]


def test_unresolved_in_one_variant_is_an_error() -> None:
    corpus = make_corpus(plain_has_table=False)
    issues = check_claims(
        [claim(1, claim_type="tabular", topic="agriculture", evidence=[TABLE_EV])], corpus=corpus
    )
    errs = [i for i in issues if i.code == "unresolved"]
    assert len(errs) == 1 and "plain_text" in errs[0].message


def test_resolution_needs_every_number_and_most_tokens() -> None:
    corpus = make_corpus()
    # tokens all present but the number is not: no chunk
    ev = EvidenceRef(
        doc_id="pes2526_ch07", page=2, quote="average CPI inflation stood at 6.3 percent"
    )
    assert resolve_evidence(ev, corpus, "table_aware") == []
    # number present, but under 60 % of the words: no chunk
    ev = EvidenceRef(doc_id="pes2526_ch07", page=2, quote="tea coffee sugar salt rice 6.2")
    assert resolve_evidence(ev, corpus, "table_aware") == []


def test_relaxed_resolution_ignores_the_document() -> None:
    corpus = make_corpus()
    ev = EvidenceRef.model_validate(dict(TEXT_EV, doc_id="pes2526_overview"))
    assert resolve_evidence(ev, corpus, "table_aware") == []
    assert resolve_evidence(ev, corpus, "table_aware", relaxed=True)


# --------------------------------------------------------------------------- gates


def _mix(n_s: int, n_r: int, n_nei: int, n_tab: int, n_ur: int) -> list:
    out, i = [], 0
    for label, count in (("SUPPORTED", n_s), ("REFUTED", n_r), ("NEI", n_nei)):
        for _ in range(count):
            i += 1
            kw: dict = {"label": label, "claim_text": f"Claim number {i} about the economy."}
            if i <= n_tab and label != "NEI":
                kw.update(claim_type="tabular", evidence=[TABLE_EV])
            if i <= n_ur:
                kw.update(language="ur", parallel_of="C9999")
            out.append(claim(i, **kw))
    return out


def test_final_gate_passes_on_the_target_mix() -> None:
    claims = _mix(240, 240, 120, 300, 120)
    splits = {c.fact_id: Split.TRAIN for c in claims}
    assert gate_issues(claims, ClaimsSettings(), gate="final", splits=splits) == []


def test_final_gate_failures() -> None:
    claims = _mix(300, 200, 100, 100, 20)
    got = {i.code for i in gate_issues(claims, ClaimsSettings(), gate="final", splits={})}
    assert got == {"label_mix", "tabular_share", "language_mix", "unsplit_facts"}
    small = _mix(10, 10, 5, 12, 5)
    assert "total" in {i.code for i in gate_issues(small, ClaimsSettings(), gate="final")}


def test_pilot_gate() -> None:
    assert gate_issues(_mix(20, 20, 0, 0, 0), ClaimsSettings(), gate="pilot") == []
    issues = gate_issues(_mix(10, 10, 5, 0, 0), ClaimsSettings(), gate="pilot")
    assert [i.code for i in issues] == ["pilot_size"]


def test_validate_file(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"
    rows = [claim(1).model_dump(mode="json"), {"claim_id": "C0002", "label": "SUPPORTED"}]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n\nnot json\n")
    report = validate_file(path)
    assert not report.ok
    assert [(i.code, i.line) for i in report.errors] == [("schema", 2), ("not_json", 4)]
    assert len(report.claims) == 1


def test_loaded_claims_carry_lines() -> None:
    issues = check_claims([LoadedClaim(claim(1, evidence=[]), line=7)])
    assert issues[0].line == 7 and str(issues[0]).startswith("ERROR")


def test_first_label_column_is_enough_when_unique() -> None:
    from claimcheck.claims.validate import TableView

    t = TableView(
        table_id="pes2425_ch03:p13:t1",
        doc_id="pes2425_ch03",
        page=13,
        header=["Minerals", "Unit", "2023-24"],
        labels=["Coal 000 M.T", "Natural Gas 000 MMCFT", "2025 Flood", "2025 Storm"],
        cells=[
            ["Coal", "000 M.T", "19,260"],
            ["Natural Gas", "000 MMCFT", "1,141"],
            ["2025", "Flood", "1"],
            ["2025", "Storm", "2"],
        ],
        lead=2,
    )
    assert t.find_cell("Natural Gas", "2023-24") == ("1,141", None)
    assert t.find_cell("Natural Gas 000 MMCFT", "2023-24") == ("1,141", None)
    assert "ambiguous" in (t.find_cell("2025", "2023-24")[1] or "")
