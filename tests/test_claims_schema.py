"""Claim records, ids, de-duplication key, files and the review CSV round trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claimcheck.claims.io import (
    CSV_COLUMNS,
    ClaimsFileError,
    CsvRow,
    claim_to_row,
    merge_review,
    parse_decision,
    read_claims,
    read_csv,
    read_drafts,
    read_splits,
    row_to_claim,
    write_claims,
    write_csv,
    write_drafts,
    write_splits,
)
from claimcheck.claims.schema import (
    CLAIM_ID_RE,
    FACT_ID_RE,
    PROPOSAL_COLUMNS,
    DraftRecord,
    DraftStatus,
    dedup_key,
    evidence_match_text,
    format_id,
    next_number,
    row_leaf,
    topic_for,
)
from claimcheck.schemas import EvidenceRef, Split
from tests.claims_fixtures import TABLE_EV, TEXT_EV, claim, manifest_entry


def test_ids() -> None:
    assert format_id("C", 7) == "C0007"
    assert next_number(["C0001", "C0012", "bad", None], CLAIM_ID_RE) == 13
    assert next_number([], FACT_ID_RE) == 1


def test_dedup_key_normalises_numbers_units_and_years() -> None:
    a = dedup_key("Inflation was 4.5% in FY2025.")
    assert a == dedup_key("inflation was 4.5 per cent in 2024-25")
    assert a == dedup_key("Inflation was 4.5 percent in FY25")
    assert dedup_key("Exports were Rs 1,234.5 billion") == dedup_key(
        "Exports were Rs 1234.5 billion"
    )
    assert a != dedup_key("Inflation was 4.6% in FY2025.")


def test_row_leaf_and_match_text() -> None:
    assert row_leaf("A AGRICULTURE / 1. Crops") == "1. Crops"
    assert evidence_match_text(EvidenceRef.model_validate(TABLE_EV)) == "Wheat 28,975"
    assert evidence_match_text(EvidenceRef.model_validate(TEXT_EV)) == TEXT_EV["quote"]


def test_topic_for() -> None:
    assert topic_for(manifest_entry("pes2526_ch07", 7)) == "inflation"
    assert topic_for(manifest_entry("pes2526_overview", None)) is None


def test_proposal_columns_are_the_eight_of_the_proposal() -> None:
    assert PROPOSAL_COLUMNS == (
        "claim_id",
        "claim_text",
        "label",
        "evidence_ids",
        "source",
        "page",
        "topic",
        "claim_type",
    )


# --------------------------------------------------------------------------- files


def test_claims_jsonl_round_trip_keeps_urdu(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"
    ur = claim(
        2,
        language="ur",
        parallel_of="C0001",
        fact_id="F0001",
        claim_text="مالی سال 2026 میں مہنگائی 1.5 فیصد رہی۔",
    )
    write_claims(path, [ur, claim(1)])
    text = path.read_text(encoding="utf-8")
    assert "مہنگائی" in text  # not \u-escaped
    back = read_claims(path)
    assert [c.claim_id for c in back] == ["C0001", "C0002"]
    assert back[1] == ur


def test_read_claims_reports_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"
    path.write_text(json.dumps(claim(1).model_dump(mode="json")) + "\n{\n[1]\n")
    with pytest.raises(ClaimsFileError, match="line 2.*line 3"):
        read_claims(path)


def test_drafts_and_splits_round_trip(tmp_path: Path) -> None:
    d = DraftRecord(
        claim=claim(1),
        source_chunk_id="0123456789abcdef",
        source_kind="text",
        checks=["older_edition"],
    )
    write_drafts(tmp_path / "drafts.jsonl", [d])
    assert read_drafts(tmp_path / "drafts.jsonl") == [d]
    write_splits(tmp_path / "splits.json", {"F0002": Split.TEST, "F0001": Split.TRAIN})
    assert read_splits(tmp_path / "splits.json") == {"F0001": Split.TRAIN, "F0002": Split.TEST}
    assert list(json.loads((tmp_path / "splits.json").read_text())) == ["F0001", "F0002"]
    assert read_splits(tmp_path / "missing.json") == {}


# --------------------------------------------------------------------------- CSV


def test_csv_row_round_trip_with_two_evidence_items() -> None:
    c = claim(
        3,
        claim_type="tabular",
        topic="agriculture",
        evidence=[TABLE_EV, TEXT_EV],
        notes="two sources",
        annotator="AR",
        origin="human",
    )
    row = claim_to_row(c, decision="accept")
    assert set(row) == set(CSV_COLUMNS)
    assert row["ev_table_id"] == TABLE_EV["table_id"] and json.loads(row["ev_more"])
    assert row_to_claim(row) == c


def test_csv_file_round_trip(tmp_path: Path) -> None:
    claims = [
        claim(1),
        claim(2, label="NEI", claim_text="Tea imports doubled in FY2026."),
        claim(
            4,
            language="roman_ur",
            parallel_of="C0001",
            fact_id="F0001",
            claim_text="FY2026 mein mehngai 1.5 percent rahi.",
        ),
    ]
    path = tmp_path / "review.csv"
    write_csv(path, [claim_to_row(c) for c in claims])
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")  # BOM for spreadsheets
    rows = read_csv(path)
    assert [r.line for r in rows] == [2, 3, 4]
    assert [row_to_claim(r.data) for r in rows] == claims


def test_spreadsheet_quirks_are_tolerated() -> None:
    row = claim_to_row(claim(1))
    row.update(ev_page="2.0", label="supported", language="EN")
    c = row_to_claim(row)
    assert c.evidence[0].page == 2 and c.label.value == "SUPPORTED"


def test_decision_values() -> None:
    assert parse_decision(" Accept ").value == "accept"
    assert parse_decision("").value == ""
    with pytest.raises(ValueError):
        parse_decision("maybe")


def _rows(*rows: dict[str, str]) -> list[CsvRow]:
    return [CsvRow(i + 2, r) for i, r in enumerate(rows)]


def test_merge_review_round_trip_changes_nothing() -> None:
    claims = [claim(1), claim(2, label="REFUTED", claim_text="Inflation was 9 percent in FY26.")]
    blank = merge_review(_rows(*(claim_to_row(c) for c in claims)), claims, [])
    assert blank.claims == claims and blank.counts["unchanged"] == 2
    accepted = merge_review(
        _rows(*(claim_to_row(c, decision="accept") for c in claims)), claims, []
    )
    assert accepted.claims == claims and not accepted.problems


def test_merge_review_accept_edit_reject_and_new() -> None:
    drafts = [
        DraftRecord(
            claim=claim(1, origin="llm_draft:m:claim-draft-v1"),
            source_chunk_id="a" * 16,
            source_kind="text",
        ),
        DraftRecord(
            claim=claim(
                2,
                origin="llm_draft:m:claim-draft-v1",
                claim_text="Inflation fell to 2 percent in FY2026 in Pakistan.",
            ),
            source_chunk_id="b" * 16,
            source_kind="text",
        ),
        DraftRecord(
            claim=claim(3, claim_text="Inflation rose to 30 percent in FY2026 overall."),
            source_chunk_id="c" * 16,
            source_kind="text",
        ),
    ]
    edited = claim_to_row(drafts[1].claim, decision="edit")
    edited["claim_text"] = "Inflation fell to 2.1 percent in FY2026 in Pakistan."
    new_nei = dict.fromkeys(CSV_COLUMNS, "")
    new_nei.update(
        decision="accept",
        claim_text="Pakistan's tea imports doubled in FY2026.",
        label="NEI",
        language="en",
        claim_type="textual",
        reasoning_type="trend",
        topic="trade",
    )
    new_ur = dict.fromkeys(CSV_COLUMNS, "")
    new_ur.update(
        decision="accept",
        claim_text="مالی سال 2026 میں مہنگائی 1.5 فیصد رہی۔",
        label="SUPPORTED",
        language="ur",
        parallel_of="C0001",
        claim_type="textual",
        reasoning_type="numeric",
        topic="inflation",
        ev_doc_id=TEXT_EV["doc_id"],
        ev_page="2",
        ev_quote=TEXT_EV["quote"],
    )
    rows = _rows(
        claim_to_row(drafts[0].claim, decision="accept"),
        edited,
        claim_to_row(drafts[2].claim, decision="reject"),
        new_nei,
        new_ur,
        dict(claim_to_row(drafts[0].claim), decision="perhaps"),
    )
    result = merge_review(rows, [], drafts, annotator="AR")
    assert result.problems == [(7, "decision 'perhaps': use accept, edit, reject or leave blank")]
    by_id = {c.claim_id: c for c in result.claims}
    assert set(by_id) == {"C0001", "C0002", "C0004", "C0005"}
    assert by_id["C0001"].annotator == "AR"
    assert by_id["C0002"].claim_text.startswith("Inflation fell to 2.1")
    assert by_id["C0002"].origin == "llm_draft_edited:m:claim-draft-v1"
    assert by_id["C0004"].origin == "human" and by_id["C0004"].fact_id == "F0004"
    assert by_id["C0005"].fact_id == "F0001"  # parallel_of joins the original's fact
    status = {d.claim.claim_id: d.status for d in result.drafts}
    assert status == {
        "C0001": DraftStatus.ACCEPTED,
        "C0002": DraftStatus.ACCEPTED,
        "C0003": DraftStatus.REJECTED,
    }
    assert result.counts == {"accepted": 1, "edited": 1, "rejected": 1, "added": 2, "unchanged": 0}


def test_merge_review_reports_invalid_rows() -> None:
    row = claim_to_row(claim(1), decision="accept")
    row["label"] = "TRUE"
    result = merge_review(_rows(row), [], [])
    assert result.claims == [] and result.problems and result.problems[0][0] == 2


def test_new_rows_get_their_ids_written_back(tmp_path: Path) -> None:
    from claimcheck.claims.io import fill_assigned_ids

    new = dict.fromkeys(CSV_COLUMNS, "")
    new.update(
        decision="accept",
        claim_text="Pakistan's tea imports doubled in FY2026.",
        label="NEI",
        language="en",
        claim_type="textual",
        reasoning_type="trend",
        topic="trade",
    )
    path = tmp_path / "review.csv"
    write_csv(path, [claim_to_row(claim(1), decision="accept"), new])
    first = merge_review(read_csv(path), [], [])
    assert first.assigned == {3: ("C0002", "F0002", "human")}
    fill_assigned_ids(path, first.assigned)
    rows = read_csv(path)
    assert (rows[1].data["claim_id"], rows[1].data["fact_id"], rows[1].data["origin"]) == (
        "C0002",
        "F0002",
        "human",
    )
    again = merge_review(rows, first.claims, [])
    assert again.counts["added"] == 0 and again.assigned == {}
    assert sorted(again.claims, key=lambda c: c.claim_id) == sorted(
        first.claims, key=lambda c: c.claim_id
    )
