"""`claims draft` with the fake LLM (sampling, code checks on evidence, translations)."""

from __future__ import annotations

from pathlib import Path

from claimcheck.claims.draft import (
    DraftSource,
    IdAllocator,
    draft_source,
    read_skips,
    run_draft,
    sample_sources,
    skips_path_for,
    table_rows,
)
from claimcheck.claims.io import read_drafts
from claimcheck.config import ClaimsSettings
from claimcheck.llm.client import LLMUnavailable
from claimcheck.llm.prompts.claim_draft import PROMPT_VERSION
from claimcheck.schemas import ChunkKind, ClaimLanguage, ClaimType, Verdict
from tests.claims_fixtures import (
    INFLATION_PAGE,
    TABLE_CHUNK_TEXT,
    chunk_record,
    make_corpus,
    manifest_entry,
)
from tests.fakes import FakeLLM

TEXT_CHUNK = chunk_record(INFLATION_PAGE)
TABLE_CHUNK = chunk_record(
    TABLE_CHUNK_TEXT,
    doc_id="pes2526_ch02",
    page_start=5,
    kind=ChunkKind.TABLE,
    table_id="pes2526_ch02:p5:t1",
)
NO_TRANSLATION = ClaimsSettings(translate_share=0.0)
ALWAYS_TRANSLATE = ClaimsSettings(translate_share=1.0)

TEXT_ANSWER = {
    "usable": True,
    "reasoning_type": "numeric",
    "topic": "inflation",
    "supported": {
        "claim": "Average CPI inflation was 6.2 percent in July-April FY2026.",
        "evidence": {"quote": "average CPI inflation stood at 6.2 percent"},
    },
    "refuted": {
        "claim": "Average CPI inflation was 8.4 percent in July-April FY2026.",
        "mutation": "number",
        "evidence": {"quote": "average CPI inflation stood at 6.2 percent"},
    },
    "translations": {
        "supported": {
            "ur": "جولائی تا اپریل اوسط مہنگائی 6.2 فیصد رہی۔",
            "roman_ur": "July-April mein mehngai 6.2 percent rahi.",
        },
        "refuted": {
            "ur": "جولائی تا اپریل اوسط مہنگائی 8.4 فیصد رہی۔",
            "roman_ur": "July-April mein mehngai 8.4 percent rahi.",
        },
    },
}

TABLE_ANSWER = {
    "usable": True,
    "reasoning_type": "numeric",
    "supported": {
        "claim": "Pakistan produced 28,975 thousand tonnes of wheat in 2025-26.",
        "evidence": {"quote": 28975, "row_label": "wheat", "col_label": "2025-26 P"},
    },
    "refuted": {
        "claim": "Pakistan produced 9,323 thousand tonnes of wheat in 2025-26.",
        "mutation": "entity",
        "evidence": {"quote": "9,323", "row_label": "Rice", "col_label": "2025-26 P"},
    },
}


def _draft(source, answer, settings=NO_TRANSLATION, edition_latest="2025-26"):
    llm = FakeLLM([answer])
    entry = manifest_entry(source.chunk.doc_id, 7 if source.chunk.doc_id.endswith("07") else 2)
    ids = IdAllocator(next_claim=1, next_fact=1)
    out = draft_source(
        source, llm, make_corpus(), entry, ids, settings, latest_edition=edition_latest
    )
    return out, llm


def test_text_draft_takes_page_and_document_from_metadata() -> None:
    (records, reason, requests), llm = _draft(DraftSource(TEXT_CHUNK), TEXT_ANSWER)
    assert reason is None and requests == 1
    sup, ref = (r.claim for r in records)
    assert (sup.claim_id, sup.fact_id, ref.claim_id, ref.fact_id) == (
        "C0001",
        "F0001",
        "C0002",
        "F0001",
    )
    assert sup.label is Verdict.SUPPORTED and ref.label is Verdict.REFUTED
    assert sup.claim_type is ClaimType.TEXTUAL and sup.topic == "inflation"
    ev = sup.evidence[0]
    assert (ev.doc_id, ev.page, ev.table_id) == ("pes2526_ch07", 2, None)
    assert records[1].mutation.value == "number"
    assert sup.origin == f"llm_draft:fake-large:{PROMPT_VERSION}"
    assert all(not r.checks for r in records), [r.checks for r in records]
    # the verifier model drafts, and the passage goes in as a JSON string
    assert "Passage (JSON string)" in llm.calls[0][1]["content"]


def test_invented_quote_drops_the_draft() -> None:
    answer = dict(
        TEXT_ANSWER,
        supported={
            "claim": "Average CPI inflation was 6.2 percent in July-April FY2026.",
            "evidence": {"quote": "CPI inflation averaged 6.2 percent"},
        },
    )
    (records, reason, _), _ = _draft(DraftSource(TEXT_CHUNK), answer)
    assert records == [] and reason.startswith("supported evidence: quote not in the passage")


def test_unusable_passage_is_skipped() -> None:
    (records, reason, _), _ = _draft(
        DraftSource(TEXT_CHUNK), {"usable": False, "skip_reason": "only headings"}
    )
    assert records == [] and reason == "not usable: only headings"


def test_table_draft_uses_the_parsed_cell() -> None:
    (records, reason, _), _ = _draft(DraftSource(TABLE_CHUNK, "Wheat"), TABLE_ANSWER)
    assert reason is None
    sup, ref = (r.claim for r in records)
    ev = sup.evidence[0]
    assert (ev.table_id, ev.page, ev.row_label, ev.col_label, ev.quote) == (
        "pes2526_ch02:p5:t1",
        5,
        "Wheat",
        "2025-26 P",
        "28,975",
    )
    assert sup.claim_type is ClaimType.TABULAR and sup.topic == "agriculture"
    assert ref.evidence[0].row_label == "Rice" and records[1].mutation.value == "entity"


def test_wrong_cell_drops_the_draft_and_bad_twin_keeps_the_original() -> None:
    wrong = dict(
        TABLE_ANSWER,
        supported={
            "claim": "Pakistan produced 31,806 thousand tonnes of wheat in 2025-26.",
            "evidence": {"quote": "31,806", "row_label": "Wheat", "col_label": "2025-26 P"},
        },
    )
    (records, reason, _), _ = _draft(DraftSource(TABLE_CHUNK), wrong)
    assert records == [] and "cell is '28,975'" in reason

    bad_twin = dict(
        TABLE_ANSWER,
        refuted={
            "claim": "Pakistan produced 9,000 thousand tonnes of wheat in 2025-26.",
            "mutation": "number",
            "evidence": {"quote": "9,000", "row_label": "Barley", "col_label": "2025-26 P"},
        },
    )
    (records, reason, _), _ = _draft(DraftSource(TABLE_CHUNK), bad_twin)
    assert reason is None and [r.claim.label for r in records] == [Verdict.SUPPORTED]


def test_checks_flag_numbers_for_the_reviewer() -> None:
    answer = dict(TEXT_ANSWER)
    answer["supported"] = dict(TEXT_ANSWER["supported"], claim="Inflation was 7 percent in FY2026.")
    answer["refuted"] = dict(TEXT_ANSWER["refuted"], claim="Inflation was 6.2 percent in FY2026.")
    (records, _, _), _ = _draft(DraftSource(TEXT_CHUNK), answer, edition_latest="2026-27")
    sup_checks, ref_checks = records[0].checks, records[1].checks
    assert any(c.startswith("number_not_in_evidence") for c in sup_checks)
    assert any(c.startswith("refuted_number_matches") for c in ref_checks)
    assert any(c.startswith("older_edition") for c in sup_checks)


def test_translations_pair_with_their_originals() -> None:
    (records, _, _), _ = _draft(DraftSource(TEXT_CHUNK), TEXT_ANSWER, settings=ALWAYS_TRANSLATE)
    claims = [r.claim for r in records]
    assert len(claims) == 4
    langs = {c.language for c in claims[2:]}
    assert langs == {ClaimLanguage.UR, ClaimLanguage.ROMAN_UR}
    by_id = {c.claim_id: c for c in claims}
    for t in claims[2:]:
        original = by_id[t.parallel_of]
        assert original.language is ClaimLanguage.EN
        assert (t.fact_id, t.label, t.evidence) == (
            original.fact_id,
            original.label,
            original.evidence,
        )
    # Urdu goes to one twin and Roman Urdu to the other
    assert {by_id[t.parallel_of].label for t in claims[2:]} == {Verdict.SUPPORTED, Verdict.REFUTED}


def test_table_rows_and_sampling() -> None:
    assert table_rows(TABLE_CHUNK_TEXT) == ["Wheat", "Rice", "A MAJOR / Cotton"]
    texts = [
        chunk_record(
            INFLATION_PAGE + f" Extra sentence {i} with 2.{i} percent.", doc_id=f"pes2526_ch0{i}"
        )
        for i in range(1, 4)
    ]
    tables = [
        chunk_record(
            TABLE_CHUNK_TEXT + f"\n| Maize {i} | 1 | 2 | 3 |",
            doc_id=f"pes2526_ch0{i}",
            kind=ChunkKind.TABLE,
            table_id=f"pes2526_ch0{i}:p2:t1",
        )
        for i in range(1, 4)
    ]
    short = chunk_record("Only 3 words.", doc_id="pes2526_ch09")
    chunks = texts + tables + [short]
    got = sample_sources(chunks, 6, seed=1, used=set(), min_text_words=10)
    assert [s.chunk.kind.value for s in got] == ["table", "text"] * 3
    assert len({s.chunk.doc_id for s in got if s.chunk.kind is ChunkKind.TEXT}) == 3
    assert all(s.focus_row for s in got if s.chunk.kind is ChunkKind.TABLE)
    assert short.chunk_id not in {s.chunk.chunk_id for s in got}
    assert got == sample_sources(list(reversed(chunks)), 6, seed=1, used=set(), min_text_words=10)
    used = {got[0].chunk.chunk_id}
    again = sample_sources(chunks, 6, seed=1, used=used, min_text_words=10)
    assert got[0].chunk.chunk_id not in {s.chunk.chunk_id for s in again}
    only_text = sample_sources(chunks, 10, seed=1, used=set(), kind="text", min_text_words=10)
    assert len(only_text) == 3
    known = sample_sources(chunks, 10, seed=1, used=set(), kind="table", known_tables=set())
    assert known == []


def test_run_draft_saves_drafts_and_skips_and_continues_ids(tmp_path: Path) -> None:
    claims_path, drafts_path = tmp_path / "claims.jsonl", tmp_path / "drafts.jsonl"
    llm = FakeLLM([{"usable": False, "skip_reason": "no fact"}, TEXT_ANSWER])
    other = chunk_record(INFLATION_PAGE + " More text with 1.1 percent.", doc_id="pes2526_ch07")
    manifest = {"pes2526_ch07": manifest_entry()}
    run = run_draft(
        5,
        llm,
        make_corpus(),
        [TEXT_CHUNK, other],
        manifest,
        claims_path=claims_path,
        drafts_path=drafts_path,
        settings=ClaimsSettings(translate_share=0.0, min_text_words=5),
        latest_edition="2025-26",
        kind="text",
    )
    assert run.stopped is None and run.llm_requests == 2
    assert len(run.records) == 2 and len(run.skipped) == 1
    assert [d.claim.claim_id for d in read_drafts(drafts_path)] == ["C0001", "C0002"]
    assert list(read_skips(skips_path_for(drafts_path)).values()) == ["not usable: no fact"]
    # a second run has nothing left to sample
    again = run_draft(
        5,
        FakeLLM([]),
        make_corpus(),
        [TEXT_CHUNK, other],
        manifest,
        claims_path=claims_path,
        drafts_path=drafts_path,
        settings=ClaimsSettings(min_text_words=5),
        latest_edition="2025-26",
        kind="text",
    )
    assert again.records == [] and again.llm_requests == 0


def test_run_draft_stops_when_the_llm_is_unavailable(tmp_path: Path) -> None:
    llm = FakeLLM([LLMUnavailable([])])
    run = run_draft(
        3,
        llm,
        make_corpus(),
        [TEXT_CHUNK],
        {"pes2526_ch07": manifest_entry()},
        claims_path=tmp_path / "c.jsonl",
        drafts_path=tmp_path / "d.jsonl",
        settings=ClaimsSettings(min_text_words=5),
        latest_edition="2025-26",
    )
    assert run.stopped and run.records == []
    assert not (tmp_path / "d.jsonl").exists()


def test_translation_quota_holds_the_share() -> None:
    from claimcheck.claims.draft import TranslationQuota

    q = TranslationQuota(0.34)
    picks = [q.next() for _ in range(30)]
    assert sum(t for t, _ in picks) in (10, 11)
    urdu_first = [u for t, u in picks if t]
    assert urdu_first[:4] == [True, False, True, False]  # Urdu alternates between the twins


def test_flags_ignore_calendar_years_and_odd_hyphens() -> None:
    from claimcheck.claims.draft import changed_numbers, clean_text, numbers_match
    from claimcheck.schemas import EvidenceRef

    assert clean_text("Units in July‑March 2025‑26 ’s") == "Units in July-March 2025-26 's"
    ev = EvidenceRef(doc_id="d", page=1, quote="guarantees were Rs 3,448 billion at end March 2025")
    assert numbers_match("Guarantees were Rs 4,448 billion at end March 2025.", ev) is False
    assert numbers_match("Guarantees were Rs 3,448 billion at end March 2025.", ev) is True
    changed = changed_numbers(
        "Guarantees were Rs 3,448 billion at end March 2025.",
        "Guarantees were Rs 4,448 billion at end March 2025.",
    )
    assert [q.number for q in changed] == [4448]


def test_year_mutation_with_non_breaking_hyphens_is_not_flagged() -> None:
    answer = dict(
        TABLE_ANSWER,
        supported={
            "claim": "Pakistan produced 28,975 thousand tonnes of wheat in 2025‑26.",
            "evidence": {"quote": "28,975", "row_label": "Wheat", "col_label": "2025-26 P"},
        },
        refuted={
            "claim": "Pakistan produced 28,975 thousand tonnes of wheat in 2024‑25.",
            "mutation": "year",
            "evidence": {"quote": "31,806", "row_label": "Wheat", "col_label": "2024-25"},
        },
    )
    (records, reason, _), _ = _draft(DraftSource(TABLE_CHUNK), answer)
    assert reason is None and records[1].checks == []
    assert records[0].claim.claim_text.endswith("in 2025-26.")


# --------------------------------------------------------------------------- claim-draft-v2


def test_context_before_gives_the_text_above_the_passage() -> None:
    from claimcheck.claims.draft import context_before

    lead = "Electricity. During July-March FY 2026, total consumption stood at 83,143 GWh."
    body = (
        "The commercial sector consumption rose from 6,898 GWh to 7,044 GWh while "
        "industrial use fell."
    )
    corpus = make_corpus()
    corpus.pages[("pes2526_ch14", 4)] = (lead + "\n" + body, lead + " " + body)
    chunk = chunk_record(body, doc_id="pes2526_ch14", page_start=4)
    assert context_before(chunk, corpus) == lead
    assert (
        context_before(chunk, corpus, words=3) == "stood at 83,143 GWh."[-len("at 83,143 GWh.") :]
    )
    assert context_before(TABLE_CHUNK, corpus) == ""
    other = chunk_record(
        "Words that appear on no page at all here.", doc_id="pes2526_ch14", page_start=4
    )
    assert context_before(other, corpus) == ""


def test_partial_year_check() -> None:
    from claimcheck.claims.draft import partial_year_check
    from claimcheck.schemas import EvidenceRef

    ev = EvidenceRef(doc_id="d", page=1, quote="from 6,898 GWh to 7,044 GWh")
    lead = ["During July-March FY 2026, total consumption stood at 83,143 GWh."]
    flag = partial_year_check("The commercial sector used 7,044 GWh in FY2026.", ev, lead)
    assert flag and flag.startswith("partial_year_missing") and "July-March" in flag
    ok = "The commercial sector used 7,044 GWh during July-March FY2026."
    assert partial_year_check(ok, ev, lead) is None
    assert partial_year_check("The sector used 7,044 GWh in FY2026.", ev, [""]) is None
    table_ev = EvidenceRef(
        doc_id="d",
        page=1,
        table_id="d:p1:t1",
        quote="5",
        row_label="Wheat",
        col_label="No. of Units 2025-26 (July-March)",
    )
    assert partial_year_check("Wheat was 5 in 2025-26.", table_ev, [])
    nine = EvidenceRef(
        doc_id="d", page=1, quote="In first 9M of FY 2026, debt stood at Rs 57.6 trillion"
    )
    assert partial_year_check("Debt was Rs 57.6 trillion in FY2026.", nine, [])
    assert (
        partial_year_check(
            "Debt was Rs 57.6 trillion in the first nine months of FY2026.", nine, []
        )
        is None
    )


def test_translation_checks() -> None:
    from claimcheck.claims.draft import translation_checks
    from tests.claims_fixtures import claim

    en = claim(1, claim_text="Average CPI inflation was 6.2 percent in FY2026.")
    good = translation_checks(en, "مالی سال 2026 میں اوسط مہنگائی 6.2 فیصد رہی۔", ClaimLanguage.UR)
    assert good == ["translation: check meaning and every number"]
    bad = translation_checks(en, "فیڈ 2026 میں اوسط مہنگائی 6.3 فیصد رہی۔", ClaimLanguage.UR)
    codes = [c.split(":")[0] for c in bad]
    assert "translation_numbers_differ" in codes and "urdu_fy_transliterated" in codes
    urdu_digits = translation_checks(
        en, "مالی سال 2026 میں مہنگائی ۶.۲ فیصد رہی۔", ClaimLanguage.UR
    )
    assert len(urdu_digits) == 1
    year = translation_checks(en, "FY2025 mein mehngai 6.2 percent rahi.", ClaimLanguage.ROMAN_UR)
    assert any(c.startswith("translation_period_differs") for c in year)


def test_reasoning_rotation_asks_for_the_kind_most_below_target() -> None:
    from claimcheck.claims.draft import ReasoningRotation

    r = ReasoningRotation({"numeric": 0.5, "trend": 0.3, "comparison": 0.2})
    asked = []
    for _ in range(10):
        kind = r.wanted()
        asked.append(kind)
        r.record(kind)
    from collections import Counter

    assert Counter(asked) == {"numeric": 5, "trend": 3, "comparison": 2}
    assert ReasoningRotation({}).wanted() is None


def test_v2_messages_carry_context_caption_and_wanted_kind() -> None:
    from claimcheck.claims.draft import ReasoningRotation

    lead = "During July-March FY 2026, average CPI inflation was low."
    corpus = make_corpus()
    page = lead + " " + INFLATION_PAGE
    corpus.pages[("pes2526_ch07", 2)] = (page, page)
    llm = FakeLLM([TEXT_ANSWER])
    rotation = ReasoningRotation({"trend": 1.0})
    records, reason, _ = draft_source(
        DraftSource(TEXT_CHUNK),
        llm,
        corpus,
        manifest_entry(),
        IdAllocator(1, 1),
        NO_TRANSLATION,
        latest_edition="2025-26",
        rotation=rotation,
    )
    user = llm.calls[0][1]["content"]
    assert '"wanted_reasoning": "trend"' in user
    assert "context_before" in user and "During July-March FY 2026" in user
    assert rotation.counts == {"numeric": 1}  # the model wrote a numeric claim
    # the draft names July-April, a part-year period: no partial-year flag
    assert reason is None and not any(c.startswith("partial_year") for c in records[0].checks)

    table_llm = FakeLLM([TABLE_ANSWER])
    draft_source(
        DraftSource(TABLE_CHUNK),
        table_llm,
        make_corpus(),
        manifest_entry("pes2526_ch02", 2),
        IdAllocator(1, 1),
        NO_TRANSLATION,
        latest_edition="2025-26",
    )
    assert '"caption": "TABLE 2.1 PRODUCTION OF CROPS"' in table_llm.calls[0][1]["content"]


def test_chunks_holding_drafted_evidence_are_not_sampled_again(tmp_path: Path) -> None:
    from claimcheck.claims.draft import evidence_chunks

    claims_path, drafts_path = tmp_path / "claims.jsonl", tmp_path / "drafts.jsonl"
    manifest = {"pes2526_ch07": manifest_entry()}
    settings = ClaimsSettings(translate_share=0.0, min_text_words=5)
    run_draft(
        1,
        FakeLLM([TEXT_ANSWER]),
        make_corpus(),
        [TEXT_CHUNK],
        manifest,
        claims_path=claims_path,
        drafts_path=drafts_path,
        settings=settings,
        latest_edition="2025-26",
        kind="text",
    )
    drafts = read_drafts(drafts_path)
    corpus = make_corpus()
    assert evidence_chunks(drafts, corpus) == {corpus.chunks("table_aware")[0].chunk_id}
    # the same passage under a new chunk id (an index rebuild) is not drafted again
    rebuilt = chunk_record(TEXT_CHUNK.text + " ", doc_id="pes2526_ch07")
    rebuilt = rebuilt.model_copy(update={"chunk_id": "f" * 16})
    corpus.chunk_views["table_aware"][0].chunk_id = "f" * 16
    again = run_draft(
        1,
        FakeLLM([]),
        corpus,
        [rebuilt],
        manifest,
        claims_path=claims_path,
        drafts_path=drafts_path,
        settings=settings,
        latest_edition="2025-26",
        kind="text",
    )
    assert again.records == [] and again.llm_requests == 0


def test_v2_flag_edge_cases() -> None:
    from claimcheck.claims.draft import changed_numbers, partial_period_of

    original = "Literacy for age 10+ rose by 3.0 percentage points between 2018-19 and 2024-25."
    mutated = "Literacy for age 10+ rose by 10.0 percentage points between 2018-19 and 2024-25."
    assert [q.number for q in changed_numbers(original, mutated)] == [10.0]
    assert partial_period_of(["Zakat paid between July and March FY2025 rose."]) is not None
    assert partial_period_of(["Zakat paid in FY2025 rose 65 percent."]) is None
