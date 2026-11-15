"""chunk: windows, sentences, table chunks and the run over synthetic documents (no corpus).

Tokens are counted with `simple_token_counter` (an offline stand-in for the embedding model's
WordPiece tokenizer); the corpus gate uses the real tokenizer.
"""

import json
import random

import pytest

from claimcheck.config import ChunkingSettings
from claimcheck.ingest.chunk import (
    Unit,
    boilerplate_keys,
    build_chunks,
    chunk_id,
    context_line,
    continues,
    linearise,
    md_cells,
    numeric_share,
    overlap_words,
    page_units,
    read_chunks,
    simple_token_counter,
    split_sentences,
    table_drafts,
    windows,
    write_chunks,
)
from claimcheck.ingest.tables import to_markdown
from claimcheck.schemas import ChunkKind, Manifest, ManifestEntry, Variant

CFG = ChunkingSettings()
COUNT = simple_token_counter
RNG = random.Random(4)
VOCAB = [
    "economy",
    "growth",
    "percent",
    "revenue",
    "exports",
    "wheat",
    "cotton",
    "investment",
    "rupee",
    "policy",
]


def sentence(n: int) -> str:
    words = [RNG.choice(VOCAB) for _ in range(n)]
    return " ".join(words).capitalize() + "."


def prose(n_sentences: int, lo: int = 6, hi: int = 28) -> str:
    paras, cur = [], []
    for k in range(n_sentences):
        cur.append(sentence(RNG.randint(lo, hi)))
        if k % 5 == 4:
            paras.append(" ".join(cur))
            cur = []
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(paras)


def units_of(n_sentences: int) -> list[Unit]:
    return page_units([(1, prose(n_sentences))])


# --------------------------------------------------------------------------- sentences


@pytest.mark.parametrize(
    "text, n",
    [
        ("Exports rose 3.7 percent. Imports fell.", 2),
        ("It cost Rs. 5 billion. Then it fell.", 2),
        ("See Table No. 4 for data. Fine.", 2),
        ("Sources: i.e. the U.S. Treasury. Next one.", 2),
        ("Mr. M. Aurangzeb spoke. He left.", 2),
        ("The list: i. Wheat rose. ii. Cotton fell.", 2),
        ("Growth was 2.89 percent in 2025-26. 2026 will differ.", 2),
        ("Phase II. The next phase began.", 2),
        ("Steps were taken. a) Wheat rose. b) Cotton fell.", 3),
        ("Is it so? Yes! It is.", 3),
        ("No end punctuation here", 1),
    ],
)
def test_split_sentences(text: str, n: int) -> None:
    parts = split_sentences(text)
    assert len(parts) == n, parts
    assert " ".join(parts) == text


def test_page_units_join_open_sentence_across_pages() -> None:
    units = page_units([(1, "Heading\n\nThe deficit was"), (2, "lower this year. New point.")])
    assert [u.text for u in units] == ["Heading", "The deficit was lower this year.", "New point."]
    assert (units[1].page_start, units[1].page_end) == (1, 2)
    assert (units[2].page_start, units[2].page_end) == (2, 2)
    assert units[1].sep == "\n\n" and units[2].sep == " "


def test_page_units_keep_a_list_sentence_whole() -> None:
    # pes2526_ch09 p6: extraction put list item (iii) in a paragraph of its own
    text = (
        "9.4 Domestic Debt\n\nDomestic debt comprises of three main categories: (i) permanent "
        "debt (medium and long-term); (ii) floating debt (short-term); and\n\n(iii) unfunded "
        "debt (primarily made up of various instruments available under National Savings "
        "Schemes).\n\nFurther in line with the Public Debt Act, 1944, the Government issues "
        "securities."
    )
    units = page_units([(6, text)])
    assert [u.text for u in units] == [
        "9.4 Domestic Debt",
        "Domestic debt comprises of three main categories: (i) permanent debt (medium and "
        "long-term); (ii) floating debt (short-term); and (iii) unfunded debt (primarily made "
        "up of various instruments available under National Savings Schemes).",
        "Further in line with the Public Debt Act, 1944, the Government issues securities.",
    ]


@pytest.mark.parametrize(
    "prev, nxt, joined",
    [
        ("Exports rose by 5 percent, while", "Imports fell.", True),
        ("The measures included:", "• Lower tariffs.", False),
        ("2.1 Agriculture Performance 2025-26", "During 2025-26, the sector grew.", False),
        ("The deficit was", "lower this year.", True),
        ("(i) wheat;", "(ii) cotton.", True),
        ("It ended.", "next starts lower.", False),
        ("the government has retired Rs", "1,939.3 billion to SBP.", True),
        ("compromise was reached at US$", "300 billion per year by 2035.", True),
        ("According to the IMF World Economic Outlook (April", "2026), prices rise.", True),
        ("29 licenses for the operation of LPG Storage", "& Filling Plants were issued.", True),
        ("2013-14 9 ,199 2 ,789", "2014-15 9 ,204 2 ,891", False),
        ("III. Unfunded Debt", "Unfunded debt is raised.", False),
        ("Total funds mobilized between July-March FY", "2026 in the PSX amounted to Rs 18.", True),
        ("Sales rose in April", "2025 compared with last year.", True),
        ("Agriculture Performance", "2.1 Crops", False),
        ("perceived enforcement of those laws. The report assesses", "10 key areas across.", True),
        ("decline from 9.10", "12.5 percent of output.", True),
        ("Water Sector Projects", "2.2-e Agricultural Credit", False),
        ("The programme covers", "1. Crops", False),
    ],
)
def test_continues(prev: str, nxt: str, joined: bool) -> None:
    assert continues(prev, nxt) is joined


def test_page_units_new_page_capital_starts_paragraph() -> None:
    units = page_units([(1, "Ends without stop"), (2, "Another paragraph.")])
    assert [(u.text, u.page_start) for u in units] == [
        ("Ends without stop", 1),
        ("Another paragraph.", 2),
    ]


# --------------------------------------------------------------------------- windows


def test_window_and_overlap_sizes() -> None:
    units = units_of(400)
    ws = windows(units, CFG.window_words, CFG.overlap_words)
    assert len(ws) > 20
    sizes = [sum(u.n_words for u in w) for w in ws]
    assert max(sizes) <= CFG.window_words
    # every window but the last is filled to within one sentence of the limit
    assert min(sizes[:-1]) >= CFG.window_words - 28
    overlaps = [overlap_words(a, b) for a, b in zip(ws, ws[1:], strict=False)]
    assert all(0 <= o <= 2 * CFG.overlap_words for o in overlaps)
    assert abs(sum(overlaps) / len(overlaps) - CFG.overlap_words) <= 8
    # windows end at sentence ends and cover every sentence in order
    seen = []
    for w in ws:
        for u in w:
            if not seen or u is not seen[-1] and u not in seen[-len(w) :]:
                seen.append(u)
    assert [u.text for u in seen] == [u.text for u in units]
    assert all(w[-1].text.endswith(".") for w in ws)


def test_long_sentence_is_split_to_window() -> None:
    long = Unit(" ".join(["word"] * 400), 1, 1, "\n\n")
    ws = windows([long, Unit("Short one.", 1, 1)], 150, 30)
    assert all(sum(u.n_words for u in w) <= 150 for w in ws)
    assert sum(u.n_words for w in ws for u in w if u.text != "Short one.") >= 400


def test_tiny_overlap_target_without_fitting_tail() -> None:
    units = [Unit(" ".join(["w"] * 100), 1, 1), Unit(" ".join(["v"] * 100), 1, 1)]
    ws = windows(units, 150, 30)
    assert [len(w) for w in ws] == [1, 1]  # the last unit alone is over 60 words: no overlap


# --------------------------------------------------------------------------- tables


HEADER = ["S.No.", "Item", "2023-24", "2024-25 P", "% Change"]
ROWS = [[str(i + 1), f"Item {i} / sub {i}", 1000.5 + i, 1200 + i, 3.25] for i in range(40)]


def table_record(tid="pes2526_ch01:p3:t1", rows=ROWS, cont=None, caption="Table 1.1: Things"):
    printed = [
        [r[0], r[1], f"{r[2]:,.1f}", f"{r[3]:,}#" if k == 0 else f"{r[3]:,}", "3.25"]
        for k, r in enumerate(rows)
    ]
    return {
        "table_id": tid,
        "doc_id": tid.split(":")[0],
        "page": int(tid.split(":p")[1].split(":")[0]),
        "caption": caption,
        "units": "Rs million",
        "header": HEADER,
        "rows": rows,
        "markdown": to_markdown(HEADER, printed, ["# : provisional", "Source: PBS"]),
        "continued_from": cont,
        "flags": ["ruled", "lead:2", "index_column"],
    }


def test_md_cells_unescapes_pipes() -> None:
    assert md_cells("| a | b \\| c |  |") == ["a", "b | c", ""]


def test_linearise_row_label_and_headers() -> None:
    cells = ["3", "Exports / Goods", "1,234.5", "-", "", "7.1"]
    header = ["S.No.", "Item", "2023-24", "2024-25", "Q", "% Change"]
    out = linearise(header, ["lead:2", "index_column"], cells)
    assert out == "Exports / Goods, 2023-24: 1,234.5; % Change: 7.1"
    assert linearise(["", "A"], [], ["Wheat", "5"]) == "Wheat, A: 5"


def test_table_rows_whole_header_everywhere_and_budget() -> None:
    t = table_record()
    ctx = context_line("2025-26", "Growth and Investment")
    drafts = table_drafts(t, None, ctx, CFG, COUNT)
    assert len(drafts) > 1
    header_line = t["markdown"].split("\n")[0]
    seen: list[int] = []
    for d in drafts:
        assert d.kind == ChunkKind.TABLE and d.table_id == t["table_id"]
        assert d.n_tokens <= CFG.table_max_tokens
        assert d.embed_text.startswith(f"PES 2025-26 · Growth and Investment\n{t['caption']}")
        assert header_line in d.text
        assert "Units: Rs million" in d.text and "Source: PBS" in d.text
        a, b = d.rows
        seen.extend(range(a, b))
        assert d.embed_text.count("\n") == 2 + (b - a)  # context, caption, units, rows
    assert seen == list(range(len(ROWS)))  # every row in exactly one chunk, in order
    body = [
        ln for d in drafts for ln in d.text.split("\n") if ln.startswith("| ") and "---" not in ln
    ]
    rows_md = [ln for ln in body if ln != header_line]
    assert rows_md == t["markdown"].split("\n\n")[0].split("\n")[2:]
    assert "1,200#" in drafts[0].text  # footnote mark kept in the displayed row
    assert "Item 0 / sub 0, 2023-24: 1,000.5; 2024-25 P: 1,200#; % Change: 3.25 (percent)" in (
        drafts[0].embed_text
    )


def banking_table() -> dict:
    """Table 5.6 of PES 2025-26 (pes2526_ch05 p13), as extraction reads it: 'Rs Billion' over the
    amounts, then a 'Key FSIs (%)' band over four ratio rows (checked against the PDF)."""
    header = ["Variables", "CY20", "CY21"]
    rows = [
        ["Profit After Tax (ytd)", 244, 264],
        ["Non-performing loans (net)", 97, 75],
        ["Key FSIs (%) / NPLs to Loans (Gross)", 9.2, 7.9],
        ["Key FSIs (%) / Net NPLs to Net Loans", 1.2, 0.7],
        ["Key FSIs (%) / Capital Adequacy Ratio (all banks)", 18.6, 16.7],
        ["Key FSIs (%) / Advances to Deposit Ratio", 44.8, 46.6],
    ]
    printed = [[r[0], str(r[1]), str(r[2])] for r in rows]
    return {
        "table_id": "pes2526_ch05:p13:t1",
        "doc_id": "pes2526_ch05",
        "page": 13,
        "caption": "Table 5.6: Highlights of the Banking Sector Industry",
        "units": "Rs Billion",
        "header": header,
        "rows": rows,
        "markdown": to_markdown(header, printed, ["Source: SBP"]),
        "continued_from": None,
        "flags": ["ruled", "row_units"],
    }


def test_mixed_units_banking_ratios_never_under_rupees() -> None:
    t = banking_table()
    (d,) = table_drafts(t, None, "PES 2025-26 · Money and Credit", CFG, COUNT)
    line = "Units: Rs Billion; percent for rows under 'Key FSIs (%)'"
    assert line in d.text.split("\n") and line in d.embed_text.split("\n")
    assert "Non-performing loans (net), CY20: 97; CY21: 75" in d.embed_text  # an amount
    assert "Key FSIs (%) / NPLs to Loans (Gross), CY20: 9.2 (percent); CY21: 7.9 (percent)" in (
        d.embed_text
    )
    # a chunk holding only ratio rows names no rupee unit at all
    small = CFG.model_copy(update={"table_max_tokens": 70})
    drafts = table_drafts(t, None, "PES 2025-26 · Money and Credit", small, COUNT)
    assert len(drafts) > 2
    last = drafts[-1]
    assert last.rows[0] >= 2
    assert "Rs Billion" not in last.text and "Rs Billion" not in last.embed_text
    assert "Units: percent for rows under 'Key FSIs (%)'" in last.text


def test_mixed_units_public_debt_and_percent_change() -> None:
    # Table 9.1 (pes2526_ch09 p3): rows under '(As percent of GDP)' in a 'Rs billion' table
    header = ["", "FY24", "FY25"]
    rows = [
        ["Domestic Debt", 47160, 54471],
        ["(As percent of GDP) / Domestic Debt", 47.3, 49.0],
    ]
    t = banking_table() | {
        "units": "Rs billion",
        "caption": "Table 9.1: Public Debt",
        "header": header,
        "rows": rows,
        "markdown": to_markdown(header, [[r[0], f"{r[1]:,}", str(r[2])] for r in rows], []),
    }
    (d,) = table_drafts(t, None, "PES 2025-26 · Public Debt", CFG, COUNT)
    assert "Units: Rs billion; As percent of GDP for rows under '(As percent of GDP)'" in d.text
    assert "Domestic Debt, FY24: 47,160; FY25: 54471\n" in d.embed_text + "\n"
    assert "FY24: 47.3 (As percent of GDP)" in d.embed_text
    # Table 3.9 (pes2526_ch03 p8): no table unit; 51.3 for cars is a percentage, 76,339 is not
    header = ["Category", "No. of Units 2024-25", "No. of Units 2025-26", "No. of Units % Change"]
    rows = [["Cars", 50467, 76339, 51.3]]
    t = banking_table() | {
        "units": None,
        "caption": "Table 3.9: Automobile Production",
        "header": header,
        "rows": rows,
        "markdown": to_markdown(header, [["Cars", "50,467", "76,339", "51.3"]], []),
    }
    (d,) = table_drafts(t, None, "PES 2025-26 · Manufacturing and Mining", CFG, COUNT)
    assert "Units: percent for column 'No. of Units % Change'" in d.text
    assert "No. of Units 2025-26: 76,339; No. of Units % Change: 51.3 (percent)" in d.embed_text


def test_continued_table_uses_first_page_caption() -> None:
    root = table_record()
    cont = table_record("pes2526_ch01:p4:t1", ROWS[:3], root["table_id"], caption=None)
    cont["units"] = None
    (d,) = table_drafts(cont, root, "PES 2025-26 · X", CFG, COUNT)
    assert d.text.startswith(
        "Table 1.1: Things (continued from p3)\nUnits: Rs million; percent for column '% Change'\n"
    )
    assert (d.page_start, d.page_end) == (4, 4)


def test_fallback_table_is_windowed_table_chunk() -> None:
    lines = [" ".join(f"{RNG.randint(1, 9999):,}" for _ in range(12)) for _ in range(60)]
    t = table_record() | {
        "header": [],
        "rows": [],
        "markdown": "\n".join(lines),
        "flags": ["fallback_plain", "reason:ragged"],
    }
    drafts = table_drafts(t, None, "PES 2025-26 · X", CFG, COUNT)
    assert len(drafts) > 2
    for d in drafts:
        assert d.kind == ChunkKind.TABLE and d.table_id == t["table_id"]
        assert d.n_tokens <= CFG.table_max_tokens
        assert d.text.startswith("Table 1.1: Things\nUnits: Rs million\n")
        assert all(ln in lines for ln in d.text.split("\n")[2:])  # whole lines


# --------------------------------------------------------------------------- run


def manifest() -> Manifest:
    def entry(doc_id: str, edition: str, no: int, title: str) -> ManifestEntry:
        return ManifestEntry(
            doc_id=doc_id,
            edition=edition,
            kind="chapter",
            chapter_no=no,
            title=title,
            source_url="https://example.org/x.pdf",
            filename=f"{doc_id}.pdf",
        )

    return Manifest(
        documents=[
            entry("pes2526_ch01", "2025-26", 1, "Growth and Investment"),
            entry("pes2526_ch02", "2025-26", 2, "Agriculture"),
            entry("pes2425_ch01", "2024-25", 1, "Growth and Investment"),
        ]
    )


def corpus() -> tuple[list[dict], list[dict]]:
    rng_state = RNG.getstate()
    RNG.seed(11)
    pages = []
    for doc in ("pes2526_ch01", "pes2526_ch02"):
        for page in range(1, 6):
            body = prose(14)
            raw = f"Pakistan Economic Survey 2025-26\n{body.replace(chr(10) * 2, chr(10))}\n{page}"
            pages.append(
                {"doc_id": doc, "page": page, "text_body": body, "text_raw": raw, "flags": []}
            )
    # the 2024-25 chapter repeats one page of 2025-26 and adds one of its own
    pages.append(dict(pages[0], doc_id="pes2425_ch01"))
    pages.append({"doc_id": "pes2425_ch01", "page": 2, "text_body": "Tiny.", "text_raw": "Tiny."})
    RNG.setstate(rng_state)
    tables = [
        table_record("pes2526_ch01:p3:t1"),
        table_record("pes2425_ch01:p3:t1"),  # same table, same text: a duplicate
    ]
    return pages, tables


def build(variant: Variant = Variant.TABLE_AWARE):
    pages, tables = corpus()
    keys = {"pes2526_ch01": {("top", "pakistan economic survey #-#")}}
    return build_chunks(
        manifest(), pages, tables, variant, CFG, COUNT, latest_edition="2025-26", keys=keys
    )


def test_build_records_contract() -> None:
    records, report = build()
    assert report["kinds"]["text"] > 5 and report["kinds"]["table"] > 1
    for r in records:
        assert r.n_tokens <= CFG.max_tokens
        assert r.n_words > 0 and r.text.strip()
        assert r.embed_text.startswith(f"PES {r.edition} · {r.chapter_title}\n")
        assert r.bm25_tokens
        assert r.chapter_title and r.edition and r.page_start <= r.page_end
        if r.kind == ChunkKind.TEXT:
            assert r.table_id is None and r.n_words <= CFG.window_words
        else:
            assert r.table_id
    assert len({r.chunk_id for r in records}) == len(records)


def test_chunk_ids_identical_on_two_runs(tmp_path) -> None:
    a, _ = build()
    b, _ = build()
    assert [r.chunk_id for r in a] == [r.chunk_id for r in b]
    write_chunks(a, tmp_path / "chunks.parquet")
    assert read_chunks(tmp_path / "chunks.parquet") == a
    assert chunk_id("table_aware", "d", 1, "text", 0, "x") != chunk_id(
        "plain_text", "d", 1, "text", 0, "x"
    )


def test_duplicates_removed_latest_edition_kept() -> None:
    records, report = build()
    texts = [r.text for r in records]
    assert len(texts) == len(set(texts))
    assert report["n_duplicates_removed"] >= 2  # the repeated page's window and the table
    dup_docs = {d["doc_id"] for d in report["duplicates"]}
    assert dup_docs == {"pes2425_ch01"}
    assert not any(r.doc_id == "pes2425_ch01" and r.kind == ChunkKind.TABLE for r in records)


def test_near_empty_document_dropped() -> None:
    m = Manifest(documents=manifest().documents[:1])
    pages = [{"doc_id": "pes2526_ch01", "page": 1, "text_body": "TAX REPORT 2025", "flags": []}]
    records, report = build_chunks(
        m, pages, [], Variant.TABLE_AWARE, CFG, COUNT, latest_edition="2025-26"
    )
    assert records == [] and report["n_short_dropped"] == 1


def test_text_never_crosses_documents() -> None:
    records, _ = build()
    pages, _ = corpus()
    body = {}
    for p in pages:
        body.setdefault(p["doc_id"], []).append(" ".join(p["text_body"].split()))
    for r in records:
        if r.kind == ChunkKind.TEXT:
            assert " ".join(r.text.split()) in " ".join(body[r.doc_id])


def test_plain_text_variant_windows_raw_text() -> None:
    records, report = build(Variant.PLAIN_TEXT)
    assert report["kinds"] == {"text": len(records), "table": 0}
    for r in records:
        if r.doc_id == "pes2526_ch01":
            assert "Pakistan Economic Survey" not in r.text  # header dropped (page keys)
        assert r.n_words <= CFG.window_words


def test_boilerplate_keys_and_numeric_share() -> None:
    keys = boilerplate_keys({"boilerplate": ["bottom: #", "top: growth and investment"]})
    assert keys == {("bottom", "#"), ("top", "growth and investment")}
    assert numeric_share("2010-11 8,901 2,365 548 wheat") == pytest.approx(0.8)
    assert numeric_share("") == 0.0


def test_report_is_json_serialisable() -> None:
    _, report = build()
    json.dumps(report)


# --------------------------------------------------------------------------- fallback tables


def _fallback(markdown: str, caption: str = "TABLE 12.12 EMPLOYED PERSONS") -> dict:
    return {
        "table_id": "x:p1:t1",
        "page": 1,
        "caption": caption,
        "units": None,
        "header": [],
        "rows": [],
        "markdown": markdown,
        "flags": ["fallback_plain", "reason:merged_labels"],
        "continued_from": None,
    }


NUMBERED = "\n".join(
    [
        "TABLE 12.12",
        "Major Industry Division | Pakistan | Punjab",
        "Total | 100.0 | 58.1",
        *[
            line
            for k in range(1, 30)
            for line in (
                f"{k}. | Industry number {k} and",
                f"Wrapped part {k} | {k}.5 | {k}.1",
                "Tail",
            )
        ],
        "Source: Pakistan Bureau of Statistics",
    ]
)


def test_fallback_rows_keep_wrapped_labels_and_repeat_the_header() -> None:
    """Round-3 review (Table 12.12): a chunk never starts inside a numbered row, and every chunk
    repeats the header lines and the notes."""
    small = CFG.model_copy(update={"table_max_tokens": 120})
    drafts = table_drafts(_fallback(NUMBERED), None, "PES 2024-25 · Population", small, COUNT)
    assert len(drafts) > 3
    for d in drafts:
        lines = d.text.split("\n")
        assert lines[0] == "TABLE 12.12 EMPLOYED PERSONS"
        assert lines[1] == "Major Industry Division | Pakistan | Punjab"
        assert lines[2].startswith(("Total", "1. ", "2. ", "3. ")) or lines[2][0].isdigit()
        assert d.text.endswith("Source: Pakistan Bureau of Statistics")
        assert d.n_tokens <= 120
        for k in range(1, 30):  # a numbered row is whole in the chunk that has its number
            if f"{k}. | Industry number {k} and" in lines:
                i = lines.index(f"{k}. | Industry number {k} and")
                assert (
                    lines[i + 1] == f"Wrapped part {k} | {k}.5 | {k}.1" and lines[i + 2] == "Tail"
                )


def test_fallback_section_heading_is_carried_into_later_chunks() -> None:
    """Table 12.6: 'Punjab' heads the rows after it; a chunk that starts inside its rows shows
    it, a chunk that starts at it does not repeat it."""
    rows = ["Total | Urban | Rural", "2023 | 2017 | 1998", "Pakistan"]
    for region in ("Pakistan", "Punjab", "Sindh"):
        if region != "Pakistan":
            rows.append(region)
        rows += [f"{sex} | 57.75 | 60.65 | 56.10" for sex in ("Both", "Male", "Female")]
    rows.append("- : Not available | Source: Pakistan Bureau of Statistics")
    small = CFG.model_copy(update={"table_max_tokens": 60})
    drafts = table_drafts(
        _fallback("\n".join(rows), "TABLE 12.6 LITERACY"), None, "PES", small, COUNT
    )
    assert len(drafts) >= 3
    for d in drafts:
        lines = d.text.split("\n")
        assert lines[1:3] == ["Total | Urban | Rural", "2023 | 2017 | 1998"]
        assert lines[3] in ("Pakistan", "Punjab", "Sindh"), lines  # the section is named
        assert lines.count(lines[3]) == 1


def test_fallback_without_value_lines_is_windowed_as_prose() -> None:
    from claimcheck.ingest.chunk import fallback_plan

    assert fallback_plan(["Hospitals", "(2025)"], None) is None
    plan = fallback_plan(["a. Inspection: | 8,822", "b. Fines: Approx Rs | 2,258,150"], "Lead-in:")
    assert plan is not None and plan.groups == [
        ["a. Inspection: | 8,822"],
        ["b. Fines: Approx Rs | 2,258,150"],
    ]


def test_split_notice_keeps_the_content() -> None:
    from claimcheck.ingest.chunk import split_notice

    notice = "[Check source page p7-8: layout not read reliably (chart_text_in_prose)]"
    assert split_notice(f"{notice}\nThe text.") == (notice, "The text.")
    assert split_notice("The text.") == (None, "The text.")
    assert split_notice("[Check this]\nThe text.") == (None, "[Check this]\nThe text.")


def test_a_text_grid_fallback_repeats_its_headings_in_every_chunk() -> None:
    """Table 2.16: a table of text cells read by its grid, one line per row."""
    rows = [
        f"Project {k} | Progress of project {k} is steady | Impact {k} on water" for k in range(12)
    ]
    md = "\n".join(["Project / Initiative | Description / Progress | Impact / Purpose", *rows])
    small = CFG.model_copy(update={"table_max_tokens": 80})
    t = _fallback(md, "Table 2.16: Water Sector Projects")
    drafts = table_drafts(t, None, "PES 2024-25 · Agriculture", small, COUNT)
    assert len(drafts) > 2
    seen = []
    for d in drafts:
        lines = d.text.split("\n")
        assert lines[:2] == [
            "Table 2.16: Water Sector Projects",
            "Project / Initiative | Description / Progress | Impact / Purpose",
        ]
        seen += lines[2:]
    assert seen == rows  # every row once, whole, in order
