"""extract: layout primitives, and the four real fixture pages.

The fixture tests run the whole per-document pipeline (extract + clean) on single pages cut
from the corpus, so they need no local corpus.
"""

import json
import re
from functools import cache
from pathlib import Path

import pytest

from claimcheck.ingest.extract import (
    FIGURE_CAPTION_RE,
    TABLE_CAPTION_RE,
    DocResult,
    Line,
    Word,
    extract_document,
    find_gutter,
    find_partial_gutter,
    group_lines,
    token_kind,
    word_count_mismatch,
)

FIXTURES = Path(__file__).parent / "fixtures" / "pdf"
SPEC = json.loads((FIXTURES / "fixtures.json").read_text(encoding="utf-8"))
TABLE_FIXTURES = [f for f in SPEC if f["table_bbox"]]


def words_of(text: str, top: float, x0: float, size: float = 10.0, gap: float = 3.0) -> list:
    out, x = [], x0
    for tok in text.split("|"):
        tok = tok.strip()
        for i, part in enumerate(tok.split()):
            w = 5.0 * len(part)
            out.append(Word(part, x, x + w, top, top + size, size))
            x += w + (gap if i < len(tok.split()) - 1 else 0)
        x += 40.0  # wide gap between cells
    return out


# --------------------------------------------------------------------------- primitives


@pytest.mark.parametrize(
    ("token", "kind"),
    [("1,234.5", "num"), ("(0.97)", "num"), ("-7.9", "num"), ("2024-25", "num"),
     ("FY2026", "num"), ("1607#", "num"), ("-", "ph"), ("P", "ph"), ("n.a.", "ph"),
     ("Health", "text"), ("2026.", "text")],
)  # fmt: skip
def test_token_kind(token: str, kind: str) -> None:
    assert token_kind(token) == kind


def test_group_lines_merges_superscripts_and_splits_rows() -> None:
    ws = [
        Word("payments", 100, 140, 100, 111, 11),
        Word("1", 141, 145, 98, 104, 6.5),  # raised footnote marker
        Word("next", 100, 120, 113, 124, 11),
    ]
    lines = group_lines(ws)
    assert [ln.text for ln in lines] == ["payments 1", "next"]


@pytest.mark.parametrize(
    ("row", "tabular"),
    [
        ("Health | 2.8 | 14.5 | 8.6", True),
        ("2010-11 | 22.72 | 331.02 | 137.16", True),
        ("(i) Amount Accepted | 175,664 | 908,867 | - | -", True),
        ("JAPAN | 2,589", True),
        ("Total assets of the pension industry in billions | 138", True),
        ("inflation eased steadily from elevated levels at the", False),
        ("prices rose by 3.7 percent and 4.1 percent in the year", False),
        ("Group | Weights", False),
    ],
)
def test_tabular_rows(row: str, tabular: bool) -> None:
    assert Line(words_of(row, 100, 85)).tabular is tabular


def test_find_gutter_two_columns_and_single_column() -> None:
    sentence = "the economy grew steadily during the year"  # 83..276 pt wide
    two, one = [], []
    for i in range(30):
        top = 100 + 13 * i
        two += words_of(sentence, top, 83, size=11)
        two += words_of(sentence, top, 317, size=11)
        one += words_of(f"{sentence} with more words to fill the full width", top, 63, size=11)
    gutter = find_gutter(two, 612)
    assert gutter is not None and gutter.x0 < 306 < gutter.x1
    assert find_gutter(one, 612) is None


def test_caption_patterns() -> None:
    assert TABLE_CAPTION_RE.match("Table 7.1: Disaggregated CPI")
    assert TABLE_CAPTION_RE.match("TABLE 2.1 B (Continued)")
    assert TABLE_CAPTION_RE.match("TABLE 1.8 b")
    assert not TABLE_CAPTION_RE.match("Table 1.4 shows that mining grew")
    assert FIGURE_CAPTION_RE.match("Fig-7.1 :Month-Wise CPI Trend")
    assert FIGURE_CAPTION_RE.match("Fig-4(a): Growth in Current Expenditure")
    assert not FIGURE_CAPTION_RE.match("Fig-8.1B further shows that imports")


def test_word_count_mismatch() -> None:
    assert not word_count_mismatch(100, 100, 0.2)
    assert not word_count_mismatch(120, 100, 0.2)
    assert word_count_mismatch(121, 100, 0.2)
    assert word_count_mismatch(3, 0, 0.2)


# --------------------------------------------------------------------------- fixtures


@cache
def extracted(name: str) -> DocResult:
    fx = next(f for f in SPEC if f["file"] == name)
    return extract_document(FIXTURES / name, fx["doc_id"])


def occurs(needle: str, haystack: str) -> bool:
    """Whole-token match, as used to cut the fixtures."""
    pattern = r"(?<![\w.,])" + re.escape(needle) + r"(?![\w]|[.,]\d)"
    return re.search(pattern, haystack) is not None


def test_fixture_files_are_four_real_pages() -> None:
    kinds = {f["kind"] for f in SPEC}
    assert kinds == {"prose", "ruled_table", "borderless_table", "continued_table"}
    for f in SPEC:
        assert (FIXTURES / f["file"]).stat().st_size > 10_000


@pytest.mark.parametrize("fx", TABLE_FIXTURES, ids=lambda f: f["kind"])
def test_text_body_has_no_table_cell(fx: dict) -> None:
    """The prose text of a table page contains none of its cell strings.
    Cells the page's own prose also quotes are excluded (they prove nothing); placeholder
    cells such as '-' are too short to test."""
    body = extracted(fx["file"]).records[0].text_body
    testable = [c for c in fx["cells"] if c not in fx["also_in_prose"] and len(c) >= 2]
    assert len(testable) >= 0.5 * len(fx["cells"])
    leaked = [c for c in testable if occurs(c, body)]
    assert leaked == []


@pytest.mark.parametrize("fx", TABLE_FIXTURES, ids=lambda f: f["kind"])
def test_text_body_has_no_table_row(fx: dict) -> None:
    body = re.sub(r"\s+", " ", extracted(fx["file"]).records[0].text_body)
    for row in fx["rows"]:
        if len(row) >= 2:
            assert " ".join(row) not in body


@pytest.mark.parametrize("fx", TABLE_FIXTURES, ids=lambda f: f["kind"])
def test_table_cells_stay_in_text_raw(fx: dict) -> None:
    """text_raw is pdfplumber's plain extraction: the table is still there."""
    raw = extracted(fx["file"]).records[0].text_raw
    assert all(occurs(c, raw) for c in fx["cells"] if len(c) >= 2 and " " not in c)


@pytest.mark.parametrize("fx", SPEC, ids=lambda f: f["kind"])
def test_prose_is_kept_and_rejoined(fx: dict) -> None:
    body = re.sub(r"\s+", " ", extracted(fx["file"]).records[0].text_body)
    for snippet in fx["prose"]:
        assert snippet in body


@pytest.mark.parametrize("fx", SPEC, ids=lambda f: f["kind"])
def test_printed_page_and_flags(fx: dict) -> None:
    res = extracted(fx["file"])
    rec = res.records[0]
    assert rec.page == 1 and rec.doc_id == fx["doc_id"]
    assert rec.printed_page == fx["printed_page"]
    assert "no_text_layer" not in rec.flags
    assert ("has_table" in rec.flags) is (fx["table_bbox"] is not None)
    if fx["printed_page"]:
        assert not re.search(rf"(?m)^{fx['printed_page']}$", rec.text_body)


@pytest.mark.parametrize("fx", TABLE_FIXTURES, ids=lambda f: f["kind"])
def test_table_region_covers_the_table(fx: dict) -> None:
    regions = extracted(fx["file"]).report["regions"]["1"]
    tables = [r for r in regions if r["kind"] == "table"]
    x0, top, x1, bottom = fx["table_bbox"]
    assert any(
        r["bbox"][0] <= x0 + 4 and r["bbox"][1] <= top + 4
        and r["bbox"][2] >= x1 - 4 and r["bbox"][3] >= bottom - 4
        for r in tables
    )  # fmt: skip


def test_prose_page_reading_order() -> None:
    body = extracted("pes2526_ch07_p2.pdf").records[0].text_body
    paragraphs = body.split("\n\n")
    assert paragraphs[:2] == ["Chapter 7", "Inflation"]
    # The left column is read before the right one, across the column break.
    assert body.index("A comparison of monthly") < body.index("The underlying inflation")
    assert "two_column" in extracted("pes2526_ch07_p2.pdf").records[0].flags


def test_column_width_table_keeps_the_prose_beside_it() -> None:
    res = extracted("pes2526_ch03_p10.pdf")
    (region,) = [r for r in res.report["regions"]["1"] if r["kind"] == "table"]
    assert region["bbox"][0] > 300  # the right column only
    assert region["caption"].startswith("Table 3.10")


# ------------------------------------------------------------------- drawn layout


def _page(rects=(), curves=(), lines=(), width=612.0, height=792.0):
    from types import SimpleNamespace

    def box(b, **extra):
        return {"x0": b[0], "top": b[1], "x1": b[2], "bottom": b[3], **extra}

    return SimpleNamespace(
        width=width,
        height=height,
        rects=[box(b, fill=True, non_stroking_color=(0.8, 0.9, 0.6), stroke=False) for b in rects],
        curves=[box(b) for b in curves],
        lines=[box(b) for b in lines],
    )


SENTENCE = "the economy grew steadily during the year"


def _two_columns(top: float, n: int) -> list:
    out = []
    for i in range(n):
        out += words_of(SENTENCE, top + 13 * i, 83, size=11)
        out += words_of(SENTENCE, top + 13 * i, 317, size=11)
    return out


def test_a_chart_frame_is_a_figure_and_its_labels_leave_the_columns() -> None:
    """Fig-6.3 (pes2425_ch06 p5): a frame full of curves beside the prose is a chart."""
    from claimcheck.ingest.extract import chart_frames

    frame = (317.0, 279.0, 529.0, 495.0)
    wedges = [(360 + 10 * k, 330, 470 + 2 * k, 465) for k in range(8)]
    labels = words_of("Banks, 18.7", 350, 330, size=6) + words_of("Cement, 8.3", 360, 460, size=6)
    page = _page(rects=[frame], curves=wedges)
    assert chart_frames(page, labels, 11.0) == [frame]
    prose_inside = labels + [w for k in range(4) for w in words_of(SENTENCE, 300 + 13 * k, 320, 11)]
    assert chart_frames(page, prose_inside, 11.0) == []  # a box of text, not a chart
    assert chart_frames(_page(rects=[frame], curves=wedges[:3]), labels, 11.0) == []


def test_footnotes_across_the_gutter_do_not_move_it() -> None:
    """A page whose right column is short: 6 pt footnotes cross the gutter at the foot."""
    words = []
    for i in range(25):
        words += words_of(SENTENCE, 100 + 13 * i, 83, size=11)
    for i in range(3):
        words += words_of(SENTENCE, 100 + 13 * i, 317, size=11)
    for i in range(4):
        words += words_of("note " * 40, 700 + 7 * i, 92, size=6)
    gutter = find_gutter(words, 612)
    assert gutter is not None and 280 < gutter.gx < 317


def test_a_text_panel_is_read_on_its_own_and_the_columns_around_it_in_order() -> None:
    """Box-1 (pes2526_ch12 p10): two text columns above a shaded box of full-width lines."""
    from claimcheck.ingest.extract import reading_order, text_panels

    above = []
    for i in range(10):
        above += words_of(f"L{i} {SENTENCE}", 98 + 12.6 * i, 83, size=11)
        above += words_of(f"R{i} {SENTENCE}", 98 + 12.6 * i, 317, size=11)
    box = (83.0, 234.0, 529.0, 730.0)
    inside = words_of("Box-1: Women, Business and the Law", 242, 174, size=10)
    for i in range(30):
        inside += words_of(f"box {i} " + " ".join([SENTENCE] * 2), 256 + 12 * i, 88, size=10)
    words = above + inside
    (panel,) = text_panels(_page(rects=[box]), words, 11.0, [])
    assert panel == box
    lines, gutter = reading_order(words, 612, [], [panel])
    texts = [ln.text for ln in lines]
    assert texts[:10] == [f"L{i} {SENTENCE}" for i in range(10)]
    assert texts[10:20] == [f"R{i} {SENTENCE}" for i in range(10)]
    assert texts[20] == "Box-1: Women, Business and the Law"
    assert texts[21].startswith("box 0 ") and len(texts) == 51
    # without the panel the box's full-width lines hide the page-level gutter, but the
    # partial gutter still reads the columns one after the other
    plain, _ = reading_order(words, 612, [])
    assert [ln.text for ln in plain[:20]] == texts[:20]


def test_two_columns_over_full_width_text_are_read_column_by_column() -> None:
    """pes2526_ch04 p10: two columns over a full-width box that is not found as a panel (the
    figure region above overlaps its frame). The box's lines fill the gutter's valley, so
    the page-level gutter is lost and the columns used to be read across, line by line."""
    from claimcheck.ingest.extract import reading_order

    words = []
    for i in range(10):
        words += words_of(f"L{i} {SENTENCE}", 98 + 12.6 * i, 83, size=11)
        words += words_of(f"R{i} {SENTENCE}", 98 + 12.6 * i, 317, size=11)
    for i in range(25):
        words += words_of(f"box {i} " + " ".join([SENTENCE] * 2), 480 + 12 * i, 88, size=11)
    assert find_gutter(words, 612) is None
    gutter = find_partial_gutter(words, 612)
    assert gutter is not None and 276 <= gutter.gx <= 317
    texts = [ln.text for ln in reading_order(words, 612, [])[0]]
    assert texts[:10] == [f"L{i} {SENTENCE}" for i in range(10)]
    assert texts[10:20] == [f"R{i} {SENTENCE}" for i in range(10)]
    assert texts[20].startswith("box 0 ") and len(texts) == 45


def test_columns_over_many_small_footnotes_are_not_headings() -> None:
    """pes2425_ch02 p9: more words of 7 pt footnotes than of 11 pt body text. The display
    heading cut-off (1.4 x the most common size) fell below the body size, so every body line
    was read as a full-width heading and the columns interleaved."""
    from claimcheck.ingest.extract import reading_order

    words = []
    for i in range(6):
        words += words_of(f"L{i} {SENTENCE}", 98 + 12.6 * i, 83, size=11)
        words += words_of(f"R{i} {SENTENCE}", 98 + 12.6 * i, 317, size=11)
    for i in range(12):
        words += words_of(f"note {i} " + " ".join([SENTENCE] * 2), 540 + 8 * i, 88, size=7)
    texts = [ln.text for ln in reading_order(words, 612, [])[0]]
    assert texts[:6] == [f"L{i} {SENTENCE}" for i in range(6)]
    assert texts[6:12] == [f"R{i} {SENTENCE}" for i in range(6)]


def test_short_last_lines_of_a_single_column_page_make_no_gutter() -> None:
    words = []
    for i in range(30):
        top = 100 + 13 * i
        if i % 5 == 4:  # a paragraph's last line, ending before the middle of the page
            words += words_of(SENTENCE, top, 63, size=11)
        else:
            words += words_of(" ".join([SENTENCE] * 2) + " and more", top, 63, size=11)
    assert find_partial_gutter(words, 612) is None


def test_a_caption_over_a_drawn_grid_of_text_is_a_table_region() -> None:
    """Table 2.16 (pes2425_ch02 p19): text cells only, so no number rows find it."""
    from claimcheck.ingest.extract import captioned_grids

    words = words_of("Table 2.16: Water Sector Projects", 211, 86, size=10)
    words += words_of("Project | Description | Impact", 228, 86, size=10)
    rules = [(83, 205, 529, 205.5), (83, 279, 529, 279.5)]
    columns = [
        (198, 222, 199, 279),
        (198, 282, 199, 330),
        (358, 222, 359, 279),
        (358, 282, 359, 330),
    ]
    page = _page(rects=[*rules, *columns])
    (region,) = captioned_grids(page, group_lines(words), [])
    assert region.kind == "table" and region.caption == "Table 2.16: Water Sector Projects"
    assert region.bbox[1] < 206 and region.bbox[3] >= 330 and region.bbox[0] <= 83
    assert captioned_grids(_page(rects=rules), group_lines(words), []) == []  # no column rules


def test_a_table_found_in_part_takes_the_grid_under_its_caption() -> None:
    """Table 16.6, pes2425_ch16 p15: number rows found two columns of the first rows only."""
    from claimcheck.ingest.extract import Region, captioned_grids

    words = words_of("Table 16.6: Status of Projects", 255, 86, size=10)
    rules = [(83, 264, 529, 264.5), (83, 744, 529, 744.5)]
    columns = [(x, 268, x + 1, 744) for x in (105, 209, 398, 479)]
    partial = Region("table", (82.0, 250.0, 211.0, 355.0), "Table 16.6: Status of Projects", 3,
                     "aligned")  # fmt: skip
    (region,) = captioned_grids(_page(rects=[*rules, *columns]), group_lines(words), [partial])
    assert region.source == "aligned+grid" and region.caption == partial.caption
    assert region.bbox[2] >= 529 and region.bbox[3] >= 744
    # a grid that does not hold the region leaves it alone
    elsewhere = Region("table", (82.0, 250.0, 600.0, 900.0), partial.caption, 3, "aligned")
    (same,) = captioned_grids(_page(rects=[*rules, *columns]), group_lines(words), [elsewhere])
    assert same == elsewhere
