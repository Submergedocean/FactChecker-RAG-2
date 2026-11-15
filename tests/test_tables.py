"""tables: one test per table rule, on synthetic regions and on the real fixtures.

Synthetic regions are built from `Word` boxes (5 pt per character, 10 pt type), so every
rule is exercised without a PDF. The fixture tests run the whole page pipeline (region
finder, then table extraction) on single pages cut from the corpus.
"""

import json
from functools import cache
from pathlib import Path

import pdfplumber
import pytest

from claimcheck.ingest.extract import Region, Word, extract_page
from claimcheck.ingest.tables import (
    ParsedTable,
    RegionParser,
    Rules,
    caption_key,
    column_unit,
    despace,
    link_continued,
    marker_legends,
    mend_numbers,
    page_tables,
    parse_cell,
    row_label,
    row_unit,
    table_words,
    to_markdown,
    word_kind,
)
from claimcheck.schemas import TableRecord

FIXTURES = Path(__file__).parent / "fixtures" / "pdf"
SPEC = json.loads((FIXTURES / "fixtures.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- helpers


def put(text: str, x: float, top: float, size: float = 10.0) -> list[Word]:
    """The words of one cell, laid out left to right from x."""
    out, cx = [], x
    for part in text.split():
        w = 5.0 * len(part)
        out.append(Word(part, cx, cx + w, top, top + size, size))
        cx += w + 3.0
    return out


def grid(
    rows: list[list[str]], xs: list[float], top: float = 100.0, pitch: float = 12.0
) -> list[Word]:
    """Rows of cells; cell j starts at xs[j]; an empty string leaves the cell out."""
    words: list[Word] = []
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if cell:
                words += put(cell, xs[j], top + i * pitch)
    return words


def box_of(words: list[Word], pad: float = 4.0) -> tuple[float, float, float, float]:
    return (
        min(w.x0 for w in words) - pad,
        min(w.top for w in words) - pad,
        max(w.x1 for w in words) + pad,
        max(w.bottom for w in words) + pad,
    )


def parse(
    words: list[Word], rules: Rules | None = None, caption: str | None = None
) -> list[ParsedTable]:
    region = Region("table", box_of(words), caption, 5, "aligned")
    return RegionParser(words, region, rules or Rules()).parse()


XS = [80.0, 200.0, 280.0, 360.0]  # a label column and three value columns


def simple_table() -> list[Word]:
    rows = [
        ["Table 4.1: Revenue"],
        ["Rs million"],
        ["Item", "FY2024", "FY2025", "FY2026"],
        ["Customs", "1,000", "1,100", "1,250"],
        ["Sales Tax", "5,000", "5,500", "6,000"],
        ["Excise", "300", "(310)", "-"],
        ["Total", "6,300", "6,910", "7,250"],
    ]
    return grid(rows, XS)


# --------------------------------------------------------------------------- numbers


@pytest.mark.parametrize(
    ("text", "value", "printed"),
    [
        ("(1,234.5)", -1234.5, "-1,234.5"),  # accounting negative
        ("(0.97)", -0.97, "-0.97"),
        ("−3.7", -3.7, "−3.7"),  # Unicode minus
        ("1,234", 1234, "1,234"),
        ("1,234,567", 1234567, "1,234,567"),
        ("45.60", 45.6, "45.60"),
        ("3.7%", 3.7, "3.7%"),
        ("1607#", 1607, "1607#"),  # footnote marker kept in the printed form only
        ("5.5 P", 5.5, "5.5 P"),  # status letter
    ],
)
def test_parse_cell_numbers(text: str, value: float, printed: str) -> None:
    v, p = parse_cell(text)
    assert v == value
    assert p == printed


def test_parse_cell_integers_and_text() -> None:
    assert isinstance(parse_cell("1,234")[0], int)
    assert isinstance(parse_cell("1,234.0")[0], float)
    assert parse_cell("-") == ("-", "-")  # placeholders stay text
    assert parse_cell("n.a.")[0] == "n.a."
    assert parse_cell("") == (None, "")
    assert parse_cell("2010-11")[0] == "2010-11"  # a fiscal year is a label, not a number
    assert parse_cell("12 34")[0] == "12 34"  # two numbers in one cell are not one number


def test_accounting_negative_in_a_table() -> None:
    (t,) = parse(simple_table())
    excise = next(r for r in t.rows if r[0] == "Excise")
    assert excise == ["Excise", 300, -310, "-"]
    assert next(r for r in t.printed if r[0] == "Excise")[2] == "-310"


def test_word_kinds() -> None:
    assert word_kind("1,234.5") == "num"
    assert word_kind("2024-25") == "year"
    assert word_kind("FY2026") == "year"
    assert word_kind("-") == "ph"
    assert word_kind("−") == "ph"  # Unicode minus as a placeholder
    assert word_kind("P") == "marker"
    assert word_kind("(000") == "text"  # a fragment of '(000 Nos)'
    assert word_kind("(000)") == "text"  # thousands, not a negative zero
    assert word_kind("Total") == "text"


def test_mend_numbers_joins_split_digits() -> None:
    words = [Word("3", 232.2, 234.8, 100, 106, 5.1), Word("8,265,131", 234.9, 255.9, 100, 106, 5.1)]
    (w,) = mend_numbers(words)
    assert w.text == "38,265,131"
    far = [Word("3", 100, 105, 100, 106, 5), Word("8,265", 120, 140, 100, 106, 5)]
    assert len(mend_numbers(far)) == 2  # a real gap between two cells


def test_despace_letter_spaced_labels() -> None:
    assert despace("E x p o r t s (fob)") == "Exports (fob)"
    assert despace("A B") == "A B"


# --------------------------------------------------------------------------- columns


def test_text_alignment_finds_the_columns_of_a_borderless_table() -> None:
    (t,) = parse(simple_table())
    assert "aligned" in t.flags and "ruled" not in t.flags
    assert t.header == ["Item", "FY2024", "FY2025", "FY2026"]
    assert [r[0] for r in t.rows] == ["Customs", "Sales Tax", "Excise", "Total"]
    assert t.rows[0] == ["Customs", 1000, 1100, 1250]
    assert t.rows[3] == ["Total", 6300, 6910, 7250]


def test_vertical_rules_give_the_columns_of_a_ruled_table() -> None:
    # Right-aligned numbers far from their centred headers: only the drawn rules tie them.
    rows = [
        ["Description", "Total number", "Total Assets"],
        ["Mutual Funds", "409", "4,536"],
        ["Pension Funds", "66", "138"],
        ["Total", "475", "4,674"],
    ]
    words: list[Word] = []
    for i, row in enumerate(rows):
        top = 100 + 14 * i
        words += put(row[0], 82, top)
        if i == 0:
            words += put(row[1], 232, top) + put(row[2], 372, top)
        else:
            words += put(row[1], 290 - 5 * len(row[1]), top) + put(
                row[2], 430 - 5 * len(row[2]), top
            )
    rules = Rules(v=((80.0, 95, 160), (230.0, 95, 160), (300.0, 95, 160), (440.0, 95, 160)))
    (t,) = parse(words, rules)
    assert "ruled" in t.flags
    assert t.header == ["Description", "Total number", "Total Assets"]
    assert t.rows == [
        ["Mutual Funds", 409, 4536],
        ["Pension Funds", 66, 138],
        ["Total", 475, 4674],
    ]


# --------------------------------------------------------------------------- header, caption, units


def test_caption_and_units_line_are_captured_not_left_in_the_header() -> None:
    (t,) = parse(simple_table())
    assert t.caption == "Table 4.1: Revenue"
    assert t.units == "Rs million"
    assert "Rs" not in " ".join(t.header)


def test_units_line_above_the_top_rule_and_units_in_the_caption() -> None:
    rows = [["Table 3.10: Cement Despatches"], ["Million Tonnes"]]
    words = grid(rows, [80.0, 330.0], top=90) + grid(
        [
            ["Years", "Capacity", "Exports"],
            ["2015-16", "45.62", "5.87"],
            ["2016-17", "46.75", "4.66"],
        ],
        [80.0, 230.0, 330.0],
        top=124,
    )
    rules = Rules(h=((118.0, 70, 400),))
    (t,) = parse(words, rules)
    assert t.units == "Million Tonnes"
    assert t.header == ["Years", "Capacity", "Exports"]
    words = grid([["TABLE 7.4 AVERAGE PRICES (Price in Rs.)"]], [80.0], top=90) + grid(
        [["Item", "2024", "2025"], ["Wheat", "100.5", "110.5"], ["Rice", "200.5", "210.5"]],
        [80.0, 230.0, 330.0],
        top=110,
    )
    (t,) = parse(words)
    assert t.units == "(Price in Rs.)"


def test_stacked_header_cells_are_joined_top_to_bottom() -> None:
    rows = [
        ["", "Total number", "Total Assets"],
        ["Description", "of Entities", "(Rs billion)"],
        ["Mutual Funds", "409", "4,536"],
        ["Pension Funds", "66", "138"],
    ]
    (t,) = parse(grid(rows, [80.0, 230.0, 360.0]))
    assert t.header == ["Description", "Total number of Entities", "Total Assets (Rs billion)"]
    assert t.rows[0] == ["Mutual Funds", 409, 4536]


def test_group_header_spans_the_columns_under_its_underline() -> None:
    # 'Indices' sits over General/Food/Core; its underline is what ties it to the three.
    words = (
        put("Indices", 235, 100)
        + put("Year", 80, 112)
        + put("General", 200, 112)
        + put("Food", 280, 112)
        + put("Core", 360, 112)
        + grid(
            [
                ["2010-11", "1.5", "2.5", "3.5"],
                ["2011-12", "1.6", "2.6", "3.6"],
                ["2012-13", "1.7", "2.7", "3.7"],
            ],
            XS,
            top=130,
        )
    )
    rules = Rules(h=((111.0, 190, 400),))
    (t,) = parse(words, rules)
    assert t.header == ["Year", "Indices General", "Indices Food", "Indices Core"]


def test_multi_row_header_with_a_shaded_bar() -> None:
    words = (
        put("Headline", 215, 100)
        + put("General", 200, 114)
        + put("Food", 280, 114)
        + put("Core", 360, 114)
        + put("Year", 80, 114)
        + grid(
            [
                ["2010", "1.5", "2.5", "3.5"],
                ["2011", "1.6", "2.6", "3.6"],
                ["2012", "1.7", "2.7", "3.7"],
            ],
            XS,
            top=132,
        )
    )
    rules = Rules(shades=((190.0, 98.0, 420.0, 112.0),))
    (t,) = parse(words, rules)
    assert t.header[1:] == ["Headline General", "Headline Food", "Headline Core"]


def test_a_bar_over_every_value_column_is_a_units_line() -> None:
    words = put("Year", 80, 100) + put("A", 210, 100) + put("B", 290, 100) + put("C", 370, 100)
    words += put("(Base Year : 2007-08 = 100)", 200, 112)
    words += grid(
        [
            ["2010", "1.5", "2.5", "3.5"],
            ["2011", "1.6", "2.6", "3.6"],
            ["2012", "1.7", "2.7", "3.7"],
        ],
        XS,
        top=128,
    )
    rules = Rules(shades=((190.0, 110.0, 430.0, 124.0),))
    (t,) = parse(words, rules)
    assert t.units == "(Base Year : 2007-08 = 100)"
    assert t.header == ["Year", "A", "B", "C"]


# --------------------------------------------------------------------------- row labels


def test_blank_row_labels_are_forward_filled() -> None:
    rows = [
        ["Country", "Item", "2024", "2025"],
        ["Pakistan", "Exports", "100.5", "120.5"],
        ["", "Imports", "80.5", "90.5"],
        ["India", "Exports", "300.5", "310.5"],
    ]
    # the second column is text: a two-column row label, and both stay columns
    (t,) = parse(grid(rows, [80.0, 160.0, 260.0, 340.0]))
    assert "label_filled" in t.flags and "lead:2" in t.flags and "index_column" not in t.flags
    assert t.header == ["Country", "Item", "2024", "2025"]
    assert [r[:2] for r in t.rows] == [
        ["Pakistan", "Exports"],
        ["Pakistan", "Imports"],  # the blank country is the one above
        ["India", "Exports"],
    ]
    assert [row_label(t.flags, r) for r in t.rows] == [
        "Pakistan Exports",
        "Pakistan Imports",
        "India Exports",
    ]
    assert t.rows[0][2:] == [100.5, 120.5] and t.lead == 2


def test_wrapped_and_split_labels() -> None:
    rows = [
        ["Item", "2024", "2025"],
        ["General", "8.86", "9.31"],
        ["Government", "", ""],
        ["Consumption", "", ""],
        ["Total", "93.59", "92.85"],
        ["Exports of Goods and", "4.25", "5.25"],
        ["Services", "", ""],
        ["Imports", "8.84", "8.82"],
    ]
    (t,) = parse(grid(rows, [80.0, 260.0, 340.0], pitch=10.0))
    assert [r[0] for r in t.rows] == [
        "General Government Consumption",
        "Total",
        "Exports of Goods and Services",
        "Imports",
    ]


def test_heading_scope_follows_the_indent() -> None:
    rows = [
        ["Item", "2024", "2025"],
        ["GROWTH RATES", "", ""],
        ["  Agriculture", "5.1", "6.1"],
        ["  Industry", "4.1", "3.1"],
        ["SAVINGS", "", ""],
        ["  Domestic", "7.1", "8.1"],
        ["  National", "9.1", "9.2"],
    ]
    words = []
    for i, row in enumerate(rows):
        indent = 90.0 if row[0].startswith(" ") else 80.0
        for j, cell in enumerate(row):
            if cell:
                words += put(cell.strip(), indent if j == 0 else [0, 260.0, 340.0][j], 100 + 12 * i)
    (t,) = parse(words)
    assert [r[0] for r in t.rows] == [
        "GROWTH RATES / Agriculture",
        "GROWTH RATES / Industry",
        "SAVINGS / Domestic",
        "SAVINGS / National",
    ]


def test_period_label_applies_to_the_rows_below_it() -> None:
    rows = [
        ["Year", "CPI", "Core"],
        ["2024-25", "264.0", "216.9"],
        ["July-April", "", ""],
        ["2024-25", "264.5", "215.7"],
        ["2025-26", "280.4", "231.1"],
    ]
    (t,) = parse(grid(rows, [80.0, 260.0, 340.0]))
    assert [r[0] for r in t.rows] == ["2024-25", "July-April / 2024-25", "July-April / 2025-26"]


def test_index_column_in_front_of_the_label() -> None:
    rows = [
        ["S.No.", "Item", "Weight", "Change"],
        ["1", "Food", "10.69", "-3.21"],
        ["2", "Beverages", "3.84", "1.70"],
        ["3", "Tobacco", "2.11", "4.20"],
    ]
    (t,) = parse(grid(rows, [80.0, 120.0, 260.0, 340.0]))
    # the two source columns stay two columns: the item number is text, not a value
    assert t.header == ["S.No.", "Item", "Weight", "Change"]
    assert t.rows[0] == ["1", "Food", 10.69, -3.21]
    assert t.printed[2] == ["3", "Tobacco", "2.11", "4.20"]
    assert "index_column" in t.flags and t.lead == 2


def test_marker_joins_its_value_cell() -> None:
    rows = [
        ["Year", "Credit", "Seed"],
        ["2010-11", "2,630", "331.0"],
        ["2011-12", "2,938", "346.4"],
        ["2012-13", "3,362", "327.1"],
    ]
    words = grid(rows, [80.0, 200.0, 300.0]) + put("*", 176.0, 136.0)
    (t,) = parse(words)
    assert t.rows[2][1:] == [3362, 327.1] or t.rows[2][1] == 2938
    assert any("*" in cell for row in t.printed for cell in row)


# --------------------------------------------------------------------------- stacked tables


def test_a_new_header_block_starts_a_new_table() -> None:
    top = [
        ["Year", "A", "B"],
        ["2010", "1.5", "2.5"],
        ["2011", "1.6", "2.6"],
        ["2012", "1.7", "2.7"],
    ]
    low = [
        ["Year", "C", "D"],
        ["2013", "3.5", "4.5"],
        ["2014", "3.6", "4.6"],
        ["2015", "3.7", "4.7"],
    ]
    xs = [80.0, 260.0, 340.0]
    words = grid(top, xs, top=100) + grid(low, xs, top=170)
    tables = parse(words)
    assert len(tables) == 2 and all("stacked" in t.flags for t in tables)
    assert tables[0].header == ["Year", "A", "B"] and tables[1].header == ["Year", "C", "D"]
    assert tables[0].rows[-1] == ["2012", 1.7, 2.7] or tables[0].rows[-1][0] == "2012"
    assert tables[1].rows[0][1:] == [3.5, 4.5]
    # the boxes partition the region
    assert tables[0].bbox[3] == pytest.approx(tables[1].bbox[1])


def test_a_section_title_is_a_heading_not_a_new_table() -> None:
    xs = [80.0, 260.0, 340.0]
    rows = [["Item", "2024", "2025"], ["Loans", "1.5", "2.5"], ["Grants", "1.6", "2.6"]]
    words = (
        grid(rows, xs, top=100)
        + put("REPAYMENTS", 220, 140)
        + grid([["Loans", "3.5", "4.5"], ["Grants", "3.6", "4.6"]], xs, top=152)
    )
    (t,) = parse(words)
    assert [r[0] for r in t.rows] == [
        "Loans",
        "Grants",
        "REPAYMENTS / Loans",
        "REPAYMENTS / Grants",
    ]


def test_a_stacked_table_without_a_header_reuses_the_one_above() -> None:
    xs = [80.0, 260.0, 340.0]
    words = grid(
        [
            ["Year", "A", "B"],
            ["2010", "1.5", "2.5"],
            ["2011", "1.6", "2.6"],
            ["2012", "1.7", "2.7"],
        ],
        xs,
        top=100,
    )
    words += put("(Base Year : 2015-16 = 100)", 200, 160)
    words += grid(
        [["2013", "3.5", "4.5"], ["2014", "3.6", "4.6"], ["2015", "3.7", "4.7"]], xs, top=176
    )
    # no second header row: only a bar, so there is one table with the bar as its heading
    tables = parse(words)
    assert len(tables) == 1


# --------------------------------------------------------------------------- fallback


def test_a_sparse_table_falls_back_to_plain_text() -> None:
    xs = [80.0, 200.0, 280.0, 360.0]
    rows = [["Item", "A", "B", "C"]]
    rows += [["x1", "1.5", "", ""], ["x2", "1.6", "", ""], ["x3", "", "2.5", ""]]
    rows += [["x4", "", "2.6", ""], ["x5", "", "", "3.5"], ["x6", "", "", "3.6"]]
    (t,) = parse(grid(rows, xs))
    assert "fallback_plain" in t.flags and "reason:sparse" in t.flags
    assert t.rows == [] and t.header == []
    assert "x1" in t.plain and "3.6" in t.plain


def test_a_ragged_table_falls_back_to_plain_text() -> None:
    xs = [80.0, 160.0, 220.0, 280.0, 340.0, 400.0]
    rows = [["Item", "A", "B", "C", "D", "E"]]
    full = ["1.5", "2.5", "3.5", "4.5", "5.5"]
    rows += [["r1", *full], ["r2", *full], ["r3", *full]]
    rows += [["r4", "9.5", "2.7", "", "", ""], ["r5", "8.5", "2.8", "", "", ""]]
    (t,) = parse(grid(rows, xs))
    assert "fallback_plain" in t.flags


def test_a_table_with_one_row_is_plain_text() -> None:
    (t,) = parse(grid([["Item", "A"], ["Total", "5.5"]], [80.0, 200.0]))
    assert "fallback_plain" in t.flags


def test_columns_with_the_same_header_are_ambiguous() -> None:
    rows = [
        ["Item", "Total", "Total"],
        ["a", "1.5", "2.5"],
        ["b", "1.6", "2.6"],
        ["c", "1.7", "2.7"],
    ]
    (t,) = parse(grid(rows, [80.0, 260.0, 340.0]))
    assert "reason:header_ambiguous" in t.flags


def test_markdown_and_records_for_fallback_and_cells() -> None:
    md = to_markdown(["A", "B"], [["x", "1"], ["y|z", "2"]], ["Source: SBP"])
    assert md.split("\n")[0] == "| A | B |"
    assert "y\\|z" in md
    assert md.endswith("\n\nSource: SBP")


# --------------------------------------------------------------------------- continuation


def parsed(
    page: int, n: int, *, caption: str | None, header: list[str], continues: bool = False
) -> tuple[ParsedTable, TableRecord]:
    rows = [["2010-11", 1.5, 2.5], ["2011-12", 1.6, 2.6]]
    printed = [["2010-11", "1.5", "2.5"], ["2011-12", "1.6", "2.6"]]
    tab = ParsedTable(
        bbox=(0, 0, 1, 1),
        caption=caption,
        units=None,
        header=header,
        rows=rows,
        printed=printed,
        notes=[],
        flags=[] if any(header) else ["no_header"],
        plain="",
        continues_next=continues,
        contd_caption=bool(caption and "Continued" in caption),
        key=caption_key(caption),
    )
    rec = TableRecord(
        table_id=f"d:p{page}:t{n}",
        doc_id="d",
        page=page,
        bbox=(0, 0, 1, 1),
        caption=caption,
        header=header,
        rows=rows,
        markdown=to_markdown(header, printed, []),
        flags=list(tab.flags),
    )
    return tab, rec


def test_caption_keys() -> None:
    assert caption_key("TABLE 2.1 B (Continued) BASIC DATA") == "2.1b"
    assert caption_key("TABLE 7.1 (B) PRICE INDICES") == "7.1b"
    assert caption_key("Table 14.5: Power Projects") == "14.5"
    assert caption_key("TABLE 1.7 a GROSS FIXED") == "1.7a"
    assert caption_key(None) == ""


def test_contd_links_the_next_page_to_the_first_page() -> None:
    hdr = ["Year", "A", "B"]
    items = [
        parsed(10, 1, caption="TABLE 2.1 B DATA", header=hdr, continues=True),
        parsed(11, 1, caption="TABLE 2.1 B DATA", header=hdr, continues=True),
        parsed(12, 1, caption="TABLE 2.1 B DATA", header=hdr),
    ]
    link_continued(items)
    recs = [r for _, r in items]
    assert recs[0].continued_from is None
    assert recs[1].continued_from == "d:p10:t1"
    assert recs[2].continued_from == "d:p10:t1"  # the first page, not the page before
    assert "continued" in recs[2].flags and "continues" in recs[0].flags


def test_continuation_without_a_header_reuses_the_first_pages() -> None:
    items = [
        parsed(10, 1, caption="TABLE 2.1 B DATA", header=["Year", "A", "B"], continues=True),
        parsed(11, 1, caption="TABLE 2.1 B (Continued)", header=["", "", ""]),
    ]
    link_continued(items)
    rec = items[1][1]
    assert rec.header == ["Year", "A", "B"] and "header_inherited" in rec.flags
    assert "no_header" not in rec.flags
    assert rec.markdown.startswith("| Year | A | B |")


def test_different_table_numbers_are_not_linked() -> None:
    hdr = ["Year", "A", "B"]
    items = [
        parsed(10, 1, caption="TABLE 7.1 (A) PRICES", header=hdr, continues=True),
        parsed(11, 1, caption="TABLE 7.1 (B) PRICES", header=hdr),
    ]
    link_continued(items)
    assert items[1][1].continued_from is None


def test_a_table_not_marked_contd_is_not_linked() -> None:
    hdr = ["Year", "A", "B"]
    items = [
        parsed(10, 1, caption="TABLE 5.1 X", header=hdr),
        parsed(11, 1, caption="TABLE 5.1 X", header=hdr),
    ]
    link_continued(items)
    assert items[1][1].continued_from is None


def test_bare_number_table_takes_row_labels_from_the_previous_page() -> None:
    xs = [80.0, 260.0, 340.0]
    left = grid(
        [
            ["Item", "1960s", "1970s"],
            ["GDP", "6.8", "4.8"],
            ["Agriculture", "5.1", "2.4"],
            ["Industry", "9.9", "5.5"],
        ],
        xs,
        top=100,
    )
    right = grid(
        [["2024", "2025"], ["3.6", "3.1"], ["2.4", "2.2"], ["5.8", "6.0"]], [260.0, 340.0], top=100
    )
    (lt,) = parse(left)
    (rt,) = parse(right)
    assert "no_row_labels" in rt.flags
    lrec = TableRecord(
        table_id="d:p1:t1", doc_id="d", page=1, bbox=(0, 0, 1, 1), markdown="", flags=lt.flags
    )
    rrec = TableRecord(
        table_id="d:p2:t1", doc_id="d", page=2, bbox=(0, 0, 1, 1), markdown="", flags=rt.flags
    )
    lrec.header, rrec.header = lt.header, rt.header
    lrec.rows, rrec.rows = lt.rows, rt.rows
    link_continued([(lt, lrec), (rt, rrec)])
    assert [r[0] for r in rrec.rows] == ["GDP", "Agriculture", "Industry"]
    assert rrec.continued_from == "d:p1:t1" and "labels_inherited" in rrec.flags


def test_bare_number_table_without_a_partner_is_plain_text() -> None:
    right = grid(
        [["2024", "2025"], ["3.6", "3.1"], ["2.4", "2.2"], ["5.8", "6.0"]], [260.0, 340.0], top=100
    )
    (rt,) = parse(right)
    rrec = TableRecord(
        table_id="d:p2:t1", doc_id="d", page=2, bbox=(0, 0, 1, 1), markdown="", flags=rt.flags
    )
    link_continued([(rt, rrec)])
    assert "fallback_plain" in rrec.flags and rrec.rows == []


# --------------------------------------------------------------------------- real fixtures


@cache
def fixture_tables(name: str) -> list[TableRecord]:
    fx = next(f for f in SPEC if f["file"] == name)
    with pdfplumber.open(FIXTURES / name) as pdf:
        page = pdf.pages[0]
        layout = extract_page(page, 1)
        pairs = page_tables(page, 1, fx["doc_id"], layout.regions)
    link_continued(pairs)
    return [rec for _, rec in pairs]


def cell_values(recs: list[TableRecord]) -> set[float | str]:
    out: set[float | str] = set()
    for rec in recs:
        for row in rec.rows:
            for v in row:
                out.add(v if isinstance(v, str) else float(v))
        for row in rec.markdown.splitlines():
            out.update(c.strip() for c in row.strip("|").split("|"))
    return out


@pytest.mark.parametrize("kind", ["ruled_table", "borderless_table", "continued_table"])
def test_fixture_cells_are_recovered(kind: str) -> None:
    """Every numeric cell string of the fixture page, found by the words inside the table
    area, is a cell of a parsed table."""
    fx = next(f for f in SPEC if f["kind"] == kind)
    recs = fixture_tables(fx["file"])
    assert recs and not any("fallback_plain" in r.flags for r in recs)
    found = cell_values(recs)
    numeric = [c for c in fx["cells"] if any(ch.isdigit() for ch in c) and "-" not in c[1:]]
    missing = [
        c for c in numeric if c not in found and c.replace(",", "") not in {str(v) for v in found}
    ]
    assert len(missing) <= 0.05 * len(numeric), missing[:10]


def test_ruled_fixture_structure() -> None:
    fx = next(f for f in SPEC if f["kind"] == "ruled_table")
    (rec,) = fixture_tables(fx["file"])
    assert "ruled" in rec.flags
    assert rec.caption and rec.caption.startswith("Table 3.10")
    assert rec.units == "Million Tonnes"
    assert rec.header == [
        "Years",
        "Production Capacity",
        "Local Despatches",
        "Exports",
        "Total Despatches",
        "Capacity Utilization % age",
    ]
    by_label = {r[0]: r[1:] for r in rec.rows}
    assert by_label["2015-16"] == [45.62, 33.0, 5.87, 38.87, 85.21]
    assert by_label["2025-26 (July-March)"] == [84.58, 31.6, 6.94, 38.54, 60.15]
    assert len(rec.rows) == 11


def test_borderless_fixture_structure() -> None:
    fx = next(f for f in SPEC if f["kind"] == "borderless_table")
    recs = fixture_tables(fx["file"])
    rec = recs[0]
    assert rec.caption and rec.caption.startswith("Table 7.1")
    assert rec.units == "Percent change (Period Average)"
    assert rec.header == ["Group", "Weights", "2024-25", "2025-26"]
    by_label = {r[0]: r[1:] for r in rec.rows}
    assert by_label["CPI National"] == [100.0, 4.7, 6.2]
    assert by_label["Recreation & culture"] == [1.6, 8.0, -3.7]
    assert len(rec.rows) == 15 and {len(r) for r in rec.rows} == {len(rec.header)}


def test_continued_fixture_has_the_first_pages_caption_key() -> None:
    fx = next(f for f in SPEC if f["kind"] == "continued_table")
    (rec,) = fixture_tables(fx["file"])
    assert caption_key(rec.caption) == "2.1b"
    assert "continued" not in rec.flags  # a single page has no first page to link to
    assert rec.header[0] == "Fiscal Year"
    assert rec.rows[0][0] == "2010-11"


# --------------------------------------------------------------------------- regression cases


def indented(spec: list[tuple[str, float, list[str]]], xs: list[float]) -> list[Word]:
    """Rows whose labels start at their own x; the values go to the columns `xs`."""
    words = put("Items", 80, 100) + put("2023", xs[0], 100) + put("2024", xs[1], 100)
    for i, (label, x, vals) in enumerate(spec):
        top = 116 + 12 * i
        words += put(label, x, top)
        for xv, v in zip(xs, vals, strict=False):
            words += put(v, xv, top)
    return words


def test_indented_rows_keep_their_parents_in_the_label() -> None:
    spec = [
        ("Current Expenditure", 80, ["100", "110"]),
        ("Federal", 95, ["60", "70"]),
        ("Defence", 110, ["20", "25"]),
        ("Others", 110, ["10", "12"]),
        ("Provincial", 95, ["40", "40"]),
        ("Development", 80, ["30", "31"]),
    ]
    (t,) = parse(indented(spec, [280.0, 360.0]))
    assert [r[0] for r in t.rows] == [
        "Current Expenditure",
        "Current Expenditure / Federal",
        "Current Expenditure / Federal / Defence",
        "Current Expenditure / Federal / Others",
        "Current Expenditure / Provincial",
        "Development",
    ]


def test_the_first_period_row_of_a_year_does_not_become_the_parent_of_the_others() -> None:
    spec = [
        ("2018-19: Q1", 80, ["1.5", "2.5"]),
        ("Q2", 95, ["1.6", "2.6"]),
        ("Q3", 95, ["1.7", "2.7"]),
        ("2019-20: Q1", 80, ["1.8", "2.8"]),
        ("Q2", 95, ["1.9", "2.9"]),
    ]
    (t,) = parse(indented(spec, [280.0, 360.0]))
    assert [r[0] for r in t.rows] == [
        "2018-19: Q1",
        "2018-19: Q2",
        "2018-19: Q3",
        "2019-20: Q1",
        "2019-20: Q2",
    ]


def test_jitter_or_too_many_levels_is_not_a_hierarchy() -> None:
    jitter = [(f"Year {i}", 80 + (i * 3) % 5, ["1.5", "2.5"]) for i in range(5)]
    (t,) = parse(indented(jitter, [280.0, 360.0]))
    assert [r[0] for r in t.rows] == [f"Year {i}" for i in range(5)]
    deep = [(f"Item {i}", 80 + 12 * i, ["1.5", "2.5"]) for i in range(6)]
    (t,) = parse(indented(deep, [280.0, 360.0]))
    assert [r[0] for r in t.rows] == [f"Item {i}" for i in range(6)]


def test_a_group_title_over_some_columns_is_not_the_units_line() -> None:
    words = (
        put("Rs million", 330, 88)
        + put("% Change", 295, 100)
        + put("Year", 80, 112)
        + put("Level", 200, 112)
        + put("2024-25", 280, 112)
        + put("2025-26", 360, 112)
        + grid(
            [
                ["2010", "1.5", "2.5", "3.5"],
                ["2011", "1.6", "2.6", "3.6"],
                ["2012", "1.7", "2.7", "3.7"],
            ],
            XS,
            top=130,
        )
    )
    framed = Rules(h=((98.0, 70, 400), (111.0, 270, 400)))  # top border, group underline
    (t,) = parse(words, framed)
    assert t.units == "Rs million"
    assert t.header == ["Year", "Level", "% Change 2024-25", "% Change 2025-26"]
    # outside any frame too: '% Change' names the columns under it, it is never a unit
    (t,) = parse(words, Rules(h=((111.0, 270, 400),)))
    assert t.units == "Rs million"
    assert t.header[2:] == ["% Change 2024-25", "% Change 2025-26"]


def test_row_unit_comes_from_a_bracketed_qualifier_heading() -> None:
    assert row_unit("(As percent of GDP) / Domestic Debt") == "As percent of GDP"
    assert row_unit("FISCAL POLICY / (as % of GDP mp) / Defence") == "as % of GDP mp"
    assert row_unit("Banks / DFI") is None
    assert row_unit("Domestic Debt") is None
    assert row_unit("(growth %)") is None  # the row's own text, not a path element
    # A heading that ends with a unit, a percent-change band, a '% of GDP' row
    assert row_unit("Key FSIs (%) / NPLs to Loans (Gross)") == "percent"
    assert row_unit("Sales (Rs million) / Cement") == "Rs million"
    assert row_unit("Percent Change / Apr 25/ Apr 24") == "percent"
    assert row_unit("A. Total Revenue / % of GDP") == "% of GDP"
    assert row_unit("% of GDP") == "% of GDP"
    assert row_unit("Non-performing loans (net)") is None
    assert row_unit("GROWTH RATE (at constant fc) % / GDP") is None
    assert row_unit("STOCK EXCHANGE (growth %) / Turnover") is None
    assert row_unit("Percentage of households") is None
    spec = [
        ("Domestic Debt", 80, ["20,732", "23,283"]),
        ("External Debt", 80, ["11,976", "13,116"]),
        ("(As percent of GDP)", 80, []),
        ("Domestic Debt", 95, ["47.3", "49.0"]),
        ("External Debt", 95, ["27.3", "27.6"]),
    ]
    (t,) = parse(indented(spec, [280.0, 360.0]))
    assert [r[0] for r in t.rows][2:] == [
        "(As percent of GDP) / Domestic Debt",
        "(As percent of GDP) / External Debt",
    ]
    assert "row_units" in t.flags


class FakePage:
    def __init__(self, *words: tuple[str, float, float]) -> None:
        self.raw = [
            {
                "text": text,
                "x0": x,
                "x1": x + 5.0 * len(text),
                "top": 100.0,
                "bottom": 100.0 + size,
                "size": size,
                "upright": True,
            }
            for text, x, size in words
        ]

    def extract_words(self, **_: object) -> list[dict]:
        return self.raw


def test_tiny_footnote_symbols_are_kept_but_tiny_words_are_not() -> None:
    page = FakePage(("1607", 100, 8.1), ("#", 124, 3.9), ("p", 140, 3.0), ("x", 150, 1.0))
    assert [w.text for w in table_words(page)] == ["1607", "#"]  # type: ignore[arg-type]
    assert [w.text for w in table_words(FakePage(("*", 5, 1.5)))] == []  # type: ignore[arg-type]


def test_a_tiny_footnote_mark_stays_with_its_number() -> None:
    words = grid(
        [["Year", "Income", "Seed"], ["2023-24", "1607", "331.0"], ["2024-25", "1751", "346.4"]],
        XS,
        top=100,
    )
    words += put("#", 200 + 21, 113.0, size=4.0)  # after 1607
    (t,) = parse(words)
    assert t.rows[0] == ["2023-24", 1607, 331.0]  # the value is still a number
    assert t.printed[0][1] == "1607#" and t.printed[1][1] == "1751"
    assert "| 1607# |" in to_markdown(t.header, t.printed, [])


def test_a_continued_table_gets_the_legend_of_the_marks_it_uses() -> None:
    notes = ["# : 2023 Population Census", "*: At average exchange rate", "mp : Market prices"]
    assert marker_legends(notes) == {
        "#": "2023 Population Census",
        "*": "At average exchange rate",
    }
    hdr = ["Year", "A", "B"]
    items = [
        parsed(10, 1, caption="TABLE 2.1 B DATA", header=hdr, continues=True),
        parsed(11, 1, caption="TABLE 2.1 B DATA", header=hdr),
    ]
    items[0][0].notes = notes
    items[1][0].printed = [["2010-11", "1.5#", "2.5"], ["2011-12", "1.6", "2.6"]]
    link_continued(items)
    tab, rec = items[1]
    assert tab.notes == ["# : 2023 Population Census"]  # only the mark this table uses
    assert rec.markdown.endswith("# : 2023 Population Census")
    assert "| 1.5# |" in rec.markdown


def test_a_heading_under_a_rule_that_underlines_the_labels_spans_them_all() -> None:
    # 'Average (Annual)' sits under the line of decade labels and their underline, which is
    # what ties it to all five decades (not to the two its own x range happens to touch)
    xs = [80.0, 160.0, 200.0, 240.0, 280.0, 320.0, 370.0, 420.0]
    words = grid(
        [["Indicators", "1960s", "1970s", "1980s", "1990s", "2000s", "2009-10", "2010-11"]],
        xs,
        top=100,
    )
    words += put("Average (Annual)", 200, 113)
    words += grid([[f"Row {i}", *[f"{i}.{j}" for j in range(7)]] for i in range(4)], xs, top=132)
    (t,) = parse(words, Rules(h=((111.0, 150, 350),)))
    assert t.header[1:6] == [
        f"{d} Average (Annual)" for d in ("1960s", "1970s", "1980s", "1990s", "2000s")
    ]
    assert t.header[6:] == ["2009-10", "2010-11"]


def test_a_group_heading_takes_what_the_shaded_group_next_to_it_leaves() -> None:
    xs = [80.0, 160.0, 200.0, 240.0, 280.0, 320.0]
    words = put("Average", 180, 100) + put("Jul-Mar", 292, 100)
    words += grid([["Item", "Mon", "Tue", "Wed", "Thu", "Fri"]], xs, top=114)
    words += grid([[f"Row {i}", *[f"{i}.{j}" for j in range(5)]] for i in range(4)], xs, top=132)
    rules = Rules(h=((111.0, 130, 345),), shades=((270.0, 98.0, 345.0, 111.0),))
    (t,) = parse(words, rules)
    assert t.header == [
        "Item",
        "Average Mon",
        "Average Tue",
        "Average Wed",
        "Jul-Mar Thu",
        "Jul-Mar Fri",
    ]


def test_column_unit_comes_from_the_header() -> None:
    assert column_unit("No. of Units % Change") == "percent"
    assert column_unit("% Change 2024-25/ 2023-24") == "percent"
    assert column_unit("Share of External Debt in Total Public Debt (%)") == "percent"
    assert column_unit("Percentage Share") == "percent"
    assert column_unit("Gross Buy (Rs million)") == "Rs million"
    assert column_unit("2024-25 (July-March)") is None
    assert column_unit("No. of Units 2025-26 (July-March)") is None
    assert column_unit("Weight") is None


def test_column_number_row_is_not_data() -> None:
    from claimcheck.ingest.tables import _column_numbers

    assert _column_numbers([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert _column_numbers([None, None, "‐", None])
    assert not _column_numbers([1, 2, 4])
    assert not _column_numbers([2015, 2016, 2017])
    assert not _column_numbers([1.0, 2.0, 3.0])
    assert not _column_numbers([3, 7, 1])


def test_units_without_a_unit_word_become_a_caption_qualifier() -> None:
    from claimcheck.ingest.tables import _units_or_qualifier
    from claimcheck.schemas import TableRecord

    def rec(units: str) -> TableRecord:
        return TableRecord(
            table_id="d:p1:t1",
            doc_id="d",
            page=1,
            bbox=(0, 0, 1, 1),
            caption="TABLE 12.4 X",
            units=units,
            markdown="",
        )

    for heading in ("1998 Census*", "Rural", "(Jul-Mar)", "2023 2024"):
        r = rec(heading)
        _units_or_qualifier(r)
        assert (r.units, r.caption) == (None, f"TABLE 12.4 X - {heading}")
    for unit in (
        "Rs million",
        "(in Millimetres)",
        "Amount in Rs '000",
        "(% of GDP)",
        "Base Year 2015-16=100 Urban",
        "Quantity in thousand MT; Value in million US $",
        "MMCFD",
        "000 Tonnes",
        "Million Acre Feet",
    ):
        r = rec(unit)
        _units_or_qualifier(r)
        assert (r.units, r.caption) == (unit, "TABLE 12.4 X")


def test_value_tail_and_absorbed_value_repair() -> None:
    """A value read into the row label ('Afghanistan 181', Table 16.1)."""
    from claimcheck.ingest.tables import _return_absorbed_values, value_tail

    assert value_tail("Residential / 101-200 Units -") == "-"
    assert value_tail("Up to 50 Units - Life Line 3.95") == "3.95"
    assert value_tail("Jul-24 77,886.99") == "77,886.99"
    assert value_tail("Afghanistan 181") == "181"
    for label in ("FY 2017", "Q1 2024", "SDG 1", "India BSE Sensex 30", "2019-20", "Total"):
        assert value_tail(label) is None, label
    assert value_tail("Credit and Loans with the IMF (Other than Reserves) 0", strict=False) == "0"
    rows = [["Afghanistan 181", None, 0.496], ["Pakistan", 168, 0.544], ["India", 130, 0.685]]
    printed = [
        ["Afghanistan 181", "", "0.496"],
        ["Pakistan", "168", "0.544"],
        ["India", "130", "0.685"],
    ]
    _return_absorbed_values(rows, printed, 1)
    assert rows[0] == ["Afghanistan", 181, 0.496] and printed[0][:2] == ["Afghanistan", "181"]
    # the cell already holds a value, or the column holds no numbers: nothing moves
    rows = [["Mules 1996 & 2006", 0.2, 1.0], ["Horses", 0.4, 2.0]]
    _return_absorbed_values(
        rows, [["Mules 1996 & 2006", "0.2", "1.0"], ["Horses", "0.4", "2.0"]], 1
    )
    assert rows[0][0] == "Mules 1996 & 2006"


def test_spread_header_needs_a_broken_first_word() -> None:
    from claimcheck.ingest.tables import _spread

    merged = ["Years", "Unemp- Employed Agricul- Mining loyed", "Unemp- Employed Agricul- Mining"]
    assert _spread(merged)
    group = ["Item", "% Change 2024-25/ 2023-24", "% Change 2025-26/ 2024-25"]
    assert not _spread(group)  # a group heading printed over two columns


# --------------------------------------------------------------------------- more regression cases


def labels(t: ParsedTable, lead: int = 1) -> list:
    return [r[:lead] if lead > 1 else r[0] for r in t.rows]


def test_items_without_values_are_left_out() -> None:
    """Table 14.5: 'HOBC (...)', 'Aviation gasoline (100LL)', JP-1's sub-items and 'JP-4' have
    no prices on the page; they are neither headings nor wrapped lines of the row above."""
    spec = [
        ("Motor Gasoline", 80, ["269.43", "260.96"]),
        ("HOBC (Automotive 100 Octane)", 80, []),
        ("Kerosene", 80, ["177.39", "171.77"]),
        ("HSD", 80, ["272.77", "266.07"]),
        ("LDO", 80, ["160.53", "157.02"]),
        ("Aviation gasoline (100LL)", 80, []),
        ("JP-1:", 80, ["202.26", "192.20"]),
        ("i) For sale to PIA", 96, []),
        ("ii) For sale to PIA foreign", 94, []),
        ("flights & foreign airline", 103, []),
        ("JP-4", 80, []),
        ("JP-8", 80, ["198.68", "192.17"]),
    ]
    (t,) = parse(indented(spec, [300.0, 360.0]))
    assert labels(t) == ["Motor Gasoline", "Kerosene", "HSD", "LDO", "JP-1:", "JP-8"]
    assert t.rows[1][1:] == [177.39, 171.77]


def test_a_heading_at_the_rows_indent_is_kept() -> None:
    """'Liabilities' after a gap and a Total row, over rows at its own indent, stays a heading."""
    words = indented(
        [
            ("Investments", 80, ["10", "11"]),
            ("Total Assets", 80, ["30", "31"]),
        ],
        [280.0, 360.0],
    )
    words += put("Liabilities", 80, 160)  # a blank line above it
    words += put("Deposits", 80, 172) + put("20", 280, 172) + put("21", 360, 172)
    words += put("Borrowings", 80, 184) + put("5", 280, 184) + put("6", 360, 184)
    (t,) = parse(words)
    assert labels(t)[-2:] == ["Liabilities / Deposits", "Liabilities / Borrowings"]


def test_wrapped_label_lines_index_and_broken_words() -> None:
    """Table 8.6: '16 | Grains, pulses' over '& flour': one label, '16' in the index column;
    'Manu-' + 'factures' mended because the page prints 'manufactures'."""
    rows = [
        ["", "Items", "2023", "2024"],
        ["15", "Edible oils", "212", "238"],
        ["12", "Iron, steel & Manu-", "", ""],
        ["", "factures thereof", "228", "344"],
        ["16", "Grains, pulses", "", ""],
        ["", "& flour", "110", "72"],
        ["17", "Other manufactures", "1,340", "1,750"],
    ]
    (t,) = parse(grid(rows, [72.0, 100.0, 280.0, 360.0]))
    assert "index_column" in t.flags
    assert [r[:3] for r in t.rows] == [
        ["15", "Edible oils", 212],
        ["12", "Iron, steel & Manufactures thereof", 228],
        ["16", "Grains, pulses & flour", 110],
        ["17", "Other manufactures", 1340],
    ]


def test_a_total_printed_between_its_two_lines_labels_both() -> None:
    """Table 9.4: 'TOTAL (I)' sits between its Principal and Interest lines."""
    words: list[Word] = []
    top = 100.0
    for country, p, i in (("1. Austria", "2.81", "1.45"), ("2. Belgium", "2.02", "1.09")):
        words += put(country, 80, top, 6) + put("Principal", 170, top, 6) + put(p, 280, top, 6)
        words += put("Interest", 170, top + 7.5, 6) + put(i, 280, top + 7.5, 6)
        top += 15
    words += put("Principal", 170, top, 6) + put("4.83", 280, top, 6)
    words += put("TOTAL (I)", 110, top + 3.75, 6)
    words += put("Interest", 170, top + 7.5, 6) + put("2.54", 280, top + 7.5, 6)
    (t,) = parse(words)
    assert [r[:3] for r in t.rows[-2:]] == [
        ["TOTAL (I)", "Principal", 4.83],
        ["TOTAL (I)", "Interest", 2.54],
    ]
    assert t.rows[1][:2] == ["2. Belgium", "Principal"] or t.rows[2][:2] == [
        "2. Belgium",
        "Principal",
    ]


def test_numbering_beats_indent_and_sums_of_sections_are_top_level() -> None:
    """Table 1.5: '2. Livestock' printed deeper than '1. Crops' is its sibling; '3. ... & Gas'
    over 'Distribution' is one label; 'GDP (A + B + C)' is under no section; 'Federal (A+B)'
    over 'A.' and 'B.' sums the rows below it."""
    v = ["1.0", "2.0"]
    spec = [
        ("A. AGRICULTURE", 80, v),
        ("1. Crops", 90, v),
        ("Important Crops", 100, v),
        ("2. Livestock", 100, v),
        ("B. INDUSTRY", 80, v),
        ("1. Mining", 90, v),
        ("2. Manufacturing", 90, v),
        ("Large Scale", 100, v),
        ("3. Electricity Generation & Gas", 90, []),
        ("Distribution", 90, v),
        ("4. Construction", 90, v),
        ("C. SERVICES", 80, v),
        ("1. Trade", 90, v),
        ("GDP (A + B + C)", 90, v),
        ("Indirect Taxes", 100, v),
        ("GDP Deflator", 90, v),
    ]
    (t,) = parse(indented(spec, [300.0, 360.0]))
    assert labels(t) == [
        "A. AGRICULTURE",
        "A. AGRICULTURE / 1. Crops",
        "A. AGRICULTURE / 1. Crops / Important Crops",
        "A. AGRICULTURE / 2. Livestock",
        "B. INDUSTRY",
        "B. INDUSTRY / 1. Mining",
        "B. INDUSTRY / 2. Manufacturing",
        "B. INDUSTRY / 2. Manufacturing / Large Scale",
        "B. INDUSTRY / 3. Electricity Generation & Gas Distribution",
        "B. INDUSTRY / 4. Construction",
        "C. SERVICES",
        "C. SERVICES / 1. Trade",
        "GDP (A + B + C)",
        "GDP (A + B + C) / Indirect Taxes",
        "GDP Deflator",
    ]


def test_sums_sections() -> None:
    from claimcheck.ingest.tables import sums_sections

    assert sums_sections("GDP {Total of GVA at bp (A + B + C)}", ["A. AGRI", "C. SERVICES"])
    assert sums_sections("TOTAL (I+II+III+IV+V)", ["IV. FUNDS", "V. GLOBAL BONDS"])
    assert not sums_sections("Federal (A+B)", ["I. Tax Revenues"])  # A and B come below
    assert not sums_sections("Exports (fob)", ["A. AGRI"])


def test_a_wrapped_line_goes_to_the_label_column_it_is_printed_in() -> None:
    """Risk indicators table: 'Debt (%)' and 'Government Securities (%)' end the indicator,
    not the risk name in the first column."""
    rows = [
        ["Risk", "Indicator", "2023", "2024"],
        ["Currency Risk", "Share of External", "36.7", "32.6"],
        ["", "Debt (%)", "", ""],
        ["Concentration Risk", "Share of Shariah Instruments in", "11.5", "12.4"],
        ["", "Government Securities (%)", "", ""],
        ["Interest Rate Risk", "Share of Fixed Rate Debt", "19.0", "17.7"],
    ]
    (t,) = parse(grid(rows, [80.0, 210.0, 400.0, 460.0]))
    assert [r[:3] for r in t.rows] == [
        ["Currency Risk", "Share of External Debt (%)", 36.7],
        ["Concentration Risk", "Share of Shariah Instruments in Government Securities (%)", 11.5],
        ["Interest Rate Risk", "Share of Fixed Rate Debt", 19.0],
    ]


def test_a_line_back_at_the_outer_indent_starts_the_next_label() -> None:
    """Indicators p5: 'Technical / Vocational' over 'Institutions' after the indented 'Female'
    row is the first line of the next label, not the end of 'Female'."""
    v = ["5.4", "10.6"]
    spec = [
        ("High Schools", 80, v),
        ("Male", 90, v),
        ("Female", 90, v),
        ("Technical / Vocational", 80, []),
        ("Institutions", 80, v),
        ("Male", 90, v),
        ("Literacy Rate", 80, v),
    ]
    (t,) = parse(indented(spec, [300.0, 360.0]))
    assert labels(t) == [
        "High Schools",
        "High Schools / Male",
        "High Schools / Female",
        "Technical / Vocational Institutions",
        "Technical / Vocational Institutions / Male",
        "Literacy Rate",
    ]


def test_a_section_heading_printed_just_above_the_region() -> None:
    body = indented([("Exports (fob)", 80, ["-3.9", "-8.8"])], [300.0, 360.0])
    body = [w for w in body if w.top > 110]  # rows only: the region starts at the first row
    heading = put("TRADE AND PAYMENTS (growth %)", 80, 100)
    region = Region("table", box_of(body), None, 1, "aligned")
    found = RegionParser(body + heading, region, Rules())._heading_above()
    assert found is not None and found.text == "TRADE AND PAYMENTS (growth %)"
    caption = put("TABLE 8.1", 80, 100)
    assert RegionParser(body + caption, region, Rules())._heading_above() is None


def test_paren_meaning_reads_the_note() -> None:
    from claimcheck.ingest.tables import paren_meaning

    note = "* : Weighted average rates shown in parentheses represent Private Sector"
    assert paren_meaning([None, note + " Source: State Bank of Pakistan"]) == "Private Sector"
    assert (
        paren_meaning(["Note: Percentage distribution is given in parenthesis"])
        == "Percentage distribution"
    )
    assert paren_meaning(["Figures in parentheses are negative"]) is None
    assert paren_meaning(["Source: PBS", None]) is None


def test_vertical_runs_read_sideways_labels() -> None:
    from types import SimpleNamespace

    from claimcheck.ingest.tables import vertical_runs

    chars = []
    for k, ch in enumerate("Protected"):  # turned anticlockwise: the first letter at the bottom
        top = 200 - 6 * k
        chars.append(
            {
                "text": ch,
                "x0": 50.0,
                "x1": 58.0,
                "top": top,
                "bottom": top + 5,
                "upright": False,
                "matrix": (0, 1, -1, 0, 0, 0),
            }
        )
    chars.append({"text": "x", "x0": 90.0, "x1": 95.0, "top": 100, "bottom": 108, "upright": True})
    (w,) = vertical_runs(SimpleNamespace(chars=chars))
    assert w.text == "Protected" and w.x0 == 50.0 and w.top == 152 and w.bottom == 205
