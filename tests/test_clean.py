"""clean: headers/footers, printed page numbers, broken lines and paragraphs."""

from claimcheck.ingest.clean import (
    Vocabulary,
    assemble,
    boilerplate_keys,
    clean_document,
    join_lines,
    line_key,
    printed_page_number,
    strip_boilerplate_text,
)
from claimcheck.ingest.extract import Line, PageLayout, Word

H = 792.0


def mk_line(text: str, top: float, x0: float = 83.0, size: float = 11.0, side: str = "L",
            band: int = 0) -> Line:  # fmt: skip
    words, x = [], x0
    for tok in text.split():
        width = 5.0 * len(tok)
        words.append(Word(tok, x, x + width, top, top + size, size))
        x += width + 3.0
    line = Line(words)
    line.side, line.band = side, band  # type: ignore[assignment]
    return line


def mk_page(n: int, lines: list[Line]) -> PageLayout:
    return PageLayout(n, 612.0, H, "x", lines, [], 100, 11.0, None, 0)


def chapter(n_pages: int = 10, appendix: int = 0) -> list[PageLayout]:
    """Pages with alternating running titles, a page number, body text, and statistical
    appendix pages that have no running title."""
    pages = []
    for n in range(1, n_pages + 1):
        title = "Pakistan Economic Survey 2025-26" if n % 2 else "Inflation"
        lines = [
            mk_line(title, 60),
            mk_line(f"Body text of page {n} about prices.", 300),
            mk_line(str(118 + n), 765, x0=300),
        ]
        pages.append(mk_page(n, lines))
    for k in range(appendix):
        # Statistical tables: their captions lie in table regions, not in body lines.
        n = n_pages + k + 1
        pages.append(
            mk_page(n, [mk_line(f"Note {k} on the table.", 400), mk_line(f"{118 + n}", 765)])
        )
    return pages


def test_line_key_masks_digits() -> None:
    assert line_key("Pakistan Economic Survey 2025-26") == "pakistan economic survey #-#"
    assert line_key(" 121 ") == "#"


def test_alternating_headers_and_page_numbers_are_boilerplate() -> None:
    keys = boilerplate_keys(chapter())
    assert ("top", "pakistan economic survey #-#") in keys
    assert ("top", "inflation") in keys
    assert ("bottom", "#") in keys
    assert not any("body text" in k for _, k in keys)


def test_appendix_pages_without_headers_do_not_hide_them() -> None:
    # 6 text pages and 8 appendix pages: each running title is on 3 of 7 odd (even) pages,
    # under 60 %, but on every page of its parity that has text in the top band.
    keys = boilerplate_keys(chapter(6, appendix=8))
    assert ("top", "inflation") in keys
    assert ("top", "pakistan economic survey #-#") in keys
    assert not any(k.startswith("note") for _, k in keys)


def test_short_documents_keep_their_lines() -> None:
    assert boilerplate_keys(chapter(2)) == set()


def test_printed_page_number() -> None:
    page = chapter(3)[1]
    assert printed_page_number(page) == "120"
    lone = mk_page(1, [mk_line("Some text", 300)])
    assert printed_page_number(lone) is None


def test_printed_page_number_inside_repeated_footer() -> None:
    footer = "HIGHLIGHTS - PAKISTAN ECONOMIC SURVEY - 2025-26"
    pages = [
        mk_page(n, [mk_line("Text", 300), mk_line(f"{n + 10} {footer}", 770)]) for n in range(1, 6)
    ]
    keys = boilerplate_keys(pages)
    assert printed_page_number(pages[0], keys) == "11"
    assert printed_page_number(pages[0]) is None  # unknown footer: no guess


def test_clean_document_drops_boilerplate_and_keeps_body() -> None:
    cleaned, _ = clean_document(chapter())
    assert cleaned[2].text_body == "Body text of page 3 about prices."
    assert cleaned[2].printed_page == "121"
    assert cleaned[2].n_boilerplate == 2


def test_strip_boilerplate_text() -> None:
    keys = boilerplate_keys(chapter())
    raw = "Inflation\nBody line one\nBody line two\n12\n121"
    assert strip_boilerplate_text(raw, keys) == "Body line one\nBody line two"


VOCAB = Vocabulary(
    {"inflation", "supply", "chain", "pass", "through", "year", "on", "the"},
    {"pass-through", "year-on-year"},
)


def test_join_lines_hyphenation() -> None:
    assert join_lines("the infla-", "tion rate", VOCAB) == "the inflation rate"
    assert join_lines("improved supply-", "chain links", VOCAB) == "improved supply-chain links"
    assert join_lines("cost pass-", "through of", VOCAB) == "cost pass-through of"
    assert join_lines("when year-on-", "year CPI", VOCAB) == "when year-on-year CPI"
    assert join_lines("July-", "March", VOCAB) == "July-March"
    assert join_lines("COVID-", "19", VOCAB) == "COVID-19"
    assert join_lines("in high-", "frequency data", VOCAB) == "in high-frequency data"
    assert join_lines("price", "stability", VOCAB) == "price stability"


def test_vocabulary_from_pages() -> None:
    page = mk_page(1, [mk_line("a pass-through effect and infla-", 300)])
    v = Vocabulary.from_pages([page])
    assert "pass-through" in v.hyphenated
    assert {"pass", "through", "effect"} <= v.words
    assert "infla" not in v.words  # the split word at a line end is not a word


def test_assemble_paragraphs_and_columns() -> None:
    lines = [
        mk_line("Heading", 100),
        mk_line("First paragraph starts here and", 130),
        mk_line("goes on to the end.", 143),
        mk_line("Second paragraph after a gap", 175),
        mk_line("continues in the", 188),
        mk_line("right column without a break.", 100, x0=317, side="R"),
        mk_line(" A bullet in a symbol font.", 130, x0=317, side="R"),
    ]
    text = assemble(lines, VOCAB)
    assert text.split("\n\n") == [
        "Heading",
        "First paragraph starts here and goes on to the end.",
        "Second paragraph after a gap continues in the right column without a break.",
        "• A bullet in a symbol font.",
    ]


def test_assemble_keeps_heading_apart_from_capitalised_paragraph() -> None:
    lines = [
        mk_line("7.1 Consumer Price Index", 100),
        mk_line("During July-April FY 2026, the headline CPI rose.", 125),
    ]
    assert assemble(lines, VOCAB).split("\n\n") == [
        "7.1 Consumer Price Index",
        "During July-April FY 2026, the headline CPI rose.",
    ]


def test_assemble_breaks_on_font_size_change() -> None:
    lines = [mk_line("Chapter 7", 100, size=32), mk_line("text follows", 140)]
    assert assemble(lines, VOCAB) == "Chapter 7\n\ntext follows"


def test_two_pages_opening_alike_are_not_a_header() -> None:
    opening = "The Pakistan Economic Survey is the flagship annual publication of the"
    pages = chapter(12)
    for n in (5, 7):
        pages[n - 1].lines.insert(0, mk_line(opening, 72))
    assert not any("flagship" in k for _, k in boilerplate_keys(pages))


def test_short_annex_header_on_two_pages() -> None:
    pages = [
        mk_page(n, [mk_line("Quarterly National Accounts" if n % 2 else "Survey", 60),
                    mk_line(f"Body {n}", 300)])
        for n in range(1, 6)
    ]  # fmt: skip
    keys = boilerplate_keys(pages)
    assert ("top", "survey") in keys


def test_a_line_starting_with_a_number_and_a_word_carries_on_the_sentence() -> None:
    """Box-1, pes2526_ch12 p10: a box read on its own puts each line in a block of its own;
    'The report assesses' / '10 key areas across …' is still one sentence. A numbered heading
    after an open line starts a paragraph."""
    lines = [
        mk_line("enforcement of those laws. The report assesses", 100, side="F", band=0),
        mk_line(
            "10 key areas across women's working lives, including Safety.", 112, side="F", band=1
        ),  # fmt: skip
        mk_line("Agricultural financing", 124, side="F", band=2),
        mk_line("2.2-e Agricultural Credit", 136, side="F", band=3),
    ]
    assert assemble(lines, VOCAB).split("\n\n") == [
        "enforcement of those laws. The report assesses 10 key areas across women's working "
        "lives, including Safety.",
        "Agricultural financing",
        "2.2-e Agricultural Credit",
    ]
