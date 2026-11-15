"""Document-level cleaning of extracted pages.

- Headers and footers: lines in the top or bottom edge band whose text (digits masked)
  repeats on 60 % or more of a document's pages. Shares are counted over all pages and over
  odd and even pages separately, because the Survey alternates the survey title (even pages)
  and the chapter title (odd pages).
- Printed page number: an edge line holding only a number or a roman numeral. It goes to
  `printed_page` and is dropped from the text.
- Broken lines: the lines of one column block are re-joined into paragraphs. A word split at
  a line end is re-joined ('infla-' + 'tion' -> 'inflation'); a compound keeps its hyphen
  ('supply-' + 'chain') when the document writes it hyphenated or both halves are words.
- A paragraph cut by a column, band or region change is re-joined when the first part does
  not end a sentence and the second starts in lower case.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from claimcheck.textnorm import clean_unicode

if TYPE_CHECKING:
    from claimcheck.ingest.extract import Line, PageLayout

Edge = Literal["top", "bottom"]

EDGE_BAND = 0.12  # share of the page height at the top and at the bottom
REPEAT_SHARE = 0.6  # lines repeating on 60 %+ of a document's pages
MIN_PAGES_FOR_REPEAT = 3
FULL_LINE_WIDTH = 150.0  # a body-text column line; headings are shorter
PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?(\d{1,4}|[ivxlcdm]{1,7})$", re.IGNORECASE)
_EMBEDDED_NUMBER_RE = re.compile(r"^(\d{1,3})\s+\D.*|.*\s(\d{1,3})$")
_SENTENCE_END = tuple('.!?:;"”)')
# A line that starts with a number followed by a lower-case word carries on a sentence
# ('10 key areas', '12.5 percent of', '2026 in the'); a numbered heading or item does not
# ('2.2-e Agricultural Credit', '1. Crops', '3) Credit cards', '17.6 Provincial').
NUMBER_THEN_WORD_RE = re.compile(r"^\d[\d,]*(?:\.\d+)?%?\s+[a-z]")
_BULLET_RE = re.compile(r"^(?:[•▪●◦‣⁃➢*-]|\(?[a-z]\)|\(?[ivx]+\))\s")
_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
_PRIVATE_USE_RE = re.compile("[-]")


# First halves of compounds that keep their hyphen even when the document never writes them
# on their own mid-line ('high-' + 'frequency').
COMPOUND_HEADS = frozenset(
    [
        "high",
        "low",
        "non",
        "self",
        "well",
        "long",
        "short",
        "medium",
        "mid",
        "large",
        "small",
        "cross",
        "multi",
        "semi",
        "year",
        "month",
        "quarter",
        "week",
        "day",
        "state",
        "cost",
        "market",
        "sector",
        "risk",
    ]
)


def line_key(text: str) -> str:
    """Comparison key for header/footer detection: case-folded, digits masked."""
    text = re.sub(r"\d+", "#", clean_unicode(text).casefold())
    return re.sub(r"\s+", " ", text).strip(" .,:;-|")


def edge_of(line: Line, height: float, band: float = EDGE_BAND) -> Edge | None:
    if line.bottom <= band * height:
        return "top"
    if line.top >= (1 - band) * height:
        return "bottom"
    return None


def boilerplate_keys(
    pages: list[PageLayout], share: float = REPEAT_SHARE, band: float = EDGE_BAND
) -> set[tuple[Edge, str]]:
    """(edge, key) of the header and footer lines of one document.

    A line is a header (footer) if its key appears on at least `share` of the pages that
    have any text in that edge band, counting all pages and odd and even pages separately
    (running titles alternate). Pages without text there, such as statistical tables at the
    end of a chapter, do not dilute the share. A line must repeat on at least 3 pages (2 in
    documents of up to 6 pages); documents under 3 pages are left alone."""
    with_text = [p for p in pages if not p.no_text_layer]
    if len(with_text) < MIN_PAGES_FOR_REPEAT:
        return set()
    seen: dict[tuple[Edge, str], set[int]] = defaultdict(set)
    has_edge: dict[Edge, set[int]] = {"top": set(), "bottom": set()}
    for p in with_text:
        for line in p.lines:
            edge = edge_of(line, p.height, band)
            if edge:
                seen[(edge, line_key(line.text))].add(p.page)
                has_edge[edge].add(p.page)
    keys = set()
    min_count = 2 if len(with_text) <= 6 else 3  # short annexes repeat a title only twice
    for (edge, key), on_pages in seen.items():
        if not key or len(on_pages) < min_count:
            continue
        pool = has_edge[edge]
        for group in (pool, {n for n in pool if n % 2}, {n for n in pool if not n % 2}):
            if len(group) >= 2 and len(on_pages & group) >= share * len(group):
                keys.add((edge, key))
                break
    return keys


def printed_page_number(
    page: PageLayout, keys: set[tuple[Edge, str]] | None = None, band: float = EDGE_BAND
) -> str | None:
    """The page number printed in the header or footer, as printed ('120', 'xii'): an edge
    line holding only the number, or a number at either end of a repeated header/footer
    ('11 HIGHLIGHTS - PAKISTAN ECONOMIC SURVEY - 2025-26')."""
    best: tuple[int, float, str] | None = None
    for line in page.lines:
        edge = edge_of(line, page.height, band)
        if not edge:
            continue
        text = line.text.strip()
        m = PAGE_NUMBER_RE.match(text)
        number = m.group(1) if m else None
        if number is None and keys and (edge, line_key(text)) in keys:
            m = _EMBEDDED_NUMBER_RE.match(text)
            number = (m.group(1) or m.group(2)) if m else None
        if number is None:
            continue
        # Prefer the footer, then the line nearest the page edge.
        dist = page.height - line.bottom if edge == "bottom" else line.top
        rank = (0 if edge == "bottom" else 1, dist, number)
        if best is None or rank < best:
            best = rank
    return best[2] if best else None


def is_boilerplate(
    line: Line, page: PageLayout, keys: set[tuple[Edge, str]], band: float = EDGE_BAND
) -> bool:
    edge = edge_of(line, page.height, band)
    if edge is None:
        return False
    return (edge, line_key(line.text)) in keys or bool(PAGE_NUMBER_RE.match(line.text.strip()))


# --------------------------------------------------------------------------- paragraphs


@dataclass
class Vocabulary:
    """Words of one document, used to undo line-end hyphenation."""

    words: set[str]
    hyphenated: set[str]

    @classmethod
    def from_pages(cls, pages: list[PageLayout]) -> Vocabulary:
        words: set[str] = set()
        hyphenated: set[str] = set()
        for p in pages:
            for line in p.lines:
                tokens = _WORD_RE.findall(line.text)
                if line.text.rstrip().endswith("-") and tokens:
                    tokens = tokens[:-1]  # the split word at the line end
                for tok in tokens:
                    low = tok.lower()
                    if "-" in low:
                        hyphenated.add(low)
                        words.update(low.split("-"))
                    else:
                        words.add(low)
        return cls(words, hyphenated)


def join_lines(a: str, b: str, vocab: Vocabulary) -> str:
    """Join two consecutive lines of a paragraph, undoing line-end hyphenation."""
    a, b = a.rstrip(), b.lstrip()
    if not a:
        return b
    if not b:
        return a
    if a.endswith("-") and len(a) >= 2 and a[-2].isalpha():
        if b[0].islower():
            left_m = re.search(r"([A-Za-z]+(?:-[A-Za-z]+)*)-$", a)
            right_m = re.match(r"([A-Za-z]+)", b)
            whole = left_m.group(1).lower() if left_m else ""
            left = whole.rsplit("-", 1)[-1]
            right = right_m.group(1).lower() if right_m else ""
            if (
                {f"{whole}-{right}", f"{left}-{right}"} & vocab.hyphenated
                or "-" in whole
                or (left in COMPOUND_HEADS and len(right) >= 3)
            ):
                return a + b  # 'pass-' + 'through', 'year-on-' + 'year', 'high-' + 'frequency'
            if left + right in vocab.words:
                return a[:-1] + b
            if len(left) >= 3 and left in vocab.words and right in vocab.words:
                return a + b
            return a[:-1] + b  # 'infla-' + 'tion'

        return a + b  # 'July-' + 'March', 'COVID-' + '19'
    if a.endswith(("–", "—", "/")):
        return a + b
    return f"{a} {b}"


def _continues(prev_text: str, next_line: Line, next_text: str, new_block: bool) -> bool:
    """True if the next line carries on the sentence that `prev_text` left open: it starts in
    lower case or, at the top of a new column block, it is a full line of text."""
    if not prev_text or prev_text.rstrip().endswith(_SENTENCE_END):
        return False
    if next_text[:1].islower() or NUMBER_THEN_WORD_RE.match(next_text):
        return True  # '… The report assesses' / '10 key areas across …'
    return (
        new_block and next_line.x1 - next_line.x0 >= FULL_LINE_WIDTH and not next_text[:1].isdigit()
    )


def _clean_line_text(text: str) -> str:
    text = clean_unicode(text).strip()
    if _PRIVATE_USE_RE.match(text):
        text = "• " + text[1:].lstrip()  # symbol-font bullet
    return _PRIVATE_USE_RE.sub("", text)


def assemble(lines: list[Line], vocab: Vocabulary) -> str:
    """Body lines in reading order -> paragraphs separated by blank lines."""
    paragraphs: list[str] = []
    cur = ""
    prev: Line | None = None
    for line in lines:
        text = _clean_line_text(line.text)
        if not text:
            continue
        if prev is None:
            cur = text
            prev = line
            continue
        same_block = line.side == prev.side and line.band == prev.band
        gap = line.top - prev.bottom
        size_change = abs(line.size - prev.size) >= 1.0
        bullet = bool(_BULLET_RE.match(text))
        new_par = (
            not same_block
            or gap > 0.5 * max(line.size, prev.size)
            or gap < -0.5 * line.size  # moved up: a new column or a sidebar
            or size_change
            or bullet
        )
        if (
            new_par
            and not size_change
            and not bullet
            and _continues(cur, line, text, new_block=not same_block)
        ):
            new_par = False
        if new_par:
            paragraphs.append(cur)
            cur = text
        else:
            cur = join_lines(cur, text, vocab)
        prev = line
    if cur:
        paragraphs.append(cur)
    return "\n\n".join(re.sub(r"\s+", " ", p).strip() for p in paragraphs if p.strip())


# --------------------------------------------------------------------------- pages


@dataclass
class CleanPage:
    printed_page: str | None
    text_body: str
    n_boilerplate: int


def clean_document(
    pages: list[PageLayout], share: float = REPEAT_SHARE, band: float = EDGE_BAND
) -> tuple[list[CleanPage], set[tuple[Edge, str]]]:
    """Clean every page of one document; also returns the header/footer keys found."""
    keys = boilerplate_keys(pages, share, band)
    vocab = Vocabulary.from_pages(pages)
    out = []
    for p in pages:
        body = [ln for ln in p.lines if not is_boilerplate(ln, p, keys, band)]
        out.append(
            CleanPage(
                printed_page=printed_page_number(p, keys, band),
                text_body=assemble(body, vocab),
                n_boilerplate=len(p.lines) - len(body),
            )
        )
    return out, keys


def strip_boilerplate_text(text: str, keys: set[tuple[Edge, str]]) -> str:
    """Drop header/footer lines, and a page number on the first or last two lines, from plain
    text such as `text_raw` (the plain_text ablation can reuse a document's keys)."""
    plain = {k for _, k in keys}
    lines = text.split("\n")
    n = len(lines)
    kept = [
        ln
        for i, ln in enumerate(lines)
        if line_key(ln) not in plain
        and not (PAGE_NUMBER_RE.match(ln.strip()) and (i < 2 or i >= n - 2))
    ]
    return "\n".join(kept)
