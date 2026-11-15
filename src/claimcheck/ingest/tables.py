"""Table-aware extraction.

Extraction finds table *regions* (caption, header rows, body rows, notes) and removes them from the
page's prose. This module reads each region whole and turns it into a `TableRecord`:

- **Columns.** Vertical rules drawn in the PDF (Word-style ruled tables) give the column
  edges. Borderless tables fall back to text alignment: the x-ranges of the cells of the body
  rows are projected onto the page, and the empty stripes between them separate the columns.
- **Header.** Every header line above the first body row is assigned to the columns it
  covers (by overlap, or by the underline / shaded bar drawn under or behind it, which is how
  a group header such as "Indices" over four columns is recognised) and the pieces of a column
  are joined top to bottom: "Indices" + "General" -> "Indices General". A bar that spans every
  value column is a sub-title or units line ("(Base Year : 2007-08 = 100)").
- **Rows.** A row label is the label-column text; label-only lines are headings (indent
  decides their scope), wrapped label lines or a label split above and below the numbers.
  Blank labels are forward-filled. Numbers are parsed: "(1,234.5)" -> -1234.5, thousands
  separators and trailing footnote markers are dropped; "-" and "n.a." stay as text.
- **Stacked tables.** A new header block in the middle of a region (statistical appendix
  pages print several tables under one caption) starts a new table with its own header.
- **Continued tables.** "(Contd.)" at the bottom of a table and the same table number at the
  top of the next page link the pages: `continued_from` is the first page's table id, and a
  continuation that printed no header reuses the first page's.
- **Fallback.** A table that comes out ragged or sparse, or whose cells do not line up, is
  not emitted as cells: it is flagged `fallback_plain` and `markdown` holds the region's plain
  text instead, so the chunker treats it as text and never indexes wrong cells.

`TableRecord.markdown`: the table as a Markdown pipe table (header row,
separator, one line per row, numbers as printed with accounting negatives written "-1,234.5"),
then, after one blank line, the table's source and note lines. For a `fallback_plain` table
it is the region's plain text, one line per row with the cells joined by " | ".

Flags: `fallback_plain`, `ruled` / `aligned` (how the columns were found), `stacked` (one of
several tables parsed from one region), `continued`, `continues`, `header_inherited`,
`labels_inherited`, `label_filled`, `no_header`, `lead:N` (N label columns before the values,
each a column; see `lead_columns`), `index_column` (the first of them is the S.No.), `row_units`
(a bracketed qualifier such as '(As percent of GDP)' gives some rows their own unit, see
`row_unit`), plus `reason:<name>` on a fallback table.

A row label is the heading path of the row: label-only heading lines above it and the rows of
the same table it is indented under ('Current Expenditure / Federal / Defence'). Footnote marks
after a number stay in the printed cell; a continued table gets the legend of its marks from
the first page's notes.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from claimcheck.ingest.extract import (
    MIN_WORD_SIZE,
    NOTE_RE,
    TABLE_CAPTION_RE,
    BBox,
    Line,
    Region,
    Segment,
    Word,
    _drop_superscripts,
    group_lines,
)
from claimcheck.schemas import DocKind, Manifest, TableRecord
from claimcheck.textnorm import clean_unicode, is_number_token, normalize_ws, parse_number

if TYPE_CHECKING:
    from pdfplumber.page import Page

# --------------------------------------------------------------------------- constants

# A value cell holds a number or one of these placeholders; markers belong to a neighbour.
PLACEHOLDERS = {
    "-", "–", "—", "−", "‐", "‑", "‒", "―", "--", "..", "...", "…", "n.a", "n.a.", "na", "n/a",
    "nil", "neg",
}  # fmt: skip
MARKERS = {
    "p",
    "r",
    "e",
    "f",
    "x",
    "(p)",
    "(r)",
    "(e)",
    "(f)",
    "*",
    "**",
    "#",
    "%",
    "(%)",
    "^",
    "@",
}
_FOOTNOTE_CHARS = "*#^@†‡"
_YEARISH_RE = re.compile(
    r"^(?:fy\s?)?['’]?(?:19|20)\d{2}(?:[-–/](?:\d{2}|\d{4}))?[a-z*#]?$"
    r"|^(?:19|20)\d{2}s$|^\((?:19|20)\d{2}(?:[-–/]\d{2})?\)$",
    re.I,
)
UNITS_RE = re.compile(
    r"^\(?\s*(?:in\s+)?(?:rs\b|rs\.|us\s?\$|\$|pkr\b|pak(?:istan)?\s+rupees|million|billion|"
    r"trillion|crore|lakh|percent|per\s?cent|%|number|nos\b|no\.|mw\b|gw|tonnes?|tons?|"
    r"hectares?|acres?|kg\b|litres?|liters?|index|indices|base\s+year|price|value|growth|"
    r"000|thousand|ratio|share|us\s*dollars?|\d+\s?=\s?100)",
    re.I,
)
_QUALIFIER_RE = re.compile(
    r"^\(\s*(?:as\s+(?:a\s+)?%|as\s+percent|in\s|growth|%|per\s|rs\b|us\s?\$|million|billion|"
    r"percent|share|ratio|index|base|average|annual|net|cumulative|cumulative|in\s+million)",
    re.I,
)
_PERIOD_RE = re.compile(
    r"^\(?\s*(?:jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?)\b"
    r"[\w\s\-–/.,()]*\)?$",
    re.I,
)
STRICT_UNITS_RE = re.compile(
    r"^\(?\s*(?:in\s+)?(?:base\s+year\b|rs\b|rs\.|us\s?\$|\$|pkr\b|million|billion|trillion|crore|lakh|percent|"
    r"per\s?cent|%|numbers?\b|nos\b|mmcfd|mw\b|tonnes?\b|hectares?\b)",
    re.I,
)
CONTD_RE = re.compile(r"\(\s*(?:contd|continued|cont'd)\s*[.…\s]*\)", re.I)
_CAPTION_KEY_RE = re.compile(
    r"^\s*(?:TABLE|Table)\s*[-:.]?\s*(\d+(?:[.\-]\d+)*|[IVX]+)\s*\(?([A-Za-z])?\)?(?![A-Za-z])"
)
_CONNECTOR_END_RE = re.compile(
    r"(?:[,&/\-]|\b(?:and|of|the|in|for|to|on|by|with|from|at|per))\s*$", re.I
)

COL_PAD = 2.0  # pt added around a band when it is compared with header text
MIN_DATA_ROWS = 2


# --------------------------------------------------------------------------- word and cell kinds


def word_kind(text: str) -> Literal["num", "year", "ph", "marker", "text"]:
    """Classify one word of a table line."""
    t = text.strip()
    low = t.lower()
    if low in PLACEHOLDERS:
        return "ph"
    if low in MARKERS:
        return "marker"
    if t.startswith("(") != t.endswith(")"):  # '(000', 'tonnes)': a fragment of a unit
        return "text"
    if re.fullmatch(r"\(0+\)", t):  # '(000)': thousands, not a number
        return "text"
    if _YEARISH_RE.match(t):
        return "year"
    core = t.rstrip(_FOOTNOTE_CHARS)
    if core and is_number_token(core):
        return "num"
    return "text"


@dataclass
class Cell:
    """One cell-sized piece of a line: words between two wide gaps (numbers one by one)."""

    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def x0(self) -> float:
        return self.words[0].x0

    @property
    def x1(self) -> float:
        return self.words[-1].x1

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def top(self) -> float:
        return min(w.top for w in self.words)

    @property
    def bottom(self) -> float:
        return max(w.bottom for w in self.words)

    @cached_property
    def kind(self) -> Literal["value", "period", "text", "marker"]:
        kinds = [word_kind(w.text) for w in self.words]
        if "text" in kinds:
            return "text"
        if "num" in kinds or "ph" in kinds:
            return "value"
        if "year" in kinds:
            return "period"
        return "marker"


def line_cells(line: Line) -> list[Cell]:
    """Cells of a line. A segment of only numbers and placeholders is split into one cell per
    word (tightly set columns), except the leftmost one of a multi-cell line: '2022-23 R' is a
    row label."""
    segs: list[Segment] = line.segments
    cells: list[Cell] = []
    for i, seg in enumerate(segs):
        kinds = [word_kind(w.text) for w in seg.words]
        only_values = "text" not in kinds
        n_numbers = sum(1 for k in kinds if k in ("num", "year", "ph"))
        label_like = i == 0 and len(segs) >= 2 and n_numbers <= 1
        if only_values and not label_like and len(seg.words) > 1:
            cells.extend(Cell([w]) for w in seg.words)
        elif kinds.count("year") >= 3 and not label_like:
            part: list[Word] = []
            for w, k in zip(seg.words, kinds, strict=True):
                if part and word_kind(part[-1].text) == "year" and k == "text":
                    cells.append(Cell(part))  # 'FY 2016' | 'FY 2017'
                    part = []
                part.append(w)
            cells.append(Cell(part))
        else:
            cells.append(Cell(list(seg.words)))
    return cells


_SPACED_RE = re.compile(r"(?<!\S)(?:[A-Za-z] ){3,}[A-Za-z](?!\S)")


def despace(text: str) -> str:
    """'E x p o r t s (fob)' -> 'Exports (fob)': letters set one by one with wide spacing."""
    return _SPACED_RE.sub(lambda m: m.group(0).replace(" ", ""), text)


def _is_upper(text: str) -> bool:
    """Mostly capitals, not counting letters in brackets ('GDP DEFLATOR (growth %)')."""
    letters = [c for c in re.sub(r"\([^)]*\)", "", text) if c.isalpha()]
    return len(letters) >= 3 and sum(c.isupper() for c in letters) >= 0.7 * len(letters)


def _inside(w: Word, box: BBox) -> bool:
    return box[0] <= (w.x0 + w.x1) / 2 <= box[2] and box[1] <= w.cy <= box[3]


# --------------------------------------------------------------------------- numbers


def parse_cell(
    text: str, *, brackets_negative: bool = True
) -> tuple[int | float | str | None, str]:
    """Value and printed form of one cell.

    '(1,234.5)' -> (-1234.5, '-1,234.5'); '1,234' -> (1234, '1,234'); '3.7%' -> (3.7, '3.7%');
    a trailing footnote marker or status letter ('1607#', '5.5 P') is dropped from the value
    and kept in the printed form; '-' and 'n.a.' stay as text; blank is None. Without
    `brackets_negative` (the table's note gives brackets another meaning, see
    `paren_meaning`) '(17.7)' -> (17.7, '(17.7)').
    """
    t = clean_unicode(text).strip()
    if not t:
        return None, ""
    core = re.sub(r"\s*(?:[" + re.escape(_FOOTNOTE_CHARS) + r"]+|\(?[PRFE]\)?)$", "", t).strip()
    value = parse_number(core) if core else None
    if value is None or not re.search(r"\d", core):
        return t, t
    if re.search(r"\d\s+\d", core):  # two numbers in one cell
        return t, t
    printed = t
    if core.startswith("(") and core.endswith(")"):
        if not brackets_negative:
            value = abs(value)
        else:
            printed = "-" + t.replace(core, core[1:-1], 1)
    has_point = "." in core
    if not has_point and float(value).is_integer() and abs(value) < 2**53:
        return int(value), printed
    return value, printed


# --------------------------------------------------------------------------- words


MARKER_SYMBOLS = frozenset(_FOOTNOTE_CHARS)
MIN_MARKER_SIZE = 2.0


def table_words(page: Page) -> list[Word]:
    """Upright words of a page for table reading: `page_words`, plus footnote symbols set in
    type too small for it (the '#' after 1607 is 4 pt): they belong to the number beside them."""
    raw = page.extract_words(
        x_tolerance=3, y_tolerance=3, keep_blank_chars=False, extra_attrs=["size", "upright"]
    )
    words = []
    for w in raw:
        text = w["text"].strip()
        if not w["upright"] or not text:
            continue
        if w["size"] < MIN_WORD_SIZE and not (
            w["size"] >= MIN_MARKER_SIZE and set(text) <= MARKER_SYMBOLS
        ):
            continue
        words.append(Word(w["text"], w["x0"], w["x1"], w["top"], w["bottom"], w["size"]))
    return words


def vertical_runs(page: Page) -> list[Word]:
    """Text set sideways (a label in a tall merged cell, `Protected` beside the tariff rows of
    Table 14.4, `A. CONSUMER PRICE INDEX` beside a block of Table 7.2), one Word per run read
    in its own direction. `table_words` keeps upright words only; this is for the plain text
    of a `fallback_plain` table, where such a label is the only context of the rows beside it."""
    cols: dict[int, list[dict[str, Any]]] = {}
    for c in page.chars:
        if c.get("upright", True):
            continue
        key = next((k for k in cols if abs(k - c["x0"]) <= 1.5), None)
        cols.setdefault(round(c["x0"]) if key is None else key, []).append(c)
    out: list[Word] = []
    for chars in cols.values():
        chars.sort(key=lambda c: c["top"])
        runs: list[list[dict[str, Any]]] = []
        for c in chars:
            height = c["x1"] - c["x0"]  # the font size of sideways text
            if runs and c["top"] - runs[-1][-1]["bottom"] <= 0.6 * height:
                runs[-1].append(c)
            else:
                runs.append([c])
        for run in runs:
            text = "".join(c["text"] for c in run)
            if run[0]["matrix"][1] > 0:  # turned anticlockwise: read from the bottom up
                text = text[::-1]
            text = normalize_ws(text)
            if sum(ch.isalnum() for ch in text) < 2:
                continue
            out.append(
                Word(
                    text,
                    min(c["x0"] for c in run),
                    max(c["x1"] for c in run),
                    run[0]["top"],
                    run[-1]["bottom"],
                    max(c["x1"] - c["x0"] for c in run),
                )
            )
    return sorted(out, key=lambda w: (w.top, w.x0))


# --------------------------------------------------------------------------- drawn rules


@dataclass(frozen=True)
class Rules:
    """Horizontal and vertical rules and shaded bars drawn inside a table region."""

    h: tuple[tuple[float, float, float], ...] = ()  # (y, x0, x1)
    v: tuple[tuple[float, float, float], ...] = ()  # (x, y0, y1)
    shades: tuple[BBox, ...] = ()


def _luminance(color: Any) -> float | None:
    """Brightness 0 (black) .. 1 (white) of a PDF colour; None for a pattern or unknown."""
    if color is None:
        return 1.0
    if isinstance(color, int | float):
        return float(color)
    try:
        vals = [float(v) for v in color]
    except (TypeError, ValueError):
        return None
    if not vals:
        return 1.0
    if len(vals) == 4:  # CMYK
        return 1.0 - min(1.0, max(vals[:3]) + vals[3])
    return statistics.fmean(vals[:3])


def _is_white(color: Any) -> bool:
    lum = _luminance(color)
    return lum is not None and lum >= 0.97


def _merge_runs(
    items: list[tuple[float, float, float]], pos_tol: float, gap_tol: float
) -> list[tuple[float, float, float]]:
    """Join collinear pieces: (position, start, end) with equal position (+-pos_tol) that
    touch (gap <= gap_tol)."""
    out: list[list[float]] = []
    for pos, a, b in sorted(items):
        for o in out:
            if abs(o[0] - pos) <= pos_tol and a <= o[2] + gap_tol and b >= o[1] - gap_tol:
                o[1], o[2] = min(o[1], a), max(o[2], b)
                break
        else:
            out.append([pos, a, b])
    return [(o[0], o[1], o[2]) for o in out]


def drawn_rules(page: Page, box: BBox, pad: float = 4.0) -> Rules:
    """Rules (thin filled rectangles or lines) and shaded bars that touch `box`."""
    x0, t, x1, b = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    h: list[tuple[float, float, float]] = []
    v: list[tuple[float, float, float]] = []
    shades: list[BBox] = []
    for r in page.rects:
        if r["x1"] < x0 or r["x0"] > x1 or r["bottom"] < t or r["top"] > b:
            continue
        w, ht = r["x1"] - r["x0"], r["bottom"] - r["top"]
        fill = r.get("non_stroking_color") if r.get("fill") else None
        if r.get("fill") and _is_white(fill) and not r.get("stroke"):
            continue
        if ht <= 2.5 and w >= 6:
            h.append(((r["top"] + r["bottom"]) / 2, r["x0"], r["x1"]))
        elif w <= 2.5 and ht >= 6:
            v.append(((r["x0"] + r["x1"]) / 2, r["top"], r["bottom"]))
        elif r.get("fill") and not _is_white(fill) and 5 <= ht <= 30 and w >= 20:
            lum = _luminance(fill)
            if lum is not None and lum >= 0.2:  # not a black bar
                shades.append((r["x0"], r["top"], r["x1"], r["bottom"]))
    for ln in page.lines:
        if ln["x1"] < x0 or ln["x0"] > x1 or ln["bottom"] < t or ln["top"] > b:
            continue
        if abs(ln["bottom"] - ln["top"]) <= 1.5 and ln["x1"] - ln["x0"] >= 6:
            h.append(((ln["top"] + ln["bottom"]) / 2, ln["x0"], ln["x1"]))
        elif abs(ln["x1"] - ln["x0"]) <= 1.5 and ln["bottom"] - ln["top"] >= 6:
            v.append(((ln["x0"] + ln["x1"]) / 2, ln["top"], ln["bottom"]))
    return Rules(
        h=tuple(_merge_runs(h, 1.2, 3.0)),
        v=tuple(_merge_runs(v, 1.2, 3.0)),
        shades=tuple(shades),
    )


# --------------------------------------------------------------------------- columns


@dataclass
class Band:
    """One table column: its data extent and the edges used to place header text."""

    x0: float
    x1: float
    lo: float = 0.0
    hi: float = 0.0
    support: int = 0
    text_col: bool = False
    ruled: bool = False

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def core(self) -> tuple[float, float]:
        """Where header text of this column can sit: the cell edges of a ruled table, the data
        extent of a borderless one."""
        return (self.lo, self.hi) if self.ruled else (self.x0, self.x1)


def _set_edges(bands: list[Band], left: float, right: float) -> None:
    for i, band in enumerate(bands):
        band.lo = left if i == 0 else (bands[i - 1].x1 + band.x0) / 2
        band.hi = right if i == len(bands) - 1 else (band.x1 + bands[i + 1].x0) / 2


def aligned_bands(
    intervals: list[tuple[int, float, float]], n_rows: int, left: float, right: float
) -> list[Band]:
    """Columns from text alignment: x-ranges of the body cells (row id, x0, x1) projected onto
    the page; stripes covered by (almost) no row separate the columns."""
    if not intervals:
        return []
    lo = int(np.floor(min(a for _, a, _ in intervals)))
    hi = int(np.ceil(max(b for _, _, b in intervals)))
    cover = np.zeros((n_rows, hi - lo + 2), dtype=bool)
    for row, a, b in intervals:
        cover[row, int(np.floor(a)) - lo : int(np.ceil(b)) - lo + 1] = True
    tolerated = int(0.04 * n_rows)
    count = cover.sum(axis=0)
    mask = count > tolerated
    bands: list[Band] = []
    start: int | None = None
    for x in range(len(mask) + 1):
        on = x < len(mask) and bool(mask[x])
        if on and start is None:
            start = x
        elif not on and start is not None:
            rows = int(cover[:, start:x].any(axis=1).sum())
            bands.append(Band(lo + start, lo + x - 1, support=rows))
            start = None
    merged: list[Band] = []
    for band in bands:
        if merged and band.x0 - merged[-1].x1 < 3.0:  # a narrow stripe inside one column
            merged[-1].x1 = band.x1
            merged[-1].support = max(merged[-1].support, band.support)
        else:
            merged.append(band)
    min_support = max(2, int(np.ceil(0.12 * n_rows)))
    merged = [b for b in merged if b.support >= min_support] or merged
    _set_edges(merged, left, right)
    return merged


def ruled_bands(
    seps: list[float], cells: list[Cell], left: float, right: float
) -> list[Band] | None:
    """Columns between vertical rules. None if the rules cut through the body cells."""
    edges = [left, *sorted(seps), right]
    bands = [
        Band(edges[i], edges[i + 1], edges[i], edges[i + 1], ruled=True)
        for i in range(len(edges) - 1)
    ]
    crossing = 0
    for c in cells:
        if any(c.x0 + 1.5 < s < c.x1 - 1.5 for s in seps):
            crossing += 1
    if cells and crossing > 0.05 * len(cells):
        return None
    for band in bands:
        inside = [c for c in cells if band.lo <= c.cx < band.hi]
        band.support = len(inside)
        if inside:
            band.x0, band.x1 = min(c.x0 for c in inside), max(c.x1 for c in inside)
    kept = [b for b in bands if b.support > 0]
    if len(kept) < 2:
        return None
    # An empty stripe between rules (a gap column) is absorbed by the neighbour on its left.
    for i, band in enumerate(bands):
        if band.support == 0 and i > 0:
            bands[i - 1].hi = band.hi
    bands = [b for b in bands if b.support > 0]
    return bands


def vertical_separators(
    rules: Rules, box: BBox, top: float, bottom: float, min_cover: float = 0.6
) -> list[float]:
    """x of the vertical rules that run through (most of) the body rows, border rules aside."""
    span = max(bottom - top, 1.0)
    seps = []
    for x, y0, y1 in rules.v:
        if x <= box[0] + 6 or x >= box[2] - 6:
            continue
        covered = max(0.0, min(y1, bottom) - max(y0, top))
        if covered >= min_cover * span:
            seps.append(x)
    return [m[0] for m in _merge_runs([(x, 0.0, 1.0) for x in seps], 2.5, 0.0)]


# --------------------------------------------------------------------------- parsing


@dataclass
class _Row:
    """A body row before values are parsed."""

    cells: dict[int, list[Cell]]
    top: float
    bottom: float
    label: str = ""
    indent: float | None = None
    path: list[str] = field(default_factory=list)
    suffix: list[str] = field(default_factory=list)
    filled: bool = False
    line: Line | None = None
    end: float = 0.0  # bottom of the last line attached to this row
    parts: list[str] = field(default_factory=list)  # text of each label column
    prefix: list[str] = field(default_factory=list)  # label lines above the numbers
    index: str | None = None  # the S.No. / item number cell, when the table has an index column
    texts: list[str] = field(default_factory=list)  # label cells: heading path joined to the first
    col_suffix: dict[int, list[str]] = field(default_factory=dict)  # wrapped lines of later columns


@dataclass
class _Heading:
    indent: float
    text: str
    had_rows: bool = False
    rank: int = 0  # enumerator depth at one indent: `enum_rank`


@dataclass
class ParsedTable:
    """One table parsed from a region (before ids and continuation links)."""

    bbox: BBox
    caption: str | None
    units: str | None
    header: list[str]
    rows: list[list[Any]]
    printed: list[list[str]]
    notes: list[str]
    flags: list[str]
    plain: str
    continues_next: bool = False
    contd_caption: bool = False
    key: str = ""
    n_cols: int = 0
    row_tops: list[float] = field(default_factory=list)  # y of each body row on the page
    top: float = 0.0
    lead: int = 1  # cells before the values: the label, or the index column and the label


def _fake_header(text: str) -> bool:
    """A header cell no table prints: words of a sentence ('price monitoring', 'national GDP.
    Meanwhile, …') or a chart label with its value ('Whoesale& retail trade, 14.5')."""
    words = text.replace("\uf07d", " ").split()
    sentence = len(words) >= 2 and words[0].isalpha() and words[0][:1].islower()
    return sentence or bool(re.search(r"[A-Za-z],\s+\d+\.\d", text))


def _duplicates(header: list[str], lead: int = 1) -> bool:
    """Two value columns with the same non-empty header: the column cannot be told apart."""
    names = [h for h in header[lead:] if h]
    return len(set(names)) < len(names)


def _spread(header: list[str], lead: int = 1) -> bool:
    """Adjacent value columns whose headers start with the same two words, the first a word
    broken at the line end: the first header line of several columns read as one cell
    ('Unemp- Employed Agricul- Mining', Table 12.10). A group heading printed over columns
    never starts with a broken word."""
    heads = [h.split() for h in header[lead:]]
    return any(
        len(a) >= 3 and len(b) >= 3 and a[:2] == b[:2] and a[0].endswith("-")
        for a, b in zip(heads, heads[1:], strict=False)
    )


_ROW_UNIT_RE = re.compile(
    r"^(?:as\s+(?:a\s+)?(?:%|percent|per\s?cent)|in\s+(?:%|percent|per\s?cent|rs\b|us\s?\$|"
    r"million|billion|trillion|crore|lakh)|%|percent|per\s?cent|rs\b|us\s?\$|\$|million\b|"
    r"billion\b|trillion\b)",
    re.I,
)


# A row the table's note marks as a percentage ('… (Percentage distribution)', the companion
# rows `_label_bracket_rows` writes for Table 12.5).
_PERCENT_ROW_TAIL_RE = re.compile(r"\((?:percentage|percent|%)\s+(?:distribution|share)\)$", re.I)
_OWN_PERCENT_RE = re.compile(r"^(?:as\s+(?:a\s+)?)?(?:%|percent\b|per\s?cent\b)", re.I)
_TAIL_UNIT_RE = re.compile(r"\(([^()]*)\)\s*$")
_CHANGE_HEADING_RE = re.compile(r"^(?:%|percent(?:age)?|per\s?cent)\s+change$", re.I)
# 'Percent Change on Average Basis' over the value columns: a column heading, not the units
_CHANGE_TITLE_RE = re.compile(r"^(?:%|percent(?:age)?|per\s?cent)\s+change\b(?!\s*\()", re.I)


def row_unit(label: str) -> str | None:
    """The unit a row is in, when its label names one; None otherwise. That unit replaces the
    column's and the table's for the row (flag `row_units`). In this order:

    - the row's own label starts with a percent unit: '% of GDP' -> '% of GDP';
    - the nearest heading of its path that is a bracketed qualifier naming a unit
      ('(As percent of GDP) / Domestic Debt' -> 'As percent of GDP') or ends with one
      ('Key FSIs (%) / NPLs to Loans (Gross)' -> 'percent'), is a percent-change band
      ('Percent Change / Apr 25/ Apr 24' -> 'percent') or starts with a percent unit ('As % of
      total expenditures / Defence' -> 'As % of total expenditures').

    The row's own bracketed text ('(growth %)') is not a unit of the row."""
    parts = [p.strip() for p in label.split(" / ")]
    if _OWN_PERCENT_RE.match(parts[-1]):
        return parts[-1]
    if _PERCENT_ROW_TAIL_RE.search(parts[-1]):  # 'Sindh (Percentage distribution)', Table 12.5
        return "percent"
    for text in reversed(parts[:-1]):
        if _CHANGE_HEADING_RE.match(text):
            return "percent"
        if _OWN_PERCENT_RE.match(text):  # 'As % of total expenditures' over the share rows
            return text
        if text.startswith("(") and text.endswith(")") and _ROW_UNIT_RE.match(text[1:-1].strip()):
            return text[1:-1].strip()
        m = _TAIL_UNIT_RE.search(text)
        if m and _ROW_UNIT_RE.match(m.group(1).strip()):
            inner = m.group(1).strip()
            return "percent" if _PERCENT_RE.fullmatch(inner) else inner
    return None


_PERCENT_RE = re.compile(r"%|\bper\s?cent\b|\bpercent(?:age)?\b", re.I)


def column_unit(header: str) -> str | None:
    """The unit a column's header names: a bracketed unit at its end ('Gross Buy (Rs million)'
    -> 'Rs million'), else 'percent' when it says so ('% Change', 'No. of Units % Change',
    'Share (%)'), else None. For a value the order is: the row's `row_unit`, this, then the
    table's `units`."""
    m = re.search(r"\(([^()]*)\)\s*$", header)
    if m and _ROW_UNIT_RE.match(m.group(1).strip()):
        inner = m.group(1).strip()
        return "percent" if _PERCENT_RE.fullmatch(inner) else inner
    return "percent" if _PERCENT_RE.search(header) else None


# The last word of a label that is really a value: a dash, a number with a decimal point or
# thousands commas, or an integer of three digits or more that is not a year ('Units -', 'Life
# Line 3.95', 'Jul-24 77,886.99', 'Afghanistan 181'; not 'FY 2017', 'Q1 2024', 'SDG 1').
_VALUE_TAIL_RE = re.compile(
    r"\s([-–]|[-–]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-–]?\d+\.\d+|[-–]?\d{3,})$"
)
_NUMBER_TAIL_RE = re.compile(r"\s([-–]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-–]?\d+(?:\.\d+)?)$")
_YEAR_RE = re.compile(r"(?:19|20)\d\d")


def value_tail(label: Any, *, strict: bool = True) -> str | None:
    """The value printed at the end of a label cell ('Afghanistan 181' -> '181'), if any.
    Not strict: any number that is not a year ('... (Other than Reserves) 0')."""
    if not isinstance(label, str):
        return None
    m = (_VALUE_TAIL_RE if strict else _NUMBER_TAIL_RE).search(label.split(" / ")[-1])
    if not m or _YEAR_RE.fullmatch(m.group(1)):
        return None
    return m.group(1)


def _return_absorbed_values(
    rows: list[list[Any]],
    printed: list[list[str]],
    lead: int,
    one_line: list[bool] | None = None,
) -> None:
    """A number printed at the end of the last label cell while the first value cell of the
    row is empty, in a column that holds numbers in its other rows, is that cell's value
    ('Afghanistan 181' with no HDI rank -> 'Afghanistan', 181). Only for a label printed on
    the row's own line (`one_line`): a number from another line ('Cotton' / '4000', an axis
    tick of a chart) is not the row's."""
    if lead >= (len(rows[0]) if rows else 0):
        return
    col = [r[lead] for r in rows if r[lead] not in (None, "")]
    if not col or sum(isinstance(v, int | float) for v in col) < 0.8 * len(col):
        return
    for k, (r, prt) in enumerate(zip(rows, printed, strict=True)):
        if one_line is not None and not one_line[k]:
            continue
        tail = value_tail(r[lead - 1], strict=False)
        if tail is None or r[lead] not in (None, ""):
            continue
        value, shown = parse_cell(tail)
        if not isinstance(value, int | float):
            continue
        r[lead - 1] = r[lead - 1][: -len(tail)].rstrip()
        prt[lead - 1] = (
            prt[lead - 1][: -len(tail)].rstrip() if prt[lead - 1].endswith(tail) else (r[lead - 1])
        )
        r[lead], prt[lead] = value, shown


def _column_numbers(vals: list[Any]) -> bool:
    """A row of the column numbers some tables print under their header ('1 2 3 ... 10'), or
    of placeholders only: no value of the table."""
    nums = [v for v in vals if v not in (None, "", "-", "‐", "–")]
    if not nums:
        return True
    ints = all(isinstance(v, int) and not isinstance(v, bool) for v in nums)
    return ints and len(nums) >= 3 and nums == list(range(1, len(nums) + 1))


_PAREN_NOTE_RE = re.compile(r"(?i)[^.;\n]*\bparenthes[ie]s\b[^.;\n]*")
_PAREN_NEGATIVE_RE = re.compile(r"(?i)negative|minus|decline|decrease|deficit|outflow|loss")


def paren_meaning(texts: Iterable[str | None]) -> str | None:
    """What numbers in brackets mean when a table's caption or notes say it is not 'negative':
    'Percentage distribution is given in parenthesis' -> 'Percentage distribution'; 'rates
    shown in parentheses represent Private Sector' -> 'Private Sector'. None when no text says
    so or it says they are negative (the default reading, '(1.2)' -> -1.2)."""
    for text in texts:
        for m in _PAREN_NOTE_RE.finditer(text or ""):
            phrase = normalize_ws(m.group(0))
            if _PAREN_NEGATIVE_RE.search(phrase):
                return None
            got = re.search(r"(?i)\brepresents?\s+(.+)$", phrase) or re.search(
                r"(?i)^\W*(?:note\s*:\s*)?(.+?)\s+(?:is|are)\s+(?:given|shown)\s+in\s+parenthes",
                phrase,
            )
            meaning = got.group(1) if got else "in parentheses"
            meaning = re.split(r"(?i)\s+(?:source|note)s?\s*:", meaning)[0]  # next note line
            return re.sub(r"^[\W_]+|[\W_]+$", "", meaning) or "in parentheses"
    return None


def _label_bracket_rows(
    rows: list[list[Any]], printed: list[list[str]], n_index: int, lead: int, meaning: str
) -> None:
    """A row whose numbers are all in brackets, under a row that is not, is that row's
    companion ('Sindh' 140,914 ... / '(17.7)' ...): it takes the row's label plus the meaning
    of the brackets ('Sindh (Percentage distribution)'). A name wrapped onto the companion
    ('Khyber' / 'Pakhtunkhwa') is joined on both rows."""
    for i in range(1, len(rows)):
        r, p = rows[i], printed[i]
        nums = [j for j in range(lead, len(r)) if isinstance(r[j], int | float)]
        if not nums or not all(p[j].lstrip().startswith("(") for j in nums):
            continue
        prev, pp = rows[i - 1], printed[i - 1]
        pnums = [j for j in range(lead, len(prev)) if isinstance(prev[j], int | float)]
        if not pnums or all(pp[j].lstrip().startswith("(") for j in pnums):
            continue
        cols = list(range(n_index, lead))
        for c in cols:
            own, above = (
                r[c] if isinstance(r[c], str) else "",
                prev[c] if isinstance(prev[c], str) else "",
            )
            if c == cols[0] and own and own != above:
                head, _, leaf = above.rpartition(" / ")
                own_leaf = own.rpartition(" / ")[2]
                joined = own_leaf if own_leaf.startswith(leaf) else f"{leaf} {own_leaf}"
                above = f"{head} / {joined}" if head else joined
                prev[c] = pp[c] = above
            r[c] = p[c] = above
        last = max((c for c in cols if r[c]), default=cols[0])
        r[last] = p[last] = f"{r[last]} ({meaning})"


_ENUM_RE = re.compile(r"^\(?([a-z]|[ivx]{1,4}|\d{1,2}|[A-Z])[).](?:\s|(?=[A-Z]))")


_ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]


def next_in_sequence(a: str, b: str) -> bool:
    """Whether enumerator `b` follows `a` in one style: '14' -> '15', 'a' -> 'b', 'A' -> 'B',
    'i' -> 'ii'."""
    if a.isdigit() and b.isdigit():
        return int(b) == int(a) + 1
    roman = a.lower() in _ROMAN and b.lower() in _ROMAN and a.islower() == b.islower()
    if roman and _ROMAN.index(b.lower()) == _ROMAN.index(a.lower()) + 1:
        return True
    return (
        len(a) == len(b) == 1
        and a.isalpha()
        and a.islower() == b.islower()
        and ord(b) == ord(a) + 1
    )


def enum_rank(text: str, last_letter: str | None = None) -> int:
    """Depth of an enumerated heading when levels share one indent (Table 5.7, 2025-26):
    0 none or a capital ('B.'), 1 a lowercase letter '(c)', 2 a roman numeral '(ii)', 3 a
    number '(1)'. '(i)' right after the letter '(h)' is the letter i."""
    m = _ENUM_RE.match(text)
    if not m:
        return 0
    tok = m.group(1)
    if tok.isdigit():
        return 3
    if tok[0].isupper():
        return 0
    if re.fullmatch(r"[ivx]+", tok):
        follows = last_letter and len(tok) == 1 and ord(tok) == ord(last_letter) + 1
        return 1 if follows else 2
    return 1 if len(tok) == 1 else 0


def lead_columns(flags: list[str]) -> int:
    """Cells before the values in each row: 1 (the label), or the number in the `lead:N` flag
    (the label columns of the table, an index column among them)."""
    return next((int(f[5:]) for f in flags if f.startswith("lead:")), 1)


def row_label(flags: list[str], row: list[Any]) -> str:
    """The row's label: its text cells before the values, the index column left out. The
    heading path and parent rows are in the first of them ('Section / Parent / Item')."""
    skip = 1 if "index_column" in flags else 0
    return " ".join(str(c) for c in row[skip : lead_columns(flags)] if c not in (None, ""))


_LEGEND_RE = re.compile(r"(?:^|\s)([*#^@†‡]+)\s*:\s*(.+?)(?=\s+[*#^@†‡]+\s*:|$)")
_TRAILING_MARK_RE = re.compile(r"[*#^@†‡]+$")


def marker_legends(notes: list[str]) -> dict[str, str]:
    """Footnote legends in a table's notes: '# : 2023 Population Census' -> {'#': '2023 ...'}."""
    out: dict[str, str] = {}
    for line in notes:
        for m in _LEGEND_RE.finditer(line):
            out.setdefault(m.group(1), normalize_ws(m.group(2)))
    return out


def _markers_used(printed: list[list[str]]) -> set[str]:
    """Footnote marks that end a printed cell (a value '1607#' or a label 'Others**')."""
    used: set[str] = set()
    for row in printed:
        for cell in row:
            m = _TRAILING_MARK_RE.search(cell.strip())
            if m:
                used.add(m.group(0))
    return used


# two numbered items in one label: the lines of neighbouring items read into one row
_TWO_ITEMS_RE = re.compile(r"(?:^|\s)\d{1,2}\.\s.*\s\d{1,2}\.(?:\s|$)")
_BASE_YEAR_LINE_RE = re.compile(r"(?i)^\(?\s*base\s+year\b")
_OPERATOR_LINE_RE = re.compile(r"(?i)^(?:plus|less|add|minus)\s*:?$")
_FIRST_ITEM_RE = re.compile(r"(?i)^\(?(?:i|a|1)\)\s|^(?:i|a)\.\s")
_TOTAL_RE = re.compile(r"^(?:grand[- ]|sub[- ]?)?total\b", re.I)
# A short product or series code printed as a whole row label ('LDO', 'HSD', 'JP-1:'): it is
# never the first line of a wrapped label.
_IN_UNITS_RE = re.compile(r"(?i)\bin\s+(?:million|billion|trillion|thousands?|crore|lakh)\b")
_CODE_LABEL_RE = re.compile(r"(?=[A-Z0-9\-/.:&]*[A-Z])[A-Z0-9][A-Z0-9\-/.:&]{0,7}")
# An item named by a code and a description in brackets ('HOBC (Automotive 100 Octane)'): an
# item of the table, not an upper-case section heading.
_CODE_ITEM_RE = re.compile(r"[A-Z][A-Z0-9\-/]{1,7}\s*(\((?=[^()]*[a-z])[^()]*\))")
# A sum of sections: '(A + B + C)', '(I+II+III+IV+V)'.
_SECTION_SUM_RE = re.compile(r"\(\s*((?:[A-Z]{1,4})(?:\s*\+\s*[A-Z]{1,4})+)\s*\)")
_SECTION_TOK_RE = re.compile(r"^\(?([A-Z]{1,4})[).]\s")  # 'A. AGRICULTURE', 'IV. DEVELOPMENT'


def sums_sections(label: str, earlier: Iterable[str]) -> bool:
    """Whether a row label sums sections printed above it: 'GDP {Total of GVA at bp (A + B +
    C)}' after 'C. SERVICES SECTOR', 'TOTAL (I+II+III+IV+V)' after 'V. GLOBAL BONDS'. 'Federal
    (A+B)' printed over 'A. Direct Taxes' and 'B. Indirect Taxes' sums the rows below it."""
    m = _SECTION_SUM_RE.search(label)
    if not m:
        return False
    last = m.group(1).replace(" ", "").split("+")[-1]
    return any((t := _SECTION_TOK_RE.match(e)) and t.group(1) == last for e in earlier)


_FIRST_PERIOD_RE = re.compile(r"^(.*\S)\s*:\s*[A-Za-z]{1,2}\d$")
_PERIOD_CODE_RE = re.compile(r"[A-Za-z]{1,2}\d")

INDENT_STEP = 6.0  # pt: label x positions closer than this are the same indent level
MAX_INDENT_LEVELS = 4


def _indent_levels(indents: list[float]) -> dict[float, int] | None:
    """Indent level (0, 1, 2 ...) of each label x position that forms a hierarchy, or None.

    Not a hierarchy: one level only, so many x positions (centred or ragged labels) that they
    are not indents, or a level that holds a single row (a centred 'Total' among left-aligned
    rows). Positions of such a level are left out of the result."""
    xs = sorted(set(indents))
    if len(xs) < 2:
        return None
    level, last = 0, xs[0]
    out = {xs[0]: 0}
    for x in xs[1:]:
        if x - last > INDENT_STEP:
            level += 1
        out[x] = level
        last = x
    if not 1 <= level < MAX_INDENT_LEVELS:
        return None
    rows_at = Counter(out[x] for x in indents)
    return {x: lv for x, lv in out.items() if rows_at[lv] >= 2} or None


def caption_key(caption: str | None) -> str:
    """'TABLE 2.1 B (Continued) BASIC DATA' -> '2.1b'; '' when there is no table number."""
    if not caption:
        return ""
    m = _CAPTION_KEY_RE.match(caption)
    if not m:
        return ""
    return (m.group(1) + (m.group(2) or "")).lower()


def _clean_caption(text: str) -> str:
    text = clean_unicode(text).replace("\\", " ")
    text = CONTD_RE.sub(" ", text)
    return normalize_ws(text)


def _units_from_caption(caption: str) -> str | None:
    m = re.search(r"\(([^()]*)\)\s*$", caption)
    if m and UNITS_RE.match(m.group(1).strip()):
        return f"({m.group(1).strip()})"
    return None


_FRAGMENT_RE = re.compile(r"^[\d,.]+$")


def mend_numbers(words: list[Word]) -> list[Word]:
    """Join the pieces of a number that the PDF set in two text runs ('3' + '8,265,131').
    Only touching pieces (gap <= 1.2 pt) whose concatenation is a number are joined."""
    out: list[Word] = []
    for w in sorted(words, key=lambda w: w.x0):
        if out:
            prev = out[-1]
            if (
                w.x0 - prev.x1 <= 1.2
                and _FRAGMENT_RE.match(prev.text)
                and _FRAGMENT_RE.match(w.text)
                and is_number_token(prev.text + w.text)
            ):
                out[-1] = Word(prev.text + w.text, prev.x0, w.x1, prev.top, w.bottom, prev.size)
                continue
        out.append(w)
    return out


MARK_REACH = 12.0  # pt: a footnote symbol this far outside the region still marks its number


def _is_mark(w: Word) -> bool:
    return bool(w.text) and set(w.text) <= MARKER_SYMBOLS


def region_lines(words: list[Word], box: BBox) -> list[Line]:
    """Upright words inside the region as lines. Footnote superscripts are dropped, except a
    footnote symbol ('#', '*') right after a number: it belongs to that value."""
    inside = [w for w in words if _inside(w, box)]
    numbers = [w for w in inside if word_kind(w.text) in ("num", "year")]
    for w in words:  # a mark just past the region's right edge, next to a number inside it
        if (
            _is_mark(w)
            and box[2] < (w.x0 + w.x1) / 2 <= box[2] + MARK_REACH
            and box[1] <= w.cy <= box[3]
            and any(abs(n.cy - w.cy) <= 6 and 0 <= w.x0 - n.x1 <= 6 for n in numbers)
        ):
            inside.append(w)
    lines = []
    for line in group_lines(inside):
        marks = [
            w
            for k, w in enumerate(line.words)
            if _is_mark(w) and k > 0 and word_kind(line.words[k - 1].text) in ("num", "year")
        ]
        _drop_superscripts(line)
        for w in marks:
            if w not in line.words:
                line.words.append(w)
        line.words.sort(key=lambda w: w.x0)
        if line.words:
            lines.append(Line(mend_numbers(line.words)))
    return sorted(lines, key=lambda ln: ln.top)


def _value_cells(cells: list[Cell], label_limit: float) -> list[Cell]:
    """Cells of a line that are body values: numbers or placeholders right of the label
    column. A cell left of the limit is a row label even when it is a number."""
    return [c for c in cells if c.kind == "value" and c.x0 >= label_limit - 3.0]


def label_limit(lines: list[Line]) -> tuple[float, float | None, float]:
    """Left edge of the first value column, the typical x of a row label, and where labels end.

    The first is where the first value cell sits to the right of the row label (the first
    text or period cell) in lines that have both; the second is that label's median x; the
    third is the right edge reached by 90 % of the labels."""
    xs, lx, ex = [], [], []
    for ln in lines:
        cells = line_cells(ln)
        label = next((i for i, c in enumerate(cells) if c.kind in ("text", "period")), None)
        if label is None:
            continue
        vals = [c for c in cells[label + 1 :] if c.kind == "value"]
        if vals:
            xs.append(vals[0].x0)
            lx.append(cells[label].x0)
            ex.append(cells[label].x1)
    if not xs:
        # No line has a label next to values: a table of bare numbers (its row labels are on
        # another page) when it has rows of values, else there is nothing to parse.
        has_values = sum(1 for ln in lines if any(c.kind == "value" for c in line_cells(ln)))
        return (float("-inf") if has_values >= 2 else float("inf")), None, float("inf")
    return float(np.percentile(xs, 5)), float(statistics.median(lx)), float(np.percentile(ex, 90))


@dataclass
class _Block:
    header_lines: list[Line]
    items: list[tuple[str, Line]] = field(default_factory=list)  # data | label | cont


class RegionParser:
    """Parse one table region into one or more tables."""

    def __init__(
        self,
        words: list[Word],
        region: Region,
        rules: Rules,
        *,
        min_fill: float = 0.5,
        max_ragged: float = 0.25,
        doc_words: Counter[str] | None = None,
        vertical: list[Word] | None = None,
    ) -> None:
        self.region = region
        self.paren_meaning: str | None = None  # brackets are not negatives (`paren_meaning`)
        self.box = region.bbox
        self.vertical = [w for w in vertical or [] if _inside(w, self.box)]
        self.rules = rules
        self.min_fill = min_fill
        self.max_ragged = max_ragged
        self.all_words = words
        self.vocab = {re.sub(r"\W", "", w.text.lower()) for w in words}
        # how often each word is printed in the document (on this page without `doc_words`):
        # the second half of a word broken at a line end ('loyed', 'ture') is rare, a real word
        # ('adjusted', 'on', 'term') recurs
        self.word_count = doc_words or Counter(re.sub(r"\W", "", w.text.lower()) for w in words)
        self.lines = region_lines(words, self.box)
        self.cells = {id(ln): line_cells(ln) for ln in self.lines}
        self.limit, self.label_x, self.label_end = label_limit(self.lines)
        first_cells = [self.cells[id(ln)][0].x0 for ln in self.lines if self.cells[id(ln)]]
        self.left_edge = min(first_cells, default=self.box[0])
        self.flags: list[str] = []

    # ---- helpers

    def _is_data(self, line: Line) -> bool:
        cells = self.cells[id(line)]
        return bool(_value_cells(cells, self.limit)) and not self._header_leak(cells)

    def _header_leak(self, cells: list[Cell]) -> bool:
        """Header text that holds numbers ('Groups | Rs 8000 | 12000'): words over the value
        columns but nothing where the row labels are."""
        if self.label_x is None or self.limit in (float("inf"), float("-inf")):
            return False
        text_over = any(c.kind == "text" and c.cx >= self.limit - 2 for c in cells)
        has_label = any(
            c.kind in ("text", "period") and abs(c.x0 - self.label_x) <= 40 for c in cells
        )
        return text_over and not has_label

    def plain_text(self) -> str:
        """The region line by line, cells joined with ' | '. A sideways label (`vertical_runs`)
        is a line of its own before the first line of the ruled cell it sits in (the rows it
        heads), else before the first line beside it."""
        texts = [" | ".join(c.text for c in self.cells[id(ln)]) for ln in self.lines]
        before: dict[int, list[str]] = {}
        for w in self.vertical:
            top, bottom = w.top, w.bottom
            cx = (w.x0 + w.x1) / 2
            across = [y for y, a, b in self.rules.h if a - 1 <= cx <= b + 1]
            above = max((y for y in across if y <= w.top + 1), default=None)
            below = min((y for y in across if y >= w.bottom - 1), default=None)
            if above is not None and below is not None:
                top, bottom = above, below
            at = next((i for i, ln in enumerate(self.lines) if top <= ln.cy <= bottom), None)
            if at is None:
                at = next(
                    (i for i, ln in enumerate(self.lines) if ln.top >= w.top - 1), len(self.lines)
                )
            before.setdefault(at, []).append(w.text)
        out: list[str] = []
        for i, text in enumerate([*texts, None]):
            out.extend(before.get(i, []))
            if text is not None:
                out.append(text)
        return "\n".join(out)

    def grid_rows_text(self) -> str | None:
        """A table drawn as a grid of text cells (Table 2.16, Water Sector Projects): one line
        per ruled row, the text of each cell (between the column rules) joined line by line,
        cells separated by ' | '; lines above the first and under the last rule as printed.
        None when the region has no such grid (column rules over half its height, three or
        more rules across most of its width)."""
        x0, top, x1, bottom = self.box
        width, height = x1 - x0, bottom - top

        def cluster(vals: list[float]) -> list[float]:
            out: list[float] = []
            for v in sorted(vals):
                if not out or v - out[-1] > 3:
                    out.append(v)
            return out

        inner = [(x, y0, y1) for x, y0, y1 in self.rules.v if x0 + 6 < x < x1 - 6]
        xs = [
            x
            for x in cluster([x for x, _, _ in inner])
            if sum(y1 - y0 for vx, y0, y1 in inner if abs(vx - x) <= 3) >= 0.5 * height
        ]
        ys = cluster([y for y, a, b in self.rules.h if b - a >= 0.6 * width])
        if not xs or len(ys) < 3:
            return None
        edges = [x0 - 1, *xs, x1 + 1]

        def text_of(lines: list[Line]) -> list[str]:
            return [" | ".join(c.text for c in self.cells[id(ln)]) for ln in lines]

        out = [  # the caption is the table's caption, not a line of its text
            t
            for t in text_of([ln for ln in self.lines if ln.cy < ys[0]])
            if not TABLE_CAPTION_RE.match(t.replace(" | ", " "))
        ]
        slabs = [
            [ln for ln in self.lines if a < ln.cy < b] for a, b in zip(ys, ys[1:], strict=False)
        ]
        filled = [rows for rows in slabs if rows]
        if len(filled) < 3 or max(map(len, filled)) > 8 or sum(map(len, filled)) > 4 * len(filled):
            return None  # the rules do not separate the rows (Table 7.2: months between rules)
        for rows in filled:
            cells = [
                normalize_ws(
                    " ".join(
                        w.text for ln in rows for w in ln.words if lo <= (w.x0 + w.x1) / 2 < hi
                    )
                )
                for lo, hi in zip(edges, edges[1:], strict=False)
            ]
            row = " | ".join(cells)
            if not TABLE_CAPTION_RE.match(row.replace(" | ", " ")):  # caption inside the rule
                out.append(row)
        out += text_of([ln for ln in self.lines if ln.cy > ys[-1]])
        return "\n".join(out)

    def fallback(self, reason: str, caption: str | None = None) -> list[ParsedTable]:
        caption = caption if caption is not None else self.region.caption
        plain = None
        if reason in ("too_few_rows", "few_numbers", "text_grid"):  # text cells: rows by grid
            plain = self.grid_rows_text()
        plain = plain or self.plain_text()
        return [
            ParsedTable(
                bbox=self.box,
                caption=_clean_caption(caption) if caption else None,
                units=None,
                header=[],
                rows=[],
                printed=[],
                notes=[],
                flags=["fallback_plain", f"reason:{reason}"],
                plain=plain,
                continues_next=bool(CONTD_RE.search(plain)),
                contd_caption=bool(caption and CONTD_RE.search(caption)),
                key=caption_key(caption),
            )
        ]

    # ---- title, body, tail

    def _split_title(self, zone: list[Line]) -> tuple[str | None, list[str], list[Line]]:
        """Caption text, units found on the caption line, and the lines left for the header."""
        caption: str | None = None
        units: list[str] = []
        i = 0
        if zone:
            first = self.cells[id(zone[0])]
            text0 = first[0].text if first else ""
            if TABLE_CAPTION_RE.match(" ".join(c.text for c in first)):
                caption = text0
                for extra in first[1:]:
                    if UNITS_RE.match(extra.text) or extra.x0 - first[0].x1 > 40:
                        units.append(extra.text)
                    else:
                        caption += " " + extra.text
                i = 1
                cap_x0 = first[0].x0
                number_only = bool(
                    re.fullmatch(
                        r"(?i)\s*table\s*[-:.]?\s*[\d.]+\s*\(?[a-z]?\)?",
                        CONTD_RE.sub(" ", caption).strip(),
                    )
                )
                cap_upper = _is_upper(caption)
                while i < len(zone) and i < 4:
                    cells = self.cells[id(zone[i])]
                    if (
                        len(cells) == 1
                        and abs(cells[0].x0 - cap_x0) <= 3
                        and not self._is_data(zone[i])
                        and (number_only or (cap_upper and _is_upper(cells[0].text)))
                    ):
                        caption += " " + cells[0].text
                        number_only = False
                        i += 1
                    else:
                        break
        while i < len(zone):  # a lone units line under the caption: 'Million Tonnes'
            cells = self.cells[id(zone[i])]
            if len(cells) != 1 or self._is_data(zone[i]) or not self._units_line(zone[i], cells[0]):
                break
            if i + 1 < len(zone) and self._group_title(zone[i], cells[0]):
                break
            units.append(clean_unicode(cells[0].text))
            i += 1
        if caption is None and self.region.caption:
            caption = self.region.caption
        if caption is None:
            caption, above_units = self._caption_above()
            if not units:
                units.extend(above_units)
        if caption:
            caption = _clean_caption(caption)
            tail_units = _units_from_caption(caption)
            if tail_units and not units:
                units.append(tail_units)
        return caption, units, zone[i:]

    def _caption_above(self, max_gap: float = 60.0) -> tuple[str | None, list[str]]:
        """A 'TABLE n' caption printed just above the region (the region starts at the units
        line): the caption line and at most two short title lines under it, with nothing else
        between them and the table ('TABLE 9.7' / 'TOTAL LOANS AND CREDITS CONTRACTED'). A
        units line among them ('US$ Million') is returned as units, not caption."""
        x0, top, x1, _ = self.box
        near = [
            w
            for w in self.all_words
            if top - max_gap <= w.cy < top - 1 and x0 - 8 <= w.x0 and w.x1 <= x1 + 8
        ]
        texts = [normalize_ws(" ".join(w.text for w in ln.words)) for ln in group_lines(near)]
        starts = [k for k, t in enumerate(texts) if TABLE_CAPTION_RE.match(t)]
        if not starts:
            return None, []
        k = starts[-1]
        units = [t for t in texts[k + 1 :] if STRICT_UNITS_RE.match(t) and len(t.split()) <= 4]
        title = [t for t in texts[k + 1 :] if t not in units]
        if len(title) > 2 or any(
            len(t.split()) > 12 or sum(is_number_token(x) for x in t.split()) > 1 for t in title
        ):
            return None, []
        return " ".join([texts[k], *title]), units

    def _lead_in(self, max_gap: float = 120.0) -> str | None:
        """The prose that introduces a list printed as a 'table' in a text column: the
        paragraph right above the region when it ends with ':' ('In the month of Ramadan,
        2025, District Administration has taken action as under:'), with the paragraph
        before it ('… throughout Balochistan.'). None when no such sentence is there."""
        x0, top, x1, _ = self.box
        near = [
            w
            for w in self.all_words
            if top - max_gap <= w.cy < top - 1 and x0 - 8 <= w.x0 and w.x1 <= x1 + 8
        ]
        lines = group_lines(near)
        if not lines:
            return None
        paras: list[list[Line]] = [[lines[0]]]
        for prev, ln in zip(lines, lines[1:], strict=False):
            if ln.top - prev.bottom > 0.4 * ln.size:  # more space than between lines
                paras.append([])
            paras[-1].append(ln)
        text = [
            normalize_ws(clean_unicode(" ".join(w.text for ln in p for w in ln.words)))
            for p in paras
        ]
        if top - paras[-1][-1].bottom > 2 * paras[-1][-1].size or not text[-1].endswith(":"):
            return None
        return " ".join(text[-2:]).replace("\uf07d", "").strip()

    def _heading_above(self, max_gap: float = 16.0) -> Line | None:
        """A section heading printed just above a region without caption or header (the region
        starts at the first row): one line within `max_gap` pt, inside the label column, in
        capitals or ending with a bracketed unit ('TRADE AND PAYMENTS (growth %)', indicators
        p3), and no caption, units or 'Contd.' line."""
        x0, top, x1, _ = self.box
        near = [
            w
            for w in self.all_words
            if top - max_gap <= w.cy < top - 1 and x0 - 8 <= w.x0 and w.x1 <= x1 + 8
        ]
        lines = group_lines(near)
        if len(lines) != 1:
            return None
        line = Line(mend_numbers(lines[0].words))
        cells = line_cells(line)
        text = normalize_ws(clean_unicode(" ".join(c.text for c in cells)))
        if not cells or line.x1 > self.limit + 2 or any(c.kind == "value" for c in cells):
            return None
        if TABLE_CAPTION_RE.match(text) or STRICT_UNITS_RE.match(text) or CONTD_RE.search(text):
            return None
        if not (_is_upper(text) or re.search(r"%\s*\)$", text)):
            return None
        if self.region.caption and text in self.region.caption:
            return None
        self.cells[id(line)] = cells
        return line

    def _lines_above(self, max_gap: float = 120.0) -> list[Line]:
        """Header lines printed above the region (a table whose header the region detection
        left out): short cells over the value columns, within `max_gap` pt of the first row,
        no prose and no page title."""
        x0, top, x1, _ = self.box
        near = [
            w
            for w in self.all_words
            if top - max_gap <= w.cy < top - 1 and x0 - 5 <= (w.x0 + w.x1) / 2 <= x1 + 5
        ]
        lines = []
        for line in group_lines(near):
            _drop_superscripts(line)
            if not line.words:
                continue
            line = Line(mend_numbers(line.words))
            cells = line_cells(line)
            body = statistics.median(ln.size for ln in self.lines)
            if line.size > 1.35 * body or any(s.prose for s in line.segments):
                continue
            lines.append((line, cells))
        lines.sort(key=lambda lc: -lc[0].top)
        kept: list[Line] = []
        edge = top
        for line, cells in lines:  # walk upwards while the lines stay close together
            if edge - line.bottom > 45 or any(c.kind == "value" for c in cells):
                break  # a line of numbers is the end of another table, not a header
            kept.append(line)
            self.cells[id(line)] = cells
            edge = line.top
        kept.sort(key=lambda ln: ln.top)
        if not any(len(self.cells[id(ln)]) >= 2 and self._over_values(ln) for ln in kept):
            return []
        return kept

    def rows_below(self, other_boxes: list[BBox]) -> int:
        """Table rows printed right under the region that the region left out (the finder
        stopped early): lines directly below it, at the table's row pitch, that hold value
        cells under the table's columns. Source and note lines end the search."""
        x0, _, x1, bottom = self.box
        tops = [ln.top for ln in self.lines]
        gaps = [b - a for a, b in zip(tops, tops[1:], strict=False) if 3 < b - a < 40]
        pitch = statistics.median(gaps) if gaps else 12.0
        near = [
            w
            for w in self.all_words
            if w.cy > bottom
            and x0 - 5 <= (w.x0 + w.x1) / 2 <= x1 + 5
            and not any(_inside(w, b) for b in other_boxes)
        ]
        edge, found = bottom, 0
        right = x0 + 0.35 * (x1 - x0)
        for line in group_lines(near):
            if line.top - edge > 1.8 * pitch:
                break
            cells = line_cells(Line(mend_numbers(line.words)))
            text = " ".join(c.text for c in cells)
            tabular = len(cells) >= 2 and any(c.kind == "value" and c.cx >= right for c in cells)
            if NOTE_RE.match(text) or not tabular:
                break  # a note, prose, a page number: the table really ended above
            found += 1
            edge = line.bottom
        return found

    def _units_line(self, line: Line, cell: Cell) -> bool:
        """A single-cell line of units text above the header: in brackets, right-aligned, or
        right above a long horizontal rule (the table's top border)."""
        text = cell.text.strip()
        flush_right = cell.cx >= self.box[0] + 0.6 * (self.box[2] - self.box[0])
        if _IN_UNITS_RE.search(text) and len(text.split()) <= 4 and flush_right:
            return True  # 'Population in million' flush right (Table 12.2)
        if not UNITS_RE.match(text) or _CHANGE_TITLE_RE.match(text):
            return False  # 'Percent Change' names what a column holds (Table 7.7)
        if STRICT_UNITS_RE.match(text) and len(text.split()) <= 4:
            return True  # 'Rs million', 'Million Hectares', 'In nos.' under the caption
        width = self.box[2] - self.box[0]
        right_aligned = cell.cx >= self.box[0] + 0.6 * width
        bracketed = text.startswith("(") and text.endswith(")")
        rule_below = any(
            0 <= y - line.bottom <= 30 and x1 - x0 >= 0.8 * width for y, x0, x1 in self.rules.h
        )
        return right_aligned or bracketed or rule_below

    def _group_title(self, line: Line, cell: Cell) -> bool:
        """A lone cell inside the table's frame (a full-width rule right above it) with a short
        rule right under it, over some of the columns only, is a column-group heading ('%
        Change' over the last two columns), not the table's units, which sit above the frame."""
        width = self.box[2] - self.box[0]
        framed = any(
            0 <= line.top - y <= 14 and x1 - x0 >= 0.8 * width for y, x0, x1 in self.rules.h
        )
        return framed and any(
            -3 <= y - line.bottom <= 8  # an underline may cut into the line box
            and x1 - x0 < 0.7 * width
            and x0 - 6 <= cell.x0
            and cell.x1 <= x1 + 6
            for y, x0, x1 in self.rules.h
        )

    def _notes(self, tail: list[Line]) -> tuple[list[str], list[Line]]:
        """Source/note lines under the body; label-only lines that are not notes stay."""
        notes: list[str] = []
        rest: list[Line] = []
        for ln in tail:
            text = " ".join(c.text for c in self.cells[id(ln)])
            if NOTE_RE.match(text) or CONTD_RE.search(text):
                text = normalize_ws(CONTD_RE.sub(" ", text))
                if text:
                    notes.append(text)
            else:
                rest.append(ln)
        return notes, rest

    # ---- main entry

    def parse(self) -> list[ParsedTable]:
        if not self.lines:
            return self.fallback("empty")
        data_idx = [i for i, ln in enumerate(self.lines) if self._is_data(ln)]
        if len(data_idx) < MIN_DATA_ROWS:
            return self.fallback("too_few_rows")
        first, last = data_idx[0], data_idx[-1]
        caption, units, header_lines = self._split_title(self.lines[:first])
        if not any(self._header_row(ln) for ln in header_lines):
            header_lines = [*self._lines_above(), *header_lines]
        tail_notes, tail_rest = self._notes(self.lines[last + 1 :])
        self.paren_meaning = paren_meaning([caption, self.region.caption, *tail_notes])
        continues = any(
            CONTD_RE.search(" ".join(c.text for c in self.cells[id(ln)]))
            for ln in self.lines[last + 1 :]
        )
        body = self.lines[first : last + 1] + tail_rest
        blocks = self._blocks(header_lines, body)
        heads = blocks[0].header_lines
        if heads and len(self.cells[id(heads[-1])]) == 1:
            last = self.cells[id(heads[-1])][0].text
            shape = re.sub(r"[^a-z]", "", last.lower())
            repeated = any(  # '2017 Census' heads a later block of rows: a section heading
                kind == "label" and re.sub(r"[^a-z]", "", ln.text.lower()) == shape
                for b in blocks
                for kind, ln in b.items
            )
            if shape and repeated and not _UNIT_WORD_RE.search(last):
                blocks[0].header_lines = heads[:-1]
                blocks[0].items.insert(0, ("label", heads[-1]))  # '1998 Census*' (Table 12.4)
        if caption is None and not any(ln.top >= self.box[1] for ln in blocks[0].header_lines):
            above = self._heading_above()  # the header, if any, was printed above it
            if above is not None and all(above.top > ln.bottom for ln in blocks[0].header_lines):
                # 'TRADE AND PAYMENTS (growth %)' over the first rows of a continued page
                blocks[0].items.insert(0, ("label", above))
        tables: list[ParsedTable] = []
        for k, block in enumerate(blocks):
            tab = self._parse_block(block, k, len(blocks))
            if tab is None:
                return self.fallback("unresolved", caption)
            tables.append(tab)
        for prev, tab in zip(tables, tables[1:], strict=False):
            if not any(tab.header) and len(prev.header) == len(tab.header) and any(prev.header):
                tab.header = list(prev.header)
                tab.flags = [f for f in tab.flags if f != "no_header"] + ["header_inherited"]
        # region-level facts go on the right tables
        tables[0].caption = caption
        tables[0].key = caption_key(caption)
        tables[0].contd_caption = bool(
            self.region.caption and CONTD_RE.search(self.region.caption)
        ) or bool(caption and re.search(r"(?i)continued|contd", self.region.caption or ""))
        for t in tables[1:]:
            t.caption = caption
            t.key = tables[0].key
        if units:
            above = " ".join(units)
            own = tables[0].units
            if own and _BASE_YEAR_LINE_RE.match(own) and not _BASE_YEAR_LINE_RE.match(above):
                # '(Rs/Unit)' above the header and '(Base Year : 2007-08 = 100)' under it
                # (Table 7.4 (D)): keep both, the unit first
                tables[0].units = f"{above} {own}"
            else:
                tables[0].units = own or above
        for t in tables[1:]:
            t.units = t.units or (tables[0].units if len(blocks) == 1 else None)
        tables[-1].notes = tail_notes
        tables[-1].continues_next = continues
        if len(tables) > 1:
            for t in tables:
                t.flags.append("stacked")
        # partition the region: boundaries halfway between consecutive blocks
        tops = [self._block_top(b) for b in blocks]
        bottoms = [self._block_bottom(b) for b in blocks]
        edges = [self.box[1]]
        for k in range(len(blocks) - 1):
            edges.append((bottoms[k] + tops[k + 1]) / 2)
        edges.append(self.box[3])
        for k, t in enumerate(tables):
            t.bbox = (self.box[0], edges[k], self.box[2], edges[k + 1])
        bad = self._check(tables)
        if bad:
            return self.fallback(bad, caption)
        if caption is None and any(
            _fake_header(h) for t in tables if "no_row_labels" not in t.flags for h in t.header
        ):
            # a list in a prose column or chart labels read as a table (pes2425_ch07 p14):
            # plain text, under the sentence that introduces it when there is one
            return self.fallback("prose_header", self._lead_in())
        text_table = all("aligned" in t.flags and len(t.header) - t.lead <= 1 for t in tables)
        # read by alignment, but drawn column rules run through its labels: a grid of text
        # cells whose columns the labels merged (Table 16.6, pes2425_ch16 p15)
        if text_table and self._rules_cross_labels() and self.grid_rows_text():
            return self.fallback("text_grid", caption)
        return tables

    def _rules_cross_labels(self) -> bool:
        """Two or more drawn column rules, each over half the region's height, between the
        left edge of the labels and the first value column."""
        height = self.box[3] - self.box[1]
        if self.limit in (float("inf"), float("-inf")):
            return False
        xs: dict[int, float] = {}
        for x, y0, y1 in self.rules.v:
            if self.left_edge + 6 < x < self.limit - 6:
                xs[round(x)] = xs.get(round(x), 0.0) + (y1 - y0)
        cols = [x for x, h in xs.items() if h >= 0.5 * height]
        return len({round(x / 4) for x in cols}) >= 2

    # ---- blocks

    def _blocks(self, header_lines: list[Line], body: list[Line]) -> list[_Block]:
        """Split the body at header blocks: text lines over value columns, followed by rows."""
        body_size = statistics.median(ln.size for ln in body) if body else 10.0
        header_lines = [ln for ln in header_lines if ln.size <= 1.35 * body_size]
        last_over = max(
            (i for i, ln in enumerate(header_lines) if self._over_values(ln)), default=-1
        )
        lead_labels = header_lines[last_over + 1 :]
        header_lines = header_lines[: last_over + 1]
        if header_lines:  # 'Year' under 'Financial', beside the 'MMFB NRSP-B …' line (16.11)
            beside = [
                ln
                for ln in lead_labels
                if ln.top < header_lines[-1].bottom
                and len(ln.text.split()) <= 2
                and not _ENUM_RE.match(ln.text)  # 'a. Paris Club Countries' heads rows
                and not _is_upper(ln.text)
            ]
            header_lines = [*header_lines, *beside]
            lead_labels = [ln for ln in lead_labels if ln not in beside]
        blocks = [_Block(header_lines, [("label", ln) for ln in lead_labels])]
        pending: list[Line] = []
        cur = blocks[0]
        for line in body:
            if self._is_data(line):
                if pending and any(self._header_row(ln) for ln in pending) and cur.items:
                    cur = _Block(pending)
                    blocks.append(cur)
                    pending = []
                else:
                    for ln in pending:
                        cur.items.append(("label", ln))
                    pending = []
                cur.items.append(("data", line))
                continue
            if self._over_values(line) or pending:
                pending.append(line)
            else:
                cur.items.append(("label", line))
        for ln in pending:  # trailing text lines: labels of the last row
            cur.items.append(("label", ln))
        for blk in blocks[1:]:  # a new block's header ends at its last line over the values
            idx = max(
                (i for i, ln in enumerate(blk.header_lines) if self._over_values(ln)), default=-1
            )
            blk.items = [("label", ln) for ln in blk.header_lines[idx + 1 :]] + blk.items
            blk.header_lines = blk.header_lines[: idx + 1]
        return blocks

    def _header_row(self, line: Line) -> bool:
        """A line of two or more cells over the value columns: column headers. A single cell
        (a section title such as 'REPAYMENTS', a units bar) is not."""
        cells = self.cells[id(line)]
        return len(cells) >= 2 and self._over_values(line)

    def _over_values(self, line: Line) -> bool:
        """A text line with a cell over the value columns (header text, bar, wrapped text). A
        heading that starts at the left edge of the label column is not one, however long."""
        if self.limit == float("inf"):
            return False
        for c in self.cells[id(line)]:
            starts_left = c.x0 < self.limit - 12 and c.x0 <= self.left_edge + 30
            right_of_labels = c.x0 >= self.label_end + 2
            if (c.cx >= self.limit - 2 or right_of_labels) and not starts_left:
                return True
        return False

    def _block_top(self, block: _Block) -> float:
        lines = [*block.header_lines, *(ln for _, ln in block.items)]
        return min(ln.top for ln in lines)

    def _block_bottom(self, block: _Block) -> float:
        lines = [*block.header_lines, *(ln for _, ln in block.items)]
        return max(ln.bottom for ln in lines)

    # ---- one block

    def _parse_block(self, block: _Block, index: int, n_blocks: int) -> ParsedTable | None:
        data_lines = [ln for kind, ln in block.items if kind == "data"]
        if not data_lines:
            return None
        all_cells = [c for ln in data_lines for c in self.cells[id(ln)]]
        left, right = self.box[0] - 1, self.box[2] + 1
        top = min(ln.top for ln in data_lines)
        bottom = max(ln.bottom for ln in data_lines)
        flags: list[str] = []
        bands: list[Band] | None = None
        seps = vertical_separators(self.rules, self.box, top, bottom)
        if seps:
            bands = ruled_bands(seps, all_cells, left, right)
            if bands:
                flags.append("ruled")
        if not bands:
            intervals = [
                (r, c.x0, c.x1)
                for r, ln in enumerate(data_lines)
                for c in self.cells[id(ln)]
                if c.kind != "marker"
            ]
            bands = aligned_bands(intervals, len(data_lines), left, right)
            flags.append("aligned")
        if len(bands) < 2:
            return None
        n_lab, n_index = self._n_label(data_lines, bands)
        if n_lab >= len(bands) or (n_lab == 0 and len(bands) < 2):
            return None
        lead = max(n_lab, 1)  # cells before the values: every label column stays a column
        rows = self._rows(block, bands, n_lab, n_index)
        header, units = self._header(block, bands, n_lab, n_index=n_index)
        if _duplicates(header, lead) or _spread(header, lead):
            retry, retry_units = self._header(
                block, bands, n_lab, split_words=True, n_index=n_index
            )
            if not _duplicates(retry, lead):
                header, units = retry, retry_units
        values, printed, extra = self._values(rows, bands, n_lab)
        flags.extend(extra)
        if not any(header):
            flags.append("no_header")
        table_units = " ".join(units) if units else None
        out_rows: list[list[Any]] = []
        out_printed: list[list[str]] = []
        one_line: list[bool] = []  # the label is printed on the row's own line only
        for k, (row, vals, prt) in enumerate(zip(rows, values, printed, strict=True)):
            head: list[Any] = [*([row.index or None] if n_index else []), *row.texts]
            if k == 0 and not any(head) and _column_numbers(vals):
                continue  # '1 2 3 ... 10' printed under the header: column numbers, not data
            out_rows.append([*head, *vals])
            out_printed.append([*(h or "" for h in head), *prt])
            one_line.append(not row.suffix and not row.prefix)
        _return_absorbed_values(out_rows, out_printed, lead, one_line)
        if self.paren_meaning:
            flags.append("brackets_not_negative")
            _label_bracket_rows(out_rows, out_printed, n_index, lead, self.paren_meaning)
        if lead > 1:
            flags.append(f"lead:{lead}")
        if n_index:
            flags.append("index_column")
        if any(row_unit(r[n_index]) for r in out_rows):
            flags.append("row_units")
        if any(r.filled for r in rows):
            flags.append("label_filled")
        if n_lab == 0:
            flags.append("no_row_labels")
        pt = ParsedTable(
            bbox=self.box,
            caption=None,
            units=table_units,
            header=header,
            rows=out_rows,
            printed=out_printed,
            notes=[],
            flags=flags,
            plain=self._block_plain(block),
            n_cols=len(bands) - n_lab + lead,
            row_tops=[r.top for r in rows],
            top=top,
            lead=lead,
        )
        return pt

    def _block_plain(self, block: _Block) -> str:
        lines = sorted([*block.header_lines, *(ln for _, ln in block.items)], key=lambda ln: ln.top)
        return "\n".join(" | ".join(c.text for c in self.cells[id(ln)]) for ln in lines)

    # ---- header

    def _shade_span(
        self,
        cell: Cell,
        line: Line,
        *,
        n_cells: int,
        underline: bool,
        value: tuple[float, float],
        cells: list[Cell] | None = None,
    ) -> tuple[float, float] | None:
        """x-range of the narrowest shaded bar behind, or underline below, this header cell.
        A bar or rule that spans every value column is the table's frame, not a group header,
        unless it holds a single cell. An underline only counts when more header text sits
        below it (the bottom border of the header is not a group underline)."""
        best: tuple[float, float] | None = None
        width = self.box[2] - self.box[0]
        full = value[1] - value[0]

        def spans_all(x0: float, x1: float) -> bool:
            return x0 <= value[0] + 6 and x1 >= value[1] - 6 and full > 0

        def crowded(x0: float, x1: float) -> bool:
            """Several cells of the line sit in the bar: it is a row bar, not a group header."""
            return cells is not None and sum(1 for c in cells if x0 - 2 <= c.cx <= x1 + 2) > 1

        for x0, t, x1, b in self.rules.shades:
            if t - 1 <= line.cy <= b + 1 and x0 - 2 <= cell.x0 and cell.x1 <= x1 + 2:
                if (spans_all(x0, x1) and n_cells > 1) or crowded(x0, x1):
                    continue
                if best is None or x1 - x0 < best[1] - best[0]:
                    best = (x0, x1)
        if underline:
            for y, x0, x1 in self.rules.h:
                if (
                    -3 <= y - cell.bottom <= 9
                    and x0 - 3 <= cell.cx <= x1 + 3
                    and x1 - x0 < 0.97 * width
                    and x1 - x0 >= cell.x1 - cell.x0 - 4
                    and not spans_all(x0, x1)
                    and not crowded(x0, x1)
                    and (best is None or x1 - x0 < best[1] - best[0])
                ):
                    best = (x0, x1)
        if best is None:
            best = self._over_span(cell, line, n_cells, underline, value, cells, spans_all)
        return best

    def _underlines_labels(self, y: float, x0: float, x1: float, below: Line) -> bool:
        """The rule at `y` is the underline of a line of at least three header cells above it
        (the decade labels), so the line below it holds a heading for all of them."""
        for ln in self.lines:
            if ln is below or not (-3 <= y - ln.bottom <= 9):
                continue
            if sum(1 for c in self.cells[id(ln)] if x0 - 2 <= c.cx <= x1 + 2) >= 3:
                return True
        return False

    def _over_span(
        self,
        cell: Cell,
        line: Line,
        n_cells: int,
        underline: bool,
        value: tuple[float, float],
        cells: list[Cell] | None,
        spans_all: Callable[[float, float], bool],
    ) -> tuple[float, float] | None:
        """Group headings whose extent is a rule but not a plain underline.

        - A rule right *above* a header cell, wider than the cell and over some columns only,
          is the cell's span ('Average (Annual)' under the line of decade labels).
        - A rule under the whole header line, shared by several cells: the cell takes what
          the shaded cells of the line leave of it ('(Average During the Year)' next to the
          shaded 'July-March')."""
        width = self.box[2] - self.box[0]
        for y, x0, x1 in self.rules.h:
            if (
                -1 <= cell.top - y <= 5
                and x0 - 3 <= cell.cx <= x1 + 3
                and x1 - x0 >= 1.5 * (cell.x1 - cell.x0)
                and x1 - x0 < 0.97 * width
                and not spans_all(x0, x1)
                and sum(1 for c in (cells or [cell]) if x0 - 2 <= c.cx <= x1 + 2) == 1
                and self._underlines_labels(y, x0, x1, line)
            ):
                return (x0, x1)
        if not (underline and cells is not None and n_cells > 1):
            return None
        others = [c for c in cells if not c.x0 - 1 <= cell.cx <= c.x1 + 1]  # not this cell
        spans = [
            self._shade_span(c, line, n_cells=n_cells, underline=False, value=value) for c in others
        ]
        shaded = [sp for sp in spans if sp is not None]
        if not shaded or len(shaded) < len(others):  # all the others are shaded groups
            return None
        for y, x0, x1 in self.rules.h:
            if -3 <= y - cell.bottom <= 9 and x0 - 3 <= cell.cx <= x1 + 3 and spans_all(x0, x1):
                lo, hi = x0, x1
                for a, b in shaded:
                    if b <= cell.cx:
                        lo = max(lo, b)
                    elif a >= cell.cx:
                        hi = min(hi, a)
                return (lo, hi) if hi - lo >= cell.x1 - cell.x0 else None
        return None

    def _rule_span(self, cell: Cell, line: Line) -> tuple[float, float] | None:
        """In a ruled table a header cell fills the box between the vertical rules on either
        side of it at its height; a group header spans several columns."""
        xs = sorted(x for x, y0, y1 in self.rules.v if y0 - 2 <= line.top and y1 + 2 >= line.bottom)
        if not xs:
            return None
        left = max((x for x in xs if x <= cell.cx), default=self.box[0])
        right = min((x for x in xs if x >= cell.cx), default=self.box[2])
        if right - left <= 0:
            return None
        # several cells in the box: the rules mark column groups, not this cell's own box,
        # unless the cells are one phrase cut at a number ('2024-25' '(P)', Table 2.3)
        inside = sorted(
            (c for c in self.cells[id(line)] if left <= c.cx <= right), key=lambda c: c.x0
        )
        phrase = all(
            b.x0 - a.x1 <= 0.6 * line.size for a, b in zip(inside, inside[1:], strict=False)
        )
        if len(inside) > 1 and not phrase:
            return None
        return (left, right)

    def _cover(
        self,
        cell: Cell,
        line: Line,
        bands: list[Band],
        *,
        n_cells: int,
        underline: bool,
        value: tuple[float, float],
    ) -> list[int]:
        """Indices of the columns a header cell stands for."""
        span = self._shade_span(
            cell,
            line,
            n_cells=n_cells,
            underline=underline,
            value=value,
            cells=self.cells[id(line)],
        )
        if span is None and bands[0].ruled:
            span = self._rule_span(cell, line)
        if span is not None:
            cols = [
                i for i, b in enumerate(bands) if span[0] - COL_PAD <= b.cx <= span[1] + COL_PAD
            ]
            if cols:
                return cols
        lo, hi = cell.x0, cell.x1
        overlaps = [(max(0.0, min(hi, b.hi) - max(lo, b.lo)), i) for i, b in enumerate(bands)]
        best = max(overlaps)
        if best[0] <= 0:
            return [min(range(len(bands)), key=lambda i: abs(bands[i].cx - cell.cx))]
        cols = {best[1]}
        for i, b in enumerate(bands):  # a cell that covers most of another column's data spans it
            c0, c1 = b.core
            if min(hi, c1) - max(lo, c0) >= 0.7 * max(c1 - c0, 1.0):
                cols.add(i)
        return sorted(cols)

    def _header(
        self,
        block: _Block,
        bands: list[Band],
        n_lab: int,
        split_words: bool = False,
        n_index: int = 0,
    ) -> tuple[list[str], list[str]]:
        pieces: list[list[tuple[float, str]]] = [[] for _ in bands]
        units: list[str] = []
        value_lo, value_hi = bands[n_lab].lo, bands[-1].hi
        n_value = len(bands) - n_lab
        value = (value_lo, value_hi)
        content: list[tuple[Line, list[Cell]]] = []
        named = [  # header lines that name most value columns ('CY17 CY18 ... CY24')
            ln
            for ln in block.header_lines
            if sum(c.cx >= value_lo - 2 for c in self.cells[id(ln)] if c.text.strip())
            >= max(2, n_value // 2)
        ]
        for line in block.header_lines:
            cells = [c for c in self.cells[id(line)] if c.text.strip()]
            if not cells:
                continue
            over = [c for c in cells if c.cx >= value_lo - 2]
            if len(cells) >= 2 and len(over) == 1 and named and line not in named:
                # '(Rs billion)' flush right on the 'Variables' line, under the year headers:
                # the table's units, not a second line of the last column's header
                t0 = over[0].text.strip()
                if (
                    t0.startswith("(")
                    and t0.endswith(")")
                    and STRICT_UNITS_RE.match(t0)
                    and len(t0.split()) <= 4
                    and over[0].x1 >= value_hi - 8
                ):
                    units.append(clean_unicode(t0))
                    cells = [c for c in cells if c is not over[0]]
            if len(cells) == 1 and cells[0].cx >= value_lo - 2:
                c0 = cells[0]
                span = self._shade_span(c0, line, n_cells=1, underline=False, value=value)
                spans_all = span is not None and span[0] <= value_lo + 6 and span[1] >= value_hi - 6
                centred = abs(c0.cx - (value_lo + value_hi) / 2) <= 0.15 * (value_hi - value_lo)
                text0 = c0.text.strip()
                unitish = text0.startswith("(") or STRICT_UNITS_RE.match(text0)
                column_heading = bool(_CHANGE_TITLE_RE.match(text0))  # over its columns only
                if n_value >= 2 and (spans_all or (centred and unitish)) and not column_heading:
                    units.append(clean_unicode(c0.text))
                    continue
            content.append((line, cells))
        for k, (line, cells) in enumerate(content):
            below = k < len(content) - 1
            cells = self._split_at_rules(cells, line)
            if split_words:
                cells = [
                    part for c in cells for part in self._split_by_bands(c, line, bands, value)
                ]
            for c in cells:
                text = clean_unicode(c.text)
                if NOTE_RE.match(text) and c.cx < value_lo:
                    continue
                cols = self._cover(c, line, bands, n_cells=len(cells), underline=below, value=value)
                for i in cols:
                    pieces[i].append((c.top, text))
        merged: list[list[tuple[float, str]]] = [
            *(pieces[:n_lab] or [[]]),  # each label column keeps its own header
            *pieces[n_lab:],
        ]
        header = []
        for col in merged:
            seen: list[str] = []
            for _, text in sorted(col, key=lambda p: p[0]):
                if not seen or seen[-1] != text:
                    seen.append(text)
            header.append(self._join_pieces(seen))
        for i in range(1, max(n_lab, 1)):  # one header cell over two label columns: say it once
            if header[i] and header[i] in header[:i]:
                header[i] = ""
        return header, units

    def _split_by_bands(
        self, cell: Cell, line: Line, bands: list[Band], value: tuple[float, float]
    ) -> list[Cell]:
        """Second attempt at a header whose columns came out identical: a text cell without a
        bar or rule under it is cut where its words fall into different columns ('Exports
        Imports Balance' over three columns)."""
        if len(cell.words) < 2 or self._shade_span(
            cell, line, n_cells=2, underline=True, value=value, cells=self.cells[id(line)]
        ):
            return [cell]
        runs: list[list[Word]] = []
        last = None
        words = cell.words
        for k, w in enumerate(words):
            # a lone '&' or '/' goes with the word after it ('& Commu-' under 'Transport')
            pick = words[k + 1] if w.text in {"&", "/"} and k + 1 < len(words) else w
            band = self._band_of(Cell([pick]), bands)
            if band != last:
                runs.append([])
            runs[-1].append(w)
            last = band
        return [Cell(r) for r in runs] if len(runs) >= 2 else [cell]

    def _split_at_rules(self, cells: list[Cell], line: Line) -> list[Cell]:
        """Cut header cells where a vertical rule runs through the line between two words."""
        xs = [
            x
            for x, y0, y1 in self.rules.v
            if y0 - 2 <= line.top
            and y1 + 2 >= line.bottom
            and self.box[0] + 6 < x < self.box[2] - 6
        ]
        if not xs:
            return cells
        out: list[Cell] = []
        for c in cells:
            part: list[Word] = []
            for w in c.words:
                if part and any(
                    (part[-1].x0 + part[-1].x1) / 2 < x <= (w.x0 + w.x1) / 2 for x in xs
                ):
                    out.append(Cell(part))
                    part = []
                part.append(w)
            out.append(Cell(part))
        return out

    def _join_pieces(self, pieces: list[str]) -> str:
        """Join the lines of one header cell; a word broken over two lines is mended with the
        page's own vocabulary ('Desp-' + 'atches', 'Producti' + 'on')."""
        out = ""
        for piece in pieces:
            if not out:
                out = piece
                continue
            head = out.split(" ")[-1]
            first = piece.split(" ")[0]
            joined = (head.rstrip("-") + first).lower()
            tail = re.sub(r"\W", "", first).lower()
            if head.endswith("-") and first[:1].islower():
                # mend 'Unemp-' + 'loyed' when the join is a page word or the second part is
                # no word of the page; keep 'year-' + 'on-year'
                mend = (
                    joined in self.vocab
                    or self.word_count[joined] > 0  # 'agriculture' printed whole elsewhere
                    or (tail.isalpha() and self.word_count[tail] < 2)
                )
                out = out[: -len(head)] + head.rstrip("-") + piece if mend else out + piece
            elif first[:1].islower() and len(first) <= 4 and joined in self.vocab:
                out = out + piece
            else:
                out = out + " " + piece
        return normalize_ws(out)

    # ---- rows

    def _join_label(self, pieces: list[str]) -> str:
        """Join the lines of a wrapped row label; a word broken at the line end is mended when
        the joined word is printed in the document ('Manu-' + 'factures thereof')."""
        out = ""
        for piece in (p for p in pieces if p):
            head = out.split(" ")[-1] if out else ""
            first = re.sub(r"\W", "", piece.split(" ")[0]).lower()
            joined = re.sub(r"\W", "", head).lower() + first
            if (
                head.endswith("-")
                and piece[:1].islower()
                and (joined in self.vocab or self.word_count[joined] > 0)
            ):
                out = out[:-1] + piece
            else:
                out = f"{out} {piece}" if out else piece
        return normalize_ws(out)

    def _indent(self, cell: Cell) -> float:
        """Indent of a heading. A title centred over the value columns is top level; a heading
        that ends inside the label column keeps its indent ('(ii) Cut-Off Price' under '(a) 02
        Years ...', Table 5.7)."""
        inside_labels = cell.x1 <= self.limit + 2
        return cell.x0 if cell.x0 <= self.left_edge + 30 or inside_labels else self.left_edge - 5

    def _wraps(
        self,
        line: Line,
        text: str,
        prev: _Row,
        rest: list[tuple[str, Line]],
        bands: list[Band],
        n_lab: int,
        seen: set[str] | None = None,
    ) -> bool:
        """A short label-only line right under a row, followed by a row at the same indent,
        is the next line of that row's wrapped label ('General' / 'Government' / 'Consumption').
        A heading is followed by rows indented deeper, or by a row whose label the table has
        already used (`seen`): '10-14' over 'Male' / 'Female' after '10 years & over' over
        'Both Sexes' / 'Male' / 'Female' (Table 12.13) starts a new group."""
        if len(text.split()) > 5 or line.top - prev.end > 1.8 * line.size:
            return False
        if prev.indent is not None and self.cells[id(line)][0].x0 < prev.indent - 2.5:
            return False  # starts left of the row's label: not its next line
        nxt = next((ln for kind, ln in rest if kind == "data"), None)
        if nxt is None:
            return True
        label_cells = [c for c in self.cells[id(nxt)] if self._band_of(c, bands) < n_lab]
        if not label_cells:
            return False
        if seen and normalize_ws(" ".join(c.text for c in label_cells)).lower() in seen:
            return False
        return label_cells[0].x0 <= self.cells[id(line)][0].x0 + 2.5

    def _no_values_item(
        self,
        cell: Cell,
        rows: list[_Row],
        nxt_data: Line | None,
        nxt_label: str,
        bands: list[Band],
        n_index: int,
        n_lab: int,
    ) -> bool:
        """A label-only line that is neither a heading nor a wrapped line: an item the table
        lists with no values on this page ('Aviation gasoline (100LL)' between 'LDO' and 'JP-1:',
        Table 14.5) when the next row starts a new label at the same indent or left of it. A
        heading is indented less than its rows, or its rows repeat labels already used."""
        if not rows or nxt_data is None or not nxt_label or cell.x1 > self.limit + 2:
            return False  # none, or a title over the value columns
        if nxt_label.lower() in {normalize_ws(r.label).lower() for r in rows if r.label}:
            return False
        xs = [c.x0 for c in self.cells[id(nxt_data)] if n_index <= self._band_of(c, bands) < n_lab]
        return bool(xs) and cell.x0 >= xs[0] - 2.5

    def _item_role(
        self,
        text: str,
        prev: _Row,
        stack: list[_Heading],
        nxt_label: str,
        nxt_data: Line | None,
        rows: list[_Row],
        x: float = 0.0,
        nxt_x: float | None = None,
        nxt_first: str = "",
    ) -> str | None:
        """What a label-only line that starts with an enumerator ('b.', '(h)', '15.') is:
        'heading' when it continues the sequence of a heading of the block ('b. Non Paris Club
        Countries' after 'a. Paris Club Countries'), when the next row is enumerated one level
        deeper ('(h) 7 Years Maturity' over '(i) Amount Accepted'), when the next row repeats a
        label already used, or when no row follows; 'skip' when it continues the row above's
        own sequence ('15. Sweden' after '14. Italy': an item with no values here); 'prefix' when
        the next row's label continues it ('... (Hotels' / '& Restaurants)'); None
        (decided as before: a wrapped line, a prefix) otherwise."""
        tok = _ENUM_RE.match(text).group(1)  # type: ignore[union-attr]
        open_bracket = text.rfind("(") > text.rfind(")")
        if nxt_data is not None and (
            open_bracket or nxt_label[:1] in tuple("&)") or nxt_label[:1].islower()
        ):
            return "prefix"  # the first line of the next row's label
        if any((m := _ENUM_RE.match(h.text)) and next_in_sequence(m.group(1), tok) for h in stack):
            return "heading"
        if nxt_data is None:
            return "heading"
        m = _ENUM_RE.match(nxt_label)
        if m and enum_rank(nxt_label) > enum_rank(text):
            return "heading"
        if nxt_label and nxt_label.lower() in {normalize_ws(r.label).lower() for r in rows}:
            return "heading"
        own = _ENUM_RE.match(prev.parts[0] if prev.parts and prev.parts[0] else prev.label)
        if own and next_in_sequence(own.group(1), tok):
            return "skip"
        last = next(
            (m for r in reversed(rows) if (m := _ENUM_RE.match(r.parts[0] if r.parts else ""))),
            None,
        )
        if (
            last is not None
            and next_in_sequence(last.group(1), tok)
            and nxt_x is not None
            and x - 2.5 <= nxt_x <= x + 12  # same indent or hanging ('5. ANZ Bank / Standard')
            and nxt_first
            and not _ENUM_RE.match(nxt_first)
            and not _TOTAL_RE.match(nxt_first)
            and len(nxt_first.split()) <= 4
        ):  # '3. Electricity Generation & Distribution & Gas' / 'Distribution'
            return "prefix"
        return None

    def _slabs(self, items: list[tuple[str, Line]], bands: list[Band]) -> dict[int, int] | None:
        """For a ruled table whose horizontal rules separate the rows: the box (0, 1, 2 ...)
        each line of the body sits in. None when the rules do not separate the rows."""
        if not bands[0].ruled:
            return None
        width = self.box[2] - self.box[0]
        ys = sorted(y for y, x0, x1 in self.rules.h if x1 - x0 >= 0.5 * width)
        if not ys:
            return None

        def box_of(line: Line) -> int:
            return sum(1 for y in ys if y < line.cy)

        data = [ln for kind, ln in items if kind == "data"]
        if len({box_of(ln) for ln in data}) < 0.8 * len(data):
            return None
        return {id(ln): box_of(ln) for _, ln in items}

    def _n_label(self, data_lines: list[Line], bands: list[Band]) -> tuple[int, int]:
        """Number of leading columns that hold the row label (text columns, optionally after
        an index column '1', '(i)', 'A.' in front of the text; at least one) and how many of
        them are an index column (0 or 1)."""
        index_re = re.compile(r"^\(?(?:[ivxlc]+|[a-z]|\d{1,3})[.)]?$", re.I)
        by_band: dict[int, list[Cell]] = {}
        for ln in data_lines:
            for c in self.cells[id(ln)]:
                by_band.setdefault(self._band_of(c, bands), []).append(c)

        def texty(i: int) -> bool:
            cells = by_band.get(i, [])
            n_text = sum(1 for c in cells if c.kind in ("text", "period"))
            return bool(cells) and n_text >= 0.6 * len(cells)

        def indexy(i: int) -> bool:
            cells = by_band.get(i, [])
            return bool(cells) and sum(
                1 for c in cells if index_re.match(c.text.strip())
            ) >= 0.7 * len(cells)

        first = by_band.get(0, [])
        n = 1
        n_index = 0
        if len(bands) > 2 and not texty(0) and indexy(0) and texty(1):
            n = 2  # S.No | Item | values
            n_index = 1
        elif first and sum(1 for c in first if c.kind == "value") >= 0.7 * len(first):
            return 0, 0  # bare numbers: the row labels are not on this page
        while n < len(bands) - 1 and texty(n):
            n += 1
        return n, n_index

    def _band_of(self, cell: Cell, bands: list[Band]) -> int:
        best, best_ov = 0, -1.0
        for i, b in enumerate(bands):
            ov = min(cell.x1, b.hi) - max(cell.x0, b.lo)
            if ov > best_ov:
                best, best_ov = i, ov
        return best

    def _rows(self, block: _Block, bands: list[Band], n_lab: int, n_index: int = 0) -> list[_Row]:
        rows: list[_Row] = []
        stack: list[_Heading] = []
        prefix: list[str] = []
        items = block.items
        i = 0

        def code_row() -> bool:
            """The row above is labelled by a code alone ('LDO', 'JP-1:')."""
            if not rows or rows[-1].suffix or rows[-1].prefix:
                return False
            own = rows[-1].parts[0] if rows[-1].parts else rows[-1].label
            return bool(own and _CODE_LABEL_RE.fullmatch(own))

        def code_item(text: str) -> bool:
            """An item named by a code: 'HOBC (Automotive 100 Octane)', or 'LDO' under 'HSD'."""
            if _CODE_LABEL_RE.fullmatch(text) and code_row():
                return True
            m = _CODE_ITEM_RE.fullmatch(text)
            return bool(m) and not _QUALIFIER_RE.match(m.group(1))  # type: ignore[union-attr]

        def heading_like(text: str, following_rows: bool) -> bool:
            if code_item(text):  # 'HOBC (Automotive 100 Octane)': an item, not a section
                return False
            if text.rstrip().endswith(":") or _is_upper(text) or _QUALIFIER_RE.match(text):
                return True
            if re.match(r"(?i)memo(?:randum)?\b", text):  # 'Memorandum Item' over its rows
                return True
            if _BASE_YEAR_LINE_RE.match(text):  # a series break over the rows below it
                return True
            return bool(_PERIOD_RE.match(text) and following_rows)

        last_letter: list[str] = []  # the last '(a)'-style letter heading, for '(i)' after '(h)'
        opened: list[_Heading] = []  # every heading of the block, open or closed

        def push(indent: float, text: str) -> None:
            rank = enum_rank(text, last_letter[-1] if last_letter else None)
            if rank == 1:
                last_letter.append(_ENUM_RE.match(text).group(1).lower())  # type: ignore[union-attr]

            def deeper_or_sibling(h: _Heading) -> bool:
                if h.indent > indent + 2.5:
                    return True
                if abs(h.indent - indent) > 2.5:
                    return False
                if rank and h.rank:  # one indent, enumerators tell the levels apart:
                    return h.rank >= rank  # '7. IFAD' is a sibling of '6. IBRD', not a child
                return h.had_rows

            while stack and deeper_or_sibling(stack[-1]):
                stack.pop()
            stack.append(_Heading(indent, text, rank=rank))
            opened.append(stack[-1])

        def attach(row: _Row, cells: list[Cell], text: str) -> None:
            """A wrapped label line goes to the label column it is printed in ('Government
            Securities (%)' under 'Share of Shariah Compliant Instruments in', not under
            'Concentration Risk')."""
            col = self._band_of(cells[0], bands) - n_index if cells else 0
            later = 0 < col < n_lab - n_index and len(row.parts) > col and row.parts[col]
            if later and re.search(r"[A-Za-z]", text):  # a lone footnote mark stays put
                row.col_suffix.setdefault(col, []).append(text)
            else:
                row.suffix.append(text)

        def rows_follow(j: int) -> bool:
            return any(kind == "data" for kind, _ in items[j + 1 :])

        slab = self._slabs(items, bands)
        skipped_x: float | None = None  # x of the last item line dropped for having no values
        prefix_index: str | None = None  # an index printed on a prefix line ('6 | Machinery')

        while i < len(items):
            kind, line = items[i]
            cells = self.cells[id(line)]
            if kind == "label":
                text = normalize_ws(clean_unicode(" ".join(c.text for c in cells)))
                if slab is not None:  # ruled rows: the lines of one box are one label
                    mine = slab[id(line)]
                    above = next((r for r in reversed(rows) if slab[id(r.line)] == mine), None)
                    nxt_data = next(
                        (
                            (j, ln)
                            for j, (k2, ln) in enumerate(items[i + 1 :], start=i + 1)
                            if k2 == "data" and slab[id(ln)] == mine
                        ),
                        None,
                    )
                    if above is not None:
                        attach(above, cells, text)
                        i += 1
                        continue
                    if nxt_data is not None:
                        prefix.append(text)
                        i += 1
                        continue
                    if rows and rows_follow(i):  # a box of its own between rows: a band
                        push(self._indent(cells[0]), text)  # heading ('Key FSIs (%)')
                        i += 1
                        continue
                nxt = items[i + 1] if i + 1 < len(items) else None
                nxt_blank = (  # the next row prints no label of its own (an index at most)
                    nxt is not None
                    and nxt[0] == "data"
                    and not any(
                        n_index <= self._band_of(c, bands) < n_lab for c in self.cells[id(nxt[1])]
                    )
                )
                nxt_data = next((ln for k2, ln in items[i + 1 :] if k2 == "data"), None)
                nxt_label = (
                    normalize_ws(
                        " ".join(
                            c.text
                            for c in self.cells[id(nxt_data)]
                            if n_index <= self._band_of(c, bands) < n_lab
                        )
                    )
                    if nxt_data is not None
                    else ""
                )
                nxt_first = [  # the next row's cells in the first text label column
                    c
                    for c in (self.cells[id(nxt_data)] if nxt_data is not None else [])
                    if self._band_of(c, bands) == n_index
                ]
                nxt_x = nxt_first[0].x0 if nxt_first else None
                if skipped_x is not None and (
                    text[:1].islower() or cells[0].x0 > skipped_x + 2.5
                ):  # the next line of an item with no values ('flights & foreign airline')
                    i += 1
                    continue
                skipped_x = None
                if (
                    _TOTAL_RE.match(text)
                    and n_lab - n_index >= 2
                    and rows
                    and rows[-1].parts
                    and not rows[-1].parts[0]
                    and rows[-1].parts[1:]
                    and line.cy - rows[-1].line.bottom < line.size  # type: ignore[union-attr]
                ):  # 'TOTAL (I)' printed between its 'Principal' and 'Interest' lines
                    while stack and stack[-1].rank >= 1:
                        stack.pop()
                    if sums_sections(text, [h.text for h in opened]):
                        stack.clear()  # 'TOTAL (I+II+III+IV+V)' is under no section
                    rows[-1].parts[0] = text
                    rows[-1].label = normalize_ws(" ".join(rows[-1].parts))
                    rows[-1].path = [h.text for h in stack]
                    i += 1
                    continue
                if _OPERATOR_LINE_RE.match(text):  # 'Plus' / 'Net Primary Income'
                    prefix.append(text)
                    i += 1
                    continue
                if rows and _FIRST_ITEM_RE.match(nxt_label) and not nxt_blank:
                    # 'Amount Accepted' over 'i) Face value', 'ii) Discounted value'
                    push(self._indent(cells[0]), text)
                    i += 1
                    continue
                nxt_label_start = (
                    nxt is not None
                    and nxt[0] == "data"
                    and (
                        self.cells[id(nxt[1])][0].text[:1].islower()
                        or self.cells[id(nxt[1])][0].text.startswith(("(", "&"))
                    )
                )
                upper = _is_upper(text) and not _CONNECTOR_END_RE.search(text)
                if not upper and (
                    nxt_blank
                    or nxt_label_start
                    or (nxt is not None and nxt[0] == "data" and _CONNECTOR_END_RE.search(text))
                ):
                    lead_index = [c for c in cells if self._band_of(c, bands) < n_index]
                    nxt_index = (
                        nxt is not None
                        and nxt[0] == "data"
                        and any(self._band_of(c, bands) < n_index for c in self.cells[id(nxt[1])])
                    )
                    if lead_index and not nxt_index and len(cells) > len(lead_index):
                        # '6 | Machinery' over '(non-electrical)': '6' is the row's index
                        prefix_index = lead_index[0].text
                        text = normalize_ws(
                            clean_unicode(" ".join(c.text for c in cells if c not in lead_index))
                        )
                    prefix.append(text)
                elif (
                    rows
                    and _ENUM_RE.match(text)
                    and (
                        role := self._item_role(
                            text,
                            rows[-1],
                            opened,
                            nxt_label,
                            nxt_data,
                            rows,
                            cells[0].x0,
                            nxt_x,
                            " ".join(c.text for c in nxt_first),
                        )
                    )
                ):
                    if role == "heading":  # '(h) 7 Years Maturity' over '(i) Amount Accepted'
                        push(self._indent(cells[0]), text)
                    elif role == "prefix":  # '3. Accommodation ... (Hotels' / '& Restaurants)'
                        prefix.append(text)
                    # 'skip': '15. Sweden' after '14. Italy', an item with no values here
                elif heading_like(text, rows_follow(i)):
                    push(self._indent(cells[0]), text)
                elif rows and (
                    (text[:1].islower() and not (code_row() and _ENUM_RE.match(text)))
                    or text[:1] in "(&-/"
                    or _CONNECTOR_END_RE.search(rows[-1].label)
                    or (
                        not code_row()
                        and not code_item(text)
                        and self._wraps(
                            line,
                            text,
                            rows[-1],
                            items[i + 1 :],
                            bands,
                            n_lab,
                            {normalize_ws(r.label).lower() for r in rows if r.label},
                        )
                    )
                ):
                    attach(rows[-1], cells, text)
                    rows[-1].end = line.bottom
                elif (
                    rows
                    and rows[-1].indent is not None
                    and cells[0].x0 < rows[-1].indent - 2.5
                    and nxt is not None
                    and nxt[0] == "data"
                    and nxt_x is not None
                    and abs(nxt_x - cells[0].x0) <= 2.5
                    and nxt_label.lower() not in {normalize_ws(r.label).lower() for r in rows}
                    and not _CONNECTOR_END_RE.search(rows[-1].label)
                ):  # 'Technical / Vocational' over 'Institutions', back at the outer indent
                    prefix.append(text)
                elif (
                    i > 0
                    and line.top - items[i - 1][1].bottom <= line.size  # no gap: not a new part
                    and not _TOTAL_RE.match(rows[-1].label if rows else "")
                    and self._no_values_item(
                        cells[0], rows, nxt_data, nxt_label, bands, n_index, n_lab
                    )
                ):
                    skipped_x = cells[0].x0  # 'Aviation gasoline (100LL)': no values here
                else:
                    push(self._indent(cells[0]), text)
                i += 1
                continue
            # data row
            skipped_x = None
            by_band: dict[int, list[Cell]] = {}
            for c in cells:
                by_band.setdefault(self._band_of(c, bands), []).append(c)
            by_label = [by_band.pop(i, []) for i in range(n_lab)]
            label_cells = [c for cs in by_label[n_index:] for c in cs]  # text columns only
            parts = [
                despace(normalize_ws(clean_unicode(" ".join(c.text for c in cs))))
                for cs in by_label
            ]
            index = parts[0] if n_index else None
            if n_index and not index and prefix_index:
                index = prefix_index
            prefix_index = None
            parts = parts[n_index:]
            label = normalize_ws(" ".join(parts))
            indent = label_cells[0].x0 if label_cells else None
            row = _Row(
                cells=by_band,
                top=line.top,
                bottom=line.bottom,
                label=label,
                indent=indent,
                line=line,
                end=line.bottom,
                parts=parts,
                prefix=list(prefix),
                index=index,
            )
            prefix = []
            if indent is not None:
                while stack and stack[-1].indent > indent + 2.5:
                    stack.pop()
                if _is_upper(label):  # a section of its own, not a row of the heading above
                    while stack and stack[-1].indent >= indent - 2.5:
                        stack.pop()
                if _TOTAL_RE.match(label):  # 'Sub-Total A' closes the items above it
                    while stack and stack[-1].rank >= 1:
                        stack.pop()
                if sums_sections(label, [h.text for h in opened]):
                    stack.clear()  # a sum of the sections above is under none of them
                own_rank = enum_rank(label)
                if own_rank:  # '3.France' after the value-less '2. Japan': a sibling item
                    while (
                        stack
                        and stack[-1].rank >= own_rank
                        and abs(stack[-1].indent - indent) <= 2.5
                    ):
                        stack.pop()
            for h in stack:
                h.had_rows = True
            row.path = [h.text for h in stack]
            # label split around the numbers: the next line is the tail of this label
            if not label_cells and i + 1 < len(items) and items[i + 1][0] == "label":
                nxt_text = normalize_ws(
                    clean_unicode(" ".join(c.text for c in self.cells[id(items[i + 1][1])]))
                )
                after = next((ln for k2, ln in items[i + 2 :] if k2 == "data"), None)
                after_label = (
                    normalize_ws(
                        " ".join(
                            c.text
                            for c in self.cells[id(after)]
                            if n_index <= self._band_of(c, bands) < n_lab
                        )
                    ).lower()
                    if after is not None
                    else ""
                )
                seen_labels = {normalize_ws(r.label).lower() for r in rows if r.label}
                new_group = bool(after_label) and after_label in seen_labels  # 'Islamic Banking'
                if not heading_like(nxt_text, rows_follow(i + 1)) and not new_group:
                    row.suffix.append(nxt_text)
                    i += 1
            rows.append(row)
            i += 1
        # finalise labels
        prev: list[str] | None = None  # label cells of the row above
        n_text = n_lab - n_index
        for row in rows:
            parts = list(row.parts)
            total = any(_TOTAL_RE.match(p or "") for p in parts)  # 'Total' sums the groups
            if n_text > 1 and prev is not None and (not total or _TOTAL_RE.match(prev[0] or "")):
                for j in range(n_text - 1):  # a blank outer label belongs to the row above
                    if not parts[j] and prev[j]:
                        parts[j] = prev[j]
                        row.filled = True
            for col, extra in row.col_suffix.items():
                parts[col] = self._join_label([parts[col], *extra])
            first = self._join_label([*row.prefix, *parts[:1], *row.suffix])
            cells = [first, *parts[1:]]
            if not any(cells) and prev is not None and n_lab > 0:
                cells = list(prev)
                row.filled = True
            row.parts = cells
            row.label = normalize_ws(" ".join(cells))
            prev = cells
        levels = _indent_levels([r.indent for r in rows if r.indent is not None])
        chain: list[tuple[int, str]] = []  # rows that are the parents of the rows below them
        prev_path: list[str] | None = None
        done: list[str] = []  # the labels of the rows above
        for row in rows:
            done.append(row.parts[0] if row.parts else "")
            if row.path != prev_path:  # another heading scope starts: no row is a parent yet
                chain = []
            prev_path = row.path
            parents: list[str] = []
            if levels is not None and row.indent in levels and row.label and not row.filled:
                level = levels[row.indent]
                own_enum = _ENUM_RE.match(row.parts[0] if row.parts else "")
                if own_enum:  # '2. Livestock' printed deeper than '1. Crops' is its sibling
                    sib = next(
                        (
                            lv
                            for lv, text in reversed(chain)
                            if (m := _ENUM_RE.match(text))
                            and next_in_sequence(m.group(1), own_enum.group(1))
                        ),
                        None,
                    )
                    level = sib if sib is not None else level
                if sums_sections(row.label, done):  # 'GDP {Total of GVA (A + B + C)}'
                    chain = []  # sums the sections: under none of them
                while chain and chain[-1][0] >= level:
                    chain.pop()
                if not _TOTAL_RE.match(row.label):  # a total sums the rows above: no parent
                    parents = [text for _, text in chain]
                chain.append((level, row.parts[0] or row.label))  # not the unit column
            own = row.parts[0] if row.parts else ""
            first = _FIRST_PERIOD_RE.match(parents[-1]) if parents else None
            if first and _PERIOD_CODE_RE.fullmatch(own):  # '2018-19: Q1' then 'Q2', 'Q3'
                parents, own = parents[:-1], f"{first.group(1)}: {own}"
            head = " / ".join([*row.path, *parents, own]) if (row.path or parents) else own
            row.texts = [head if head.strip() else "", *row.parts[1:]]
            row.label = " ".join(t for t in row.texts if t)
        return rows

    def _values(
        self, rows: list[_Row], bands: list[Band], n_lab: int
    ) -> tuple[list[list[Any]], list[list[str]], list[str]]:
        out, printed = [], []
        merged = 0
        for row in rows:
            vals: list[Any] = []
            prt: list[str] = []
            # footnote markers and status letters join the nearest value cell of the row
            flat = [c for cs in row.cells.values() for c in cs]
            extras: dict[int, list[str]] = {}
            for m in (c for c in flat if c.kind == "marker"):
                near = [v for v in flat if v.kind in ("value", "period")]  # 1901 reads as a year
                # a footnote marker follows its number: the closest value to its left, else
                # the closest one to its right
                left = [v for v in near if v.cx < m.cx and m.x0 - v.x1 <= 60]
                right = [v for v in near if v.cx >= m.cx and v.x0 - m.x1 <= 60]
                pick = (
                    max(left, key=lambda v: v.cx)
                    if left
                    else (min(right, key=lambda v: v.cx) if right else None)
                )
                if pick is not None:
                    extras.setdefault(id(pick), []).append(m.text)
            for i in range(n_lab, len(bands)):
                cs = sorted(row.cells.get(i, []), key=lambda c: c.x0)
                texts = [c.text + "".join(extras.get(id(c), [])) for c in cs if c.kind != "marker"]
                kinds = [c.kind for c in cs if c.kind != "marker"]
                n_val = sum(1 for k in kinds if k == "value")
                if n_val > 1:
                    merged += 1
                text = normalize_ws(clean_unicode(" ".join(texts)))
                v, p = parse_cell(text, brackets_negative=self.paren_meaning is None)
                vals.append(v)
                prt.append(p)
            out.append(vals)
            printed.append(prt)
        extra = ["merged_cells"] if merged else []
        self._merged = merged
        return out, printed, extra

    # ---- quality

    def _check(self, tables: list[ParsedTable]) -> str | None:
        for t in tables:
            n_rows = len(t.rows)
            lead = t.lead
            n_cols = len(t.header) - lead
            if _duplicates(t.header, lead):
                return "header_ambiguous"
            if n_rows < MIN_DATA_ROWS or n_cols < 1:
                return "too_few_rows"
            cells = [v for r in t.rows for v in r[lead:]]
            filled = sum(1 for v in cells if v not in (None, ""))
            if filled < self.min_fill * len(cells):
                return "sparse"
            counts = [sum(1 for v in r[lead:] if v not in (None, "")) for r in t.rows]
            modal = Counter(counts).most_common(1)[0][0]
            ragged = sum(1 for c in counts if c < 0.5 * modal)
            if ragged > self.max_ragged * n_rows:
                return "ragged"
            numeric = sum(1 for v in cells if isinstance(v, int | float))
            if numeric < 0.5 * filled:
                return "few_numbers"
            if any(len(h) > 100 or len(h.split()) > 16 for h in t.header):
                return "header_unreadable"  # body text absorbed into the header
            for j in range(lead, len(t.header)):
                col = [r[j] for r in t.rows if r[j] not in (None, "")]
                n_num = sum(1 for v in col if isinstance(v, int | float))
                n_other = sum(  # words in a number column; '186 (2019)' is a legitimate value
                    1
                    for v in col
                    if isinstance(v, str)
                    and v.strip().lower() not in PLACEHOLDERS
                    and not re.search(r"\d", v)
                )
                if len(col) >= 5 and n_num >= 0.6 * len(col) and n_other > 0.05 * len(col):
                    return "mixed_cells"
            if (
                n_cols >= 3
                and any(  # one value printed between two rows (tariff 'Peak' /
                    # '1,000' / 'Off-Peak'): a row of that value alone under the same label
                    sum(v not in (None, "") for v in r[lead:]) == 1
                    and r[lead - 1]
                    in (
                        t.rows[i - 1][lead - 1] if i else None,
                        t.rows[i + 1][lead - 1] if i + 1 < n_rows else None,
                    )
                    for i, r in enumerate(t.rows)
                )
            ):
                return "shared_cells"
            merged_items = sum(
                1
                for r in t.rows
                if any(
                    isinstance(c, str) and _TWO_ITEMS_RE.search(c.split(" / ")[-1])
                    for c in r[:lead]
                )
            )
            if merged_items >= 2:  # '17. Human Health and Social 16. Work Activities'
                return "merged_labels"
            tails = sum(1 for r in t.rows if lead and value_tail(r[lead - 1]) is not None)
            if n_rows >= 3 and tails >= 0.3 * n_rows:
                return "label_absorbs_values"  # a value column read into the row labels
            if "merged_cells" in t.flags:
                if self._merged > 0.03 * len(cells):
                    return "merged_cells"
                t.flags.remove("merged_cells")
        return None


# --------------------------------------------------------------------------- output


def _esc(text: Any) -> str:
    s = "" if text is None else str(text)
    return s.replace("|", "\\|").replace("\n", " ")


def to_markdown(header: list[str], printed: list[list[str]], notes: list[str]) -> str:
    """Pipe table, then (after one blank line) the source and note lines."""
    lines = [
        "| " + " | ".join(_esc(h) for h in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in printed:
        cells = [*row, *([""] * (len(header) - len(row)))]
        lines.append("| " + " | ".join(_esc(c) for c in cells) + " |")
    text = "\n".join(lines)
    if notes:
        text += "\n\n" + "\n".join(notes)
    return text


def page_tables(
    page: Page,
    page_no: int,
    doc_id: str,
    regions: Iterable[Region],
    *,
    min_fill: float = 0.5,
    max_ragged: float = 0.25,
    infographic: bool = False,
    doc_words: Counter[str] | None = None,
) -> list[tuple[ParsedTable, TableRecord]]:
    """Every table of one page, in reading order, as (parsed table, record). On an infographic
    page (the Highlights) the 'tables' are figure labels: each region is plain text."""
    table_regions = sorted(
        (r for r in regions if r.kind == "table"), key=lambda r: (r.bbox[1], r.bbox[0])
    )
    if not table_regions:
        return []
    words = table_words(page)
    vertical = vertical_runs(page)
    out: list[tuple[ParsedTable, TableRecord]] = []
    n = 0
    for region in table_regions:
        rules = drawn_rules(page, region.bbox)
        parser = RegionParser(
            words,
            region,
            rules,
            min_fill=min_fill,
            max_ragged=max_ragged,
            doc_words=doc_words,
            vertical=vertical,
        )
        if infographic:
            parsed = parser.fallback("infographic")
        else:
            parsed = parser.parse()
            others = [r.bbox for r in table_regions if r is not region]
            if any("fallback_plain" not in t.flags for t in parsed) and parser.rows_below(others):
                parsed = parser.fallback("truncated")
        for tab in parsed:
            n += 1
            tid = f"{doc_id}:p{page_no}:t{n}"
            fallback = "fallback_plain" in tab.flags
            if fallback:
                markdown = tab.plain
                if tab.notes:
                    markdown += "\n\n" + "\n".join(tab.notes)
            else:
                markdown = to_markdown(tab.header, tab.printed, tab.notes)
            record = TableRecord(
                table_id=tid,
                doc_id=doc_id,
                page=page_no,
                bbox=tuple(round(v, 1) for v in tab.bbox),  # type: ignore[arg-type]
                caption=tab.caption,
                units=tab.units,
                header=[] if fallback else tab.header,
                rows=[] if fallback else tab.rows,
                markdown=markdown,
                continued_from=None,
                flags=list(tab.flags),
            )
            out.append((tab, record))
    _share_page_headers(out)
    return out


Item = tuple[ParsedTable, TableRecord]


def _share_page_headers(out: list[Item]) -> None:
    """A table without a header, under another table of the same page with as many columns,
    has that table's header (several tables share one set of columns)."""
    prev: Item | None = None
    for tab, rec in out:
        if "fallback_plain" in rec.flags:
            continue
        if prev is not None and "no_header" in rec.flags:
            _, prec = prev
            if any(prec.header) and len(prec.header) == len(rec.header):
                lead = tab.lead
                rec.header = [
                    *(a or b for a, b in zip(rec.header[:lead], prec.header[:lead], strict=True)),
                    *prec.header[lead:],
                ]
                rec.flags = [f for f in rec.flags if f != "no_header"] + ["header_inherited"]
                rec.markdown = to_markdown(rec.header, tab.printed, tab.notes)
        if any(rec.header):
            prev = (tab, rec)


def link_continued(items: list[Item]) -> None:
    """Set `continued_from` (the first page's table) on tables that continue an earlier page
    and let a continuation without a header reuse the first page's. `items` are one document's
    tables in reading order."""
    by_page: dict[int, list[Item]] = {}
    for item in items:
        by_page.setdefault(item[1].page, []).append(item)
    root_of: dict[str, str] = {}
    for page in sorted(by_page):
        _link_page(page, by_page, items, root_of)
        _inherit_row_labels(page, by_page, root_of)
    _inherit_legends(items)
    for tab, rec in items:
        if tab.continues_next:
            rec.flags.append("continues")


def _link_page(
    page: int, by_page: dict[int, list[Item]], items: list[Item], root_of: dict[str, str]
) -> None:
    """Link the first table of `page` to the last table of the page before it when the table
    above says '(Contd.)' (or this one says 'Continued') and both carry the same table number."""
    prev_items = by_page.get(page - 1, [])
    if not prev_items:
        return
    last_tab, last_rec = prev_items[-1]
    tab, rec = by_page[page][0]
    continues = last_tab.continues_next or tab.contd_caption
    same = last_tab.key == tab.key and (tab.key != "" or (not tab.caption and not last_tab.caption))
    if not (continues and same):
        return
    root = root_of.get(last_rec.table_id, last_rec.table_id)
    root_rec = next((r for _, r in items if r.table_id == root), None)
    if root_rec is None:
        return
    root_of[rec.table_id] = root
    rec.continued_from = root
    rec.flags.append("continued")
    if rec.units is None:
        rec.units = root_rec.units
    if "fallback_plain" in rec.flags or "fallback_plain" in root_rec.flags:
        return
    reuse = (not any(rec.header) or "no_header" in rec.flags) and len(rec.header) == len(
        root_rec.header
    )
    if reuse:
        rec.header = list(root_rec.header)
        rec.flags = [f for f in rec.flags if f != "no_header"]
        rec.flags.append("header_inherited")
        rec.markdown = to_markdown(rec.header, tab.printed, tab.notes)


def _inherit_row_labels(page: int, by_page: dict[int, list[Item]], root_of: dict[str, str]) -> None:
    """A table of bare numbers (the right half of a table that spans two pages sideways) takes
    its row labels from the table on the previous page whose rows sit at the same heights.
    Without such a partner it is not usable as cells and falls back to plain text."""
    for tab, rec in by_page[page]:
        if "no_row_labels" not in tab.flags:
            continue
        partner = None
        n = len(tab.row_tops)
        for ptab, prec in by_page.get(page - 1, []):
            if "fallback_plain" in prec.flags or "no_row_labels" in ptab.flags:
                continue
            for k in range(len(ptab.row_tops) - n + 1):  # rows k..k+n-1 line up with ours
                window = ptab.row_tops[k : k + n]
                if all(abs(a - b) <= 1.5 for a, b in zip(window, tab.row_tops, strict=True)):
                    partner = (ptab, prec, k)
                    break
            if partner:
                break
        if partner is None:
            text = tab.plain + ("\n\n" + "\n".join(tab.notes) if tab.notes else "")
            rec.header, rec.rows, rec.markdown = [], [], text
            rec.flags = [f for f in rec.flags if f != "no_row_labels"]
            rec.flags += ["fallback_plain", "reason:no_row_labels"]
            continue
        ptab, prec, k = partner
        lead = ptab.lead
        pick = slice(k, k + len(tab.rows))
        tab.rows = [[*p[:lead], *row[1:]] for p, row in zip(ptab.rows[pick], tab.rows, strict=True)]
        tab.printed = [
            [*p[:lead], *row[1:]] for p, row in zip(ptab.printed[pick], tab.printed, strict=True)
        ]
        tab.lead = lead
        rec.rows = tab.rows
        rec.header = [*prec.header[:lead], *rec.header[1:]] if prec.header else rec.header
        rec.flags += [f for f in prec.flags if f == "index_column" or f.startswith("lead:")]
        rec.markdown = to_markdown(rec.header, tab.printed, tab.notes)
        root = root_of.get(prec.table_id, prec.table_id)
        root_of[rec.table_id] = root
        rec.continued_from = root
        rec.flags = [f for f in rec.flags if f != "no_row_labels"]
        rec.flags += ["continued", "labels_inherited"]
        if rec.units is None:
            rec.units = prec.units


def _inherit_legends(items: list[Item]) -> None:
    """A continued table uses footnote marks ('#', '*') whose legend is printed under the first
    page only: add the legend lines of the marks it uses and does not explain itself."""
    by_id = {rec.table_id: tab for tab, rec in items}
    for tab, rec in items:
        if rec.continued_from is None or "fallback_plain" in rec.flags:
            continue
        root = by_id.get(rec.continued_from)
        if root is None:
            continue
        legends, own = marker_legends(root.notes), marker_legends(tab.notes)
        missing = [m for m in sorted(_markers_used(tab.printed)) if m in legends and m not in own]
        if missing:
            tab.notes.extend(f"{m} : {legends[m]}" for m in missing)
            rec.markdown = to_markdown(rec.header, tab.printed, tab.notes)


@dataclass
class DocTables:
    doc_id: str
    records: list[TableRecord]
    report: dict[str, Any]


def regions_from_report(rep: dict[str, Any]) -> dict[int, list[Region]]:
    """Rebuild the extraction's regions of one document from `extract_report.json`."""
    out: dict[int, list[Region]] = {}
    for page, regs in rep.get("regions", {}).items():
        out[int(page)] = [
            Region(r["kind"], tuple(r["bbox"]), r["caption"], r["n_rows"], r["source"])  # type: ignore[arg-type]
            for r in regs
        ]
    return out


# A word that names a unit somewhere in a units line ('Amount in Rs '000', '(in Millimetres)').
_UNIT_WORD_RE = re.compile(
    r"\b(?:rs|pkr|rupees?|us\s?\$|dollars?|million|billion|trillion|crore|lakh|thousands?|"
    r"percent(?:age)?|per\s?cent|numbers?|nos|no|mw|gw|gwh|kwh|mmcfd|tonnes?|tons?|mt|"
    r"hectares?|acres?|kgs?|litres?|liters?|ltrs?|millimetres?|mm|feet|index|indices|base\s+year|"
    r"bn|ratio|share)\b|[%$]|'000|\b000\b|\d+\s?=\s?100",
    re.I,
)


def _units_or_qualifier(rec: TableRecord) -> None:
    """A `units` text that names no unit is a section or period heading of the table ('1998
    Census*' over one block of a stacked table, 'Rural', '(Jul-Mar)'): it moves to the end of
    the caption ('TABLE 12.4 ... - 1998 Census*') so it is never shown as the unit."""
    if rec.units and not _UNIT_WORD_RE.search(rec.units):
        years = rec.units.split()
        lead = lead_columns(list(rec.flags))
        values = rec.header[lead:]
        if (
            values
            and not any(values)
            and len(years) == len(values)
            and all(_YEARISH_RE.match(y) for y in years)
        ):
            # 'Table 12.1: FP 2030 Indicators   2023   2024': the column headings printed on
            # the caption line, over the value columns that have none
            rec.header = [*rec.header[:lead], *years]
            rec.units = None
            _, rest = rec.markdown.split("\n", 1)
            rec.markdown = "| " + " | ".join(_esc(h) for h in rec.header) + " |\n" + rest
            return
        qualifier = rec.units
        rec.units = None
        rec.caption = f"{rec.caption} - {qualifier}" if rec.caption else qualifier


def document_word_counts(path: Path) -> Counter[str]:
    """How often each word (lower case; letters, digits and inner hyphens) is printed in a PDF,
    read with PyMuPDF: tells a real word from the second half of a word broken at a line end."""
    import pymupdf

    counts: Counter[str] = Counter()
    with pymupdf.open(path) as doc:
        for page in doc:
            for w in page.get_text("words"):
                counts[re.sub(r"[^\w-]", "", w[4].lower()).strip("-")] += 1  # keep 'year-on-year'
    return counts


def extract_tables(
    path: Path,
    doc_id: str,
    regions: dict[int, list[Region]],
    *,
    min_fill: float = 0.5,
    max_ragged: float = 0.25,
    infographic: bool = False,
) -> DocTables:
    """Parse every table region of one PDF."""
    import pdfplumber

    items: list[tuple[ParsedTable, TableRecord]] = []
    doc_words = document_word_counts(path)
    with pdfplumber.open(path) as pdf:
        for page_no in sorted(regions):
            if not any(r.kind == "table" for r in regions[page_no]):
                continue
            page = pdf.pages[page_no - 1]
            items.extend(
                page_tables(
                    page,
                    page_no,
                    doc_id,
                    regions[page_no],
                    min_fill=min_fill,
                    max_ragged=max_ragged,
                    infographic=infographic,
                    doc_words=doc_words,
                )
            )
            page.close()
    link_continued(items)
    records = [rec for _, rec in items]
    for rec in records:
        _units_or_qualifier(rec)
    n_fallback = sum(1 for r in records if "fallback_plain" in r.flags)
    reasons = Counter(
        f[len("reason:") :] for r in records for f in r.flags if f.startswith("reason:")
    )
    report = {
        "n_regions": sum(1 for rs in regions.values() for r in rs if r.kind == "table"),
        "n_tables": len(records),
        "n_fallback_plain": n_fallback,
        "reasons": dict(reasons),
        "n_continued": sum(1 for r in records if r.continued_from),
        "n_ruled": sum(1 for r in records if "ruled" in r.flags),
        "n_stacked": sum(1 for r in records if "stacked" in r.flags),
        "fallback": [r.table_id for r in records if "fallback_plain" in r.flags],
    }
    return DocTables(doc_id, records, report)


def _extract_job(args: tuple[str, str, dict[int, list[tuple]], float, float, bool]) -> DocTables:
    path, doc_id, raw_regions, min_fill, max_ragged, infographic = args
    regions = {
        p: [Region(k, tuple(b), c, n, s) for (k, b, c, n, s) in rs]  # type: ignore[arg-type]
        for p, rs in raw_regions.items()
    }
    return extract_tables(
        Path(path),
        doc_id,
        regions,
        min_fill=min_fill,
        max_ragged=max_ragged,
        infographic=infographic,
    )


def run_tables(
    manifest: Manifest,
    raw_dir: Path,
    extract_report: Path,
    out_tables: Path,
    out_report: Path,
    *,
    min_fill: float = 0.5,
    max_ragged: float = 0.25,
    workers: int = 0,
    on_doc: Callable[[DocTables], None] | None = None,
) -> dict[str, Any]:
    """Parse the tables of every manifest document into tables.parquet and a run report."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    from datetime import UTC, datetime

    import pyarrow as pa
    import pyarrow.parquet as pq

    rep = json.loads(extract_report.read_text(encoding="utf-8"))
    jobs = []
    for d in manifest.documents:
        regs = regions_from_report(rep["documents"][d.doc_id])
        raw = {
            p: [(r.kind, list(r.bbox), r.caption, r.n_rows, r.source) for r in rs]
            for p, rs in regs.items()
        }
        infographic = d.kind == DocKind.HIGHLIGHTS
        jobs.append((str(raw_dir / d.filename), d.doc_id, raw, min_fill, max_ragged, infographic))
    n_workers = workers or min(len(jobs), os.cpu_count() or 1)
    results: dict[str, DocTables] = {}
    if n_workers <= 1:
        for job in jobs:
            res = _extract_job(job)
            results[res.doc_id] = res
            if on_doc:
                on_doc(res)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for res in pool.map(_extract_job, jobs):
                results[res.doc_id] = res
                if on_doc:
                    on_doc(res)

    records = [r for d in manifest.documents for r in results[d.doc_id].records]
    schema = pa.schema(
        [
            ("table_id", pa.string()),
            ("doc_id", pa.string()),
            ("page", pa.int32()),
            ("bbox", pa.list_(pa.float64(), 4)),
            ("caption", pa.string()),
            ("units", pa.string()),
            ("header", pa.list_(pa.string())),
            ("rows", pa.string()),
            ("markdown", pa.string()),
            ("continued_from", pa.string()),
            ("flags", pa.list_(pa.string())),
        ]
    )
    rows = []
    for r in records:
        d = r.model_dump()
        d["rows"] = json.dumps(d["rows"], ensure_ascii=False)
        d["bbox"] = list(d["bbox"])
        rows.append(d)
    out_tables.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out_tables)

    n_fallback = sum(1 for r in records if "fallback_plain" in r.flags)
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_documents": len(manifest.documents),
        "n_regions": sum(results[d.doc_id].report["n_regions"] for d in manifest.documents),
        "n_tables": len(records),
        "n_fallback_plain": n_fallback,
        "fallback_share": round(n_fallback / max(len(records), 1), 4),
        "n_continued": sum(1 for r in records if r.continued_from),
        "reasons": dict(
            Counter(
                f[len("reason:") :] for r in records for f in r.flags if f.startswith("reason:")
            )
        ),
        "documents": {d.doc_id: results[d.doc_id].report for d in manifest.documents},
    }
    out_report.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report
