"""Page extraction with pdfplumber.

For every page this module produces a `PageLayout`:

- `text_raw`: pdfplumber's plain `extract_text()`; the plain_text ablation uses it as is.
- `regions`: table and figure boxes. pdfplumber's own table finder is unreliable on this
  corpus (it reports text boxes, chart bars and page borders as tables and misses row-label
  columns), so regions come from text alignment over pdfplumber words: rows whose cells are
  separated by wide gaps and hold numbers. A pdfplumber ruled-table box widens a region only
  when the extra area holds no prose. A region includes its caption ("Table 7.1: ...",
  "Fig-7.1 ...") and the header rows above, and the source/note lines below.
- `lines`: every word outside the regions, grouped into lines and put in reading order
  column by column (left column, then right, between full-width elements).

Header/footer removal, printed page numbers and paragraph assembly need the whole document
and live in `clean.py`.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from claimcheck.schemas import Manifest, PageRecord
from claimcheck.textnorm import is_number_token

if TYPE_CHECKING:
    from pdfplumber.page import Page

BBox = tuple[float, float, float, float]  # x0, top, x1, bottom (pdfplumber page coordinates)
Side = Literal["L", "R", "F"]

# Layout constants, in PDF points or as multiples of the font size.
LINE_TOL = 0.45  # words share a line if their vertical centres differ by <= LINE_TOL * size
SEG_GAP = 1.4  # a horizontal gap > SEG_GAP * size (min SEG_GAP_MIN) separates table cells
SEG_GAP_MIN = 8.0
ROW_GAP = 2.5  # rows of one table are at most ROW_GAP * size apart vertically
ROW_GAP_MIN = 12.0
MIN_WORD_SIZE = 4.0  # smaller text is chart micro-text or invisible
PROSE_MIN_WIDTH = 150.0  # a body-text column is ~212 pt wide; header cells are narrower

# A caption number ('7.1', '2.1 B', '4(a)', '7.1 (C)', 'IV') followed by the end of the line,
# punctuation or a capitalised title. 'Table 1.4 shows that ...' is prose, not a caption.
_CAP_NUM = (
    r"[-:.]?\s*(?>\d+(?:[.\-]\d+)*|[IVX]+)"
    r"(?:\s?\(?[A-Za-z]\)?(?![A-Za-z]))?+"
    r"(?:\s?\((?:[A-Za-z]|contd?\.?|continued)\))?"
)
_CAP_END = r"(?=\s*(?:$|[:.\-–(]|[A-Z]))"
TABLE_CAPTION_RE = re.compile(rf"^\s*(?:TABLE|Table)\s*{_CAP_NUM}{_CAP_END}")
FIGURE_CAPTION_RE = re.compile(
    rf"^\s*(?:Fig(?:ure)?|FIG(?:URE)?|Chart|CHART|Graph|GRAPH)\s*\.?\s*{_CAP_NUM}{_CAP_END}"
)
NOTE_RE = re.compile(
    r"^\s*(?:Sources?\b|Notes?\b|N\.?B\b|Memo\b|Provisional\b|Revised\b|Figures? in\b"
    r"|[*^@#†‡]|[-–—.]+\s?:|\S{1,12}\s?:\s)"
)
_PLACEHOLDERS = {
    "-", "–", "—", "--", "..", "...", "…", "n.a", "n.a.", "na", "n/a", "nil",
    "neg", "p", "r", "e", "(p)", "(r)", "(e)", "*", "**", "#", "%", "(%)", "x",
}  # fmt: skip
_FY_TOKEN_RE = re.compile(r"^(?:fy)?['’]?(?:\d{4}[-–/]\d{2,4}|\d{2,4})[a-z*#]?$", re.I)
_SUPERSCRIPT_RE = re.compile(r"^[\d*†‡,]+$")


# --------------------------------------------------------------------------- primitives


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2


def token_kind(token: str) -> Literal["num", "ph", "text"]:
    """Classify a word as a number, a table placeholder ('-', 'P', 'n.a.') or text."""
    t = token.strip().lower()
    core = t.rstrip("*#^@†‡")
    if core and (is_number_token(core) or _FY_TOKEN_RE.match(core)):
        return "num"
    if t in _PLACEHOLDERS:
        return "ph"
    return "text"


@dataclass
class Segment:
    """Words of one line between two wide gaps: a table cell, or a column of prose."""

    words: list[Word]

    @cached_property
    def kinds(self) -> list[str]:
        return [token_kind(w.text) for w in self.words]

    @property
    def x0(self) -> float:
        return self.words[0].x0

    @property
    def x1(self) -> float:
        return self.words[-1].x1

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def n_num(self) -> int:
        return self.kinds.count("num")

    @property
    def numeric(self) -> bool:
        return self.n_num >= 1 and "text" not in self.kinds

    @property
    def cell(self) -> bool:
        """A number or placeholder cell ('1,234', '-', 'n.a.'): no words."""
        return "text" not in self.kinds

    @property
    def prose(self) -> bool:
        """A run of words as wide as a text column (table header cells are narrower)."""
        n_text = self.kinds.count("text")
        wide = self.x1 - self.x0 >= PROSE_MIN_WIDTH or n_text >= 9
        return wide and n_text >= 5 and n_text >= 0.6 * len(self.kinds)


@dataclass
class Line:
    words: list[Word]  # sorted left to right
    side: Side = "F"
    band: int = 0

    @property
    def x0(self) -> float:
        return self.words[0].x0

    @property
    def x1(self) -> float:
        return max(w.x1 for w in self.words)

    @property
    def top(self) -> float:
        return min(w.top for w in self.words)

    @property
    def bottom(self) -> float:
        return max(w.bottom for w in self.words)

    @property
    def cy(self) -> float:
        return statistics.fmean(w.cy for w in self.words)

    @cached_property
    def size(self) -> float:
        return statistics.median(w.size for w in self.words)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @cached_property
    def segments(self) -> list[Segment]:
        gap = max(SEG_GAP_MIN, SEG_GAP * self.size)
        segs: list[list[Word]] = [[self.words[0]]]
        for prev, w in zip(self.words, self.words[1:], strict=False):
            if w.x0 - prev.x1 > gap:
                segs.append([w])
            else:
                segs[-1].append(w)
        return [Segment(s) for s in segs]

    @cached_property
    def tabular(self) -> bool:
        """A table row: numbers in cells separated by wide gaps, at most two text cells.
        A short label followed by a single number cell also counts ('JAPAN    2,589')."""
        segs = self.segments
        numeric = [s for s in segs if s.numeric]
        n_num = sum(s.n_num for s in numeric)
        if len(segs) == 1:
            return segs[0].numeric and n_num >= 3
        if n_num >= 2:
            return sum(1 for s in segs if not s.cell) <= 2
        if n_num != 1:
            return False
        if not any(s.prose for s in segs):
            return True
        # A long row label, then a single number far to the right.
        return len(segs) == 2 and segs[1].numeric and segs[1].x0 - segs[0].x1 >= 4 * self.size


def group_lines(words: list[Word]) -> list[Line]:
    """Group words into lines by vertical centre, then sort each line left to right."""
    groups: list[list[Word]] = []
    centres: list[float] = []
    for w in sorted(words, key=lambda w: (w.cy, w.x0)):
        if groups and abs(w.cy - centres[-1]) <= LINE_TOL * max(w.size, groups[-1][0].size):
            g = groups[-1]
            g.append(w)
            centres[-1] += (w.cy - centres[-1]) / len(g)
        else:
            groups.append([w])
            centres.append(w.cy)
    return [Line(sorted(g, key=lambda w: w.x0)) for g in groups]


def _row_gap(line: Line) -> float:
    return max(ROW_GAP * line.size, ROW_GAP_MIN)


def _overlaps(a0: float, a1: float, b0: float, b1: float, pad: float = 0.0) -> bool:
    return a0 < b1 + pad and b0 < a1 + pad


def _union(boxes: list[BBox]) -> BBox:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _seg_box(seg: Segment, line: Line) -> BBox:
    return (seg.x0, min(w.top for w in seg.words), seg.x1, max(w.bottom for w in seg.words))


def _inside(w: Word, box: BBox) -> bool:
    cx = (w.x0 + w.x1) / 2
    return box[0] <= cx <= box[2] and box[1] <= w.cy <= box[3]


# --------------------------------------------------------------------------- regions


@dataclass(frozen=True)
class Region:
    kind: Literal["table", "figure"]
    bbox: BBox
    caption: str | None
    n_rows: int  # table rows found by alignment (0 for a chart found by its caption)
    source: str  # "aligned", "aligned+ruled", "caption"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "bbox": [round(v, 1) for v in self.bbox],
            "caption": self.caption,
            "n_rows": self.n_rows,
            "source": self.source,
        }


@dataclass
class _Run:
    lines: list[Line]
    x0: float
    x1: float
    n_tab: int = 1
    pending: list[Line] = field(default_factory=list)

    @property
    def bottom(self) -> float:
        return (self.pending or self.lines)[-1].bottom


@dataclass
class _Cand:
    box: BBox
    n_tab: int
    size: float  # median font size of the rows
    rows: list[float] = field(default_factory=list)  # vertical centres of the rows
    caption: str | None = None
    source: str = "aligned"


def _table_segments(line: Line) -> list[Segment]:
    return [s for s in line.segments if not s.prose]


def _segs_in(line: Line, x0: float, x1: float, pad: float = 4.0) -> list[Segment]:
    return [s for s in line.segments if _overlaps(s.x0, s.x1, x0, x1, pad)]


def _find_runs(lines: list[Line]) -> list[_Run]:
    """Runs of tabular lines that line up horizontally and follow each other closely."""
    runs: list[_Run] = []
    cur: _Run | None = None

    def close() -> None:
        nonlocal cur
        if cur is not None:
            cur.pending.clear()
            runs.append(cur)
        cur = None

    for line in lines:
        tsegs = _table_segments(line)
        max_gap = _row_gap(line)
        if line.tabular and tsegs:
            x0, x1 = min(s.x0 for s in tsegs), max(s.x1 for s in tsegs)
            if (
                cur is not None
                and line.top - cur.bottom <= max_gap
                and _overlaps(x0, x1, cur.x0, cur.x1)
            ):
                cur.lines.extend(cur.pending)
                cur.pending.clear()
                cur.lines.append(line)
                cur.x0, cur.x1 = min(cur.x0, x0), max(cur.x1, x1)
                cur.n_tab += 1
            else:
                close()
                cur = _Run([line], x0, x1)
            continue
        if cur is None:
            continue
        inside = _segs_in(line, cur.x0, cur.x1)
        if not inside:
            continue
        if (
            line.top - cur.bottom <= max_gap
            and not any(s.prose for s in inside)
            and len(cur.pending) < 3
        ):
            cur.pending.append(line)  # wrapped row label, sub-heading or blank row
        else:
            close()
    close()
    return runs


def _split_streams(lines: list[Line], gx: float) -> list[list[Line]]:
    """Left-column and right-column parts of every line (a segment goes by its centre)."""
    left, right = [], []
    for line in lines:
        lw = [w for s in line.segments if (s.x0 + s.x1) / 2 < gx for w in s.words]
        rw = [w for s in line.segments if (s.x0 + s.x1) / 2 >= gx for w in s.words]
        if lw:
            left.append(Line(lw))
        if rw:
            right.append(Line(rw))
    return [left, right]


def _caption_of(segs: list[Segment]) -> str | None:
    text = " ".join(s.text for s in segs).strip()
    if TABLE_CAPTION_RE.match(text) or FIGURE_CAPTION_RE.match(text):
        return text
    return None


def _extend_up(lines: list[Line], box: BBox, max_lines: int = 10) -> tuple[list[BBox], str | None]:
    """Caption, title, units and header lines above a table.

    Walks up through closely spaced lines inside the table's x-range. If a caption
    ('Table 7.1: ...', 'TABLE 1.8 b') is reached, everything up to it belongs to the table;
    otherwise only the non-prose lines directly above it do."""
    above = sorted((ln for ln in lines if ln.cy < box[1]), key=lambda ln: ln.top, reverse=True)
    top = box[1]
    walked: list[tuple[Line, list[Segment]]] = []
    for line in above:
        if len(walked) >= max_lines or top - line.bottom > _row_gap(line) + 2:
            break
        segs = _segs_in(line, box[0], box[2])
        if not segs:
            continue
        walked.append((line, segs))
        top = line.top
        caption = _caption_of(segs)
        if caption:
            title = [
                " ".join(s.text for s in sg)
                for _, sg in reversed(walked[:-1][-2:])
                if len(sg) == 1 and not any(token_kind(w.text) == "num" for w in sg[0].words)
            ]
            boxes = [_seg_box(s, ln) for ln, sg in walked for s in sg]
            return boxes, " ".join([caption, *title]) if len(caption) < 25 else caption
    boxes = []
    for line, segs in walked:
        if any(s.prose for s in segs):
            break
        boxes.extend(_seg_box(s, line) for s in segs)
    return boxes, None


def _extend_down(lines: list[Line], box: BBox, size: float, max_lines: int = 8) -> list[BBox]:
    """Source, note and footnote lines under a table."""
    below = sorted((ln for ln in lines if ln.cy > box[3]), key=lambda ln: ln.top)
    boxes: list[BBox] = []
    bottom = box[3]
    after_source = False
    taken = 0
    for line in below:
        if taken >= max_lines or line.top - bottom > _row_gap(line) + 2:
            break
        segs = _segs_in(line, box[0], box[2])
        if not segs:
            continue
        if any(_caption_of([s]) for s in segs):
            break  # the next table or chart
        text = " ".join(s.text for s in segs)
        small = all(w.size <= size + 0.3 for s in segs for w in s.words)
        if NOTE_RE.match(text) or (small and not after_source and not any(s.prose for s in segs)):
            boxes.extend(_seg_box(s, line) for s in segs)
            bottom = line.bottom
            after_source = after_source or text.lstrip().lower().startswith("source")
            taken += 1
            continue
        break
    return boxes


def _figure_from_caption(
    lines: list[Line], cap: Line, gutter: Gutter | None, page_w: float
) -> BBox | None:
    """A chart below a 'Fig-...' caption: non-prose lines down to its 'Source:' line."""
    segs = [s for s in cap.segments if FIGURE_CAPTION_RE.match(s.text)]
    if not segs:
        return None
    x0, x1 = 0.0, page_w
    if gutter is not None:
        gx = gutter.gx
        if segs[0].x1 <= gx + 2:
            x1 = gx  # caption in the left column
        elif segs[0].x0 >= gx - 2:
            x0 = gx
    boxes = [_seg_box(s, cap) for s in _segs_in(cap, x0, x1, pad=-2)]
    bottom = cap.bottom
    found = False
    for line in sorted((ln for ln in lines if ln.cy > cap.cy), key=lambda ln: ln.top)[:60]:
        if line.top - bottom > 6 * line.size:
            break
        segs_in = _segs_in(line, x0, x1, pad=-2)
        if not segs_in:
            continue
        text = " ".join(s.text for s in segs_in)
        if any(s.prose for s in segs_in) and not NOTE_RE.match(text):
            break
        boxes.extend(_seg_box(s, line) for s in segs_in)
        bottom = line.bottom
        found = True
        if text.lstrip().lower().startswith("source"):
            break
    return _union(boxes) if found else None


def _merge_cands(cands: list[_Cand], join_gap: float, lines: list[Line]) -> list[_Cand]:
    """Merge candidates that overlap, or that are the two halves of one table (side by side
    on the same rows). A merge never takes in a paragraph of prose."""

    def joinable(a: BBox, b: BBox) -> bool:
        if _overlaps(a[0], a[2], b[0], b[2]) and _overlaps(a[1], a[3], b[1], b[3], pad=2):
            return True
        v = min(a[3], b[3]) - max(a[1], b[1])
        h_gap = max(a[0], b[0]) - min(a[2], b[2])
        if v < 0.6 * max(a[3] - a[1], b[3] - b[1]) or h_gap > join_gap:
            return False
        u = _union([a, b])
        return not any(
            s.prose
            and _box_contains_point(u, s)
            and not _box_contains_point(a, s)
            and not _box_contains_point(b, s)
            for line in lines
            for s in line.segments
        )

    items = sorted(cands, key=lambda c: c.box[1])
    changed = True
    while changed:
        changed = False
        out: list[_Cand] = []
        for c in items:
            for o in out:
                if joinable(c.box, o.box):
                    o.box = _union([c.box, o.box])
                    o.n_tab += c.n_tab
                    o.size = min(o.size, c.size)
                    o.rows += c.rows
                    o.caption = o.caption or c.caption
                    changed = True
                    break
            else:
                out.append(c)
        items = out
    return items


def _join_stacked(cands: list[_Cand], lines: list[Line], max_gap: float = 30.0) -> list[_Cand]:
    """Join two blocks of one table stacked closely on top of each other: the lines between
    them (wrapped row labels, sub-headings) are in the table's font, and none is a caption
    or a source line."""
    items = sorted(cands, key=lambda c: c.box[1])
    out: list[_Cand] = []
    for c in items:
        prev = out[-1] if out else None
        if prev is not None:
            a, b = prev.box, c.box
            gap = b[1] - a[3]
            width = min(a[2] - a[0], b[2] - b[0])
            x_overlap = min(a[2], b[2]) - max(a[0], b[0])
            between = [ln for ln in lines if a[3] < ln.cy < b[1] and _segs_in(ln, a[0], a[2])]
            if (
                0 <= gap <= max_gap
                and x_overlap >= 0.5 * width
                and c.caption is None
                and all(
                    ln.size <= max(prev.size, c.size) + 0.3
                    and not any(_caption_of([s]) for s in ln.segments)
                    and not NOTE_RE.match(ln.text)
                    for ln in between
                )
            ):
                prev.box = _union([a, b, *(_seg_box(s, ln) for ln in between for s in ln.segments)])
                prev.n_tab += c.n_tab
                prev.rows += c.rows
                continue
        out.append(c)
    return out


def _widen_across_gutter(
    cand: _Cand, streams: list[list[Line]], gx: float, prose_size: float
) -> None:
    """A table found in one column whose other column holds only its row labels spans the
    page: take the other column's words too. Labels sit on the table's rows and the other
    column holds no paragraph text."""
    box = cand.box
    if cand.n_tab < 2:
        return
    if box[0] >= gx - 4:
        other = streams[0]
    elif box[2] <= gx + 4:
        other = streams[1]
    else:
        return
    beside = [ln for ln in other if box[1] - 1 <= ln.cy <= box[3] + 1]
    if not beside:
        return
    if any(
        s.prose and statistics.median(w.size for w in s.words) >= prose_size - 0.6
        for ln in beside
        for s in ln.segments
    ):
        return  # a column-width table or chart beside a column of body text
    tol = LINE_TOL * cand.size
    aligned = sum(1 for ln in beside if any(abs(ln.cy - r) <= tol for r in cand.rows))
    if aligned < 0.6 * len(beside):
        return
    cand.box = _union([box, *(_seg_box(s, ln) for ln in beside for s in ln.segments)])


def find_regions(
    lines: list[Line],
    page_w: float,
    ruled: list[BBox] | None = None,
    gutter: Gutter | None = None,
    curves: list[BBox] | None = None,
) -> list[Region]:
    """Table and figure regions of a page. `lines` are all upright lines of the page."""
    lines = sorted(lines, key=lambda ln: (ln.top, ln.x0))
    gx = gutter.gx if gutter else None
    streams = [lines] if gx is None else _split_streams(lines, gx)

    prose_size = prose_font_size(lines)

    def body_prose_in(box: BBox) -> bool:
        return any(
            s.prose
            and statistics.median(w.size for w in s.words) >= prose_size - 0.6
            and _box_contains_point(box, s)
            for ln in lines
            for s in ln.segments
        )

    def cand_of(run: _Run, all_segments: bool = False) -> _Cand:
        boxes = [
            _seg_box(s, ln)
            for ln in run.lines
            for s in (ln.segments if all_segments else _table_segments(ln))
            if all_segments or _overlaps(s.x0, s.x1, run.x0, run.x1)
        ]
        size = statistics.median(ln.size for ln in run.lines)
        return _Cand(_union(boxes), run.n_tab, size, [ln.cy for ln in run.lines])

    cands: list[_Cand] = []
    for stream in streams:
        cands.extend(cand_of(run) for run in _find_runs(stream))
    if gx is not None:
        # Small tables whose labels sit in the left column and numbers in the right one
        # ('Total number of funds | 66') are table rows only as whole lines.
        for run in _find_runs(lines):
            straddle = all(
                any(s.x1 <= gx for s in ln.segments) and any(s.x0 >= gx for s in ln.segments)
                for ln in run.lines
                if ln.tabular
            )
            cand = cand_of(run, all_segments=True)
            if straddle and not body_prose_in(cand.box):
                cands.append(cand)
    join_gap = gutter.width + 30 if gutter else 30.0
    cands = _merge_cands(cands, join_gap, lines)
    if gx is not None:
        for c in cands:
            _widen_across_gutter(c, streams, gx, prose_size)

    for c in cands:
        up, c.caption = _extend_up(lines, c.box)
        down = _extend_down(lines, c.box, c.size)
        c.box = _union([c.box, *up, *down])
    cands = _join_stacked(cands, lines)
    cands = [c for c in cands if c.n_tab >= 3 or c.caption]

    # Charts announced by a 'Fig-' caption whose labels did not form table rows.
    for line in lines:
        cap_segs = [s for s in line.segments if FIGURE_CAPTION_RE.match(s.text)]
        if not cap_segs or any(_box_contains_point(c.box, cap_segs[0]) for c in cands):
            continue
        box = _figure_from_caption(lines, line, gutter, page_w)
        if box is not None:
            caption = " ".join(s.text for s in cap_segs)
            cands.append(_Cand(box, 0, line.size, [], caption, "caption"))

    regions = []
    for c in _merge_cands(cands, 0.0, lines):
        box, source = c.box, c.source
        if ruled:
            box, widened = _widen_with_ruled(box, ruled, lines)
            if widened:
                source = f"{source}+ruled"
        kind = _region_kind(c.caption, box, curves or [])
        regions.append(Region(kind, box, c.caption, c.n_tab, source))
    return regions


def _box_contains_point(box: BBox, seg: Segment) -> bool:
    cx, cy = (seg.x0 + seg.x1) / 2, seg.words[0].cy
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _widen_with_ruled(box: BBox, ruled: list[BBox], lines: list[Line]) -> tuple[BBox, bool]:
    """Union with an overlapping pdfplumber ruled-table box if that captures no prose."""
    widened = False
    for rb in ruled:
        ix = min(box[2], rb[2]) - max(box[0], rb[0])
        iy = min(box[3], rb[3]) - max(box[1], rb[1])
        if ix <= 0 or iy <= 0:
            continue
        smaller = min((box[2] - box[0]) * (box[3] - box[1]), (rb[2] - rb[0]) * (rb[3] - rb[1]))
        if smaller <= 0 or ix * iy < 0.2 * smaller:
            continue
        cand = _union([box, rb])
        if cand == box:
            continue
        gains_prose = any(
            s.prose and _box_contains_point(cand, s) and not _box_contains_point(box, s)
            for line in lines
            for s in line.segments
        )
        if not gains_prose:
            box, widened = cand, True
    return box, widened


def _region_kind(caption: str | None, box: BBox, curves: list[BBox]) -> Literal["table", "figure"]:
    if caption:
        return "table" if TABLE_CAPTION_RE.match(caption) else "figure"
    inside = sum(
        1
        for c in curves
        if c[0] >= box[0] - 2 and c[2] <= box[2] + 2 and c[1] >= box[1] - 2 and c[3] <= box[3] + 2
    )
    return "figure" if inside >= 3 else "table"


# --------------------------------------------------------------------------- reading order


@dataclass(frozen=True)
class Gutter:
    """The empty strip between two text columns."""

    x0: float  # right edge of the left column
    x1: float  # left edge of the right column
    gx: float  # the split line

    @property
    def width(self) -> float:
        return self.x1 - self.x0


def prose_font_size(lines: list[Line]) -> float:
    """Most common font size of the page's prose (0 if it has none)."""
    sizes = Counter(
        round(w.size * 2) / 2 for ln in lines for s in ln.segments if s.prose for w in s.words
    )
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _coverage(words: list[Word], width: int) -> np.ndarray:
    """How many words span each integer x in [0, width)."""
    cov = np.zeros(width + 1, dtype=np.int32)
    for w in words:
        a, b = max(0, int(np.ceil(w.x0))), min(width - 1, int(np.floor(w.x1)))
        if a <= b:
            cov[a] += 1
            cov[b + 1] -= 1
    return np.cumsum(cov)[:width]


def _longest_run(mask: np.ndarray) -> tuple[int, int] | None:
    """[start, end) of the longest run of True values."""
    best, start = None, None
    for i, v in enumerate([*mask.tolist(), False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def find_gutter(words: list[Word], page_w: float) -> Gutter | None:
    """The gutter between two text columns, if the page has one.

    Only prose words (runs of words as wide as a column) define the columns, so tables and
    boxes that cross the gutter do not hide it: the gutter is the valley of prose coverage in
    the middle of the page (at most half the typical coverage, at least 6 pt wide), bordered
    by prose lines that end at its left edge or start at its right edge. The split line is
    then placed where the fewest words of any kind cross the valley."""
    if len(words) < 20:
        return None
    lines = group_lines(words)
    # Body text only: long table labels in a smaller font also look like prose.
    body = prose_font_size(lines)
    prose_segs = [
        s
        for ln in lines
        for s in ln.segments
        if s.prose
        and statistics.median(w.size for w in s.words) >= body - 0.6
        and not _caption_of([s])  # a figure caption is centred across the gutter
    ]
    prose = [w for s in prose_segs for w in s.words]
    if len(prose) < 20:
        return None
    width = int(page_w) + 1
    cov_p = _coverage(prose, width)
    lo, hi = int(0.3 * page_w), int(0.75 * page_w)
    ref = float(np.median(cov_p[cov_p > 0]))
    mid = cov_p[lo : hi + 1]
    low = int(mid.min())
    if low > 0.5 * ref:
        return None
    run = _longest_run(mid <= low + 0.1 * ref)
    if run is None or run[1] - run[0] < 6:
        return None
    a, b = lo + run[0], lo + run[1]
    edges = sum(1 for s in prose_segs if abs(s.x1 - a) <= 12 or abs(s.x0 - b) <= 12)
    if edges < 3:
        return None  # a wide gap inside a table (labels | numbers) is not a gutter
    # footnotes in small type run across the gutter at the foot of the page: only text of
    # body size places the split
    cov_a = _coverage([w for w in words if w.size >= body - 0.6], width)[a:b]
    sub = _longest_run(cov_a == cov_a.min())
    gx = a + (sub[0] + sub[1]) / 2 if sub else (a + b) / 2
    left = [w.x1 for w in words if w.x1 <= gx]
    right = [w.x0 for w in words if w.x0 >= gx]
    if len(left) < 5 or len(right) < 5:
        return None
    return Gutter(max(left), min(right), gx)


def find_partial_gutter(words: list[Word], page_w: float) -> Gutter | None:
    """The gutter of a page whose two-column text shares the page with full-width text
    or with full-width tables (pes2526_ch04 p10: two columns over a full-width box whose lines
    fill the valley that `find_gutter` looks for; pes2526_ch02 p19: a few lines of two columns
    between tables). The evidence is the lines themselves: at least 3 lines hold two prose
    segments of body size side by side, split near the middle of the page, and the right-hand
    segments start at one x (the right column's edge). The short last lines of a single-column
    page's paragraphs have nothing beside them, so they make no gutter."""
    lines = group_lines(words)
    body = prose_font_size(lines)
    lo, hi = 0.3 * page_w, 0.75 * page_w
    pairs: list[tuple[float, float]] = []
    for ln in lines:
        segs = [
            s
            for s in ln.segments
            if s.prose and statistics.median(w.size for w in s.words) >= body - 0.6
        ]
        for a, b in zip(segs, segs[1:], strict=False):
            if lo <= (a.x1 + b.x0) / 2 <= hi:
                pairs.append((a.x1, b.x0))
    if len(pairs) < 3:
        return None
    edge = statistics.median(b for _, b in pairs)
    pairs = [(a, b) for a, b in pairs if abs(b - edge) <= 4]
    if len(pairs) < 3:
        return None
    a, b = max(a for a, _ in pairs), min(b for _, b in pairs)
    if b - a < 6:
        return None
    # the edges come from the paired lines: full-width lines have words near the split, and a
    # narrow gutter would leave their wide justified spaces unmerged (`_merge_side_lines`)
    return Gutter(a, b, (a + b) / 2)


def _merge_side_lines(full: list[Line], left: list[Line], right: list[Line], gap: float) -> None:
    """Attach left/right fragments to full-width lines (or to each other) when they sit on
    the same baseline with an ordinary word gap between them. Mutates the lists."""

    def same_row(a: Line, b: Line) -> bool:
        return abs(a.cy - b.cy) <= LINE_TOL * max(a.size, b.size)

    for line in full:
        for pool in (left, right):
            for frag in list(pool):
                near = (
                    frag.x1 <= line.x0 and line.x0 - frag.x1 <= gap
                    if pool is left
                    else frag.x0 >= line.x1 and frag.x0 - line.x1 <= gap
                )
                if same_row(line, frag) and near:
                    line.words = sorted(line.words + frag.words, key=lambda w: w.x0)
                    pool.remove(frag)
    for lf in list(left):
        for rf in list(right):
            if same_row(lf, rf) and 0 <= rf.x0 - lf.x1 <= gap:
                full.append(Line(sorted(lf.words + rf.words, key=lambda w: w.x0)))
                left.remove(lf)
                right.remove(rf)
                break


def reading_order(
    words: list[Word],
    page_w: float,
    regions: list[Region],
    panels: list[BBox] | None = None,
) -> tuple[list[Line], Gutter | None]:
    """Lines in reading order. With a gutter: left column, then right column, within each
    band between full-width lines or full-width regions. A text panel (`text_panels`) is read
    on its own, in its own column order, and its lines come where it stands on the page.
    Returns (lines, gutter)."""
    panels = panels or []
    if panels:
        outside = [w for w in words if not any(_inside(w, p) for p in panels)]
        seps = [Region("figure", p, None, 0, "panel") for p in panels]  # cuts the columns
        lines, gutter = reading_order(outside, page_w, [*regions, *seps])
        for p in sorted(panels, key=lambda b: b[1]):
            inner, _ = reading_order([w for w in words if _inside(w, p)], page_w, [])
            for ln in inner:
                ln.side = "F"
            mid = (p[1] + p[3]) / 2
            across = gutter is None or p[0] < gutter.gx < p[2]
            side = "L" if gutter is not None and p[2] <= gutter.gx else "R"
            mine = [i for i, ln in enumerate(lines) if across or ln.side in (side, "F")]
            at = next((i for i in mine if lines[i].cy > mid), (mine[-1] + 1) if mine else 0)
            lines[at:at] = inner
        return lines, gutter
    gutter = find_gutter(words, page_w) or find_partial_gutter(words, page_w)
    if gutter is None:
        lines = sorted(group_lines(words), key=lambda ln: ln.top)
        for i, line in enumerate(lines):
            line.side, line.band = "F", i
        return lines, None
    gx = gutter.gx
    # Display headings ('Chapter 7' in 32 pt) span the page in reading order. The scale is
    # the largest size of a quarter of the words: on a page with many small footnotes
    # (pes2425_ch02 p9) the most common size alone would make the body text 'headings'.
    sizes = Counter(round(w.size) for w in words)
    major = [s for s, n in sizes.items() if n >= 0.25 * len(words)]
    heading = 1.4 * max(major or [sizes.most_common(1)[0][0]])
    left = group_lines([w for w in words if w.x1 <= gx and w.size < heading])
    right = group_lines([w for w in words if w.x0 >= gx and w.size < heading])
    full = group_lines([w for w in words if (w.x0 < gx < w.x1) or w.size >= heading])
    _merge_side_lines(full, left, right, gap=min(max(0.6 * gutter.width, 6.0), 14.0))

    separators: list[tuple[float, Line | None]] = [(ln.cy, ln) for ln in full]
    # a region crosses the gutter when it reaches well into both columns (a column-width
    # frame drawn a few points past the gutter does not)
    reach = 12.0
    separators += [
        ((r.bbox[1] + r.bbox[3]) / 2, None)
        for r in regions
        if r.bbox[0] < gx - reach and r.bbox[2] > gx + reach
    ]
    # two figures or tables facing each other across the gutter cut both columns too
    lefts = [r.bbox for r in regions if r.bbox[2] <= gx + reach and r.bbox[0] < gx - reach]
    rights = [r.bbox for r in regions if r.bbox[0] >= gx - reach and r.bbox[2] > gx + reach]
    for a in lefts:
        for b in rights:
            ov = min(a[3], b[3]) - max(a[1], b[1])
            if ov >= 0.5 * min(a[3] - a[1], b[3] - b[1]):
                separators.append(((max(a[1], b[1]) + min(a[3], b[3])) / 2, None))
    separators.sort(key=lambda s: s[0])
    cuts = [s[0] for s in separators]

    def band_of(line: Line) -> int:
        return int(np.searchsorted(cuts, line.cy))

    out: list[Line] = []
    bands: dict[int, dict[str, list[Line]]] = {}
    for side, pool in (("L", left), ("R", right)):
        for line in pool:
            line.side = side  # type: ignore[assignment]
            line.band = band_of(line)
            bands.setdefault(line.band, {"L": [], "R": []})[side].append(line)
    for k in range(len(separators) + 1):
        for side in ("L", "R"):
            out.extend(sorted(bands.get(k, {}).get(side, []), key=lambda ln: ln.top))
        if k < len(separators) and separators[k][1] is not None:
            sep = separators[k][1]
            sep.side, sep.band = "F", k
            out.append(sep)
    return out, gutter


# --------------------------------------------------------------------------- page


@dataclass
class PageLayout:
    page: int  # 1-based
    width: float
    height: float
    text_raw: str
    lines: list[Line]  # body lines (outside regions) in reading order
    regions: list[Region]
    n_chars: int
    body_size: float
    gutter: Gutter | None
    dropped_words: int  # rotated, tiny, superscript markers and chart labels
    review: list[str] = field(default_factory=list)  # why the page needs its source page

    @property
    def no_text_layer(self) -> bool:
        return self.n_chars == 0

    @property
    def two_column(self) -> bool:
        return (
            self.gutter is not None
            and any(ln.side == "L" for ln in self.lines)
            and any(ln.side == "R" for ln in self.lines)
        )


def page_words(page: Page) -> tuple[list[Word], int]:
    """Upright words of at least MIN_WORD_SIZE pt, plus the count of words dropped."""
    raw = page.extract_words(
        x_tolerance=3, y_tolerance=3, keep_blank_chars=False, extra_attrs=["size", "upright"]
    )
    words, dropped = [], 0
    for w in raw:
        if not w["upright"] or w["size"] < MIN_WORD_SIZE or not w["text"].strip():
            dropped += 1
            continue
        words.append(Word(w["text"], w["x0"], w["x1"], w["top"], w["bottom"], w["size"]))
    return words, dropped


def body_font_size(page: Page) -> float:
    sizes = Counter(round(c["size"] * 2) / 2 for c in page.chars if c["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _drop_superscripts(line: Line) -> int:
    """Remove footnote markers: small, raised digits or asterisks inside a line."""
    if len(line.words) < 2:
        return 0
    size = line.size
    base = statistics.median(w.bottom for w in line.words if w.size >= 0.85 * size)
    keep = [
        w
        for w in line.words
        if not (
            w.size <= 0.8 * size and _SUPERSCRIPT_RE.match(w.text) and w.bottom < base - 0.2 * size
        )
    ]
    dropped = len(line.words) - len(keep)
    line.words = keep
    return dropped


def _is_chart_label(line: Line, body_size: float, regions: list[Region]) -> bool:
    """Leftover chart text: a small line holding only numbers or placeholders, or any small
    non-prose line level with a figure (axis labels, legends)."""
    if line.size > body_size - 1.5:
        return False
    if all(token_kind(w.text) != "text" for w in line.words):
        return True
    return not any(s.prose for s in line.segments) and any(
        r.kind == "figure" and _overlaps(line.top, line.bottom, r.bbox[1], r.bbox[3], pad=6)
        for r in regions
    )


_ORDINALS = {"st", "nd", "rd", "th"}


def _unreliable(line: Line, body_size: float) -> str | None:
    """Why a body line cannot be trusted as read, or None: prose with two or more words of
    chart-label size inside it ('… share of 18.7 percent followed by Oil & Gas Fig-6.3 Top
    Five Sectors …') holds labels of a chart the layout rules did not find."""
    if len(line.words) < 4:
        return None
    small = [  # ordinal suffixes ('29 th', '2 nd') are set small in running text
        w for w in line.words if w.size <= 0.75 * line.size and w.text not in _ORDINALS
    ]
    prose = any(s.prose for s in line.segments)
    if prose and len(small) >= 2 and line.size >= body_size - 1.5:
        return "chart_text_in_prose"
    return None


def _ruled_boxes(page: Page) -> list[BBox]:
    boxes = []
    for t in page.find_tables():
        n_cols = max((len(r.cells) for r in t.rows), default=0)
        if len(t.rows) >= 2 and n_cols >= 2:
            boxes.append(tuple(float(v) for v in t.bbox))
    return boxes  # type: ignore[return-value]


def _curve_boxes(page: Page) -> list[BBox]:
    return [(c["x0"], c["top"], c["x1"], c["bottom"]) for c in page.curves]


def _contains(outer: BBox, inner: BBox, pad: float = 2.0) -> bool:
    return (
        inner[0] >= outer[0] - pad
        and inner[1] >= outer[1] - pad
        and inner[2] <= outer[2] + pad
        and inner[3] <= outer[3] + pad
    )


def _rect_boxes(page: Page) -> list[BBox]:
    return [(r["x0"], r["top"], r["x1"], r["bottom"]) for r in page.rects]


def _prose_lines_in(box: BBox, words: list[Word], body_size: float) -> int:
    """Lines of body-size prose inside a box."""
    inside = [w for w in words if _inside(w, box)]
    return sum(
        1
        for ln in group_lines(inside)
        if any(s.prose for s in ln.segments) and ln.size >= body_size - 0.6
    )


def chart_frames(page: Page, words: list[Word], body_size: float) -> list[BBox]:
    """Drawn frames that hold a chart (Fig-6.3, a pie chart in a white frame beside the text):
    a rectangle at least 80 x 60 pt with 6 or more drawn curves inside covering a quarter of
    it, and fewer than 3 lines of body prose. Its words are the chart's labels and caption."""
    curves = _curve_boxes(page)
    frames: list[BBox] = []
    for box in sorted(_rect_boxes(page), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        w, h = box[2] - box[0], box[3] - box[1]
        if w < 80 or h < 60 or w > 0.95 * page.width or any(_contains(f, box) for f in frames):
            continue
        inner = [c for c in curves if _contains(box, c)]
        if len(inner) < 6:
            continue
        cover = _union(inner)
        if (cover[2] - cover[0]) * (cover[3] - cover[1]) < 0.25 * w * h:
            continue
        inside = [w for w in words if _inside(w, box)]
        if _prose_lines_in(box, words, body_size - 1.5) >= 3:  # box text is a size smaller
            continue
        if any(w.size >= body_size + 2 for w in inside):  # a display heading
            continue
        if len(inside) <= 6 and sum(token_kind(w.text) == "num" for w in inside) <= 1:
            continue  # a title banner ('Chapter 1', 'Annexure - I'), not a chart's labels
        frames.append(box)
    return frames


def _figures_to_frames(
    page: Page, regions: list[Region], words: list[Word], body_size: float
) -> list[Region]:
    """A figure found from its caption or labels that sits in a drawn frame (a bar chart has
    few curves) takes the whole frame, so its legend and source line leave the text too
    (Fig-8.4.b, pes2425_ch08 p17). Not when the frame holds prose."""
    out = []
    for r in regions:
        if r.kind == "figure" and r.source != "frame":
            area = (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1])
            for f in _rect_boxes(page):
                w, h = f[2] - f[0], f[3] - f[1]
                if w < 80 or h < 60 or w > 0.95 * page.width or area <= 0:
                    continue
                ix = min(f[2], r.bbox[2]) - max(f[0], r.bbox[0])
                iy = min(f[3], r.bbox[3]) - max(f[1], r.bbox[1])
                holds = ix > 0 and iy > 0 and ix * iy >= 0.6 * area and w * h <= 4 * area
                if holds and _prose_lines_in(f, words, body_size - 1.5) < 3:
                    r = Region(r.kind, _union([r.bbox, f]), r.caption, r.n_rows, r.source)
                    break
        out.append(r)
    return out


def text_panels(page: Page, words: list[Word], body_size: float, taken: list[BBox]) -> list[BBox]:
    """Shaded or framed text boxes ('Box-1: Women, Business and the Law 2026 …'): a
    rectangle at least 250 x 50 pt holding 3 or more lines of prose, not a table or figure
    region and not the page background. Each is read as a zone of its own, so its lines do not
    hide the columns of the text around it."""
    panels: list[BBox] = []
    for box in sorted(_rect_boxes(page), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        w, h = box[2] - box[0], box[3] - box[1]
        if w < 250 or h < 50 or (w > 0.9 * page.width and h > 0.8 * page.height):
            continue
        if any(_contains(p, box) for p in panels):
            continue
        if any(
            min(box[2], t[2]) > max(box[0], t[0]) and min(box[3], t[3]) > max(box[1], t[1])
            for t in taken
        ):
            continue
        if _prose_lines_in(box, words, body_size - 1.5) < 3:
            continue
        panels.append(box)
    return panels


def _vertical_rules(page: Page) -> list[BBox]:
    """Thin vertical rules drawn as rectangles or lines."""
    out = [b for b in _rect_boxes(page) if b[2] - b[0] <= 2.5 and b[3] - b[1] >= 4]
    out += [
        (ln["x0"], ln["top"], ln["x1"], ln["bottom"])
        for ln in page.lines
        if abs(ln["x1"] - ln["x0"]) <= 2.5 and ln["bottom"] - ln["top"] >= 4
    ]
    return out


def _horizontal_rules(page: Page) -> list[BBox]:
    out = [b for b in _rect_boxes(page) if b[3] - b[1] <= 2.5 and b[2] - b[0] >= 20]
    out += [
        (ln["x0"], ln["top"], ln["x1"], ln["bottom"])
        for ln in page.lines
        if abs(ln["bottom"] - ln["top"]) <= 2.5 and ln["x1"] - ln["x0"] >= 20
    ]
    return out


_REFERENCE_RE = re.compile(r"^\s*(?:TABLE|Table)\s*[\d.]+\d\s*(?:\([A-Za-z]\))?\s*[.,;)]\s*$")


def _grid_under(
    line: Line, cap: Segment, verticals: list[BBox], horizontals: list[BBox]
) -> BBox | None:
    """The drawn grid under a caption line: from the rule over the caption to the end of the
    column rules that start within 30 pt under it, across the horizontal rules; None without two
    column rules and 40 pt of height."""
    start = [v for v in verticals if line.bottom - 2 <= v[1] <= line.bottom + 30]
    xs = sorted({round((v[0] + v[2]) / 2) for v in start})
    if len(xs) < 2:
        return None
    bottom = max(v[3] for v in start)
    grew = True
    while grew:  # follow the column rules down, cell by cell
        grew = False
        for v in verticals:
            chained = round((v[0] + v[2]) / 2) in xs and bottom - 1 <= v[1] <= bottom + 6
            if chained and v[3] > bottom:
                bottom, grew = v[3], True
    top_rules = [h for h in horizontals if line.top - 14 <= h[1] <= bottom + 3]
    if bottom - line.top < 40 or not top_rules:
        return None
    x0 = min(min(h[0] for h in top_rules), cap.x0)
    x1 = max(h[2] for h in top_rules)
    top = min(min(h[1] for h in top_rules), line.top) - 1
    return (x0, top, x1, bottom + 1)


def captioned_grids(page: Page, lines: list[Line], regions: list[Region]) -> list[Region]:
    """`regions` plus the tables a 'Table n' caption over a drawn grid shows (`_grid_under`):
    a grid whose cells hold text, not numbers, is a new region (Table 2.16, Water Sector
    Projects); a table region found from some of its number rows that the grid under its own
    caption contains takes the whole grid (Table 16.6, pes2425_ch16 p15: only two columns of
    the first rows were found, the rest was read as body text)."""
    verticals = _vertical_rules(page)
    horizontals = _horizontal_rules(page)
    out = list(regions)
    for line in lines:
        cap = next((s for s in line.segments if TABLE_CAPTION_RE.match(s.text)), None)
        if cap is None or _REFERENCE_RE.match(cap.text):  # 'Table 2.4.': a sentence's end
            continue
        holder = next((k for k, r in enumerate(out) if _box_contains_point(r.bbox, cap)), None)
        if holder is not None and out[holder].kind != "table":
            continue
        box = _grid_under(line, cap, verticals, horizontals)
        if box is None:
            continue
        if holder is not None:
            r = out[holder]
            area = (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1])
            grid_area = (box[2] - box[0]) * (box[3] - box[1])
            if _contains(box, r.bbox, pad=6) and grid_area >= 1.5 * area:
                out[holder] = Region("table", box, r.caption, r.n_rows, f"{r.source}+grid")
            continue
        if any(
            _overlaps(box[0], box[2], r.bbox[0], r.bbox[2])
            and _overlaps(box[1], box[3], r.bbox[1], r.bbox[3])
            for r in out
        ):
            continue  # the rules belong to a table already found
        out.append(Region("table", box, cap.text, 0, "caption+grid"))
    return out


def _body_words(words: list[Word], lines: list[Line], regions: list[Region]) -> list[Word]:
    """Words outside every region. Safety net: prose in a larger font than a region's own
    text (a paragraph of the neighbouring column caught by a chart or table box) is kept.
    Captions and source lines are not kept."""
    inside: dict[int, list[Word]] = {}
    for w in words:
        for i, r in enumerate(regions):
            if _inside(w, r.bbox):
                inside.setdefault(i, []).append(w)
                break
    if not inside:
        return words
    region_size = {i: statistics.median(w.size for w in ws) for i, ws in inside.items()}
    region_of = {id(w): i for i, ws in inside.items() for w in ws}
    keep: set[int] = set()
    for line in lines:
        for s in line.segments:
            ids = [region_of.get(id(w)) for w in s.words]
            i = next((x for x in ids if x is not None), None)
            if (
                i is not None
                and s.prose
                and statistics.median(w.size for w in s.words) >= region_size[i] + 0.5
                and not _caption_of([s])
                and not NOTE_RE.match(s.text)
            ):
                keep.update(id(w) for w in s.words)
    return [w for w in words if id(w) not in region_of or id(w) in keep]


def extract_page(page: Page, page_no: int) -> PageLayout:
    """Lay out one pdfplumber page (see module docstring)."""
    n_chars = sum(1 for c in page.chars if c["text"].strip())
    if not n_chars:
        return PageLayout(page_no, page.width, page.height, "", [], [], 0, 10.0, None, 0)
    text_raw = page.extract_text() or ""
    words, dropped = page_words(page)
    body_size = body_font_size(page)
    # charts in a drawn frame: their labels form neither table rows nor columns of the text
    frames = chart_frames(page, words, body_size)
    frame_words = [w for w in words if any(_inside(w, f) for f in frames)]
    free = [w for w in words if not any(_inside(w, f) for f in frames)]
    all_lines = group_lines(free)
    gutter = find_gutter(free, page.width)
    regions = find_regions(
        all_lines,
        page.width,
        ruled=_ruled_boxes(page),
        gutter=gutter,
        curves=_curve_boxes(page),
    )
    regions = captioned_grids(page, all_lines, regions)
    for f in frames:
        cap = next(
            (
                ln.text
                for ln in group_lines([w for w in frame_words if _inside(w, f)])
                if FIGURE_CAPTION_RE.match(ln.text)
            ),
            None,
        )
        box = f
        if cap is None:  # a caption printed just above the frame ('Fig - 8.2 D.(i): …')
            above = next(
                (
                    ln
                    for ln in all_lines
                    if FIGURE_CAPTION_RE.match(ln.text)
                    and -2 <= f[1] - ln.bottom <= 24
                    and _overlaps(ln.x0, ln.x1, f[0], f[2])
                ),
                None,
            )
            if above is not None:
                cap = above.text
                box = (min(f[0], above.x0), above.top - 1, max(f[2], above.x1), f[3])
        regions.append(Region("figure", box, cap, 0, "frame"))
    regions = _figures_to_frames(page, regions, words, body_size)
    body = _body_words(words, group_lines(words), regions)
    panels = text_panels(page, body, body_size, [r.bbox for r in regions])
    lines, gutter = reading_order(body, page.width, regions, panels)
    kept: list[Line] = []
    for line in lines:
        dropped += _drop_superscripts(line)
        if not line.words:
            continue
        if _is_chart_label(line, body_size, regions):
            dropped += len(line.words)
            continue
        kept.append(line)
    review = sorted({why for line in kept if (why := _unreliable(line, body_size))})
    return PageLayout(
        page=page_no,
        width=float(page.width),
        height=float(page.height),
        text_raw=text_raw,
        lines=kept,
        regions=regions,
        n_chars=n_chars,
        body_size=body_size,
        gutter=gutter,
        dropped_words=dropped,
        review=review,
    )


# --------------------------------------------------------------------------- documents

PAGE_SCHEMA_FIELDS = ("doc_id", "page", "printed_page", "text_body", "text_raw", "flags")


def pymupdf_word_counts(path: Path) -> list[int]:
    """Words per page according to PyMuPDF (the cross-check engine)."""
    import pymupdf

    with pymupdf.open(path) as doc:
        return [len(page.get_text("words")) for page in doc]


def word_count_mismatch(plumber: int, mupdf: int, tolerance: float) -> bool:
    """pdfplumber's count is more than `tolerance` away from PyMuPDF's."""
    if plumber == mupdf:
        return False
    return abs(plumber - mupdf) > tolerance * max(mupdf, 1)


@dataclass
class DocResult:
    doc_id: str
    records: list[PageRecord]
    report: dict[str, Any]


def extract_document(
    path: Path,
    doc_id: str,
    *,
    share: float = 0.6,
    band: float = 0.12,
    tolerance: float = 0.20,
) -> DocResult:
    """Extract and clean every page of one PDF into pages.parquet records."""
    import pdfplumber

    from claimcheck.ingest.clean import clean_document

    layouts: list[PageLayout] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            layouts.append(extract_page(page, i))
            page.close()  # free the page's cached objects
    cleaned, keys = clean_document(layouts, share=share, band=band)
    mupdf = pymupdf_word_counts(path)

    records, mismatches, no_text, regions = [], [], [], {}
    review: dict[str, list[str]] = {}
    for lay, page_clean, n_mupdf in zip(layouts, cleaned, mupdf, strict=True):
        n_plumber = len(lay.text_raw.split())
        flags = []
        if lay.no_text_layer:
            flags.append("no_text_layer")
            no_text.append(lay.page)
        if word_count_mismatch(n_plumber, n_mupdf, tolerance):
            flags.append("word_count_mismatch")
            mismatches.append({"page": lay.page, "pdfplumber": n_plumber, "pymupdf": n_mupdf})
        if any(r.kind == "table" for r in lay.regions):
            flags.append("has_table")
        if any(r.kind == "figure" for r in lay.regions):
            flags.append("has_figure")
        if lay.two_column:
            flags.append("two_column")
        if not lay.no_text_layer and not page_clean.text_body:
            flags.append("empty_body")
        if lay.review:  # the text of this page needs its source page
            flags.append("source_review")
            review[str(lay.page)] = lay.review
        if lay.regions:
            regions[str(lay.page)] = [r.to_dict() for r in lay.regions]
        records.append(
            PageRecord(
                doc_id=doc_id,
                page=lay.page,
                printed_page=page_clean.printed_page,
                text_body=page_clean.text_body,
                text_raw=lay.text_raw,
                flags=flags,
            )
        )
    report = {
        "n_pages": len(layouts),
        "boilerplate": sorted(f"{edge}: {key}" for edge, key in keys),
        "no_text_layer": no_text,
        "word_count_mismatch": mismatches,
        "n_tables": sum(1 for lay in layouts for r in lay.regions if r.kind == "table"),
        "n_figures": sum(1 for lay in layouts for r in lay.regions if r.kind == "figure"),
        "printed_pages": sum(1 for c in cleaned if c.printed_page),
        "regions": regions,
        "source_review": review,
    }
    return DocResult(doc_id, records, report)


def _extract_job(args: tuple[str, str, float, float, float]) -> DocResult:
    path, doc_id, share, band, tolerance = args
    return extract_document(Path(path), doc_id, share=share, band=band, tolerance=tolerance)


def run_extract(
    manifest: Manifest,
    raw_dir: Path,
    out_pages: Path,
    out_report: Path,
    *,
    share: float = 0.6,
    band: float = 0.12,
    tolerance: float = 0.20,
    workers: int = 0,
    on_doc: Callable[[DocResult], None] | None = None,
) -> dict[str, Any]:
    """Extract every manifest document (in parallel) into pages.parquet and a run report."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    import pyarrow as pa
    import pyarrow.parquet as pq

    docs = manifest.documents
    jobs = [(str(raw_dir / d.filename), d.doc_id, share, band, tolerance) for d in docs]
    n_workers = workers or min(len(jobs), os.cpu_count() or 1)
    results: dict[str, DocResult] = {}
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

    rows = [rec.model_dump() for d in docs for rec in results[d.doc_id].records]
    schema = pa.schema(
        [
            ("doc_id", pa.string()),
            ("page", pa.int32()),
            ("printed_page", pa.string()),
            ("text_body", pa.string()),
            ("text_raw", pa.string()),
            ("flags", pa.list_(pa.string())),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    out_pages.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_pages)

    expected = sum(d.n_pages or 0 for d in docs)
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_documents": len(docs),
        "n_pages": len(rows),
        "expected_pages": expected,
        "word_count_tolerance": tolerance,
        "no_text_layer": [
            {"doc_id": d.doc_id, "page": p}
            for d in docs
            for p in results[d.doc_id].report["no_text_layer"]
        ],
        "word_count_mismatch": [
            {"doc_id": d.doc_id, **m}
            for d in docs
            for m in results[d.doc_id].report["word_count_mismatch"]
        ],
        "documents": {d.doc_id: results[d.doc_id].report for d in docs},
    }
    out_report.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report
