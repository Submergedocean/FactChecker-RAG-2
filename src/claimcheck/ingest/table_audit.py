"""Corpus-wide audit of parsed tables against the PDF, read independently with PyMuPDF.

The table parser works on pdfplumber words; this module re-reads each page with PyMuPDF and checks
every non-fallback table of `tables.parquet` for the defects that a check of values alone does
not catch:

`row_line`        a row with two or more numbers is not printed on one line in that order
`cell_column`     a number sits under another column than the one it is stored in (by the
                  right edge of the column's other numbers): a value shifted past a blank cell
`label_row`       the row's own label words are not on the row's line or its wrapped lines,
                  but nearer another row (a label attached to the wrong row)
`header_column`   none of a column header's own words is printed over that column, or most
                  of them over another column (headers of neighbouring columns mixed up)
`merged_label`    one label holds two numbered items ('14. Italy 15. Sweden')

Findings are leads for review, not proof: `audit_tables` returns them with the evidence; the
corpus tests keep the reviewed counts from growing.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from claimcheck.ingest.tables import lead_columns
from claimcheck.textnorm import is_number_token, parse_number

LINE_TOL = 3.5  # pt: words whose centres differ by less are on one printed line
CHECKS = ("row_line", "cell_column", "label_row", "header_column", "merged_label")


@dataclass(frozen=True)
class PdfWord:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass(frozen=True)
class Finding:
    table_id: str
    check: str
    row: int | None
    col: int | None
    detail: str


def _num(text: str, brackets_negative: bool = True) -> float | None:
    t = text.strip().rstrip("*#^@†‡%,;")
    if not t or not re.search(r"\d", t):
        return None
    if not brackets_negative and t.startswith("(") and t.endswith(")"):
        t = t[1:-1]
    try:
        v = parse_number(t)
    except ValueError:
        return None
    return None if v is None else round(float(v), 6)


_TOKEN_RE = re.compile(r"[a-z][a-z0-9&'-]*|\d+(?:-\d+)+", re.I)


def tokens(text: str) -> list[str]:
    """Words of a label or header, lower case: letters, and number ranges ('15-19'); plain
    numbers and punctuation are left out."""
    out = []
    for t in _TOKEN_RE.findall(text or ""):
        t = t.lower().strip("-'")
        if t[:1].isalpha():
            t = t.rstrip("0123456789")  # a footnote digit printed on the word: 'Meat4'

        if len(t) >= 2 and not is_number_token(t):
            out.append(t)
    return out


@dataclass
class _RowHit:
    y: float
    xs: dict[int, PdfWord]  # column -> the PDF word of its number


def _match_rows(
    t: dict[str, Any], words: Sequence[PdfWord], lead: int
) -> tuple[list[_RowHit | None], list[Finding]]:
    """Find each row's printed line: its numbers left to right on one line, in order."""
    neg = "brackets_not_negative" not in t["flags"]
    nums = [(w, _num(w.text, neg)) for w in words]
    nums = [(w, v) for w, v in nums if v is not None]
    hits: list[_RowHit | None] = []
    findings: list[Finding] = []
    used: set[tuple[float, float]] = set()
    for i, row in enumerate(t["rows"]):
        cells = [
            (j, round(float(v), 6))
            for j, v in enumerate(row)
            if j >= lead and isinstance(v, int | float) and not isinstance(v, bool)
        ]
        if not cells:
            hits.append(None)
            continue
        best = None
        for w0, v0 in nums:
            if v0 != cells[0][1] or (w0.x0, w0.cy) in used:
                continue
            line = sorted(
                ((w, v) for w, v in nums if abs(w.cy - w0.cy) <= LINE_TOL and w.x0 >= w0.x0),
                key=lambda wv: wv[0].x0,
            )
            got: dict[int, PdfWord] = {}
            k = 0
            for w, v in line:
                if k < len(cells) and v == cells[k][1]:
                    got[cells[k][0]] = w
                    k += 1
            if k == len(cells):
                best = _RowHit(w0.cy, got)
                break
        if best is None:
            if len(cells) >= 2:
                findings.append(Finding(t["table_id"], "row_line", i, None, str(row[:3])))
            hits.append(None)
            continue
        for w in best.xs.values():
            used.add((w.x0, w.cy))
        hits.append(best)
    return hits, findings


def _cell_columns(t: dict[str, Any], hits: Sequence[_RowHit | None]) -> list[Finding]:
    """A number closer (by right edge) to another column's numbers than to its own column's."""
    edges: dict[int, list[float]] = defaultdict(list)
    for h in hits:
        if h:
            for j, w in h.xs.items():
                edges[j].append(w.x1)
    med = {j: statistics.median(xs) for j, xs in edges.items() if len(xs) >= 2}
    out = []
    for i, h in enumerate(hits):
        if not h:
            continue
        for j, w in h.xs.items():
            if j not in med:
                continue
            own = abs(w.x1 - med[j])
            k, other = min(((k, abs(w.x1 - m)) for k, m in med.items()), key=lambda km: km[1])
            if k != j and own - other > 4.0:
                out.append(
                    Finding(t["table_id"], "cell_column", i, j, f"{w.text} under column {k}")
                )
    return out


def _leaf(row: Sequence[Any], skip: int, lead: int) -> str:
    """The innermost label of a row: the last non-empty label cell (an outer label column is
    often filled down), its last path element, without a trailing '(qualifier)'."""
    cells = [str(c) for c in row[skip:lead] if isinstance(c, str) and c]
    label = cells[-1] if cells else ""
    return re.sub(r"\s*\([^()]*\)\s*$", "", label.split(" / ")[-1])  # '(Private Sector)'


def _lines(words: Sequence[PdfWord]) -> list[tuple[float, float, list[PdfWord]]]:
    """(y, height, words) of each printed line, top to bottom."""
    out: list[tuple[float, float, list[PdfWord]]] = []
    for w in sorted(words, key=lambda w: w.cy):
        if out and abs(out[-1][0] - w.cy) <= LINE_TOL:
            out[-1][2].append(w)
        else:
            out.append((w.cy, w.y1 - w.y0, [w]))
    return out


def _label_rows(
    t: dict[str, Any], words: Sequence[PdfWord], hits: Sequence[_RowHit | None], lead: int
) -> list[Finding]:
    """The words of a row's own label (its last path element) are printed left of its first
    number, on its line or on label lines chained to it at line spacing (a wrapped label)
    without crossing another row's line. Words the row shares with the row above (a label
    filled down, a period printed once) are not checked."""
    skip = 1 if "index_column" in t["flags"] else 0
    ys = [(i, h.y) for i, h in enumerate(hits) if h]
    row_ys = [y for _, y in ys]
    out = []
    for n, (i, y) in enumerate(ys):
        leaf = _leaf(t["rows"][i], skip, lead)
        want = tokens(leaf)
        if n:
            above = set(tokens(_leaf(t["rows"][ys[n - 1][0]], skip, lead)))
            want = [tok for tok in want if tok not in above]
        if not want:
            continue
        first_x = min(w.x0 for w in hits[i].xs.values())  # type: ignore[union-attr]
        lines = _lines([w for w in words if w.x1 <= first_x + 1 and abs(w.cy - y) < 60])
        own = next((k for k, (ly, _, _) in enumerate(lines) if abs(ly - y) <= LINE_TOL), None)
        block: set[str] = set()
        if own is not None:
            block |= {tok for w in lines[own][2] for tok in tokens(w.text)}
            for step in (1, -1):  # follow wrapped lines down, then up
                k = own
                while 0 <= k + step < len(lines):
                    ly, h, ws = lines[k + step]
                    gap = abs(ly - lines[k][0])
                    if gap > 1.45 * max(h, lines[k][1]) or any(
                        abs(ly - ry) <= LINE_TOL for ry in row_ys if abs(ry - y) > LINE_TOL
                    ):
                        break
                    block |= {tok for w in ws for tok in tokens(w.text)}
                    k += step
        missing = [tok for tok in want if tok not in block]
        wrong: dict[str, float] = {}
        others = [yy for k, yy in enumerate(row_ys) if k != n]
        for tok in dict.fromkeys(missing):
            ws = [w for ly, _, lw in lines for w in lw if tok in tokens(w.text)]
            if not ws:
                continue
            w = min(ws, key=lambda w: abs(w.cy - y))
            if others and min(abs(w.cy - yy) for yy in others) < abs(w.cy - y) - 0.5:
                wrong[tok] = round(w.cy - y, 1)
        if wrong:
            out.append(Finding(t["table_id"], "label_row", i, None, f"{leaf!r}: {wrong}"))
    return out


def _header_columns(
    t: dict[str, Any], words: Sequence[PdfWord], hits: Sequence[_RowHit | None], lead: int
) -> list[Finding]:
    """Each column header's own words (not shared with a neighbouring header) are printed
    over its column, and not mostly over another column."""
    first_y = min((h.y for h in hits if h), default=None)
    if first_y is None:
        return []
    spans: dict[int, tuple[float, float]] = {}
    for j in range(lead, len(t["header"])):
        ws = [h.xs[j] for h in hits if h and j in h.xs]
        if len(ws) >= 2:
            spans[j] = (min(w.x0 for w in ws) - 3, max(w.x1 for w in ws) + 3)
    top = t["bbox"][1] - 2
    zone = [w for w in words if top <= w.cy < first_y - LINE_TOL]
    out = []
    header = t["header"]
    for j, (a, b) in spans.items():
        mine = set(tokens(header[j]))
        shared = set(tokens(header[j - 1])) if j - 1 >= lead else set()
        if j + 1 < len(header):
            shared |= set(tokens(header[j + 1]))
        own = mine - shared
        if not own:
            continue
        over, elsewhere = 0, 0
        for tok in own:
            ws = [w for w in zone if tok in tokens(w.text)]
            if not ws:
                continue
            if any(w.x1 >= a and w.x0 <= b for w in ws):
                over += 1
            elif any(k != j and w.x1 >= s and w.x0 <= e for w in ws for k, (s, e) in spans.items()):
                elsewhere += 1
        if over == 0 and elsewhere > 0:
            out.append(
                Finding(
                    t["table_id"], "header_column", None, j, f"{header[j]!r}: own words elsewhere"
                )
            )
    return out


_MERGED_RE = re.compile(r"(?:^|\s)(\d{1,2})\.\s+\S.*\s(\d{1,2})\.\s+\S")


def _merged_labels(t: dict[str, Any], lead: int) -> list[Finding]:
    out = []
    for i, row in enumerate(t["rows"]):
        for c in row[:lead]:
            if not isinstance(c, str):
                continue
            m = _MERGED_RE.search(c.split(" / ")[-1])
            if m and int(m.group(2)) == int(m.group(1)) + 1:
                out.append(Finding(t["table_id"], "merged_label", i, None, c[-60:]))
    return out


def page_words(doc: Any, page: int) -> list[PdfWord]:
    return [PdfWord(w[0], w[1], w[2], w[3], w[4]) for w in doc[page - 1].get_text("words")]


def audit_table(t: dict[str, Any], words: Sequence[PdfWord]) -> list[Finding]:
    """All findings for one parsed table (rows already parsed from JSON)."""
    lead = lead_columns(list(t["flags"]))
    x0, top, x1, bottom = t["bbox"]
    inside = [w for w in words if x0 - 4 <= w.cx <= x1 + 14 and top - 4 <= w.cy <= bottom + 4]
    hits, findings = _match_rows(t, inside, lead)
    findings += _cell_columns(t, hits)
    findings += _label_rows(t, inside, hits, lead)
    findings += _header_columns(t, inside, hits, lead)
    findings += _merged_labels(t, lead)
    return findings


def audit_tables(tables: Iterable[dict[str, Any]], raw_dir: Path) -> list[Finding]:
    """Audit every non-fallback table; PDFs are opened once per document."""
    import pymupdf

    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in tables:
        if "fallback_plain" not in t["flags"] and t["rows"]:
            by_doc[t["doc_id"]].append(t)
    out: list[Finding] = []
    for doc_id, ts in by_doc.items():
        with pymupdf.open(raw_dir / f"{doc_id}.pdf") as doc:
            cache: dict[int, list[PdfWord]] = {}
            for t in ts:
                if t["page"] not in cache:
                    cache[t["page"]] = page_words(doc, t["page"])
                out.extend(audit_table(t, cache[t["page"]]))
    return out


def summary(findings: Sequence[Finding]) -> dict[str, Any]:
    by_check: dict[str, set[str]] = defaultdict(set)
    n: dict[str, int] = defaultdict(int)
    for f in findings:
        by_check[f.check].add(f.table_id)
        n[f.check] += 1
    return {
        c: {"findings": n.get(c, 0), "tables": sorted(by_check.get(c, set()))} for c in CHECKS
    } | {"all": [asdict(f) for f in findings]}
