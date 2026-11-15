"""Chunking into `chunks.parquet`, two variants (proposal 6.1).

`table_aware` (default)
    Text chunks: a sliding window of about `window_words` (150) words with about
    `overlap_words` (30) of overlap, cut at sentence ends, built from `text_body` only (with
    every table and figure region removed), never across documents; a window may cross pages.
    Table chunks: one table of tables.parquet is cut only between rows. Each chunk holds the
    caption (the first page's caption for a continued table), the units line, the header and
    whole rows, up to `table_max_tokens` (350) embedding tokens; `text` is the pipe table with
    its notes (footnotes, source, legends of marks such as `1607#`); `embed_text` linearises each
    row as "row label, column header: value; column header: value". A `fallback_plain` table
    (the parser could not read its cells safely) is windowed like prose from its `markdown`, line by
    line, and stays a table chunk of that table.
`plain_text` (ablation)
    The same text window over `text_raw` (headers and footers dropped with the page keys), with no
    table handling: tables are flattened into the text like on the PDF's plain text layer.

Every `embed_text` starts with "PES {edition} · {chapter title}". Every `n_tokens` (tokens of
`embed_text` for the embedding model, special tokens included) is at most `max_tokens`. Exact
duplicate chunks (same `text`) are removed, keeping the latest edition's copy. Text chunks with
fewer than `min_words` words (a near-empty document) are dropped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claimcheck.config import ChunkingSettings
from claimcheck.ingest.clean import NUMBER_THEN_WORD_RE, strip_boilerplate_text
from claimcheck.ingest.tables import column_unit, lead_columns, row_unit
from claimcheck.schemas import ChunkKind, ChunkRecord, Manifest, ManifestEntry, Variant
from claimcheck.textnorm import is_number_token, tokenize

log = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]
"""Embedding-model tokens of a string, special tokens included."""

VARIANTS: tuple[Variant, ...] = (Variant.TABLE_AWARE, Variant.PLAIN_TEXT)
SEP = " · "


# --------------------------------------------------------------------------- tokens


def hf_token_counter(model: str, revision: str | None = None) -> TokenCounter:
    """Token counter of the embedding model's own tokenizer (downloads tokenizer.json once)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(model, revision=revision or "main")
    tok.no_truncation()

    def count(text: str) -> int:
        return len(tok.encode(text).ids)

    return count


def simple_token_counter(text: str) -> int:
    """Offline stand-in for tests: word pieces split at punctuation and digits, plus [CLS] and
    [SEP]. It over-counts numbers the way WordPiece does ('2025-26' -> 4)."""
    return len(re.findall(r"[A-Za-z]{1,8}|\d{1,3}|[^\sA-Za-z\d]", text)) + 2


# --------------------------------------------------------------------------- sentences

# Words whose trailing period does not end a sentence ("Rs. 5", "No. 3", "Dr. Ali").
ABBREVIATIONS = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "st",
        "no",
        "nos",
        "rs",
        "fig",
        "figs",
        "vol",
        "viz",
        "vs",
        "approx",
        "govt",
        "dept",
        "est",
        "inc",
        "ltd",
        "co",
        "pvt",
        "jan",
        "feb",
        "mar",
        "apr",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "art",
        "sec",
        "para",
        "min",
        "max",
        "ref",
        "resp",
    ]
)
_ACRONYM_RE = re.compile(r"(?:[A-Za-z]{1,2}\.)+[A-Za-z]{1,2}")  # U.S, i.e, e.g, D.I
_ROMAN_RE = re.compile(r"(?:[ivx]{1,4}|[IVX]{1,4})")
# End punctuation, closing quotes/brackets, whitespace, then what can start a sentence.
_END_RE = re.compile(
    r"[.!?][\"'”’)\]]*\s+(?=[\"'“‘(\[]?[A-Z0-9■•▪●◆➢✓–-]|(?:[ivx]{1,4}|[a-h])[.)]\s)"
)


def _ends_sentence(text: str, start: int, pos: int) -> bool:
    """Whether the period at `pos` ends the sentence that began at `start`."""
    m = re.search(r"(\S+)$", text[start:pos])
    if not m:
        return True
    tok = m.group(1).lstrip("(\"'“‘[")
    if tok.lower() in ABBREVIATIONS or _ACRONYM_RE.fullmatch(tok):
        return False
    if len(tok) == 1 and tok.isalpha():  # an initial or a list letter: "M. Aurangzeb", "a."
        return False
    if _ROMAN_RE.fullmatch(tok):  # a list numeral "ii." at the start, not "Phase II."
        before = text[start : start + m.start()].rstrip()
        return bool(before) and before[-1] not in ".:;!?"
    return True


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence of one paragraph; ends exclude trailing whitespace."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _END_RE.finditer(text):
        if text[m.start()] == "." and not _ends_sentence(text, start, m.start()):
            continue
        end = len(text[: m.end()].rstrip())
        if end > start:
            spans.append((start, end))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text.rstrip())))
    return spans


def split_sentences(text: str) -> list[str]:
    return [text[a:b] for a, b in sentence_spans(text)]


@dataclass
class Unit:
    """A sentence (or a table line), the smallest piece a window holds."""

    text: str
    page_start: int
    page_end: int
    sep: str = " "  # joins it to the unit before it in the same window

    @property
    def n_words(self) -> int:
        return len(self.text.split())


@dataclass
class _Para:
    text: str
    breaks: list[tuple[int, int]] = field(default_factory=list)  # (char offset, page)

    def page_at(self, offset: int) -> int:
        page = self.breaks[0][1]
        for off, p in self.breaks:
            if off <= offset:
                page = p
        return page


_CLOSED_END = ".?!:"
# A paragraph ending like this is an unfinished sentence: '(ii) floating debt (short-term); and'
_CONNECTOR_END_RE = re.compile(
    r"(?:[,;(\-–]|\b(?:and|or|nor|but|of|the|a|an|to|in|for|with|by|as|on|at|from|that|which|"
    r"than|including|such|between|into|under|over|while|whereas|i\.e\.|e\.g\.))$",
    re.I,
)


_CURRENCY_END_RE = re.compile(r"(?:\bRs\.?|\bPKR|US\s?\$|\$)$")
# Words that are followed by a number in running text: 'July-March FY' / '2026 in the PSX'.
_BEFORE_NUMBER_RE = re.compile(
    r"\b(?:FY|fiscal\s+year|January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)$"
)


def continues(prev: str, nxt: str) -> bool:
    """Whether paragraph `nxt` continues the sentence left open by paragraph `prev`: `prev`
    ends with a connector (',', ';', 'and', 'of' ...), a currency sign ('retired Rs' /
    '1,939.3 billion'); `nxt` starts with a number and `prev` ends inside a bracket ('(April'
    / '2026), energy ...') or with 'FY' or a month ('July-March FY' / '2026 in the PSX'); or
    `prev` has no end punctuation and `nxt` starts in lower case or with '&'."""
    if _CONNECTOR_END_RE.search(prev) or _CURRENCY_END_RE.search(prev):
        return True
    if nxt[0].isdigit() and (prev.rfind("(") > prev.rfind(")") or _BEFORE_NUMBER_RE.search(prev)):
        return True
    if prev[-1] not in _CLOSED_END and NUMBER_THEN_WORD_RE.match(nxt):
        return True  # 'The report assesses' / '10 key areas across women's working lives
    return prev[-1] not in _CLOSED_END and (nxt[0].islower() or nxt[0] == "&")


def page_units(pages: Iterable[tuple[int, str]], *, paragraphs: bool = True) -> list[Unit]:
    """Sentences of a document's pages in order, each with the page(s) it lies on.

    With `paragraphs`, blank lines separate paragraphs (text_body); otherwise each page is one
    paragraph whose lines are joined (text_raw). A paragraph that leaves its sentence open
    (`continues`) is joined with the next one, on the same page or the next, so a list item
    on a line of its own ('...; and' / '(iii) unfunded debt ...') stays in its sentence."""
    paras: list[_Para] = []
    for page, text in pages:
        blocks = re.split(r"\n\s*\n", text) if paragraphs else [text]
        parts = [" ".join(b.split()) for b in blocks]
        for part in (p for p in parts if p):
            prev = paras[-1] if paras else None
            if prev and continues(prev.text, part):
                prev.breaks.append((len(prev.text) + 1, page))
                prev.text += " " + part
            else:
                paras.append(_Para(part, [(0, page)]))
    units: list[Unit] = []
    for para in paras:
        for k, (a, b) in enumerate(sentence_spans(para.text)):
            units.append(
                Unit(para.text[a:b], para.page_at(a), para.page_at(b - 1), " " if k else "\n\n")
            )
    return units


def split_long(units: list[Unit], max_words: int) -> list[Unit]:
    """Cut a unit longer than `max_words` into near-equal pieces at word boundaries."""
    out: list[Unit] = []
    for u in units:
        words = u.text.split()
        if len(words) <= max_words:
            out.append(u)
            continue
        n = math.ceil(len(words) / max_words)
        size = math.ceil(len(words) / n)
        for k in range(n):
            piece = words[k * size : (k + 1) * size]
            out.append(Unit(" ".join(piece), u.page_start, u.page_end, u.sep if k == 0 else " "))
    return out


# --------------------------------------------------------------------------- windows


def windows(units: Sequence[Unit], window_words: int, overlap_words: int) -> list[list[Unit]]:
    """Sliding windows of whole units: each holds at most `window_words` words (units are
    first cut to that size), ends at a unit end, and the next one starts with the last units
    of this one whose word count is closest to `overlap_words` (at most twice it; none when
    even the last unit alone is longer)."""
    units = split_long(list(units), window_words)
    n = [u.n_words for u in units]
    out: list[list[Unit]] = []
    i = 0
    while i < len(units):
        j, total = i, 0
        while j < len(units) and (j == i or total + n[j] <= window_words):
            total += n[j]
            j += 1
        out.append(units[i:j])
        if j >= len(units):
            break
        best, cost, back = j, overlap_words, 0
        for k in range(j - 1, i, -1):
            back += n[k]
            if back > 2 * overlap_words:
                break
            if abs(back - overlap_words) < cost:
                best, cost = k, abs(back - overlap_words)
        i = best
    return out


def join_units(units: Sequence[Unit]) -> str:
    return "".join((u.sep if k else "") + u.text for k, u in enumerate(units))


def overlap_words(a: Sequence[Unit], b: Sequence[Unit]) -> int:
    """Words the window `b` repeats from the end of window `a`."""
    shared = 0
    for k in range(1, min(len(a), len(b)) + 1):
        if [u.text for u in a[-k:]] == [u.text for u in b[:k]]:
            shared = sum(u.n_words for u in b[:k])
    return shared


# --------------------------------------------------------------------------- drafts


@dataclass
class Draft:
    """A chunk before ids, de-duplication and tokens for BM25."""

    kind: ChunkKind
    page_start: int
    page_end: int
    text: str
    embed_text: str
    n_tokens: int
    table_id: str | None = None
    rows: tuple[int, int] | None = None  # table row range [a, b) of a table chunk


def context_line(edition: str, chapter_title: str) -> str:
    return f"PES {edition}{SEP}{chapter_title}"


def _fit(units: list[Unit], make: Callable[[list[Unit]], Draft], max_tokens: int) -> list[Draft]:
    """`make(units)`, halved by units (or words) until every draft has `max_tokens` or fewer."""
    draft = make(units)
    if draft.n_tokens <= max_tokens:
        return [draft]
    if len(units) == 1:
        words = units[0].text.split()
        if len(words) < 2:
            raise ValueError(f"one word over {max_tokens} tokens: {words[:1]}")
        units = split_long(units, math.ceil(len(words) / 2))
    half = len(units) // 2
    return _fit(units[:half], make, max_tokens) + _fit(units[half:], make, max_tokens)


def text_drafts(
    units: Sequence[Unit],
    context: str,
    cfg: ChunkingSettings,
    count: TokenCounter,
    *,
    kind: ChunkKind = ChunkKind.TEXT,
    table_id: str | None = None,
    head: str = "",
    max_tokens: int | None = None,
) -> list[Draft]:
    """Window `units` and wrap each window in a draft. `head` (a table's caption and units)
    is put before the content of every window, in `text` and in `embed_text`. A window over
    `max_tokens` (default `cfg.max_tokens`) is halved until it fits."""

    def make(ws: list[Unit]) -> Draft:
        body = join_units(ws)
        text = f"{head}\n{body}" if head else body
        embed = f"{context}\n{text}"
        return Draft(kind, ws[0].page_start, ws[-1].page_end, text, embed, count(embed), table_id)

    out: list[Draft] = []
    for w in windows(units, cfg.window_words, cfg.overlap_words):
        out.extend(_fit(list(w), make, max_tokens or cfg.max_tokens))
    return out


# --------------------------------------------------------------------------- tables


def md_cells(line: str) -> list[str]:
    """Cells of one pipe-table line ('| a | b \\| c |' -> ['a', 'b | c'])."""
    inner = line.strip()
    inner = inner[1:] if inner.startswith("|") else inner
    inner = inner[:-1] if inner.endswith("|") and not inner.endswith("\\|") else inner
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", inner)]


def split_markdown(markdown: str) -> tuple[list[str], str]:
    """(pipe-table lines, notes) of a table `markdown`: the table, a blank line, the notes."""
    table, _, notes = markdown.partition("\n\n")
    return [ln for ln in table.split("\n") if ln.strip()], notes.strip()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.10g}" if abs(value) >= 1000 else f"{value:.10g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def printed_rows(table: dict[str, Any]) -> tuple[str, list[str], list[list[str]], str]:
    """Header line, row lines, row cells and notes of a parsed table. Row i is `rows[i]` with
    the printed cell text (footnote marks such as `1607#` kept)."""
    header: list[str] = list(table["header"])
    rows: list[list[Any]] = table["rows"]
    lines, notes = split_markdown(table["markdown"])
    body = lines[2:]
    cells = [md_cells(ln) for ln in body]
    if len(body) != len(rows) or any(len(c) != len(header) for c in cells):
        log.warning("%s: markdown rows do not match `rows`; rebuilt", table["table_id"])
        cells = [[_fmt(v) for v in r] for r in rows]
        body = ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in cells]
    head = "\n".join(lines[:2]) if len(lines) >= 2 else ""
    return head, body, cells, notes


_BLANK = {"", "-", "–", "--"}


def linearise(
    header: Sequence[str],
    flags: Sequence[str],
    cells: Sequence[str],
    units: Sequence[str | None] | None = None,
) -> str:
    """'row label, column header: value; column header: value' for one printed row. The label
    is the row's label columns joined (the index column left out); blank and '-' cells are
    skipped. `units[k]`, when given, follows the value of column k: '9.2 (percent)'."""
    lead = lead_columns(list(flags))
    skip = 1 if "index_column" in flags else 0
    label = " ".join(c for c in cells[skip:lead] if c)
    pairs = []
    for k in range(lead, len(cells)):
        v = cells[k]
        if v in _BLANK:
            continue
        h = header[k] if k < len(header) and header[k] else f"column {k + 1}"
        u = units[k] if units and k < len(units) else None
        pairs.append(f"{h}: {v} ({u})" if u else f"{h}: {v}")
    body = "; ".join(pairs)
    return f"{label}, {body}" if label and body else (label or body)


def _unit_key(unit: str | None) -> str:
    """Comparable form of a unit: '%', 'percent' and '(%)' are the same."""
    if not unit:
        return ""
    u = unit.strip().strip("()").strip().casefold()
    return "percent" if u in {"%", "percent", "per cent", "percentage"} else u


@dataclass(frozen=True)
class CellUnit:
    """The unit of one value and where it comes from (row, then column, then
    table). `scope` names the rows or column it covers when it is not the table's."""

    unit: str | None
    source: str  # "row", "column", "table" or "none"
    scope: str = ""


def row_unit_scope(label: str) -> tuple[str, str] | None:
    """(unit, scope) of a row whose label names a unit: the scope is the heading that gives
    it ("rows under 'Key FSIs (%)'") or the row itself ("row '% of GDP'")."""
    unit = row_unit(label)
    if not unit:
        return None
    parts = [p.strip() for p in label.split(" / ")]
    marked = re.search(r"\((?:percentage|percent|%)\s+(?:distribution|share)\)$", parts[-1], re.I)
    if marked:  # every such row of the table: 'rows marked '(Percentage distribution)''
        return unit, f"rows marked '{marked.group(0)}'"
    if row_unit(parts[-1]) == unit:
        return unit, f"rows '{parts[-1]}'"
    for k in range(len(parts) - 2, -1, -1):
        if row_unit(f"{parts[k]} / x") == unit:
            return unit, f"rows under '{parts[k]}'"
    return unit, f"rows '{label}'"


def cell_units(table: dict[str, Any], units: str | None) -> list[list[CellUnit]]:
    """The unit of every cell of a parsed table (label cells get source 'none')."""
    flags = list(table["flags"])
    lead = lead_columns(flags)
    skip = 1 if "index_column" in flags else 0
    header: list[str] = list(table["header"])
    col = [column_unit(h) if k >= lead else None for k, h in enumerate(header)]
    out = []
    for row in table["rows"]:
        first = row[skip] if len(row) > skip and isinstance(row[skip], str) else ""
        ru = row_unit_scope(first) if first else None
        cells = []
        for k in range(len(header)):
            if k < lead:
                cells.append(CellUnit(None, "none"))
            elif ru and _unit_key(ru[0]) != _unit_key(units):
                cells.append(CellUnit(ru[0], "row", ru[1]))
            elif not ru and col[k] and _unit_key(col[k]) != _unit_key(units):
                cells.append(CellUnit(col[k], "column", f"column '{header[k]}'"))
            elif units:
                cells.append(CellUnit(units, "table"))
            else:
                cells.append(CellUnit(None, "none"))
        out.append(cells)
    return out


def units_line(cells: Sequence[Sequence[CellUnit]], printed: Sequence[Sequence[str]]) -> str:
    """'Units: ...' for the value cells shown: the table's unit when some value is in it, then
    each other unit with the rows or columns it covers ('percent for rows under 'Key FSIs
    (%)''). Empty when no value has a known unit."""
    default = False
    other: dict[str, list[str]] = {}
    for row, prt in zip(cells, printed, strict=True):
        for cu, v in zip(row, prt, strict=True):
            if v in _BLANK or cu.source == "none":
                continue
            if cu.source == "table":
                default = True
                table_unit = cu.unit
            else:
                scopes = other.setdefault(cu.unit or "", [])
                if cu.scope not in scopes:
                    scopes.append(cu.scope)
    parts = [table_unit] if default else []
    for unit, scopes in other.items():
        parts.append(f"{unit} for " + ", ".join(scopes))
    return "Units: " + "; ".join(parts) if parts else ""


def table_head(caption: str | None, units: str | None, continued_from: int | None = None) -> str:
    """Caption line ('... (continued from p23)' on a continued table's later pages) and units
    line put before every chunk of a table. `units` is a whole 'Units: ...' line when it
    starts so, else the table's unit."""
    more = f"(continued from p{continued_from})" if continued_from else ""
    lines = []
    if caption or more:
        lines.append(" ".join(x for x in (caption, more) if x))
    if units:
        lines.append(units if units.startswith("Units: ") else f"Units: {units}")
    return "\n".join(lines)


# ---- fallback tables: logical rows, repeated header and section context

_NUM_CELL_RE = re.compile(r"^[-–+(]?\d[\d,]*(?:\.\d+)?\)?%?[*#@^†‡]*$")
_YEAR_CELL_RE = re.compile(r"^\(?(?:19|20)\d\d\)?$")
_ENUM_ROW_RE = re.compile(r"^\(?(?:\d{1,2}|[a-z]|[ivx]{1,4})[.)](?:\s|$)")
_SECTION_ENUM_RE = re.compile(r"^[A-Z][.)]\s")
_FALLBACK_NOTE_RE = re.compile(
    r"(?i)^(?:source\b|notes?\b|\*|#|@|†|‡|-\s*:|[A-Z]{1,6}\s*:\s|p\s*:|r\s*:)"
)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.split(" | ")]


def value_line(line: str) -> bool:
    """A line of a fallback table that holds a number (a year in a header line does not
    count)."""
    return any(
        _NUM_CELL_RE.match(c) and not _YEAR_CELL_RE.match(c) for c in _cells(line)
    ) and not _FALLBACK_NOTE_RE.match(line)


def _labelled(line: str) -> bool:
    first = _cells(line)[0]
    return bool(first) and not _NUM_CELL_RE.match(first) and first not in _BLANK


def _continuation(line: str) -> bool:
    return line[:1].islower() or line[:1] in "(&"


@dataclass
class FallbackPlan:
    """A `fallback_plain` table read back from its plain text: the header lines before the
    first value line, logical rows (a value line with the label lines wrapped around it), the
    section heading each row is under, and the notes after the last value line."""

    header: list[str]
    groups: list[list[str]]
    sections: list[list[str]]  # per group: the section heading lines in force
    starts: list[bool]  # per group: the group prints its own section heading
    notes: list[str]


def fallback_plan(lines: list[str], caption: str | None) -> FallbackPlan | None:
    """None when the text holds no value line (it is windowed as prose). Rows are grouped so
    a wrapped label stays with its numbers: in a table with numbered rows ('5. | Water Supply,
    Sewerage,' / 'Waste, Management & | 0.4 | …' / 'Remediation Activity') a row runs from
    one numbered line to the next; otherwise label lines over a line of bare numbers are its
    label, lines starting in lower case or with '(' / '&' end the row above, and other label
    lines are a section heading ('Islamabad', 'Protected') over the rows after them."""
    vals = [k for k, ln in enumerate(lines) if value_line(ln)]
    cells = [
        k
        for k, ln in enumerate(lines)
        if " | " in ln and len(_cells(ln)) >= 2 and not re.match(r"(?i)table\b", ln)
    ]
    if not vals and len(cells) >= 3 and len(cells) >= 0.6 * len(lines):
        # a table of text cells read by its grid (Table 2.16): one line per row, the first
        # its column headings
        first = cells[0]
        notes = [ln for ln in lines[first + 1 :] if _FALLBACK_NOTE_RE.match(ln)]
        rows = [ln for ln in lines[first + 1 :] if ln not in notes]
        lead = [ln for ln in lines[: first + 1] if not re.match(r"(?i)table\b", ln)]
        return FallbackPlan(
            lead, [[ln] for ln in rows], [[] for _ in rows], [False] * len(rows), notes
        )
    if not vals:
        return None
    first, last = vals[0], vals[-1]
    cap = (caption or "").casefold()
    lead = [  # the caption's own lines ('TABLE 12.6', its title) are in the head already
        ln
        for ln in lines[:first]
        if not (re.match(r"(?i)table\b", ln) or (len(ln.split()) >= 3 and ln.casefold() in cap))
    ]
    tail = lines[last + 1 :]
    k_note = next((k for k, ln in enumerate(tail) if _FALLBACK_NOTE_RE.match(ln)), len(tail))
    body = lines[first : last + 1] + tail[:k_note]
    notes = tail[k_note:]
    # the section heading in force at the first row, printed at the end of the header block
    k_sec = next(  # 'A. CONSUMER PRICE INDEX'; not a header line of cells ('S. No. | Name …')
        (k for k, ln in enumerate(lead) if _SECTION_ENUM_RE.match(ln) and " | " not in ln), None
    )
    if k_sec is None:
        k_sec = len(lead)
        while k_sec > 0 and " | " not in lead[k_sec - 1] and len(lead[k_sec - 1].split()) <= 8:
            k_sec -= 1
    header, section = lead[:k_sec], lead[k_sec:]
    groups: list[list[str]] = []
    sections: list[list[str]] = []
    starts: list[bool] = []
    numbered = sum(1 for ln in body if _ENUM_ROW_RE.match(ln)) >= 3
    pending: list[str] = []
    for ln in body:
        if numbered:
            if (
                _ENUM_ROW_RE.match(ln)
                or not groups
                or (value_line(ln) and _labelled(ln) and any(map(value_line, groups[-1])))
            ):  # a numbered line, or a labelled row after a row that has its numbers ('Total')
                groups.append([ln])
                sections.append(section)
                starts.append(False)
            else:
                groups[-1].append(ln)
            continue
        if not value_line(ln):
            pending.append(ln)
            continue
        while pending and groups and _continuation(pending[0]):
            groups[-1].append(pending.pop(0))  # '(Jul-Mar)' ends the row above
        heading = bool(pending) and _labelled(ln)
        if heading:
            section = list(pending)
        groups.append([*pending, ln])
        sections.append(section)
        starts.append(heading)
        pending = []
    if pending and groups:
        groups[-1].extend(pending)
    return FallbackPlan(header, groups, sections, starts, notes)


def fallback_drafts(
    plan: FallbackPlan,
    head: str,
    context: str,
    cfg: ChunkingSettings,
    count: TokenCounter,
    tid: str,
    page: int,
) -> list[Draft]:
    """Chunks of a `fallback_plain` table: whole logical rows, each chunk under the caption,
    the table's header lines and the section heading in force, with the notes."""

    def make(a: int, b: int, lead: bool, notes: bool) -> Draft:
        parts = [head] if head else []
        if lead:
            parts += plan.header
            if not plan.starts[a]:
                parts += plan.sections[a]
        elif a == 0:
            parts += [*plan.header, *plan.sections[0]]
        for g in plan.groups[a:b]:
            parts += g
        text = "\n".join(parts)
        if notes and plan.notes:
            text += "\n\n" + "\n".join(plan.notes)
        embed = f"{context}\n{text}"
        return Draft(ChunkKind.TABLE, page, page, text, embed, count(embed), tid)

    out: list[Draft] = []
    a, n = 0, len(plan.groups)
    while a < n:
        for lead, notes in ((True, True), (True, False), (False, False)):
            b = a + 1
            draft = make(a, b, lead, notes)
            if draft.n_tokens <= cfg.table_max_tokens:
                break
        if draft.n_tokens > cfg.table_max_tokens:  # one logical row too long: lines as prose
            units_ = [Unit(ln, page, page, "\n") for ln in plan.groups[a]]
            out += text_drafts(
                units_,
                context,
                cfg,
                count,
                kind=ChunkKind.TABLE,
                table_id=tid,
                head=head,
                max_tokens=cfg.table_max_tokens,
            )
            a += 1
            continue
        while b < n:
            bigger = make(a, b + 1, lead, notes)
            if bigger.n_tokens > cfg.table_max_tokens:
                break
            draft, b = bigger, b + 1
        out.append(draft)
        a = b
    return out


def table_drafts(
    table: dict[str, Any],
    root: dict[str, Any] | None,
    context: str,
    cfg: ChunkingSettings,
    count: TokenCounter,
) -> list[Draft]:
    """Chunks of one table of tables.parquet (`rows` already parsed from JSON). `root` is the
    first page's table of a continued table: its caption and units are used when this page
    prints none."""
    tid, page = table["table_id"], table["page"]
    caption = table.get("caption") or (root or {}).get("caption")
    units = table.get("units") or (root or {}).get("units")
    head = table_head(caption, units, root["page"] if root else None)
    flags = list(table["flags"])
    if "fallback_plain" in flags or not table["rows"]:
        lines = [ln.strip() for ln in table["markdown"].split("\n") if ln.strip()]
        if not lines:
            return []
        plan = fallback_plan(lines, caption)
        if plan is not None:
            return fallback_drafts(plan, head, context, cfg, count, tid, page)
        units_ = [Unit(ln, page, page, "\n") for ln in lines]
        return text_drafts(
            units_,
            context,
            cfg,
            count,
            kind=ChunkKind.TABLE,
            table_id=tid,
            head=head,
            max_tokens=cfg.table_max_tokens,
        )

    header_md, body, cells, notes = printed_rows(table)
    units_of = cell_units(table, units)
    suffix = [[cu.unit if cu.source in ("row", "column") else None for cu in r] for r in units_of]
    lin = [linearise(table["header"], flags, c, u) for c, u in zip(cells, suffix, strict=True)]
    cont = root["page"] if root else None

    def head_of(a: int, b: int) -> str:
        return table_head(caption, units_line(units_of[a:b], cells[a:b]), cont)

    base = count("")
    widest = head_of(0, len(body))  # names every unit of the table: a bound for any chunk
    prefix_tokens = count(f"{context}\n{widest}" if widest else context)
    row_tokens = [count(x) - base for x in lin]

    def make(a: int, b: int) -> Draft:
        h = head_of(a, b)
        parts = [h, header_md, *body[a:b]] if h else [header_md, *body[a:b]]
        text = "\n".join(parts) + (f"\n\n{notes}" if notes else "")
        prefix = f"{context}\n{h}" if h else context
        embed = prefix + "\n" + "\n".join(lin[a:b])
        return Draft(ChunkKind.TABLE, page, page, text, embed, count(embed), tid, (a, b))

    out: list[Draft] = []
    a = 0
    while a < len(body):
        b, total = a + 1, prefix_tokens + row_tokens[a]
        while b < len(body) and total + row_tokens[b] <= cfg.table_max_tokens:
            total += row_tokens[b]
            b += 1
        draft = make(a, b)
        while draft.n_tokens > cfg.table_max_tokens and b - a > 1:  # the estimate was low
            b -= 1
            draft = make(a, b)
        if draft.n_tokens > cfg.max_tokens:
            raise ValueError(f"{tid} row {a}: {draft.n_tokens} tokens > {cfg.max_tokens}")
        out.append(draft)
        a = b
    return out


# --------------------------------------------------------------------------- documents


def boilerplate_keys(doc_report: dict[str, Any]) -> set[tuple[str, str]]:
    """The extraction's header/footer keys of one document ('top: growth and investment')."""
    out: set[tuple[str, str]] = set()
    for item in doc_report.get("boilerplate", []):
        edge, _, key = item.partition(": ")
        out.add((edge, key))
    return out


def document_drafts(
    doc: ManifestEntry,
    pages: Sequence[dict[str, Any]],
    tables: Sequence[dict[str, Any]],
    variant: Variant,
    cfg: ChunkingSettings,
    count: TokenCounter,
    keys: set[tuple[str, str]] | None = None,
) -> list[Draft]:
    """All chunks of one document: text windows, then (table_aware) its tables in order."""
    context = context_line(doc.edition, doc.title)
    pages = sorted(pages, key=lambda p: p["page"])
    if variant == Variant.TABLE_AWARE:
        units = page_units((p["page"], p["text_body"]) for p in pages)
    else:
        units = page_units(
            ((p["page"], strip_boilerplate_text(p["text_raw"], keys or set())) for p in pages),
            paragraphs=False,
        )
    drafts = [d for d in text_drafts(units, context, cfg, count) if len(d.text.split()) > 0]
    if variant == Variant.TABLE_AWARE:
        by_id = {t["table_id"]: t for t in tables}
        for t in sorted(tables, key=_table_order):
            root = by_id.get(t["continued_from"]) if t.get("continued_from") else None
            drafts.extend(table_drafts(t, root, context, cfg, count))
    return drafts


def _table_order(t: dict[str, Any]) -> tuple[int, int]:
    return t["page"], int(t["table_id"].rsplit(":t", 1)[1])


def chunk_id(variant: str, doc_id: str, page: int, kind: str, ordinal: int, text: str) -> str:
    """sha1 of variant, doc, page, kind, ordinal and text; 16 hex."""
    key = "\x1f".join([variant, doc_id, str(page), kind, str(ordinal), text])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


_FY_RE = re.compile(r"\d{4}-\d{2}")


def numeric_share(text: str) -> float:
    """Share of whitespace words that are numbers or fiscal years (missed-table detector)."""
    words = [w.strip(",;:.") for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return 0.0
    num = sum(1 for w in words if is_number_token(w) or _FY_RE.fullmatch(w))
    return num / len(words)


def doc_order(manifest: Manifest, latest_edition: str) -> list[ManifestEntry]:
    """Latest edition first, then manifest order: the copy of a duplicate that is kept."""
    return sorted(manifest.documents, key=lambda d: d.edition != latest_edition)


def build_chunks(
    manifest: Manifest,
    pages: Sequence[dict[str, Any]],
    tables: Sequence[dict[str, Any]],
    variant: Variant,
    cfg: ChunkingSettings,
    count: TokenCounter,
    *,
    latest_edition: str,
    keys: dict[str, set[tuple[str, str]]] | None = None,
) -> tuple[list[ChunkRecord], dict[str, Any]]:
    """Chunk every manifest document; returns the records and the run report."""
    pages_by: dict[str, list[dict[str, Any]]] = {}
    for p in pages:
        pages_by.setdefault(p["doc_id"], []).append(p)
    tables_by: dict[str, list[dict[str, Any]]] = {}
    for t in tables:
        tables_by.setdefault(t["doc_id"], []).append(t)

    records: list[ChunkRecord] = []
    seen: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    short: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    flagged_pages = {
        (p["doc_id"], p["page"]) for p in pages if "source_review" in (p.get("flags") or [])
    }
    flagged_tables = {
        t["table_id"]: why
        for t in tables
        for why in ("prose_header", "infographic")
        if f"reason:{why}" in (t.get("flags") or [])
    }
    for doc in doc_order(manifest, latest_edition):
        drafts = document_drafts(
            doc,
            pages_by.get(doc.doc_id, []),
            tables_by.get(doc.doc_id, []),
            variant,
            cfg,
            count,
            (keys or {}).get(doc.doc_id),
        )
        ordinals: Counter[str] = Counter()
        for d in drafts:
            ordinal = ordinals[d.kind.value]
            ordinals[d.kind.value] += 1
            why = source_review_reason(d, doc.doc_id, flagged_pages, flagged_tables)
            text = f"{review_line(d, why)}\n{d.text}" if why else d.text
            n_words = len(d.text.split())
            cid = chunk_id(variant.value, doc.doc_id, d.page_start, d.kind.value, ordinal, d.text)
            if d.kind == ChunkKind.TEXT and n_words < cfg.min_words:
                short.append({"doc_id": doc.doc_id, "page": d.page_start, "text": d.text})
                continue
            if d.text in seen:
                duplicates.append({"chunk_id": cid, "doc_id": doc.doc_id, "kept": seen[d.text]})
                continue
            seen[d.text] = cid
            if why:
                review.append(
                    {
                        "chunk_id": cid,
                        "doc_id": doc.doc_id,
                        "pages": [d.page_start, d.page_end],
                        "table_id": d.table_id,
                        "reason": why,
                    }
                )
            records.append(
                ChunkRecord(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    edition=doc.edition,
                    chapter_no=doc.chapter_no,
                    chapter_title=doc.title,
                    page_start=d.page_start,
                    page_end=d.page_end,
                    kind=d.kind,
                    table_id=d.table_id,
                    text=text,
                    embed_text=d.embed_text,
                    bm25_tokens=tokenize(d.text),
                    n_words=n_words,
                    n_tokens=d.n_tokens,
                )
            )
    order = {d.doc_id: i for i, d in enumerate(manifest.documents)}
    records.sort(key=lambda r: order[r.doc_id])  # stable: document order, chunks in order
    report = _report(records, variant, cfg, duplicates, short)
    report["source_review"] = {"n": len(review), "chunks": review}
    return records, report


def source_review_reason(
    d: Draft,
    doc_id: str,
    flagged_pages: set[tuple[str, int]],
    flagged_tables: dict[str, str],
) -> str | None:
    """Why a chunk must be checked against its source page before it is used as evidence, or None:
    its page is flagged `source_review` by extraction (chart text read into the
    prose), or it comes from a table that is really prose or an infographic."""
    if d.table_id and d.table_id in flagged_tables:
        return flagged_tables[d.table_id]
    if any((doc_id, p) in flagged_pages for p in range(d.page_start, d.page_end + 1)):
        return "chart_text_in_prose"
    return None


def review_line(d: Draft, why: str) -> str:
    pages = f"p{d.page_start}" + (f"-{d.page_end}" if d.page_end != d.page_start else "")
    return f"[Check source page {pages}: layout not read reliably ({why})]"


NOTICE_RE = re.compile(
    r"^\[Check source page p\d+(?:-\d+)?: layout not read reliably \((\w+)\)\]\n"
)


def split_notice(text: str) -> tuple[str | None, str]:
    """(review notice or None, content) of a stored chunk `text`. The notice is
    one line for the reader; `n_words`, `bm25_tokens` and the chunk id are of the content."""
    m = NOTICE_RE.match(text)
    return (m.group(0).rstrip("\n"), text[m.end() :]) if m else (None, text)


def _report(
    records: list[ChunkRecord],
    variant: Variant,
    cfg: ChunkingSettings,
    duplicates: list[dict[str, str]],
    short: list[dict[str, Any]],
) -> dict[str, Any]:
    texts = [r for r in records if r.kind == ChunkKind.TEXT]
    heavy = []
    for r in texts:
        share = numeric_share(r.text)
        if share > cfg.numeric_share:
            heavy.append(
                {
                    "chunk_id": r.chunk_id,
                    "doc_id": r.doc_id,
                    "pages": [r.page_start, r.page_end],
                    "share": round(share, 3),
                }
            )
    kinds = Counter(r.kind.value for r in records)
    return {
        "variant": variant.value,
        "n_chunks": len(records),
        "kinds": {k.value: kinds.get(k.value, 0) for k in ChunkKind},
        "n_documents": len({r.doc_id for r in records}),
        "chunking": cfg.model_dump(),
        "n_words": _stats([r.n_words for r in records]),
        "n_tokens": _stats([r.n_tokens for r in records]),
        "n_tokens_by_kind": {
            k.value: _stats([r.n_tokens for r in records if r.kind == k]) for k in ChunkKind
        },
        "n_duplicates_removed": len(duplicates),
        "duplicates": duplicates,
        "n_short_dropped": len(short),
        "short_dropped": short,
        "numeric_heavy": {
            "threshold": cfg.numeric_share,
            "n": len(heavy),
            "share_of_text_chunks": round(len(heavy) / max(len(texts), 1), 4),
            "chunks": heavy,
        },
    }


def _stats(xs: list[int]) -> dict[str, float]:
    if not xs:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {"min": min(xs), "max": max(xs), "mean": round(sum(xs) / len(xs), 1)}


# --------------------------------------------------------------------------- driver


def load_inputs(
    pages_path: Path, tables_path: Path, report_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, set[tuple[str, str]]]]:
    import pyarrow.parquet as pq

    pages = pq.read_table(pages_path).to_pylist()
    tables = pq.read_table(tables_path).to_pylist()
    for t in tables:
        t["rows"] = json.loads(t["rows"]) if isinstance(t["rows"], str) else t["rows"]
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    keys = {doc: boilerplate_keys(r) for doc, r in rep["documents"].items()}
    return pages, tables, keys


CHUNK_SCHEMA_FIELDS = (
    ("chunk_id", "string"),
    ("doc_id", "string"),
    ("edition", "string"),
    ("chapter_no", "int32"),
    ("chapter_title", "string"),
    ("page_start", "int32"),
    ("page_end", "int32"),
    ("kind", "string"),
    ("table_id", "string"),
    ("text", "string"),
    ("embed_text", "string"),
    ("bm25_tokens", "list<string>"),
    ("n_words", "int32"),
    ("n_tokens", "int32"),
)


def write_chunks(records: list[ChunkRecord], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    types = {"string": pa.string(), "int32": pa.int32(), "list<string>": pa.list_(pa.string())}
    schema = pa.schema([(name, types[t]) for name, t in CHUNK_SCHEMA_FIELDS])
    rows = [r.model_dump(mode="json") for r in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def read_chunks(path: Path) -> list[ChunkRecord]:
    import pyarrow.parquet as pq

    return [ChunkRecord.model_validate(r) for r in pq.read_table(path).to_pylist()]


def run_chunk(
    variant: Variant,
    manifest: Manifest,
    pages: Sequence[dict[str, Any]],
    tables: Sequence[dict[str, Any]],
    keys: dict[str, set[tuple[str, str]]],
    cfg: ChunkingSettings,
    count: TokenCounter,
    out_chunks: Path,
    out_report: Path,
    *,
    latest_edition: str,
) -> dict[str, Any]:
    """Build one variant's chunks.parquet and its run report."""
    from datetime import UTC, datetime

    records, report = build_chunks(
        manifest, pages, tables, variant, cfg, count, latest_edition=latest_edition, keys=keys
    )
    write_chunks(records, out_chunks)
    report = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), **report}
    out_report.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report
