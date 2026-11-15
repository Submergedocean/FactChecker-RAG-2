"""`claimcheck claims validate`: every rule a claims file must pass.

Row rules (no corpus needed):

- the line parses and matches the 7.2 schema (`bad_label` for a label outside
  SUPPORTED / REFUTED / NEI);
- unique `claim_id`; no duplicate claim text after number, unit and fiscal-year
  normalisation (`dedup_key`);
- SUPPORTED and REFUTED carry evidence, NEI carries none;
- Urdu and Roman Urdu claims name an English `parallel_of` in the same fact with the same
  label; English claims name none; `parallel_of` must exist;
- a tabular claim cites at least one table cell; table evidence names row and column;
  a text quote has at least 8 characters and 2 tokens; `topic` is one of `TOPICS`;
- the claim passes the app's input guard (10-500 characters).

Corpus rules (when `data/processed` is present):

- the document and page exist; a text quote is verbatim on that page (the pipeline's citation guard,
  `quote_in_passage`); a table cell is found by `row_label` and `col_label` in that table
  and equals the quote;
- every evidence item resolves to at least one chunk in each index variant: same
  `doc_id`, page inside the chunk's page range, at least 60 % of the evidence tokens present
  and every number of the quote present. `resolve_evidence(relaxed=True)` drops the
  same-document condition (the evaluation reports both).

Dataset gates: the pilot gate (at least `pilot_size` claims, no errors) and the final gate
(about 600 claims, label mix within 5 points of 40/40/20, about half tabular, 80/20 language,
every fact in `splits.json`).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from claimcheck.claims.io import JsonLine, iter_jsonl
from claimcheck.claims.schema import (
    TOPICS,
    Claim,
    ClaimLanguage,
    ClaimType,
    EvidenceRef,
    Verdict,
    dedup_key,
    evidence_match_text,
    row_leaf,
)
from claimcheck.config import ClaimsSettings, LimitsSettings
from claimcheck.pipeline.guards import MIN_QUOTE_CHARS, MIN_QUOTE_TOKENS, match_key
from claimcheck.pipeline.guards import quote_in_passage as _quote_in_passage
from claimcheck.schemas import Split, Variant
from claimcheck.textnorm import parse_number, tokenize

Severity = Literal["error", "warning"]

MIN_TOKEN_SHARE = 0.6  # at least 60 % of the evidence tokens in the chunk


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str
    claim_id: str | None = None
    line: int | None = None

    def __str__(self) -> str:
        where = " ".join(
            p for p in (f"line {self.line}" if self.line else "", self.claim_id or "") if p
        )
        return f"{self.severity.upper():7} {self.code:24} {where + ': ' if where else ''}" + (
            self.message
        )


# --------------------------------------------------------------------------- corpus


@dataclass(frozen=True)
class TableView:
    """A table as printed: header, row labels and printed cells."""

    table_id: str
    doc_id: str
    page: int
    header: list[str]
    labels: list[str]
    cells: list[list[str]]
    lead: int  # label columns before the values

    def locate(self, row_label: str, col_label: str) -> tuple[int, int] | str:
        """(row index, column index) of the cell, or a problem message."""
        want_col = match_key(col_label)
        cols = [
            k for k in range(self.lead, len(self.header)) if match_key(self.header[k]) == want_col
        ]
        if not cols:
            return f"no column {col_label!r} in {self.table_id}"
        if len(cols) > 1:
            return f"column {col_label!r} is ambiguous in {self.table_id}"
        want_row = match_key(row_label)
        rows = [i for i, lab in enumerate(self.labels) if match_key(lab) == want_row]
        if not rows:  # the label's last segment ('1. Crops') is enough when it is unique
            rows = [i for i, lab in enumerate(self.labels) if match_key(row_leaf(lab)) == want_row]
        if not rows and self.lead > 1:
            # Several label columns ('Natural Gas | 000 MMCFT'): the first filled one is enough
            # when it is unique.
            rows = [
                i
                for i, row in enumerate(self.cells)
                if (first := next((c for c in row[: self.lead] if c), ""))
                and match_key(row_leaf(first)) == want_row
            ]
        if not rows:
            return f"no row {row_label!r} in {self.table_id}"
        if len(rows) > 1:
            return f"row {row_label!r} is ambiguous in {self.table_id}; use the full label"
        return rows[0], cols[0]

    def find_cell(self, row_label: str, col_label: str) -> tuple[str | None, str | None]:
        """(cell text, None) or (None, problem)."""
        found = self.locate(row_label, col_label)
        if isinstance(found, str):
            return None, found
        i, k = found
        row = self.cells[i]
        return (row[k] if k < len(row) else ""), None


@dataclass
class ChunkView:
    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    text: str
    _tokens: frozenset[str] | None = None

    @property
    def tokens(self) -> frozenset[str]:
        if self._tokens is None:
            self._tokens = frozenset(tokenize(self.text))
        return self._tokens


class SourceCorpus(Protocol):
    """What the corpus rules need; `Corpus` loads it from `data/`, tests build a fake."""

    def n_pages(self, doc_id: str) -> int | None: ...  # None: unknown document

    def page_texts(self, doc_id: str, page: int) -> Sequence[str]: ...

    def table(self, table_id: str) -> TableView | None: ...

    def variants(self) -> Sequence[str]: ...

    def chunks(self, variant: str, doc_id: str | None = None) -> Sequence[ChunkView]: ...


@dataclass
class Corpus:
    """Manifest, pages, tables and the chunks of both variants, loaded once."""

    docs: dict[str, int]  # doc_id -> number of pages
    pages: dict[tuple[str, int], tuple[str, ...]]
    tables: dict[str, TableView]
    chunk_views: dict[str, list[ChunkView]] = field(default_factory=dict)
    _by_doc: dict[str, dict[str, list[ChunkView]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for variant, views in self.chunk_views.items():
            grouped: dict[str, list[ChunkView]] = defaultdict(list)
            for v in views:
                grouped[v.doc_id].append(v)
            self._by_doc[variant] = dict(grouped)

    @classmethod
    def load(cls, settings: Any | None = None) -> Corpus:
        import pyarrow.parquet as pq

        from claimcheck.config import get_settings
        from claimcheck.ingest.chunk import printed_rows
        from claimcheck.ingest.tables import lead_columns, row_label
        from claimcheck.schemas import Manifest

        settings = settings or get_settings()
        paths = settings.paths
        manifest = Manifest.model_validate(json.loads(paths.manifest.read_text(encoding="utf-8")))
        docs = {d.doc_id: d.n_pages or 0 for d in manifest.documents}
        pages: dict[tuple[str, int], tuple[str, ...]] = {}
        for r in pq.read_table(
            paths.pages, columns=["doc_id", "page", "text_body", "text_raw"]
        ).to_pylist():
            pages[(r["doc_id"], int(r["page"]))] = (r["text_body"] or "", r["text_raw"] or "")
        tables: dict[str, TableView] = {}
        for t in pq.read_table(paths.tables).to_pylist():
            flags = list(t["flags"] or [])
            if "fallback_plain" in flags:
                continue
            t["header"] = list(t["header"] or [])
            t["rows"] = json.loads(t["rows"]) if isinstance(t["rows"], str) else t["rows"]
            _, _, cells, _ = printed_rows(t)
            tables[t["table_id"]] = TableView(
                table_id=t["table_id"],
                doc_id=t["doc_id"],
                page=int(t["page"]),
                header=t["header"],
                labels=[row_label(flags, c) for c in cells],
                cells=cells,
                lead=lead_columns(flags),
            )
        chunk_views: dict[str, list[ChunkView]] = {}
        for variant in Variant:
            path = paths.chunks(variant.value)
            if not path.exists():
                continue
            rows = pq.read_table(
                path, columns=["chunk_id", "doc_id", "page_start", "page_end", "text"]
            ).to_pylist()
            chunk_views[variant.value] = [ChunkView(**r) for r in rows]
        return cls(docs, pages, tables, chunk_views)

    def n_pages(self, doc_id: str) -> int | None:
        return self.docs.get(doc_id)

    def page_texts(self, doc_id: str, page: int) -> Sequence[str]:
        return self.pages.get((doc_id, page), ())

    def table(self, table_id: str) -> TableView | None:
        return self.tables.get(table_id)

    def variants(self) -> Sequence[str]:
        return list(self.chunk_views)

    def chunks(self, variant: str, doc_id: str | None = None) -> Sequence[ChunkView]:
        if doc_id is None:
            return self.chunk_views.get(variant, [])
        return self._by_doc.get(variant, {}).get(doc_id, [])


# --------------------------------------------------------------------------- evidence


def quote_in_text(quote: str, text: str) -> bool:
    return _quote_in_passage(quote, text)


def cell_equals(cell: str, quote: str) -> bool:
    """A printed cell equals the quoted cell: same text, or the same number ('8,137,860' and
    '8137860'; a trailing footnote mark such as '1607#' is ignored)."""
    if match_key(cell) == match_key(quote):
        return True
    a = parse_number(cell.rstrip("*#^@†‡ "))
    b = parse_number(quote.rstrip("*#^@†‡ "))
    return a is not None and b is not None and abs(a - b) < 1e-9


def evidence_tokens(ev: EvidenceRef) -> tuple[set[str], set[str]]:
    """(tokens of the match text, numeric tokens of the quote)."""
    tokens = set(tokenize(evidence_match_text(ev)))
    numbers = {t for t in tokenize(ev.quote) if any(ch.isdigit() for ch in t)}
    return tokens, numbers


def chunk_holds(ev_tokens: set[str], numbers: set[str], chunk: ChunkView) -> bool:
    if not ev_tokens:
        return False
    present = len(ev_tokens & chunk.tokens) / len(ev_tokens)
    return present >= MIN_TOKEN_SHARE and numbers <= chunk.tokens


def resolve_evidence(
    ev: EvidenceRef, corpus: SourceCorpus, variant: str, *, relaxed: bool = False
) -> list[str]:
    """Chunk ids of `variant` that hold this evidence. Strict: same document and
    the page inside the chunk's range. Relaxed: any document (Overview and Highlights repeat
    chapter facts)."""
    tokens, numbers = evidence_tokens(ev)
    if relaxed:
        candidates: Iterable[ChunkView] = corpus.chunks(variant)
    else:
        candidates = (
            c for c in corpus.chunks(variant, ev.doc_id) if c.page_start <= ev.page <= c.page_end
        )
    return [c.chunk_id for c in candidates if chunk_holds(tokens, numbers, c)]


def check_evidence_source(ev: EvidenceRef, corpus: SourceCorpus) -> list[tuple[str, str]]:
    """(code, message) problems of one evidence item against the PDFs' text and tables."""
    n_pages = corpus.n_pages(ev.doc_id)
    if n_pages is None:
        return [("unknown_doc", f"no document {ev.doc_id!r}")]
    if n_pages and ev.page > n_pages:
        return [("bad_page", f"{ev.doc_id} has {n_pages} pages, not {ev.page}")]
    if ev.table_id:
        table = corpus.table(ev.table_id)
        if table is None:
            return [("unknown_table", f"no parsed table {ev.table_id!r}")]
        if table.doc_id != ev.doc_id or table.page != ev.page:
            return [
                (
                    "table_mismatch",
                    f"{ev.table_id} is on {table.doc_id} p{table.page}, "
                    f"evidence says {ev.doc_id} p{ev.page}",
                )
            ]
        if not ev.row_label or not ev.col_label:
            return []  # reported by the row rules
        cell, problem = table.find_cell(ev.row_label, ev.col_label)
        if problem:
            return [("cell_not_found", problem)]
        if cell is not None and not cell_equals(cell, ev.quote):
            return [
                (
                    "cell_mismatch",
                    f"{ev.table_id} [{ev.row_label} | {ev.col_label}] is {cell!r}, "
                    f"quote is {ev.quote!r}",
                )
            ]
        return []
    texts = corpus.page_texts(ev.doc_id, ev.page)
    if not any(quote_in_text(ev.quote, t) for t in texts):
        return [("quote_not_found", f"quote not on {ev.doc_id} p{ev.page}: {ev.quote[:80]!r}")]
    return []


# --------------------------------------------------------------------------- rows


def _schema_issue(item: JsonLine, exc: ValidationError) -> Issue:
    claim_id = item.data.get("claim_id") if item.data else None
    errors = exc.errors()
    for err in errors:
        if err["loc"] and err["loc"][0] in ("label", "second_label"):
            return Issue(
                "error",
                "bad_label",
                f"{err['loc'][0]} {item.data.get(err['loc'][0])!r} is not one of "  # type: ignore[union-attr]
                "SUPPORTED, REFUTED, NEI",
                claim_id if isinstance(claim_id, str) else None,
                item.line,
            )
    first = errors[0]
    loc = ".".join(str(p) for p in first["loc"]) or "(root)"
    return Issue(
        "error",
        "schema",
        f"{loc}: {first['msg']}",
        claim_id if isinstance(claim_id, str) else None,
        item.line,
    )


@dataclass(frozen=True)
class LoadedClaim:
    claim: Claim
    line: int | None = None


def parse_lines(items: Iterable[JsonLine]) -> tuple[list[LoadedClaim], list[Issue]]:
    loaded: list[LoadedClaim] = []
    issues: list[Issue] = []
    for item in items:
        if item.data is None:
            issues.append(Issue("error", "not_json", item.error or "", None, item.line))
            continue
        try:
            loaded.append(LoadedClaim(Claim.model_validate(item.data), item.line))
        except ValidationError as exc:
            issues.append(_schema_issue(item, exc))
    return loaded, issues


def check_claims(
    claims: Sequence[LoadedClaim] | Sequence[Claim],
    *,
    corpus: SourceCorpus | None = None,
    limits: LimitsSettings | None = None,
) -> list[Issue]:
    """Row and cross-row rules, plus the corpus rules when `corpus` is given."""
    limits = limits or LimitsSettings()
    rows = [c if isinstance(c, LoadedClaim) else LoadedClaim(c) for c in claims]
    issues: list[Issue] = []

    def add(sev: Severity, code: str, msg: str, row: LoadedClaim) -> None:
        issues.append(Issue(sev, code, msg, row.claim.claim_id, row.line))

    by_id: dict[str, LoadedClaim] = {}
    for row in rows:
        if row.claim.claim_id in by_id:
            add("error", "duplicate_id", "claim_id used twice", row)
        else:
            by_id[row.claim.claim_id] = row

    seen_text: dict[str, str] = {}
    variant_slots: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        c = row.claim
        n = len(c.claim_text.strip())
        if not limits.input_min_chars <= n <= limits.input_max_chars:
            add(
                "error",
                "claim_length",
                f"{n} characters; the app accepts "
                f"{limits.input_min_chars}-{limits.input_max_chars}",
                row,
            )
        key = dedup_key(c.claim_text)
        if key in seen_text and seen_text[key] != c.claim_id:
            add("error", "duplicate_text", f"same claim as {seen_text[key]}", row)
        else:
            seen_text.setdefault(key, c.claim_id)
        variant_slots[(c.fact_id, c.language.value, c.label.value)] += 1

        # evidence and label
        if c.label is Verdict.NEI and c.evidence:
            add("error", "nei_with_evidence", "an NEI claim has no evidence", row)
        if c.label is not Verdict.NEI and not c.evidence:
            add("error", "missing_evidence", f"a {c.label.value} claim needs evidence", row)
        if c.topic not in TOPICS:
            add(
                "error",
                "unknown_topic",
                f"topic {c.topic!r} is not one of {', '.join(TOPICS)}",
                row,
            )
        has_table = any(e.table_id for e in c.evidence)
        if c.claim_type is ClaimType.TABULAR and c.label is not Verdict.NEI and not has_table:
            add("error", "tabular_without_table", "a tabular claim cites a table cell", row)
        if c.claim_type is ClaimType.TEXTUAL and has_table:
            add("warning", "textual_with_table", "a textual claim cites a table cell", row)
        for i, ev in enumerate(c.evidence, start=1):
            tag = f"evidence {i}"
            if ev.table_id:
                if not ev.row_label or not ev.col_label:
                    add(
                        "error",
                        "table_evidence_incomplete",
                        f"{tag}: table evidence needs row_label and col_label",
                        row,
                    )
                if not ev.quote.strip():
                    add("error", "short_quote", f"{tag}: the quote is the cell value", row)
            else:
                q = match_key(ev.quote)
                if len(q) < MIN_QUOTE_CHARS or len(q.split()) < MIN_QUOTE_TOKENS:
                    add(
                        "error",
                        "short_quote",
                        f"{tag}: a text quote needs {MIN_QUOTE_CHARS}+ characters and "
                        f"{MIN_QUOTE_TOKENS}+ words",
                        row,
                    )
                if ev.row_label or ev.col_label:
                    add(
                        "warning",
                        "labels_without_table",
                        f"{tag}: row/col label without table_id",
                        row,
                    )

        # translations
        if c.language is ClaimLanguage.EN:
            if c.parallel_of:
                add("error", "parallel_on_english", "English claims have no parallel_of", row)
        elif not c.parallel_of:
            add("error", "missing_parallel", f"a {c.language.value} claim names parallel_of", row)
        if c.parallel_of:
            target = by_id.get(c.parallel_of)
            if target is None:
                add("error", "orphan_parallel", f"parallel_of {c.parallel_of} does not exist", row)
            else:
                t = target.claim
                if t.language is not ClaimLanguage.EN:
                    add("error", "parallel_not_english", f"{t.claim_id} is not English", row)
                if t.fact_id != c.fact_id:
                    add(
                        "error",
                        "parallel_fact_mismatch",
                        f"fact {c.fact_id} differs from {t.claim_id}'s {t.fact_id}",
                        row,
                    )
                if t.label is not c.label:
                    add(
                        "error",
                        "parallel_label_mismatch",
                        f"label {c.label.value} differs from {t.claim_id}'s {t.label.value}",
                        row,
                    )

    for (fact, lang, label), n in variant_slots.items():
        if n > 1:
            issues.append(
                Issue(
                    "warning",
                    "fact_slot_repeated",
                    f"fact {fact} has {n} {lang} {label} claims",
                )
            )

    if corpus is not None:
        issues.extend(check_corpus(rows, corpus))
    return issues


def check_corpus(rows: Sequence[LoadedClaim], corpus: SourceCorpus) -> list[Issue]:
    issues: list[Issue] = []
    variants = list(corpus.variants())
    for row in rows:
        for i, ev in enumerate(row.claim.evidence, start=1):
            problems = check_evidence_source(ev, corpus)
            for code, msg in problems:
                issues.append(
                    Issue("error", code, f"evidence {i}: {msg}", row.claim.claim_id, row.line)
                )
            if problems:
                continue
            for variant in variants:
                if not resolve_evidence(ev, corpus, variant):
                    issues.append(
                        Issue(
                            "error",
                            "unresolved",
                            f"evidence {i} ({ev.doc_id} p{ev.page}) is in no {variant} chunk",
                            row.claim.claim_id,
                            row.line,
                        )
                    )
    return issues


# --------------------------------------------------------------------------- dataset


@dataclass
class DatasetStats:
    n: int
    labels: Counter[str]
    languages: Counter[str]
    claim_types: Counter[str]
    reasoning_types: Counter[str]
    topics: Counter[str]
    origins: Counter[str]
    n_facts: int
    n_second_label: int

    @classmethod
    def of(cls, claims: Sequence[Claim]) -> DatasetStats:
        return cls(
            n=len(claims),
            labels=Counter(c.label.value for c in claims),
            languages=Counter(c.language.value for c in claims),
            claim_types=Counter(c.claim_type.value for c in claims),
            reasoning_types=Counter(c.reasoning_type.value for c in claims),
            topics=Counter(c.topic for c in claims),
            origins=Counter((c.origin or "").split(":", 1)[0] or "unknown" for c in claims),
            n_facts=len({c.fact_id for c in claims}),
            n_second_label=sum(c.second_label is not None for c in claims),
        )

    def share(self, counter: Counter[str], key: str) -> float:
        return counter.get(key, 0) / self.n if self.n else 0.0

    def lines(self) -> list[str]:
        def fmt(counter: Counter[str]) -> str:
            return (
                ", ".join(f"{k} {v} ({v / self.n:.0%})" for k, v in sorted(counter.items()))
                if self.n
                else "-"
            )

        return [
            f"claims {self.n}, facts {self.n_facts}, second labels {self.n_second_label}",
            f"label: {fmt(self.labels)}",
            f"language: {fmt(self.languages)}",
            f"claim_type: {fmt(self.claim_types)}",
            f"reasoning_type: {fmt(self.reasoning_types)}",
            f"origin: {fmt(self.origins)}",
            f"topics: {len(self.topics)} of {len(TOPICS)}",
        ]


def gate_issues(
    claims: Sequence[Claim],
    settings: ClaimsSettings,
    *,
    gate: Literal["pilot", "final"],
    splits: dict[str, Split] | None = None,
) -> list[Issue]:
    """Dataset-level gate failures (errors only)."""
    stats = DatasetStats.of(claims)
    out: list[Issue] = []
    if gate == "pilot":
        if stats.n < settings.pilot_size:
            out.append(
                Issue(
                    "error",
                    "pilot_size",
                    f"{stats.n} claims; the pilot needs {settings.pilot_size}",
                )
            )
        return out

    lo = round(settings.target_total * (1 - settings.total_tolerance))
    hi = round(settings.target_total * (1 + settings.total_tolerance))
    if not lo <= stats.n <= hi:
        out.append(Issue("error", "total", f"{stats.n} claims; the final gate needs {lo}-{hi}"))
    for label, target in settings.label_mix.items():
        share = stats.share(stats.labels, label)
        if abs(share - target) > settings.label_tolerance + 1e-9:
            out.append(
                Issue(
                    "error",
                    "label_mix",
                    f"{label} {share:.1%}; target {target:.0%} ± {settings.label_tolerance:.0%}",
                )
            )
    tab = stats.share(stats.claim_types, ClaimType.TABULAR.value)
    if abs(tab - settings.tabular_share) > settings.tabular_tolerance + 1e-9:
        out.append(
            Issue(
                "error",
                "tabular_share",
                f"tabular {tab:.1%}; target {settings.tabular_share:.0%} "
                f"± {settings.tabular_tolerance:.0%}",
            )
        )
    en = stats.share(stats.languages, ClaimLanguage.EN.value)
    if abs(en - settings.english_share) > settings.language_tolerance + 1e-9:
        out.append(
            Issue(
                "error",
                "language_mix",
                f"English {en:.1%}; target {settings.english_share:.0%} "
                f"± {settings.language_tolerance:.0%}",
            )
        )
    splits = splits or {}
    missing = sorted({c.fact_id for c in claims} - set(splits))
    if missing:
        out.append(
            Issue(
                "error",
                "unsplit_facts",
                f"{len(missing)} fact(s) not in splits.json (first: {', '.join(missing[:5])}); "
                "run `claimcheck claims split`",
            )
        )
    return out


@dataclass
class ValidationReport:
    claims: list[Claim]
    issues: list[Issue]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_file(
    path: Path,
    *,
    corpus: SourceCorpus | None = None,
    limits: LimitsSettings | None = None,
    settings: ClaimsSettings | None = None,
    gate: Literal["pilot", "final"] | None = None,
    splits: dict[str, Split] | None = None,
) -> ValidationReport:
    loaded, issues = parse_lines(iter_jsonl(path))
    issues.extend(check_claims(loaded, corpus=corpus, limits=limits))
    claims = [r.claim for r in loaded]
    if splits:
        for fact in sorted({c.fact_id for c in claims} - set(splits)):
            issues.append(Issue("warning", "unsplit_fact", f"fact {fact} not in splits.json"))
    if gate is not None:
        issues.extend(gate_issues(claims, settings or ClaimsSettings(), gate=gate, splits=splits))
    return ValidationReport(claims, issues)
