"""Reading and writing the claims files.

- `claims.jsonl` and `drafts.jsonl`: one JSON object per line, UTF-8, Urdu kept as is.
- `splits.json`: `{fact_id: split}`, keys sorted.
- Review CSV: one claim per row, the first evidence item in `ev_*` columns and any further
  items as JSON in `ev_more`. Written with a UTF-8 BOM so spreadsheet programs show Urdu.
  The reviewer fills `decision` (`accept`, `edit`, `reject`, or blank = no change); the
  columns in `READ_ONLY_COLUMNS` are context and are ignored on import.

Writes are atomic (temporary file, then rename).
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from claimcheck.claims.schema import CLAIM_ID_RE, Claim, DraftRecord, DraftStatus, EvidenceRef
from claimcheck.schemas import Split

# --------------------------------------------------------------------------- JSONL


@dataclass(frozen=True)
class JsonLine:
    line: int  # 1-based line number in the file
    data: dict[str, Any] | None  # None when the line is not a JSON object
    error: str | None = None


def iter_jsonl(path: Path) -> Iterator[JsonLine]:
    """Every non-blank line of a JSONL file, parsed or with its parse error."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for n, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                yield JsonLine(n, None, f"not JSON: {exc.msg}")
                continue
            if not isinstance(obj, dict):
                yield JsonLine(n, None, "not a JSON object")
                continue
            yield JsonLine(n, obj)


class ClaimsFileError(ValueError):
    """A claims or drafts file that does not parse; the message lists the bad lines."""


def _read_models[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    out: list[T] = []
    problems: list[str] = []
    for item in iter_jsonl(path):
        if item.data is None:
            problems.append(f"line {item.line}: {item.error}")
            continue
        try:
            out.append(model.model_validate(item.data))
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first["loc"]) or "(root)"
            problems.append(f"line {item.line}: {loc}: {first['msg']}")
    if problems:
        shown = "; ".join(problems[:10])
        more = f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""
        raise ClaimsFileError(f"{path}: {shown}{more}")
    return out


def read_claims(path: Path) -> list[Claim]:
    """All claims; raises `ClaimsFileError` on any bad line (use `claims validate` to list)."""
    return _read_models(path, Claim)


def read_drafts(path: Path) -> list[DraftRecord]:
    return _read_models(path, DraftRecord)


def _atomic_write(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def claim_sort_key(claim_id: str) -> tuple[int, str]:
    m = CLAIM_ID_RE.match(claim_id)
    return (int(m.group(1)) if m else 10**9, claim_id)


def claim_json(claim: Claim) -> str:
    return json.dumps(claim.model_dump(mode="json"), ensure_ascii=False)


def write_claims(path: Path, claims: Iterable[Claim]) -> None:
    """Write claims sorted by id, one JSON object per line."""
    ordered = sorted(claims, key=lambda c: claim_sort_key(c.claim_id))
    _atomic_write(path, "".join(claim_json(c) + "\n" for c in ordered))


def write_drafts(path: Path, drafts: Iterable[DraftRecord]) -> None:
    ordered = sorted(drafts, key=lambda d: claim_sort_key(d.claim.claim_id))
    _atomic_write(
        path,
        "".join(json.dumps(d.model_dump(mode="json"), ensure_ascii=False) + "\n" for d in ordered),
    )


# --------------------------------------------------------------------------- splits


def read_splits(path: Path) -> dict[str, Split]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(data, dict):
        raise ClaimsFileError(f"{path}: must be a JSON object {{fact_id: split}}")
    try:
        return {str(k): Split(v) for k, v in data.items()}
    except ValueError as exc:
        raise ClaimsFileError(f"{path}: {exc}") from exc


def write_splits(path: Path, splits: dict[str, Split]) -> None:
    data = {k: splits[k].value for k in sorted(splits)}
    _atomic_write(path, json.dumps(data, indent=1, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- review CSV

EDITABLE_COLUMNS: tuple[str, ...] = (
    "claim_id",
    "fact_id",
    "decision",
    "claim_text",
    "label",
    "language",
    "parallel_of",
    "claim_type",
    "reasoning_type",
    "topic",
    "ev_doc_id",
    "ev_page",
    "ev_table_id",
    "ev_quote",
    "ev_row_label",
    "ev_col_label",
    "ev_more",
    "origin",
    "annotator",
    "second_label",
    "notes",
)
READ_ONLY_COLUMNS: tuple[str, ...] = ("status", "mutation", "checks", "pdf_link", "context")
CSV_COLUMNS: tuple[str, ...] = EDITABLE_COLUMNS + READ_ONLY_COLUMNS
_EVIDENCE_FIELDS = ("doc_id", "page", "table_id", "quote", "row_label", "col_label")


class Decision(StrEnum):
    NONE = ""  # no change
    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"


_DECISION_ALIASES = {
    "": Decision.NONE,
    "accept": Decision.ACCEPT,
    "accepted": Decision.ACCEPT,
    "a": Decision.ACCEPT,
    "y": Decision.ACCEPT,
    "yes": Decision.ACCEPT,
    "ok": Decision.ACCEPT,
    "edit": Decision.EDIT,
    "edited": Decision.EDIT,
    "e": Decision.EDIT,
    "reject": Decision.REJECT,
    "rejected": Decision.REJECT,
    "r": Decision.REJECT,
    "n": Decision.REJECT,
    "no": Decision.REJECT,
}


def parse_decision(value: str | None) -> Decision:
    key = (value or "").strip().lower()
    if key not in _DECISION_ALIASES:
        raise ValueError(f"decision {value!r}: use accept, edit, reject or leave blank")
    return _DECISION_ALIASES[key]


def claim_to_row(claim: Claim, **extra: Any) -> dict[str, str]:
    """A CSV row for `claim`; `extra` fills `decision` and the read-only columns."""
    row: dict[str, str] = dict.fromkeys(CSV_COLUMNS, "")
    data = claim.model_dump(mode="json")
    for key in EDITABLE_COLUMNS:
        if key in data and key != "evidence":
            row[key] = "" if data[key] is None else str(data[key])
    evidence = data["evidence"]
    if evidence:
        first = evidence[0]
        for f in _EVIDENCE_FIELDS:
            row[f"ev_{f}"] = "" if first[f] is None else str(first[f])
        if len(evidence) > 1:
            row["ev_more"] = json.dumps(evidence[1:], ensure_ascii=False)
    for key, value in extra.items():
        if key not in CSV_COLUMNS:
            raise KeyError(f"unknown CSV column {key!r}")
        row[key] = "" if value is None else str(value)
    return row


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def row_to_claim(row: dict[str, str]) -> Claim:
    """The claim in an edited CSV row (raises `ValueError` / `ValidationError`). `claim_id` and
    `fact_id` must already be filled in (the importer assigns them to new rows)."""
    evidence: list[dict[str, Any]] = []
    first = {f: _blank_to_none(row.get(f"ev_{f}")) for f in _EVIDENCE_FIELDS}
    if any(v is not None for v in first.values()):
        page = first["page"]
        if page is not None:
            try:
                first["page"] = int(float(page))  # spreadsheets may write "12.0"
            except ValueError as exc:
                raise ValueError(f"ev_page {page!r} is not a page number") from exc
        evidence.append(first)
    more = _blank_to_none(row.get("ev_more"))
    if more:
        try:
            extra = json.loads(more)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ev_more is not JSON: {exc.msg}") from exc
        if not isinstance(extra, list):
            raise ValueError("ev_more must be a JSON list of evidence objects")
        evidence.extend(extra)
    data: dict[str, Any] = {
        key: _blank_to_none(row.get(key))
        for key in EDITABLE_COLUMNS
        if not key.startswith("ev_") and key != "decision"
    }
    for key in ("label", "second_label"):
        if data[key] is not None:
            data[key] = data[key].upper()
    for key in ("language", "claim_type", "reasoning_type"):
        if data[key] is not None:
            data[key] = data[key].lower()
    data["evidence"] = [EvidenceRef.model_validate(e) for e in evidence]
    return Claim.model_validate(data)


def write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Rows in `CSV_COLUMNS` order, UTF-8 with BOM (spreadsheets then read Urdu correctly)."""
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
    _atomic_write(path, buf.getvalue(), encoding="utf-8-sig")


@dataclass(frozen=True)
class CsvRow:
    line: int  # spreadsheet row number (the header is row 1)
    data: dict[str, str]


def read_csv(path: Path, *, required: Sequence[str] = ("claim_id", "decision")) -> list[CsvRow]:
    """Rows of a review CSV; missing optional columns read as blank."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in required if c not in header]
        if missing:
            raise ClaimsFileError(f"{path}: missing column(s) {', '.join(missing)}")
        reader.fieldnames = header
        rows = []
        for n, raw in enumerate(reader, start=2):
            data = {k: (v if isinstance(v, str) else "") for k, v in raw.items() if k}
            if not any(v.strip() for v in data.values()):
                continue
            rows.append(CsvRow(n, data))
    return rows


@dataclass
class ReviewResult:
    claims: list[Claim]
    drafts: list[DraftRecord]
    problems: list[tuple[int, str]]  # (CSV row, message)
    counts: dict[str, int]
    # New rows: CSV row -> (claim_id, fact_id, origin) assigned, to be written back into the
    # sheet so that importing it again updates these claims instead of adding them twice.
    assigned: dict[int, tuple[str, str, str]] = field(default_factory=dict)


def _edited_origin(origin: str | None) -> str | None:
    if origin and origin.startswith("llm_draft:"):
        return "llm_draft_edited:" + origin.split(":", 1)[1]
    return origin


def merge_review(
    rows: Sequence[CsvRow],
    claims: Sequence[Claim],
    drafts: Sequence[DraftRecord],
    *,
    annotator: str | None = None,
) -> ReviewResult:
    """Apply a reviewed CSV: `accept` / `edit` upsert the row's claim into claims.jsonl (and
    mark its draft accepted), `reject` removes it (and marks its draft rejected), blank leaves
    it. A row without `claim_id` is a new hand-written claim (`origin` "human"): it gets the
    next id, and the next fact id unless `fact_id` is given or `parallel_of` names a claim.
    Rows that do not parse are reported and change nothing."""
    from claimcheck.claims.schema import FACT_ID_RE, format_id, next_number

    by_id = {c.claim_id: c for c in claims}
    draft_by_id = {d.claim.claim_id: d for d in drafts}
    # Ids written in the sheet are taken too, so a new row never reuses one.
    sheet_ids = [(r.data.get("claim_id") or "").strip() for r in rows]
    sheet_facts = [(r.data.get("fact_id") or "").strip() for r in rows]
    all_ids = list(by_id) + list(draft_by_id) + sheet_ids
    next_claim = next_number(all_ids, CLAIM_ID_RE)
    facts = [c.fact_id for c in claims] + [d.claim.fact_id for d in drafts] + sheet_facts
    next_fact = next_number(facts, FACT_ID_RE)
    problems: list[tuple[int, str]] = []
    counts = {"accepted": 0, "edited": 0, "rejected": 0, "added": 0, "unchanged": 0}
    assigned: dict[int, tuple[str, str, str]] = {}

    for row in rows:
        try:
            decision = parse_decision(row.data.get("decision"))
        except ValueError as exc:
            problems.append((row.line, str(exc)))
            continue
        claim_id = (row.data.get("claim_id") or "").strip()
        if decision is Decision.NONE:
            counts["unchanged"] += 1
            continue
        if decision is Decision.REJECT:
            if not claim_id:
                counts["unchanged"] += 1  # a new row rejected: nothing to do
                continue
            by_id.pop(claim_id, None)
            if claim_id in draft_by_id:
                draft_by_id[claim_id] = draft_by_id[claim_id].model_copy(
                    update={"status": DraftStatus.REJECTED}
                )
            counts["rejected"] += 1
            continue
        data = dict(row.data)
        new = not claim_id
        if new:
            data["claim_id"] = format_id("C", next_claim)
            if not (data.get("fact_id") or "").strip():
                target = (data.get("parallel_of") or "").strip()
                parent = by_id.get(target) or (
                    draft_by_id[target].claim if target in draft_by_id else None
                )
                if parent is not None:
                    data["fact_id"] = parent.fact_id
                else:
                    data["fact_id"] = format_id("F", next_fact)
                    next_fact += 1
            data["origin"] = (data.get("origin") or "").strip() or "human"
        if not (data.get("annotator") or "").strip() and annotator:
            data["annotator"] = annotator
        if decision is Decision.EDIT:
            data["origin"] = _edited_origin(data.get("origin")) or ""
        try:
            claim = row_to_claim(data)
        except (ValueError, ValidationError) as exc:
            msg = exc.errors()[0] if isinstance(exc, ValidationError) else None
            text = f"{'.'.join(str(p) for p in msg['loc'])}: {msg['msg']}" if msg else str(exc)
            problems.append((row.line, text))
            continue
        if new:
            next_claim += 1
            counts["added"] += 1
            assigned[row.line] = (claim.claim_id, claim.fact_id, claim.origin or "")
        else:
            counts["edited" if decision is Decision.EDIT else "accepted"] += 1
        by_id[claim.claim_id] = claim
        if claim.claim_id in draft_by_id:
            draft_by_id[claim.claim_id] = draft_by_id[claim.claim_id].model_copy(
                update={"status": DraftStatus.ACCEPTED}
            )
    return ReviewResult(
        list(by_id.values()), list(draft_by_id.values()), problems, counts, assigned
    )


def fill_assigned_ids(path: Path, assigned: dict[int, tuple[str, str, str]]) -> None:
    """Write the ids `merge_review` gave to new rows back into the reviewed CSV, keeping the
    sheet's own columns and order."""
    if not assigned:
        return
    import io

    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = [h.strip() for h in (reader.fieldnames or [])]
        reader.fieldnames = fields
        rows = [(n, raw) for n, raw in enumerate(reader, start=2)]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for n, raw in rows:
        if n in assigned:
            raw["claim_id"], raw["fact_id"], origin = assigned[n]
            if "origin" in raw:
                raw["origin"] = origin
        writer.writerow({k: (raw.get(k) or "") for k in fields})
    _atomic_write(path, buf.getvalue(), encoding="utf-8-sig")


def write_table_csv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    """A plain CSV with the given columns (proposal view, kappa sample)."""
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in columns})
    _atomic_write(path, buf.getvalue(), encoding="utf-8-sig")
