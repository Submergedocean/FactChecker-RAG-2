"""`claimcheck claims draft`: LLM-proposed claims for human review.

1. Sample sources from the `table_aware` chunks: table chunks of parsed tables and text chunks
   with at least one number, round-robin over documents and alternating table / text (so
   about half the facts are tabular), in a seeded order. Chunks already drafted are skipped,
   so repeated runs continue where the last one stopped.
2. Ask the LLM (verifier model, temperature 0, `claim-draft-v1`) for a SUPPORTED claim, a
   REFUTED twin and translations, with evidence.
3. Check everything in code; the model is never trusted to cite:
   - a text quote must be verbatim in the chunk and on one page (that page is the evidence
     page); a table cell is looked up by row and column in `tables.parquet` and must equal
     the quote; document, page and table id come from metadata, never from the model;
   - SUPPORTED: the claim's numbers should appear in the evidence (`number_not_in_evidence`);
   - REFUTED: the text must differ from the original, and a number mutation must not match
     the evidence at the claim's precision;
   - every evidence item is checked against the page text and must resolve in both variants
     (`claims.validate`).
   A draft whose SUPPORTED evidence fails is dropped; one whose REFUTED evidence fails keeps
   only the SUPPORTED claim. Remaining warnings go to `checks` for the reviewer.
4. A seeded share of facts (`claims.translate_share`) also gets one Urdu and one Roman Urdu
   claim (Urdu of one twin, Roman Urdu of the other, alternating), each `parallel_of` its
   English original and sharing its label, fact and evidence.

Drafts are appended to `data/claims/drafts.jsonl` after every source, so an interrupted run
(quota, network) keeps what it drafted.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from claimcheck.claims.io import read_claims, read_drafts, write_drafts
from claimcheck.claims.schema import (
    CLAIM_ID_RE,
    FACT_ID_RE,
    TOPICS,
    Claim,
    ClaimLanguage,
    ClaimType,
    DraftRecord,
    EvidenceRef,
    Mutation,
    ReasoningType,
    Verdict,
    dedup_key,
    format_id,
    next_number,
    topic_for,
)
from claimcheck.claims.validate import (
    SourceCorpus,
    cell_equals,
    check_evidence_source,
    quote_in_text,
    resolve_evidence,
)
from claimcheck.config import ClaimsSettings
from claimcheck.llm.client import LLMError
from claimcheck.llm.prompts.claim_draft import (
    PROMPT_VERSION,
    ClaimDraftOutput,
    DraftEvidence,
    build_messages,
)
from claimcheck.pipeline.extractor import JSONCompleter
from claimcheck.pipeline.guards import _same_number, match_key
from claimcheck.schemas import ChunkKind, ChunkRecord, ManifestEntry
from claimcheck.textnorm import (
    Quantity,
    ascii_digits,
    ascii_punct,
    clean_unicode,
    find_fiscal_years,
    find_month_ranges,
    find_quantities,
)

log = logging.getLogger(__name__)

KINDS = ("mixed", "table", "text")


# --------------------------------------------------------------------------- sampling


@dataclass(frozen=True)
class DraftSource:
    chunk: ChunkRecord
    focus_row: str | None = None


def _has_number(text: str) -> bool:
    return bool(find_quantities(text))


_ROW_RE = re.compile(r"^\|(.+)\|\s*$")


def table_rows(text: str) -> list[str]:
    """Labels of the body rows of a table chunk that hold at least one number."""
    labels = []
    lines = text.split("\n")
    body = False
    for line in lines:
        m = _ROW_RE.match(line.strip())
        if not m:
            continue
        if re.fullmatch(r"[\s|:\-]+", line.strip()):
            body = True  # the header rule
            continue
        if not body:
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if cells and cells[0] and any(_has_number(c) for c in cells[1:]):
            labels.append(cells[0])
    return labels


def sample_sources(
    chunks: Sequence[ChunkRecord],
    n: int,
    *,
    seed: int,
    used: set[str],
    kind: str = "mixed",
    min_text_words: int = 40,
    known_tables: set[str] | None = None,
) -> list[DraftSource]:
    """Up to `n` sources, round-robin over documents, alternating table and text for
    `kind="mixed"`. The order depends only on `seed` and the chunks; `used` chunk ids are
    skipped."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    pools: dict[str, dict[str, list[ChunkRecord]]] = {
        "table": defaultdict(list),
        "text": defaultdict(list),
    }
    for c in sorted(chunks, key=lambda c: c.chunk_id):
        if c.chunk_id in used:
            continue
        if c.kind is ChunkKind.TABLE:
            known = c.table_id and (known_tables is None or c.table_id in known_tables)
            if known and table_rows(c.text):
                pools["table"][c.doc_id].append(c)
        elif c.n_words >= min_text_words and _has_number(c.text):
            pools["text"][c.doc_id].append(c)
    rng = random.Random(seed)
    queues: dict[str, list[list[ChunkRecord]]] = {}
    for k, by_doc in pools.items():
        docs = sorted(by_doc)
        rng.shuffle(docs)
        for d in docs:
            rng.shuffle(by_doc[d])
        queues[k] = [by_doc[d] for d in docs]
    order = ["table", "text"] if kind == "mixed" else [kind]
    cursors = {k: 0 for k in order}
    out: list[DraftSource] = []
    turn = 0
    while len(out) < n and any(any(q) for k in order for q in queues[k]):
        k = order[turn % len(order)]
        turn += 1
        qs = queues[k]
        if not any(qs):
            continue
        while not qs[cursors[k] % len(qs)]:
            cursors[k] += 1
        chunk = qs[cursors[k] % len(qs)].pop(0)
        cursors[k] += 1
        focus = None
        if chunk.kind is ChunkKind.TABLE:
            rows = table_rows(chunk.text)
            focus = random.Random(f"{seed}:{chunk.chunk_id}").choice(rows) if rows else None
        out.append(DraftSource(chunk, focus))
    return out


# --------------------------------------------------------------------------- checks


class EvidenceProblem(Exception):
    pass


def build_evidence(chunk: ChunkRecord, ev: DraftEvidence, corpus: SourceCorpus) -> EvidenceRef:
    """The model's evidence turned into an `EvidenceRef` from metadata, or `EvidenceProblem`."""
    if chunk.kind is ChunkKind.TABLE and chunk.table_id:
        table = corpus.table(chunk.table_id)
        if table is None:
            raise EvidenceProblem(f"table {chunk.table_id} not parsed")
        if not ev.row_label or not ev.col_label:
            raise EvidenceProblem("table evidence without row_label / col_label")
        found = table.locate(ev.row_label, ev.col_label)
        if isinstance(found, str):
            raise EvidenceProblem(found)
        i, k = found
        cell = table.cells[i][k] if k < len(table.cells[i]) else ""
        if not cell_equals(cell, ev.quote):
            raise EvidenceProblem(f"cell is {cell!r}, model quoted {ev.quote!r}")
        return EvidenceRef(
            doc_id=table.doc_id,
            page=table.page,
            table_id=table.table_id,
            quote=cell,
            row_label=table.labels[i],
            col_label=table.header[k],
        )
    quote = ev.quote.strip().strip("\"'“”‘’").strip()
    if not quote_in_text(quote, chunk.text):
        raise EvidenceProblem(f"quote not in the passage: {quote[:80]!r}")
    for page in range(chunk.page_start, chunk.page_end + 1):
        if any(quote_in_text(quote, t) for t in corpus.page_texts(chunk.doc_id, page)):
            return EvidenceRef(doc_id=chunk.doc_id, page=page, quote=quote)
    raise EvidenceProblem("quote is not on a single page")


def evidence_checks(ev: EvidenceRef, corpus: SourceCorpus) -> list[str]:
    out = [f"{code}: {msg}" for code, msg in check_evidence_source(ev, corpus)]
    if not out:
        out += [f"unresolved_{v}" for v in corpus.variants() if not resolve_evidence(ev, corpus, v)]
    return out


def clean_text(text: str) -> str:
    """Drafted text as stored: Unicode clean-up, ASCII quotes and dashes (models write
    '2025\u201126' with a non-breaking hyphen), single spaces."""
    return " ".join(ascii_punct(clean_unicode(text)).split())


def _is_year(q: Quantity) -> bool:
    return (
        q.decimals == 0
        and 1900 <= q.number <= 2100
        and not (q.unit or q.currency or q.scale)
        and "," not in q.raw
    )


def claim_numbers(text: str) -> list[Quantity]:
    """The claim's quantities without bare calendar years ('March 2025'), which would match
    any quote about the same period."""
    return [q for q in find_quantities(text) if not _is_year(q)]


def numbers_match(claim_text: str, evidence: EvidenceRef) -> bool | None:
    """True if some number of the claim matches the evidence quote (rule 3 precision), None
    if the claim has no number other than its periods."""
    targets = claim_numbers(claim_text)
    if not targets:
        return None
    sources = find_quantities(evidence.quote)
    return any(_same_number(t, s) for t in targets for s in sources)


def changed_numbers(original: str, mutated: str) -> list[Quantity]:
    """Quantities of `mutated` beyond those of `original`, counted with multiplicity: "age 10+
    ... 3.0 points" -> "age 10+ ... 10.0 points" changed one number although 10 occurs in
    both."""
    left = Counter(abs(q.number) for q in claim_numbers(original))
    out = []
    for q in claim_numbers(mutated):
        if left[abs(q.number)] > 0:
            left[abs(q.number)] -= 1
        else:
            out.append(q)
    return out


# --------------------------------------------------------------------------- drafting


@dataclass
class DraftRun:
    records: list[DraftRecord] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (chunk_id, reason)
    stopped: str | None = None  # why the run stopped early (quota, provider down)
    llm_requests: int = 0


@dataclass
class IdAllocator:
    next_claim: int
    next_fact: int

    @classmethod
    def from_existing(cls, claims: Sequence[Claim], drafts: Sequence[DraftRecord]) -> IdAllocator:
        ids = [c.claim_id for c in claims] + [d.claim.claim_id for d in drafts]
        facts = [c.fact_id for c in claims] + [d.claim.fact_id for d in drafts]
        return cls(next_number(ids, CLAIM_ID_RE), next_number(facts, FACT_ID_RE))

    def claim(self) -> str:
        out = format_id("C", self.next_claim)
        self.next_claim += 1
        return out

    def fact(self) -> str:
        out = format_id("F", self.next_fact)
        self.next_fact += 1
        return out


@dataclass
class TranslationQuota:
    """Translate a fact whenever the translated share of drafted facts is below the target, so
    the language mix holds by construction (a per-fact coin flip drifted far off on small
    batches). Urdu goes to the SUPPORTED claim on every other translated fact."""

    share: float
    facts: int = 0
    translated: int = 0

    @classmethod
    def from_drafts(cls, share: float, drafts: Sequence[DraftRecord]) -> TranslationQuota:
        facts = {d.claim.fact_id for d in drafts}
        done = {d.claim.fact_id for d in drafts if d.claim.language is not ClaimLanguage.EN}
        return cls(share, len(facts), len(done))

    def next(self) -> tuple[bool, bool]:
        """(translate the next fact?, Urdu on its SUPPORTED claim?); counts the fact."""
        self.facts += 1
        if self.translated < self.share * self.facts:
            self.translated += 1
            return True, self.translated % 2 == 1
        return False, False


CONTEXT_WORDS = 120
_HEAD_WORDS = 8
_MONTH_WORD = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_PARTIAL_WORDS_RE = re.compile(
    r"(?i)\b(?:first|1st)\s+(?:nine|six|three|half|\d+)\s*(?:months?|M)\b|\b\d{1,2}M\b"
    r"|\bH[12]\b|\bQ[1-4]\b|\bquarter\b|\bhalf[- ]year\b"
    rf"|\b(?:between|from)\s+{_MONTH_WORD}\s+(?:and|to|till|until)\s+{_MONTH_WORD}\b"
)


def context_before(chunk: ChunkRecord, corpus: SourceCorpus, words: int = CONTEXT_WORDS) -> str:
    """Up to `words` words of the page text just above a text chunk (its section heading and
    lead, which often carry the period), or "" when the chunk's start is not found."""
    if chunk.kind is not ChunkKind.TEXT:
        return ""
    head = match_key(" ".join(chunk.text.split()[:_HEAD_WORDS]))
    if not head:
        return ""
    for text in corpus.page_texts(chunk.doc_id, chunk.page_start):
        flat = " ".join(text.split())
        # find the chunk's first words in the page, comparing on the match key word by word
        page_words = flat.split()
        keys = [match_key(w) for w in page_words]
        target = head.split()
        for i in range(len(keys) - len(target) + 1):
            if " ".join(k for k in keys[i : i + len(target)] if k) == " ".join(target):
                return " ".join(page_words[max(0, i - words) : i])
    return ""


def table_caption(chunk: ChunkRecord) -> str | None:
    if chunk.kind is not ChunkKind.TABLE:
        return None
    first = chunk.text.split("\n", 1)[0].strip()
    return None if first.startswith("|") else first or None


def partial_period_of(texts: Sequence[str]) -> str | None:
    """The first part-year period named in `texts` ('July-March', 'first nine months')."""
    for text in texts:
        for r in find_month_ranges(text):
            if r.partial_year:
                return r.raw
        m = _PARTIAL_WORDS_RE.search(text)
        if m:
            return m.group(0)
    return None


def partial_year_check(claim_text: str, ev: EvidenceRef, extra: Sequence[str]) -> str | None:
    """Flag a claim that reads as full-year when its source figure covers part of a year:
    the column header, the quote or the text above names a part-year period and the
    claim names none."""
    source = [ev.col_label or "", ev.quote, *extra]
    period = partial_period_of(source)
    if period and partial_period_of([claim_text]) is None:
        return (
            f"partial_year_missing: the source figure is for {period!r}; "
            "the claim names no part-year period"
        )
    return None


def translation_checks(original: Claim, text: str, lang: ClaimLanguage) -> list[str]:
    out = ["translation: check meaning and every number"]
    want = sorted({abs(q.number) for q in claim_numbers(original.claim_text)})
    got = sorted({abs(q.number) for q in claim_numbers(text)})
    if want != got:
        out.append(f"translation_numbers_differ: original {want}, translation {got}")
    fy_a = {f.end_year for f in find_fiscal_years(original.claim_text)}
    fy_b = {f.end_year for f in find_fiscal_years(ascii_digits(text))}
    if fy_a and fy_b and fy_a != fy_b:
        out.append("translation_period_differs: the fiscal year differs from the original")
    if lang is ClaimLanguage.UR and "فیڈ" in text:
        out.append("urdu_fy_transliterated: write FY as مالی سال")
    return out


@dataclass
class ReasoningRotation:
    """Asks the drafter for the kind of claim (numeric / trend / comparison) that is furthest
    below its target share among the drafted English facts, so the dataset is not all
    `numeric`. A hint only: the model writes what the passage supports."""

    mix: dict[str, float]
    counts: Counter[str] = field(default_factory=Counter)

    @classmethod
    def from_drafts(cls, mix: dict[str, float], drafts: Sequence[DraftRecord]) -> ReasoningRotation:
        seen: dict[str, str] = {}
        for d in drafts:
            if d.claim.language is ClaimLanguage.EN and d.claim.label is Verdict.SUPPORTED:
                seen[d.claim.fact_id] = d.claim.reasoning_type.value
        return cls(dict(mix), Counter(seen.values()))

    def wanted(self) -> str | None:
        if not self.mix:
            return None
        total = sum(self.counts.values()) + 1
        return max(self.mix, key=lambda k: (self.mix[k] * total - self.counts[k], self.mix[k]))

    def record(self, reasoning: str) -> None:
        self.counts[reasoning] += 1


def draft_source(
    source: DraftSource,
    llm: JSONCompleter,
    corpus: SourceCorpus,
    entry: ManifestEntry,
    ids: IdAllocator,
    settings: ClaimsSettings,
    *,
    latest_edition: str,
    quota: TranslationQuota | None = None,
    rotation: ReasoningRotation | None = None,
) -> tuple[list[DraftRecord], str | None, int]:
    """(records, skip reason, LLM requests) for one source. `quota` decides translations
    (default: a fresh quota at `settings.translate_share`); `rotation` asks for a kind of
    claim (default: none asked)."""
    chunk = source.chunk
    fixed_topic = topic_for(entry)
    page = chunk.page_start
    before = context_before(chunk, corpus)
    caption = table_caption(chunk)
    messages = build_messages(
        passage=chunk.text,
        kind=chunk.kind.value,
        edition=chunk.edition,
        document=f"{entry.title} ({entry.doc_id})",
        page=page,
        topic=fixed_topic,
        allowed_topics=TOPICS,
        focus_row=source.focus_row,
        caption=caption,
        context_before=before or None,
        wanted_reasoning=rotation.wanted() if rotation else None,
    )
    result = llm.complete_json(
        messages, ClaimDraftOutput, role="verifier", prompt_version=PROMPT_VERSION
    )
    out = result.value
    if not out.usable or out.supported is None:
        return [], f"not usable: {out.skip_reason or 'no claim'}", result.requests
    try:
        sup_ev = build_evidence(chunk, out.supported.evidence, corpus)
    except EvidenceProblem as exc:
        return [], f"supported evidence: {exc}", result.requests

    common: list[str] = []
    if chunk.edition != latest_edition:
        common.append(f"older_edition: check whether {latest_edition} revises this figure")
    topic = fixed_topic or (out.topic or "").strip().lower()
    if topic not in TOPICS:
        common.append(f"topic_unknown: model said {out.topic!r}")
    claim_type = ClaimType.TABULAR if sup_ev.table_id else ClaimType.TEXTUAL
    reasoning = out.reasoning_type or ReasoningType.NUMERIC
    origin = f"llm_draft:{result.model}:{PROMPT_VERSION}"
    meta = {
        "source_chunk_id": chunk.chunk_id,
        "source_kind": chunk.kind.value,
        "model": result.model,
        "prompt_version": PROMPT_VERSION,
    }
    fact = ids.fact()

    def make(
        text: str, label: Verdict, ev: EvidenceRef, lang: ClaimLanguage, parallel: str | None
    ) -> Claim:
        return Claim(
            claim_id=ids.claim(),
            fact_id=fact,
            claim_text=clean_text(text),
            label=label,
            language=lang,
            parallel_of=parallel,
            claim_type=claim_type,
            reasoning_type=reasoning,
            topic=topic,
            evidence=[ev],
            origin=origin,
        )

    if rotation is not None:
        rotation.record(reasoning.value if hasattr(reasoning, "value") else str(reasoning))
    period_extra = [caption or "", before[-300:]]
    sup_checks = list(common) + evidence_checks(sup_ev, corpus)
    partial = partial_year_check(clean_text(out.supported.claim), sup_ev, period_extra)
    if partial:
        sup_checks.append(partial)
    if numbers_match(out.supported.claim, sup_ev) is False:
        sup_checks.append("number_not_in_evidence: the claim's number is not in the evidence")
    sup = make(out.supported.claim, Verdict.SUPPORTED, sup_ev, ClaimLanguage.EN, None)
    records = [DraftRecord(claim=sup, checks=sup_checks, **meta)]

    ref_claim: Claim | None = None
    if out.refuted is not None:
        ref_checks = list(common)
        try:
            ref_ev = build_evidence(chunk, out.refuted.evidence, corpus)
        except EvidenceProblem as exc:
            ref_ev = None
            log.info("%s: refuted twin dropped: %s", chunk.chunk_id, exc)
        if ref_ev is not None and dedup_key(out.refuted.claim) != dedup_key(out.supported.claim):
            mutation = out.refuted.mutation
            ref_checks += evidence_checks(ref_ev, corpus)
            sup_text, ref_text = clean_text(out.supported.claim), clean_text(out.refuted.claim)
            if mutation is Mutation.NUMBER:
                changed = changed_numbers(sup_text, ref_text)
                sources = find_quantities(ref_ev.quote)
                if not changed:
                    ref_checks.append("number_unchanged: no number differs from the original")
                elif any(_same_number(q, s) for q in changed for s in sources):
                    ref_checks.append(
                        "refuted_number_matches: the changed number matches the evidence"
                    )
            if mutation is Mutation.YEAR:
                a = {f.end_year for f in find_fiscal_years(sup_text)}
                b = {f.end_year for f in find_fiscal_years(ref_text)}
                if a == b:
                    ref_checks.append("year_unchanged: the fiscal year is the same as the original")
            ref_claim = make(out.refuted.claim, Verdict.REFUTED, ref_ev, ClaimLanguage.EN, None)
            records.append(
                DraftRecord(claim=ref_claim, mutation=mutation, checks=ref_checks, **meta)
            )

    quota = quota or TranslationQuota(settings.translate_share)
    tr = out.translations
    translate, urdu_on_supported = quota.next() if tr is not None else (False, False)
    if translate and tr is not None:
        plan: list[tuple[Claim, ClaimLanguage]] = []
        if ref_claim is None:
            plan.append((sup, ClaimLanguage.UR if urdu_on_supported else ClaimLanguage.ROMAN_UR))
        else:
            first, second = (sup, ref_claim) if urdu_on_supported else (ref_claim, sup)
            plan += [(first, ClaimLanguage.UR), (second, ClaimLanguage.ROMAN_UR)]
        for original, lang in plan:
            t = tr.supported if original.label is Verdict.SUPPORTED else tr.refuted
            if t is None:
                continue
            text = t.ur if lang is ClaimLanguage.UR else t.roman_ur
            twin = make(text, original.label, original.evidence[0], lang, original.claim_id)
            records.append(
                DraftRecord(
                    claim=twin,
                    mutation=next((r.mutation for r in records if r.claim is original), None),
                    checks=list(common) + translation_checks(original, twin.claim_text, lang),
                    **meta,
                )
            )
    return records, None, result.requests


def evidence_chunks(
    drafts: Sequence[DraftRecord], corpus: SourceCorpus, variant: str = "table_aware"
) -> set[str]:
    """Chunks that already hold a drafted claim's evidence. Treated as used, so a rebuilt
    index (new chunk ids) or overlapping text windows never yield the same fact twice."""
    out: set[str] = set()
    seen: set[tuple[str, int, str | None, str]] = set()
    for d in drafts:
        for ev in d.claim.evidence:
            key = (ev.doc_id, ev.page, ev.table_id, ev.quote)
            if key in seen:
                continue
            seen.add(key)
            out.update(resolve_evidence(ev, corpus, variant))
    return out


def skips_path_for(drafts_path: Path) -> Path:
    """Sources that gave no usable draft: `{chunk_id: reason}` next to the drafts file."""
    return drafts_path.with_name("draft_skips.json")


def read_skips(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_skips(path: Path, skips: dict[str, str]) -> None:
    path.write_text(json.dumps(dict(sorted(skips.items())), indent=1) + "\n", encoding="utf-8")


def run_draft(
    n: int,
    llm: JSONCompleter,
    corpus: SourceCorpus,
    chunks: Sequence[ChunkRecord],
    manifest: dict[str, ManifestEntry],
    *,
    claims_path: Path,
    drafts_path: Path,
    settings: ClaimsSettings,
    latest_edition: str,
    kind: str = "mixed",
    seed: int | None = None,
    known_tables: set[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> DraftRun:
    """Draft from up to `n` new sources; drafts are saved after each source."""
    claims = read_claims(claims_path) if claims_path.exists() else []
    drafts = read_drafts(drafts_path) if drafts_path.exists() else []
    skips_path = skips_path_for(drafts_path)
    skips = read_skips(skips_path)
    used = {d.source_chunk_id for d in drafts} | set(skips) | evidence_chunks(drafts, corpus)
    ids = IdAllocator.from_existing(claims, drafts)
    quota = TranslationQuota.from_drafts(settings.translate_share, drafts)
    rotation = ReasoningRotation.from_drafts(settings.reasoning_mix, drafts)
    sources = sample_sources(
        chunks,
        n,
        seed=settings.draft_seed if seed is None else seed,
        used=used,
        kind=kind,
        min_text_words=settings.min_text_words,
        known_tables=known_tables,
    )
    run = DraftRun()
    for i, src in enumerate(sources, start=1):
        entry = manifest[src.chunk.doc_id]
        try:
            records, reason, requests = draft_source(
                src,
                llm,
                corpus,
                entry,
                ids,
                settings,
                latest_edition=latest_edition,
                quota=quota,
                rotation=rotation,
            )
        except LLMError as exc:
            run.stopped = f"{type(exc).__name__}: {exc}"
            break
        run.llm_requests += requests
        if reason:
            run.skipped.append((src.chunk.chunk_id, reason))
            # Remember the source so the next run moves on to new ones.
            skips[src.chunk.chunk_id] = reason
            write_skips(skips_path, skips)
        else:
            run.records.extend(records)
            drafts.extend(records)
            write_drafts(drafts_path, drafts)
        if progress:
            progress(
                f"[{i}/{len(sources)}] {src.chunk.chunk_id} {src.chunk.kind.value:5} "
                + (f"skipped: {reason}" if reason else f"{len(records)} claim(s)")
            )
    return run
