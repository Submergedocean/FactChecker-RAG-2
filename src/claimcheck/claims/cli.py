"""`claimcheck claims` commands: draft, review through CSV, validate, split, kappa.

Workflow:

    claims draft --n 20              # LLM drafts -> drafts.jsonl (spends LLM quota)
    claims export-csv review.csv     # pending drafts -> spreadsheet
    claims import-csv review.csv     # accept / edit / reject -> claims.jsonl
    claims validate [--pilot|--final]
    claims split                     # splits.json, frozen after the first run
    claims kappa-sample k.csv / claims kappa-import k.csv / claims kappa
    claims export out.csv --proposal-view
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from claimcheck.config import get_settings

app = typer.Typer(
    help="Claims dataset tooling: validate, draft, split, kappa.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Claims dataset tooling: validate, draft, split, kappa."""


def _paths() -> dict[str, Path]:
    d = get_settings().paths.claims_dir
    return {
        "claims": d / "claims.jsonl",
        "drafts": d / "drafts.jsonl",
        "splits": d / "splits.json",
    }


def _corpus_ready() -> bool:
    p = get_settings().paths
    return p.pages.exists() and p.tables.exists() and p.manifest.exists()


def _load_corpus() -> Any:
    from claimcheck.claims.validate import Corpus

    return Corpus.load(get_settings())


def _manifest() -> dict[str, Any]:
    import json

    from claimcheck.schemas import Manifest

    path = get_settings().paths.manifest
    m = Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return {d.doc_id: d for d in m.documents}


def _fail(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


# --------------------------------------------------------------------------- validate


@app.command()
def validate(
    path: Annotated[Path | None, typer.Option(help="Claims file (default data/claims).")] = None,
    corpus: Annotated[bool, typer.Option(help="Check evidence against the corpus.")] = True,
    pilot: Annotated[bool, typer.Option("--pilot", help="Pilot gate: 40+ claims.")] = False,
    final: Annotated[bool, typer.Option("--final", help="Final gate: size and mixes.")] = False,
    show: Annotated[int, typer.Option(help="Issues to print.")] = 50,
) -> None:
    """Check claims.jsonl (schema, evidence, translations, duplicates) and print its mix."""
    from claimcheck.claims.io import read_splits
    from claimcheck.claims.validate import DatasetStats, validate_file

    settings = get_settings()
    paths = _paths()
    path = path or paths["claims"]
    if not path.exists():
        _fail(f"{path} not found")
    src = None
    if corpus:
        if _corpus_ready():
            src = _load_corpus()
        else:
            typer.secho("corpus not found: evidence not checked against the PDFs", fg="yellow")
    gate = "final" if final else "pilot" if pilot else None
    report = validate_file(
        path,
        corpus=src,
        limits=settings.limits,
        settings=settings.claims,
        gate=gate,
        splits=read_splits(paths["splits"]),
    )
    for line in DatasetStats.of(report.claims).lines():
        typer.echo(line)
    for issue in report.issues[:show]:
        color = typer.colors.RED if issue.severity == "error" else typer.colors.YELLOW
        typer.secho(str(issue), fg=color)
    if len(report.issues) > show:
        typer.echo(f"... {len(report.issues) - show} more")
    typer.echo(
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
        + (f"; {gate} gate" if gate else "")
        + ("" if src is not None else "; corpus not checked")
    )
    if not report.ok:
        raise typer.Exit(1)


# --------------------------------------------------------------------------- draft


@app.command()
def draft(
    n: Annotated[int, typer.Option(help="Sources to draft from (each gives 1-4 claims).")] = 10,
    kind: Annotated[str, typer.Option(help="mixed | table | text")] = "mixed",
    seed: Annotated[int | None, typer.Option(help="Sampling seed (default from config).")] = None,
    dry_run: Annotated[bool, typer.Option(help="Show the sampled sources; no LLM call.")] = False,
) -> None:
    """Draft SUPPORTED claims, REFUTED twins and translations with the LLM (spends quota)."""
    from claimcheck.claims.draft import (
        evidence_chunks,
        read_skips,
        run_draft,
        sample_sources,
        skips_path_for,
    )
    from claimcheck.claims.io import read_drafts
    from claimcheck.ingest.chunk import read_chunks

    settings = get_settings()
    paths = _paths()
    if not _corpus_ready():
        _fail("drafting needs the corpus (data/processed)")
    chunks = read_chunks(settings.paths.chunks("table_aware"))
    corpus = _load_corpus()
    known_tables = set(corpus.tables)
    if dry_run:
        drafts = read_drafts(paths["drafts"]) if paths["drafts"].exists() else []
        used = (
            {d.source_chunk_id for d in drafts}
            | set(read_skips(skips_path_for(paths["drafts"])))
            | evidence_chunks(drafts, corpus)
        )
        sources = sample_sources(
            chunks,
            n,
            seed=settings.claims.draft_seed if seed is None else seed,
            used=used,
            kind=kind,
            min_text_words=settings.claims.min_text_words,
            known_tables=known_tables,
        )
        for s in sources:
            c = s.chunk
            focus = f"  row: {s.focus_row}" if s.focus_row else ""
            typer.echo(f"{c.chunk_id} {c.kind.value:5} {c.doc_id} p{c.page_start}{focus}")
        typer.echo(f"{len(sources)} source(s); no LLM call")
        return

    from claimcheck.llm.client import LLMClient

    llm = LLMClient.from_settings(settings)
    run = run_draft(
        n,
        llm,
        corpus,
        chunks,
        _manifest(),
        claims_path=paths["claims"],
        drafts_path=paths["drafts"],
        settings=settings.claims,
        latest_edition=settings.latest_edition,
        kind=kind,
        seed=seed,
        known_tables=known_tables,
        progress=typer.echo,
    )
    flagged = sum(bool(r.checks) for r in run.records)
    typer.echo(
        f"drafted {len(run.records)} claim(s) from {n - len(run.skipped)} source(s) "
        f"({len(run.skipped)} skipped, {flagged} with checks to review), "
        f"{run.llm_requests} LLM request(s)"
    )
    if run.stopped:
        _fail(f"stopped early: {run.stopped}")


# --------------------------------------------------------------------------- CSV review


@app.command("export-csv")
def export_csv(
    out: Annotated[Path, typer.Argument(help="CSV to write.")],
    source: Annotated[str, typer.Option("--from", help="drafts | claims")] = "drafts",
    status: Annotated[str, typer.Option(help="Drafts with this status, or 'all'.")] = "pending",
) -> None:
    """Write drafts (or claims) to a review CSV for accept / edit / reject."""
    from claimcheck.claims.io import claim_to_row, read_claims, read_drafts, write_csv

    settings = get_settings()
    paths = _paths()
    manifest = _manifest() if settings.paths.manifest.exists() else {}

    def link(claim: Any) -> str:
        if not claim.evidence:
            return ""
        ev = claim.evidence[0]
        entry = manifest.get(ev.doc_id)
        return f"{entry.source_url}#page={ev.page}" if entry else ""

    rows = []
    if source == "claims":
        for c in read_claims(paths["claims"]):
            rows.append(claim_to_row(c, pdf_link=link(c)))
    elif source == "drafts":
        drafts = read_drafts(paths["drafts"])
        if status != "all":
            drafts = [d for d in drafts if d.status.value == status]
        texts: dict[str, str] = {}
        chunk_path = settings.paths.chunks("table_aware")
        if chunk_path.exists():
            import pyarrow.parquet as pq

            wanted = {d.source_chunk_id for d in drafts}
            for r in pq.read_table(chunk_path, columns=["chunk_id", "text"]).to_pylist():
                if r["chunk_id"] in wanted:
                    texts[r["chunk_id"]] = r["text"]
        for d in drafts:
            rows.append(
                claim_to_row(
                    d.claim,
                    status=d.status.value,
                    mutation=d.mutation.value if d.mutation else "",
                    checks="; ".join(d.checks),
                    pdf_link=link(d.claim),
                    context=texts.get(d.source_chunk_id, "")[:30000],
                )
            )
    else:
        _fail("--from must be drafts or claims")
    write_csv(out, rows)
    typer.echo(f"wrote {len(rows)} row(s) to {out}")


@app.command("import-csv")
def import_csv(
    csv_path: Annotated[Path, typer.Argument(help="Reviewed CSV.")],
    annotator: Annotated[str | None, typer.Option(help="Fills empty `annotator` cells.")] = None,
    corpus: Annotated[bool, typer.Option(help="Check evidence against the corpus.")] = True,
    dry_run: Annotated[bool, typer.Option(help="Check only; write nothing.")] = False,
) -> None:
    """Apply decisions from a reviewed CSV to claims.jsonl. Nothing is written unless every
    row parses and the resulting claims file validates."""
    from claimcheck.claims.io import (
        fill_assigned_ids,
        merge_review,
        read_claims,
        read_csv,
        read_drafts,
        write_claims,
        write_drafts,
    )
    from claimcheck.claims.validate import check_claims

    settings = get_settings()
    paths = _paths()
    rows = read_csv(csv_path)
    claims = read_claims(paths["claims"]) if paths["claims"].exists() else []
    drafts = read_drafts(paths["drafts"]) if paths["drafts"].exists() else []
    result = merge_review(rows, claims, drafts, annotator=annotator)
    for line, msg in result.problems:
        typer.secho(f"row {line}: {msg}", fg=typer.colors.RED)
    src = _load_corpus() if corpus and _corpus_ready() else None
    issues = check_claims(result.claims, corpus=src, limits=settings.limits)
    errors = [i for i in issues if i.severity == "error"]
    for issue in issues[:50]:
        color = typer.colors.RED if issue.severity == "error" else typer.colors.YELLOW
        typer.secho(str(issue), fg=color)
    typer.echo(", ".join(f"{k} {v}" for k, v in result.counts.items()))
    if result.problems or errors:
        _fail(
            f"nothing written: {len(result.problems)} bad row(s), {len(errors)} error(s) "
            "in the resulting claims; fix the CSV and import again"
        )
    if dry_run:
        typer.echo("dry run: nothing written")
        return
    write_claims(paths["claims"], result.claims)
    if drafts:
        write_drafts(paths["drafts"], result.drafts)
    fill_assigned_ids(csv_path, result.assigned)
    typer.echo(f"claims.jsonl now has {len(result.claims)} claim(s)")
    if result.assigned:
        typer.echo(f"wrote {len(result.assigned)} new claim id(s) back into {csv_path}")


@app.command()
def export(
    out: Annotated[Path, typer.Argument(help="CSV to write.")],
    proposal_view: Annotated[
        bool, typer.Option("--proposal-view", help="The proposal's 8 columns (7.2).")
    ] = False,
) -> None:
    """Export claims.jsonl as CSV: the full review layout, or the proposal's exact columns
    (`evidence_ids` are table_aware chunk ids, resolved from the stored evidence)."""
    from claimcheck.claims.io import claim_to_row, read_claims, write_csv, write_table_csv
    from claimcheck.claims.schema import PROPOSAL_COLUMNS, PROPOSAL_LABELS
    from claimcheck.claims.validate import resolve_evidence

    paths = _paths()
    claims = read_claims(paths["claims"])
    if not proposal_view:
        write_csv(out, [claim_to_row(c) for c in claims])
        typer.echo(f"wrote {len(claims)} claim(s) to {out}")
        return
    if not _corpus_ready():
        _fail("--proposal-view needs the corpus to resolve evidence_ids")
    corpus = _load_corpus()
    manifest = _manifest()
    rows = []
    unresolved = 0
    for c in claims:
        ids: list[str] = []
        sources: list[str] = []
        pages: list[str] = []
        for ev in c.evidence:
            found = resolve_evidence(ev, corpus, "table_aware")
            unresolved += not found
            ids += [i for i in found if i not in ids]
            entry = manifest.get(ev.doc_id)
            src = f"PES {entry.edition} - {entry.title}" if entry else ev.doc_id
            if src not in sources:
                sources.append(src)
            if str(ev.page) not in pages:
                pages.append(str(ev.page))
        rows.append(
            {
                "claim_id": c.claim_id,
                "claim_text": c.claim_text,
                "label": PROPOSAL_LABELS[c.label],
                "evidence_ids": ";".join(ids),
                "source": "; ".join(sources),
                "page": ";".join(pages),
                "topic": c.topic,
                "claim_type": c.reasoning_type.value,
            }
        )
    write_table_csv(out, PROPOSAL_COLUMNS, rows)
    typer.echo(f"wrote {len(rows)} claim(s) to {out} ({unresolved} evidence item(s) unresolved)")


# --------------------------------------------------------------------------- split


@app.command()
def split(
    force: Annotated[bool, typer.Option(help="Regenerate instead of extending.")] = False,
) -> None:
    """Assign every fact to train / val / test (70/15/15, stratified, frozen)."""
    from claimcheck.claims.io import read_claims, read_splits, write_splits
    from claimcheck.claims.split import leaked_facts, make_splits, split_summary

    settings = get_settings()
    paths = _paths()
    claims = read_claims(paths["claims"])
    existing = {} if force else read_splits(paths["splits"])
    splits = make_splits(
        claims, settings.claims.split_ratios, seed=settings.claims.split_seed, existing=existing
    )
    leaked = leaked_facts(claims, splits)
    if leaked:
        _fail(f"facts in two splits: {', '.join(leaked[:10])}")
    write_splits(paths["splits"], splits)
    added = len(splits) - len(existing)
    for line in split_summary(claims, splits):
        typer.echo(line)
    typer.echo(f"{len(splits)} fact(s) in splits.json ({added} new)")


# --------------------------------------------------------------------------- kappa


@app.command()
def kappa(
    minimum: Annotated[float | None, typer.Option("--min", help="Fail below this.")] = None,
) -> None:
    """Cohen's kappa between `label` and `second_label`."""
    from claimcheck.claims.io import read_claims
    from claimcheck.claims.kappa import kappa_of

    settings = get_settings()
    claims = read_claims(_paths()["claims"])
    try:
        result = kappa_of(claims)
    except ValueError:
        _fail("no claim has a second_label yet (claims kappa-sample, then kappa-import)")
        return
    for line in result.lines():
        typer.echo(line)
    floor = settings.claims.kappa_min if minimum is None else minimum
    if result.n < settings.claims.kappa_sample:
        typer.secho(
            f"{result.n} pairs; the plan asks for about {settings.claims.kappa_sample}", fg="yellow"
        )
    if result.kappa < floor:
        _fail(f"kappa {result.kappa:.3f} is below {floor:.2f}")
    typer.echo(f"kappa {result.kappa:.3f} >= {floor:.2f}")


@app.command("kappa-sample")
def kappa_sample_cmd(
    out: Annotated[Path, typer.Argument(help="Blind CSV for the second annotator.")],
    n: Annotated[int | None, typer.Option(help="Claims (default from config).")] = None,
) -> None:
    """Write a blind sample (no labels, no evidence) for the second annotator."""
    from claimcheck.claims.io import read_claims, write_table_csv
    from claimcheck.claims.kappa import SAMPLE_COLUMNS, kappa_sample

    settings = get_settings()
    claims = read_claims(_paths()["claims"])
    picked = kappa_sample(
        claims, n or settings.claims.kappa_sample, seed=settings.claims.split_seed
    )
    rows = [
        {"claim_id": c.claim_id, "claim_text": c.claim_text, "language": c.language.value,
         "topic": c.topic}
        for c in picked
    ]  # fmt: skip
    write_table_csv(out, SAMPLE_COLUMNS, rows)
    typer.echo(f"wrote {len(rows)} claim(s) to {out}")


@app.command("kappa-import")
def kappa_import(
    csv_path: Annotated[Path, typer.Argument(help="The second annotator's CSV.")],
) -> None:
    """Store the second annotator's labels in `second_label`."""
    from claimcheck.claims.io import read_claims, read_csv, write_claims
    from claimcheck.schemas import Verdict

    paths = _paths()
    claims = {c.claim_id: c for c in read_claims(paths["claims"])}
    rows = read_csv(csv_path, required=("claim_id", "second_label"))
    problems, updated = [], 0
    for row in rows:
        cid = row.data.get("claim_id", "").strip()
        raw = row.data.get("second_label", "").strip().upper()
        if not raw:
            continue
        if cid not in claims:
            problems.append(f"row {row.line}: unknown claim {cid!r}")
            continue
        try:
            label = Verdict(raw)
        except ValueError:
            problems.append(f"row {row.line}: {raw!r} is not SUPPORTED, REFUTED or NEI")
            continue
        note = row.data.get("second_annotator", "").strip()
        claim = claims[cid]
        notes = claim.notes
        if note:
            tag = f"second annotator: {note}"
            notes = tag if not notes else (notes if tag in notes else f"{notes}; {tag}")
        claims[cid] = claim.model_copy(update={"second_label": label, "notes": notes})
        updated += 1
    for p in problems:
        typer.secho(p, fg=typer.colors.RED)
    if problems:
        _fail("nothing written")
    write_claims(paths["claims"], claims.values())
    typer.echo(f"stored {updated} second label(s)")
