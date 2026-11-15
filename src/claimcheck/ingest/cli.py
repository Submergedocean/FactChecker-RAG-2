"""`claimcheck ingest` commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    help="Corpus download, extraction, cleaning, tables and chunking.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Corpus download, extraction, cleaning, tables and chunking."""


@app.command()
def download(
    force: Annotated[
        bool, typer.Option("--force", help="Re-download even if the checksum matches.")
    ] = False,
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Only the first N documents.")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Download the PES PDFs into data/raw/ and write data/manifest.json (idempotent)."""
    from claimcheck.config import get_settings
    from claimcheck.ingest.download import CATALOGUE, download_corpus

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = get_settings()
    catalogue = CATALOGUE[:limit] if limit else CATALOGUE

    def progress(doc_id: str, status: str) -> None:
        typer.echo(f"  {status:<10} {doc_id}")

    report = download_corpus(
        settings.paths.raw_dir,
        settings.paths.manifest,
        settings.ingest,
        catalogue=catalogue,
        force=force,
        on_progress=progress,
    )
    typer.echo(
        f"{report.n_ok} ok | {len(report.downloaded)} downloaded | "
        f"{len(report.skipped)} skipped | {len(report.failed)} failed"
    )
    for doc_id, reason in report.failed.items():
        typer.echo(f"FAILED {doc_id}: {reason}", err=True)
    if report.failed:
        raise typer.Exit(code=1)


@app.command("cut")
def cut(
    complete_pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    doc_id: Annotated[str, typer.Argument(help="Catalogue doc_id whose link is dead.")],
    first_page: Annotated[int, typer.Argument(min=1)],
    last_page: Annotated[int, typer.Argument(min=1)],
) -> None:
    """Fallback for a dead link: cut a document out of a complete PDF by (1-based) page range.

    Writes data/raw/<doc_id>.pdf; run `download` afterwards to record its checksum.
    """
    from claimcheck.config import get_settings
    from claimcheck.ingest.download import CATALOGUE, cut_from_complete

    by_id = {c.doc_id: c for c in CATALOGUE}
    if doc_id not in by_id:
        raise typer.BadParameter(f"unknown doc_id {doc_id!r}", param_hint="doc_id")
    dest = get_settings().paths.raw_dir / by_id[doc_id].filename
    n = cut_from_complete(complete_pdf, dest, first_page, last_page)
    typer.echo(f"wrote {dest} ({n} pages)")


@app.command()
def extract(
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Only the first N documents.")
    ] = None,
    doc: Annotated[
        list[str] | None, typer.Option("--doc", help="Only these doc_ids (repeatable).")
    ] = None,
    workers: Annotated[
        int | None, typer.Option("--workers", min=1, help="Processes (default: CPU cores).")
    ] = None,
) -> None:
    """Extract and clean every page into data/processed/pages.parquet (plus a run report)."""
    from claimcheck.config import get_settings
    from claimcheck.ingest.extract import DocResult, run_extract
    from claimcheck.schemas import Manifest

    settings = get_settings()
    paths, ing = settings.paths, settings.ingest
    manifest = Manifest.model_validate_json(paths.manifest.read_text(encoding="utf-8"))
    docs = manifest.documents
    if doc:
        unknown = set(doc) - {d.doc_id for d in docs}
        if unknown:
            raise typer.BadParameter(f"unknown doc_id(s): {', '.join(sorted(unknown))}")
        docs = [d for d in docs if d.doc_id in doc]
    if limit:
        docs = docs[:limit]
    missing = [d.doc_id for d in docs if not (paths.raw_dir / d.filename).exists()]
    if missing:
        typer.echo(f"missing PDFs (run `claimcheck ingest download`): {missing}", err=True)
        raise typer.Exit(code=1)

    def progress(res: DocResult) -> None:
        r = res.report
        typer.echo(
            f"  {res.doc_id:<26} {r['n_pages']:>3} pages  {r['n_tables']:>3} tables  "
            f"{r['n_figures']:>3} figures  {len(r['word_count_mismatch']):>2} mismatched  "
            f"{len(r['no_text_layer']):>2} no text"
        )

    report = run_extract(
        Manifest(documents=docs),
        paths.raw_dir,
        paths.pages,
        paths.extract_report,
        share=ing.header_footer_share,
        band=ing.edge_band,
        tolerance=ing.word_count_tolerance,
        workers=workers or ing.workers,
        on_doc=progress,
    )
    n, expected = report["n_pages"], report["expected_pages"]
    typer.echo(
        f"{n} pages from {report['n_documents']} documents "
        f"(manifest n_pages sum {expected}) -> {paths.pages}"
    )
    typer.echo(
        f"no_text_layer: {len(report['no_text_layer'])} pages | "
        f"word count outside {ing.word_count_tolerance:.0%}: "
        f"{len(report['word_count_mismatch'])} pages | report: {paths.extract_report}"
    )
    if n != expected:
        typer.echo("page count does not match the manifest", err=True)
        raise typer.Exit(code=1)


@app.command()
def show(
    doc_id: Annotated[str, typer.Argument(help="e.g. pes2526_ch07")],
    page: Annotated[int, typer.Argument(min=1, help="1-based PDF page.")],
    raw: Annotated[bool, typer.Option("--raw", help="Also print text_raw.")] = False,
) -> None:
    """Print one page of pages.parquet: flags, printed page, regions and text_body."""
    import json

    import pyarrow.parquet as pq

    from claimcheck.config import get_settings

    paths = get_settings().paths
    if not paths.pages.exists():
        typer.echo("pages.parquet not found; run `claimcheck ingest extract`", err=True)
        raise typer.Exit(code=1)
    rows = pq.read_table(
        paths.pages, filters=[("doc_id", "=", doc_id), ("page", "=", page)]
    ).to_pylist()
    if not rows:
        typer.echo(f"no page {page} for {doc_id}", err=True)
        raise typer.Exit(code=1)
    row = rows[0]
    regions = []
    if paths.extract_report.exists():
        rep = json.loads(paths.extract_report.read_text(encoding="utf-8"))
        regions = rep["documents"].get(doc_id, {}).get("regions", {}).get(str(page), [])
    typer.echo(
        f"{doc_id} page {page} (printed {row['printed_page'] or '-'}) "
        f"flags={row['flags']} words body={len(row['text_body'].split())} "
        f"raw={len(row['text_raw'].split())}"
    )
    for r in regions:
        typer.echo(f"  {r['kind']:<6} bbox={r['bbox']} rows={r['n_rows']} {r['caption'] or ''}")
    typer.echo("-" * 80)
    typer.echo(row["text_body"])
    if raw:
        typer.echo("-" * 80 + " text_raw")
        typer.echo(row["text_raw"])


@app.command()
def tables(
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Only the first N documents.")
    ] = None,
    doc: Annotated[
        list[str] | None, typer.Option("--doc", help="Only these doc_ids (repeatable).")
    ] = None,
    workers: Annotated[
        int | None, typer.Option("--workers", min=1, help="Processes (default: CPU cores).")
    ] = None,
) -> None:
    """Extract every table region of pages.parquet's run into data/processed/tables.parquet."""
    from claimcheck.config import get_settings
    from claimcheck.ingest.tables import DocTables, run_tables
    from claimcheck.schemas import Manifest

    settings = get_settings()
    paths, ing = settings.paths, settings.ingest
    if not paths.extract_report.exists():
        typer.echo("extract_report.json not found; run `claimcheck ingest extract`", err=True)
        raise typer.Exit(code=1)
    manifest = Manifest.model_validate_json(paths.manifest.read_text(encoding="utf-8"))
    docs = manifest.documents
    if doc:
        unknown = set(doc) - {d.doc_id for d in docs}
        if unknown:
            raise typer.BadParameter(f"unknown doc_id(s): {', '.join(sorted(unknown))}")
        docs = [d for d in docs if d.doc_id in doc]
    if limit:
        docs = docs[:limit]

    def progress(res: DocTables) -> None:
        r = res.report
        share = r["n_fallback_plain"] / r["n_tables"] if r["n_tables"] else 0.0
        typer.echo(
            f"  {res.doc_id:<26} {r['n_regions']:>3} regions  {r['n_tables']:>3} tables  "
            f"{r['n_fallback_plain']:>3} fallback_plain ({share:>4.0%})  "
            f"{r['n_continued']:>2} continued"
        )

    report = run_tables(
        Manifest(documents=docs),
        paths.raw_dir,
        paths.extract_report,
        paths.tables,
        paths.tables_report,
        min_fill=ing.table_min_fill,
        max_ragged=ing.table_max_ragged,
        workers=workers or ing.workers,
        on_doc=progress,
    )
    typer.echo(
        f"{report['n_tables']} tables from {report['n_regions']} regions in "
        f"{report['n_documents']} documents -> {paths.tables}"
    )
    typer.echo(
        f"flagged fallback_plain: {report['n_fallback_plain']} "
        f"({report['fallback_share']:.1%}) | reasons: {report['reasons']} | "
        f"continued: {report['n_continued']} | report: {paths.tables_report}"
    )


def _variants(variant: str) -> list[str]:
    from claimcheck.ingest.chunk import VARIANTS

    names = [v.value for v in VARIANTS]
    if variant == "all":
        return names
    if variant not in names:
        typer.echo(f"unknown variant {variant!r}; use one of {names} or 'all'", err=True)
        raise typer.Exit(code=2)
    return [variant]


@app.command()
def chunk(
    variant: Annotated[
        str, typer.Option("--variant", help="table_aware, plain_text or all.")
    ] = "table_aware",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Chunk pages.parquet and tables.parquet into <variant>/chunks.parquet."""
    from claimcheck.config import get_settings
    from claimcheck.ingest.chunk import hf_token_counter, load_inputs, run_chunk
    from claimcheck.schemas import Manifest, Variant

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    settings = get_settings()
    paths = settings.paths
    for need, cmd in ((paths.pages, "extract"), (paths.tables, "tables")):
        if not need.exists():
            typer.echo(f"{need.name} not found; run `claimcheck ingest {cmd}`", err=True)
            raise typer.Exit(code=1)
    manifest = Manifest.model_validate_json(paths.manifest.read_text(encoding="utf-8"))
    pages, tables_, keys = load_inputs(paths.pages, paths.tables, paths.extract_report)
    count = hf_token_counter(settings.embedding.model, settings.embedding.revision)
    failed = False
    for name in _variants(variant):
        v = Variant(name)
        rep = run_chunk(
            v,
            manifest,
            pages,
            tables_,
            keys,
            settings.chunking,
            count,
            paths.chunks(name),
            paths.chunk_report(name),
            latest_edition=settings.latest_edition,
        )
        kinds = " | ".join(f"{k} {n}" for k, n in rep["kinds"].items())
        heavy = rep["numeric_heavy"]
        typer.echo(
            f"{name}: {rep['n_chunks']} chunks ({kinds}) from {rep['n_documents']} documents"
            f" -> {paths.chunks(name)}"
        )
        typer.echo(
            f"  words {rep['n_words']} | tokens {rep['n_tokens']} | duplicates removed "
            f"{rep['n_duplicates_removed']} | short dropped {rep['n_short_dropped']} | "
            f"numeric-heavy text chunks {heavy['n']} ({heavy['share_of_text_chunks']:.1%})"
        )
        if rep["n_words"]["min"] == 0 or rep["n_tokens"]["max"] > settings.chunking.max_tokens:
            typer.echo("  empty chunk or chunk over the token limit", err=True)
            failed = True
    if failed:
        raise typer.Exit(code=1)


@app.command()
def sample(
    n: Annotated[int, typer.Option("--n", min=1, help="Chunks of each kind.")] = 10,
    variant: Annotated[str, typer.Option("--variant")] = "table_aware",
    seed: Annotated[int, typer.Option("--seed")] = 2026,
    embed: Annotated[bool, typer.Option("--embed", help="Also print embed_text.")] = False,
) -> None:
    """Print N random text chunks and N random table chunks of a variant."""
    import random

    from claimcheck.config import get_settings
    from claimcheck.ingest.chunk import read_chunks

    path = get_settings().paths.chunks(_variants(variant)[0])
    if not path.exists():
        typer.echo(f"{path} not found; run `claimcheck ingest chunk`", err=True)
        raise typer.Exit(code=1)
    records = read_chunks(path)
    rng = random.Random(seed)
    for kind in ("text", "table"):
        pool = [r for r in records if r.kind.value == kind]
        for r in rng.sample(pool, min(n, len(pool))):
            pages = f"p{r.page_start}" + (f"-{r.page_end}" if r.page_end != r.page_start else "")
            typer.echo("=" * 100)
            typer.echo(
                f"[{kind}] {r.chunk_id} {r.doc_id} {pages} {r.table_id or ''} "
                f"words={r.n_words} tokens={r.n_tokens}"
            )
            typer.echo("-" * 100)
            typer.echo(r.text)
            if embed:
                typer.echo("-" * 40 + " embed_text")
                typer.echo(r.embed_text)
