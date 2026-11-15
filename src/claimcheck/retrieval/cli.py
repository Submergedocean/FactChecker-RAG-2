"""`claimcheck retrieval` commands: run one query against an index variant.

claimcheck retrieval retrieve "<query>" --mode hybrid -k 5
"""

from __future__ import annotations

import json
import textwrap
from typing import Annotated

import typer

app = typer.Typer(
    help="Hybrid (BM25 + dense, RRF) retrieval.",
    no_args_is_help=True,
)

MODES = ["dense", "sparse", "hybrid", "interleave"]


@app.callback()
def main() -> None:
    """Hybrid (BM25 + dense, RRF) retrieval."""


def _rank(r: int | None) -> str:
    return "-" if r is None else str(r)


@app.command()
def retrieve(
    query: Annotated[str, typer.Argument(help="The claim or query text.")],
    mode: Annotated[str, typer.Option("--mode", "-m", help=" | ".join(MODES))] = "hybrid",
    k: Annotated[int | None, typer.Option("-k", help="Passages to keep (default top_k).")] = None,
    variant: Annotated[
        str, typer.Option("--variant", help="table_aware or plain_text.")
    ] = "table_aware",
    full: Annotated[bool, typer.Option("--full", help="Print whole passages.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Print the ranked passages for QUERY with each method's rank and the RRF score."""
    from claimcheck.index.store import IndexMismatchError
    from claimcheck.retrieval.retriever import Retriever
    from claimcheck.schemas import Variant

    if mode not in MODES:
        typer.echo(f"unknown mode {mode!r}; use one of {MODES}", err=True)
        raise typer.Exit(code=2)
    if variant not in [v.value for v in Variant]:
        typer.echo(f"unknown variant {variant!r}", err=True)
        raise typer.Exit(code=2)
    if k is not None and k < 1:
        typer.echo("-k must be at least 1", err=True)
        raise typer.Exit(code=2)
    if not query.strip():
        typer.echo("empty query", err=True)
        raise typer.Exit(code=2)
    try:
        retriever = Retriever.from_settings(variant)
    except (IndexMismatchError, FileNotFoundError) as exc:
        typer.echo(f"{variant}: NOT LOADED: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    res = retriever.retrieve(query, mode=mode, k=k)
    if as_json:
        out = {
            "query": res.query,
            "mode": res.mode.value,
            "variant": res.variant.value,
            "index_version": res.index_version,
            "timings_ms": {m: round(v, 2) for m, v in res.timings_ms.items()},
            "passages": [p.to_evidence().model_dump(mode="json") for p in res.passages],
        }
        typer.echo(json.dumps(out, ensure_ascii=False, indent=2))
        return

    methods = list(res.candidates)
    typer.echo(
        f"{res.mode.value} | {res.variant.value} | index {res.index_version} | "
        f"candidates {', '.join(f'{m} {n}' for m, n in res.candidates.items())}"
    )
    for p in res.passages:
        c = p.chunk
        pages = (
            f"p{c.page_start}" if c.page_start == c.page_end else (f"p{c.page_start}-{c.page_end}")
        )
        ranks = "  ".join(f"{m} {_rank(p.ranks[m]):>2}" for m in methods)
        where = f"{c.doc_id} {pages} {c.kind.value}" + (f" {c.table_id}" if c.table_id else "")
        typer.echo(f"\n#{p.rank}  rrf {p.rrf:.4f}  {ranks}  | {where}")
        typer.echo(f"    {c.chapter_title} ({c.edition}) | {c.chunk_id}")
        body = " ".join(c.text.split())
        if not full:
            body = textwrap.shorten(body, width=300, placeholder=" …")
        for line in textwrap.wrap(body, width=96):
            typer.echo(f"    {line}")
    t = res.timings_ms
    typer.echo(
        "\n"
        + " | ".join(
            f"{name} {t[name]:.1f} ms"
            for name in ("encode", "dense", "bm25", "fuse", "total")
            if name in t
        )
    )
