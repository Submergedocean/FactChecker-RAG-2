"""`claimcheck index` commands: build and inspect the FAISS + BM25 index."""

from __future__ import annotations

import logging
import time
from typing import Annotated

import typer

app = typer.Typer(
    help="Embedding, BM25 and FAISS index build and inspection.",
    no_args_is_help=True,
)

VARIANT_HELP = "table_aware, plain_text or all."


@app.callback()
def main() -> None:
    """Embedding, BM25 and FAISS index build and inspection."""


def _variants(variant: str) -> list[str]:
    from claimcheck.schemas import Variant

    names = [v.value for v in Variant]
    if variant == "all":
        return names
    if variant not in names:
        typer.echo(f"unknown variant {variant!r}; use one of {names} or 'all'", err=True)
        raise typer.Exit(code=2)
    return [variant]


@app.command()
def build(
    variant: Annotated[str, typer.Option("--variant", help=VARIANT_HELP)] = "table_aware",
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Embed every chunk, ignoring cached vectors.")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Embed <variant>/chunks.parquet into faiss.index and write index_meta.json."""
    from claimcheck.config import get_settings
    from claimcheck.index.build import build_index, git_commit
    from claimcheck.index.embed import EmbeddingCache, SentenceTransformerEmbedder, cache_path
    from claimcheck.index.store import IndexPaths

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    names = _variants(variant)
    settings = get_settings()
    for name in names:
        if not settings.paths.chunks(name).exists():
            typer.echo(f"{settings.paths.chunks(name)} not found; run `claimcheck ingest chunk`")
            raise typer.Exit(code=1)
    t0 = time.perf_counter()
    embedder = SentenceTransformerEmbedder(settings.embedding, show_progress=verbose)
    typer.echo(
        f"model {embedder.model}@{embedder.revision} ({embedder.dim}-d, "
        f"{settings.embedding.device}) loaded in {time.perf_counter() - t0:.1f} s"
    )
    cpath = None if no_cache else cache_path(settings.paths.embedding_cache, embedder)
    cache = EmbeddingCache(cpath, embedder.dim)
    commit = git_commit()
    for name in names:
        res = build_index(
            name,
            IndexPaths.of(settings.paths, name),
            embedder,
            chunking=settings.chunking,
            cache=cache,
            commit=commit,
        )
        m = res.meta
        typer.echo(
            f"{name}: {m.n_chunks} vectors ({res.n_embedded} embedded, "
            f"{m.n_chunks - res.n_embedded} from cache) in {res.seconds:.1f} s | "
            f"version {m.index_version} -> {settings.paths.faiss_index(name)}"
        )


@app.command()
def info(
    variant: Annotated[str, typer.Option("--variant", help=VARIANT_HELP)] = "all",
) -> None:
    """Load each variant's index (with its checks) and print what it holds."""
    from collections import Counter

    from claimcheck.index.store import IndexMismatchError, IndexStore

    failed = False
    for name in _variants(variant):
        t0 = time.perf_counter()
        try:
            store = IndexStore.load(name)
        except (IndexMismatchError, FileNotFoundError) as exc:
            typer.echo(f"{name}: NOT LOADED: {exc}", err=True)
            failed = True
            continue
        secs = time.perf_counter() - t0
        m = store.meta
        kinds = Counter(c.kind.value for c in store.chunks)
        typer.echo(f"{name}: index {m.index_version}")
        typer.echo(
            f"  vectors {store.faiss.ntotal} | chunks {len(store)} "
            f"({', '.join(f'{k} {n}' for k, n in sorted(kinds.items()))}) | "
            f"dimension {store.faiss.d} | {m.faiss_type}"
        )
        typer.echo(f"  model {m.embedding_model} | revision {m.embedding_revision}")
        typer.echo(
            f"  bm25 documents {store.bm25.corpus_size} | built {m.built_at.isoformat()} | "
            f"commit {m.git_commit} | loaded and checked in {secs:.2f} s"
        )
    if failed:
        raise typer.Exit(code=1)
