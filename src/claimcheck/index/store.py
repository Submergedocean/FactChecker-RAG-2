"""The loaded index of one variant: chunks, FAISS and BM25, checked on load.

Files of a variant (`data/processed/<variant>/`): `chunks.parquet`, `faiss.index`
(`IndexFlatIP`, row i = row i of `chunks.parquet`) and `index_meta.json` (`IndexFileMeta`).
`IndexStore.load` refuses to load (`IndexMismatchError`) when the metadata and the files
disagree: a file changed since the build (SHA-256), a vector count, dimension or index type
that is not the one recorded, chunks whose ids or texts differ from those that were embedded,
another variant, or another embedding model, revision or prefix than the settings name.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from claimcheck.config import EmbeddingSettings, PathsSettings
from claimcheck.index.bm25 import build_bm25, top_k
from claimcheck.schemas import ChunkRecord, IndexMeta, Variant

FAISS_TYPE = "IndexFlatIP"


class IndexMismatchError(RuntimeError):
    """The index files and `index_meta.json` disagree; rebuild with `claimcheck index build`."""


class IndexFileMeta(IndexMeta):
    """`index_meta.json`: the stored fields plus what `IndexStore.load` checks against."""

    faiss_type: str = FAISS_TYPE
    normalize: bool
    query_prefix: str
    passage_prefix: str
    content_sha1: str  # every field of every chunk row, in order (`content_digest`)
    chunks_sha256: str
    faiss_sha256: str


@dataclass(frozen=True)
class IndexPaths:
    chunks: Path
    faiss: Path
    meta: Path
    chunk_report: Path

    @classmethod
    def of(cls, paths: PathsSettings, variant: Variant | str) -> IndexPaths:
        v = Variant(variant)
        return cls(
            chunks=paths.chunks(v),
            faiss=paths.faiss_index(v),
            meta=paths.index_meta(v),
            chunk_report=paths.chunk_report(v),
        )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def content_digest(rows: Sequence[dict[str, Any]]) -> str:
    """SHA-1 over every field of every chunk row, in order: ids, texts, BM25 tokens and the
    citation metadata (document, edition, chapter, pages, kind, table id, counts). Any change
    that can alter what is retrieved or cited changes it."""
    h = hashlib.sha1()
    for row in rows:
        line = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def read_meta(path: Path) -> IndexFileMeta:
    return IndexFileMeta.model_validate_json(path.read_text(encoding="utf-8"))


def read_chunk_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


@dataclass(frozen=True, eq=False)
class IndexStore:
    """Immutable after load; one per variant, shared by API, CLI and evaluation."""

    variant: Variant
    meta: IndexFileMeta
    chunks: tuple[ChunkRecord, ...]
    faiss: Any  # faiss.IndexFlatIP
    bm25: Any  # rank_bm25.BM25Okapi
    row_of: dict[str, int]

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def index_version(self) -> str:
        return self.meta.index_version

    # ------------------------------------------------------------------ search primitives

    def dense_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """The k nearest rows to one query vector as (row, inner product), best first."""
        q = np.ascontiguousarray(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))
        if q.shape[1] != self.meta.dim:
            raise ValueError(f"query has dimension {q.shape[1]}, index {self.meta.dim}")
        k = min(k, len(self.chunks))
        if k <= 0:
            return []
        scores, rows = self.faiss.search(q, k)
        return [(int(r), float(s)) for r, s in zip(rows[0], scores[0], strict=True) if r >= 0]

    def sparse_search(self, tokens: Sequence[str], k: int) -> list[tuple[int, float]]:
        """The k best BM25 rows for query tokens as (row, score), best first."""
        return top_k(self.bm25, tokens, k)

    def vector(self, row: int) -> np.ndarray:
        """The stored vector of a row."""
        return self.faiss.reconstruct(int(row))

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(
        cls,
        variant: Variant | str,
        *,
        paths: IndexPaths | None = None,
        embedding: EmbeddingSettings | None = None,
    ) -> IndexStore:
        """Load and check one variant. With `embedding` (the default: the settings), the
        index must have been built with that model, revision, normalisation and prefixes."""
        from claimcheck.index._faiss import load_faiss

        faiss = load_faiss()

        from claimcheck.config import get_settings

        v = Variant(variant)
        settings = get_settings()
        paths = paths or IndexPaths.of(settings.paths, v)
        embedding = embedding if embedding is not None else settings.embedding
        for path in (paths.chunks, paths.faiss, paths.meta):
            if not path.exists():
                raise FileNotFoundError(f"{path} not found; run `claimcheck index build`")

        try:
            meta = read_meta(paths.meta)
        except (ValueError, json.JSONDecodeError) as exc:
            raise IndexMismatchError(f"{paths.meta}: unreadable metadata ({exc})") from exc
        problems = _meta_problems(meta, v, embedding)
        if file_sha256(paths.chunks) != meta.chunks_sha256:
            problems.append(f"{paths.chunks.name} changed since the index was built")
        if file_sha256(paths.faiss) != meta.faiss_sha256:
            problems.append(f"{paths.faiss.name} changed since the index was built")
        if problems:
            raise IndexMismatchError(f"{v.value}: " + "; ".join(problems))

        index = faiss.read_index(str(paths.faiss))
        rows = read_chunk_rows(paths.chunks)
        if type(index).__name__ != meta.faiss_type or (
            index.metric_type != faiss.METRIC_INNER_PRODUCT
        ):
            problems.append(f"index type {type(index).__name__}, expected {meta.faiss_type}")
        if index.d != meta.dim:
            problems.append(f"index dimension {index.d}, metadata {meta.dim}")
        if not (index.ntotal == meta.n_chunks == len(rows)):
            problems.append(
                f"{index.ntotal} vectors, {len(rows)} chunks, metadata says {meta.n_chunks}"
            )
        if content_digest(rows) != meta.content_sha1:
            problems.append("chunk rows differ from those the index was built from")
        if problems:
            raise IndexMismatchError(f"{v.value}: " + "; ".join(problems))

        chunks = tuple(ChunkRecord.model_validate(r) for r in rows)
        row_of = {c.chunk_id: i for i, c in enumerate(chunks)}
        if len(row_of) != len(chunks):
            raise IndexMismatchError(f"{v.value}: duplicate chunk ids in {paths.chunks.name}")
        bm25 = build_bm25([c.bm25_tokens for c in chunks])
        return cls(variant=v, meta=meta, chunks=chunks, faiss=index, bm25=bm25, row_of=row_of)


def _meta_problems(meta: IndexFileMeta, v: Variant, emb: EmbeddingSettings) -> list[str]:
    out = []
    if meta.variant != v:
        out.append(f"metadata is for variant {meta.variant.value}")
    expected = {
        "embedding_model": emb.model,
        "embedding_revision": emb.revision,
        "dim": emb.dim,
        "normalize": emb.normalize,
        "query_prefix": emb.query_prefix,
        "passage_prefix": emb.passage_prefix,
    }
    for field, want in expected.items():
        got = getattr(meta, field)
        if got != want:
            out.append(f"{field} is {got!r}, settings say {want!r}")
    return out
