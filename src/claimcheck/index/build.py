"""`claimcheck index build`: embed `chunks.parquet` into `faiss.index`.

For each variant: read the chunks in file order, embed every `embed_text` (reusing cached
vectors), add the vectors to an `IndexFlatIP` in the same order (row i = chunk row
i), and write `faiss.index` and `index_meta.json`. Files are written to a temporary name and
renamed, the metadata last, so a failed build never leaves a matching pair of stale files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from claimcheck.config import REPO_ROOT, ChunkingSettings
from claimcheck.index.bm25 import BM25_PARAMS
from claimcheck.index.embed import Embedder, EmbeddingCache, embed_passages
from claimcheck.index.store import (
    FAISS_TYPE,
    IndexFileMeta,
    IndexPaths,
    content_digest,
    file_sha256,
    read_chunk_rows,
)
from claimcheck.schemas import Variant

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildResult:
    meta: IndexFileMeta
    n_embedded: int  # vectors computed in this build (the rest came from the cache)
    seconds: float


def index_version(variant: Variant, embedder: Embedder, content_sha1: str) -> str:
    """Deterministic: the same chunk rows (every field, `content_digest`) embedded by the same
    model with the same BM25 parameters give the same version, so the result cache
    stays valid across identical rebuilds and is invalidated by any other change."""
    raw = "\x00".join(
        [
            variant.value,
            embedder.model,
            embedder.revision or "",
            str(embedder.normalize),
            embedder.passage_prefix,
            embedder.query_prefix,
            json.dumps(BM25_PARAMS, sort_keys=True),
            content_sha1,
        ]
    )
    return f"{variant.value}-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def git_commit(repo: Path = REPO_ROOT) -> str | None:
    """HEAD of the repository, with `-dirty` when tracked files have uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=repo).returncode != 0
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}-dirty" if dirty else head


def chunking_params(report: Path, fallback: ChunkingSettings, variant: Variant) -> dict[str, Any]:
    """The chunking parameters the chunks were built with (the chunker's chunk_report.json), with
    `variant` set to the variant being indexed (the chunk report records the settings' default
    there, `table_aware`, also for `plain_text`)."""
    params: dict[str, Any] = fallback.model_dump(mode="json")
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        if isinstance(data.get("chunking"), dict):
            params = dict(data["chunking"])
    params["variant"] = variant.value
    return params


def build_index(
    variant: Variant | str,
    paths: IndexPaths,
    embedder: Embedder,
    *,
    chunking: ChunkingSettings,
    cache: EmbeddingCache | None = None,
    commit: str | None = None,
) -> BuildResult:
    from claimcheck.index._faiss import load_faiss

    faiss = load_faiss()

    v = Variant(variant)
    t0 = time.perf_counter()
    if not paths.chunks.exists():
        raise FileNotFoundError(f"{paths.chunks} not found; run `claimcheck ingest chunk`")
    rows = read_chunk_rows(paths.chunks)
    if not rows:
        raise ValueError(f"{paths.chunks} has no chunks")
    texts = [r["embed_text"] for r in rows]
    if any(not t.strip() for t in texts):
        raise ValueError(f"{paths.chunks} has an empty embed_text")

    vecs, n_embedded = embed_passages(embedder, texts, cache)
    if embedder.normalize:
        norms = np.linalg.norm(vecs, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"vectors are not normalised (norms {norms.min()}–{norms.max()})")
    index = faiss.IndexFlatIP(embedder.dim)
    index.add(np.ascontiguousarray(vecs, dtype=np.float32))

    paths.faiss.parent.mkdir(parents=True, exist_ok=True)
    tmp_faiss = paths.faiss.with_suffix(".index.tmp")
    faiss.write_index(index, str(tmp_faiss))
    tmp_faiss.replace(paths.faiss)
    if cache is not None:
        cache.save()

    digest = content_digest(rows)
    meta = IndexFileMeta(
        index_version=index_version(v, embedder, digest),
        variant=v,
        embedding_model=embedder.model,
        embedding_revision=embedder.revision,
        dim=embedder.dim,
        n_chunks=index.ntotal,
        chunking=chunking_params(paths.chunk_report, chunking, v),
        built_at=datetime.now(UTC).replace(microsecond=0),
        git_commit=commit,
        faiss_type=FAISS_TYPE,
        normalize=embedder.normalize,
        query_prefix=embedder.query_prefix,
        passage_prefix=embedder.passage_prefix,
        content_sha1=digest,
        chunks_sha256=file_sha256(paths.chunks),
        faiss_sha256=file_sha256(paths.faiss),
    )
    tmp_meta = paths.meta.with_suffix(".json.tmp")
    tmp_meta.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp_meta.replace(paths.meta)
    seconds = time.perf_counter() - t0
    log.info("%s: %d vectors (%d embedded) in %.1f s", v.value, index.ntotal, n_embedded, seconds)
    return BuildResult(meta=meta, n_embedded=n_embedded, seconds=seconds)
