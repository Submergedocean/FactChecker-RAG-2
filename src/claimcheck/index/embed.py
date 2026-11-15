"""Embeddings: `BAAI/bge-small-en-v1.5` at a pinned revision, CPU torch.

Passages are embedded from `embed_text` (plus `passage_prefix`, empty for BGE); queries get the
BGE query instruction (`query_prefix`). Vectors are float32 and L2-normalised, so the inner
product of `IndexFlatIP` is the cosine similarity.

Rebuilds reuse vectors: `EmbeddingCache` keeps passage vectors keyed by the model,
revision, prefix and text, in a NumPy `.npz` file (no pickle) under `data/cache/embeddings/`.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from claimcheck.config import EmbeddingSettings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    """What the index build and retrieval need from an embedding model."""

    model: str
    revision: str | None
    dim: int
    normalize: bool
    query_prefix: str
    passage_prefix: str

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) float32 vectors of passages (`passage_prefix` + text)."""
        ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) float32 vectors of queries (`query_prefix` + text)."""
        ...


class SentenceTransformerEmbedder:
    """The real embedder. Loads the model once (about 130 MB, downloaded on first use)."""

    def __init__(self, cfg: EmbeddingSettings, *, show_progress: bool = False) -> None:
        if not cfg.revision:
            raise ValueError("embedding.revision must be pinned")
        from sentence_transformers import SentenceTransformer

        self.model = cfg.model
        self.revision = cfg.revision
        self.normalize = cfg.normalize
        self.query_prefix = cfg.query_prefix
        self.passage_prefix = cfg.passage_prefix
        self.batch_size = cfg.batch_size
        self.show_progress = show_progress
        self._st = SentenceTransformer(cfg.model, revision=cfg.revision, device=cfg.device)
        dim_of = getattr(self._st, "get_embedding_dimension", None) or (
            self._st.get_sentence_embedding_dimension
        )
        self.dim = int(dim_of() or 0)
        if self.dim != cfg.dim:
            raise ValueError(f"{cfg.model} gives {self.dim}-d vectors, settings say {cfg.dim}")

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._st.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress,
        )
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])


# --------------------------------------------------------------------------- cache


def cache_path(cache_dir: Path, embedder: Embedder) -> Path | None:
    """One cache file per model and revision; none for an unpinned revision."""
    if not embedder.revision:
        return None
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", embedder.model)
    return cache_dir / f"{slug}@{embedder.revision[:12]}.npz"


def text_key(embedder: Embedder, text: str) -> str:
    """Cache key of a passage: what decides its vector besides model and revision."""
    raw = f"{embedder.passage_prefix}\x00{int(embedder.normalize)}\x00{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Passage vectors keyed by `text_key`, stored as `keys` (str) and `vectors` (float32)."""

    def __init__(self, path: Path | None, dim: int) -> None:
        self.path = path
        self.dim = dim
        self._vecs: dict[str, np.ndarray] = {}
        if path is not None and path.exists():
            with np.load(path, allow_pickle=False) as data:
                keys, vecs = data["keys"], data["vectors"]
            if vecs.ndim == 2 and vecs.shape[1] == dim and len(keys) == len(vecs):
                self._vecs = {str(k): v for k, v in zip(keys, vecs, strict=True)}
            else:
                log.warning("ignoring embedding cache %s: shape %s", path, vecs.shape)

    def __len__(self) -> int:
        return len(self._vecs)

    def get(self, key: str) -> np.ndarray | None:
        return self._vecs.get(key)

    def put(self, key: str, vec: np.ndarray) -> None:
        self._vecs[key] = vec

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(sorted(self._vecs), dtype="U40")
        vecs = (
            np.stack([self._vecs[k] for k in keys]).astype(np.float32)
            if len(keys)
            else np.zeros((0, self.dim), dtype=np.float32)
        )
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=keys, vectors=vecs)
        tmp.replace(self.path)


def embed_passages(
    embedder: Embedder, texts: Sequence[str], cache: EmbeddingCache | None = None
) -> tuple[np.ndarray, int]:
    """Vectors of `texts` in order, reusing cached ones. Returns (vectors, number embedded)."""
    out = np.zeros((len(texts), embedder.dim), dtype=np.float32)
    todo: dict[str, list[int]] = {}
    for i, text in enumerate(texts):
        key = text_key(embedder, text)
        hit = cache.get(key) if cache is not None else None
        if hit is not None:
            out[i] = hit
        else:
            todo.setdefault(key, []).append(i)
    if todo:
        firsts = [rows[0] for rows in todo.values()]
        vecs = embedder.encode_passages([texts[i] for i in firsts])
        if vecs.shape != (len(firsts), embedder.dim):
            raise ValueError(f"embedder returned {vecs.shape}, expected ({len(firsts)}, dim)")
        for (key, rows), vec in zip(todo.items(), vecs, strict=True):
            out[rows] = vec
            if cache is not None:
                cache.put(key, vec)
    return out, len(todo)
