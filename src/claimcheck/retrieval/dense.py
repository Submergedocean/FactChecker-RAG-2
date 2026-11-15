"""Dense side: the query with the BGE query instruction, FAISS inner product.

The embedder comes from `claimcheck.index.embed` (imported through `claimcheck.index`, so the
macOS one-OpenMP-thread guard runs before torch and faiss load; see `index/_faiss.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

from claimcheck.index.embed import Embedder
from claimcheck.index.store import IndexStore


@dataclass(frozen=True)
class Hit:
    """One ranked result of a single method."""

    row: int
    chunk_id: str
    rank: int  # 1-based
    score: float


def hits_of(store: IndexStore, pairs: list[tuple[int, float]]) -> list[Hit]:
    return [
        Hit(row=row, chunk_id=store.chunks[row].chunk_id, rank=i, score=score)
        for i, (row, score) in enumerate(pairs, start=1)
    ]


class DenseRetriever:
    def __init__(self, store: IndexStore, embedder: Embedder) -> None:
        if embedder.dim != store.meta.dim:
            raise ValueError(f"embedder gives {embedder.dim}-d vectors, index {store.meta.dim}")
        if (embedder.model, embedder.revision) != (
            store.meta.embedding_model,
            store.meta.embedding_revision,
        ):
            raise ValueError(
                f"embedder {embedder.model}@{embedder.revision} is not the index's "
                f"{store.meta.embedding_model}@{store.meta.embedding_revision}"
            )
        if embedder.query_prefix != store.meta.query_prefix:
            raise ValueError("embedder query prefix differs from the index metadata")
        self.store = store
        self.embedder = embedder

    def encode(self, query: str):
        """The query vector (`query_prefix` + query)."""
        return self.embedder.encode_queries([query])[0]

    def search(self, query: str, k: int) -> list[Hit]:
        return self.search_vector(self.encode(query), k)

    def search_vector(self, vec, k: int) -> list[Hit]:
        return hits_of(self.store, self.store.dense_search(vec, k))
