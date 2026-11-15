"""One retrieval interface for every mode: `dense`, `sparse`, `hybrid`, `interleave`.

`Retriever.retrieve(query, mode=..., k=...)` returns the top k passages of one index variant,
with each method's 1-based rank and the RRF score:

- `hybrid` (the system): BM25 top `bm25_k` (50) and FAISS top `dense_k` (50) fused by RRF,
  `score = Σ 1/(rrf_k + rank)` with `rrf_k` 60, ties by `chunk_id`, top `top_k` (5) kept.
- `dense`, `sparse`: one method alone (ranked by that method; `rrf` is its single term).
- `interleave` (fusion ablation): the same two lists merged round-robin, dense first.

`ranks` holds the methods that were run (`{"bm25": 2, "dense": 1}`); a rank is None when the
passage is not in that method's candidate list. The query is used as given (the pipeline's
input guard trims and normalises it); dense queries get the BGE query instruction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from claimcheck.config import RetrievalSettings
from claimcheck.index.embed import Embedder
from claimcheck.index.store import IndexStore
from claimcheck.retrieval.dense import DenseRetriever, Hit
from claimcheck.retrieval.fusion import interleave, rrf
from claimcheck.retrieval.sparse import SparseRetriever
from claimcheck.schemas import ChunkRecord, EvidencePassage, Variant

BM25 = "bm25"
DENSE = "dense"


class Mode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    INTERLEAVE = "interleave"


@dataclass(frozen=True)
class RetrievedPassage:
    rank: int  # 1-based, in the fused (or single-method) order
    chunk: ChunkRecord
    ranks: dict[str, int | None]  # methods run → rank in its candidate list
    rrf: float
    scores: dict[str, float | None]  # methods run → raw score (BM25, inner product)

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def covers(self, doc_id: str, page: int) -> bool:
        c = self.chunk
        return c.doc_id == doc_id and c.page_start <= page <= c.page_end

    def to_evidence(self, *, cited: bool = False) -> EvidencePassage:
        c = self.chunk
        return EvidencePassage(
            chunk_id=c.chunk_id,
            rank=self.rank,
            kind=c.kind,
            text=c.text,
            doc_id=c.doc_id,
            edition=c.edition,
            chapter=c.chapter_title,
            page_start=c.page_start,
            page_end=c.page_end,
            table_id=c.table_id,
            ranks=dict(self.ranks),
            rrf=self.rrf,
            cited=cited,
        )


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    mode: Mode
    variant: Variant
    index_version: str
    passages: list[RetrievedPassage]
    candidates: dict[str, int]  # methods run → candidate list length
    timings_ms: dict[str, float] = field(default_factory=dict)  # encode, dense, bm25, fuse, total

    @property
    def latency_ms(self) -> float:
        return self.timings_ms.get("total", 0.0)


class Retriever:
    """Holds one variant's `IndexStore` and an embedder; stateless per query, thread-safe for
    reads (the store is immutable)."""

    def __init__(
        self,
        store: IndexStore,
        embedder: Embedder,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self.store = store
        self.settings = settings or RetrievalSettings()
        self.dense = DenseRetriever(store, embedder)
        self.sparse = SparseRetriever(store)

    @classmethod
    def from_settings(
        cls, variant: Variant | str | None = None, *, embedder: Embedder | None = None
    ) -> Retriever:
        """Load the variant's index (default `table_aware`) and the pinned embedding model."""
        from claimcheck.config import get_settings

        settings = get_settings()
        store = IndexStore.load(variant or Variant.TABLE_AWARE)
        if embedder is None:
            from claimcheck.index.embed import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder(settings.embedding)
        return cls(store, embedder, settings.retrieval)

    @property
    def variant(self) -> Variant:
        return self.store.variant

    def retrieve(
        self, query: str, *, mode: Mode | str = Mode.HYBRID, k: int | None = None
    ) -> RetrievalResult:
        mode = Mode(mode)
        k = self.settings.top_k if k is None else k
        if k < 1:
            raise ValueError("k must be at least 1")
        if not query.strip():
            raise ValueError("empty query")
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        lists: dict[str, list[Hit]] = {}

        if mode is not Mode.SPARSE:
            t = time.perf_counter()
            vec = self.dense.encode(query)
            timings["encode"] = _ms(t)
            t = time.perf_counter()
            lists[DENSE] = self.dense.search_vector(vec, max(k, self.settings.dense_k))
            timings["dense"] = _ms(t)
        if mode is not Mode.DENSE:
            t = time.perf_counter()
            lists[BM25] = self.sparse.search(query, max(k, self.settings.bm25_k))
            timings["bm25"] = _ms(t)

        t = time.perf_counter()
        # Reported order of methods: bm25 first, as in the API example.
        rankings = {m: [h.chunk_id for h in lists[m]] for m in (BM25, DENSE) if m in lists}
        if mode is Mode.INTERLEAVE:
            fused = interleave(
                {DENSE: rankings[DENSE], BM25: rankings[BM25]}, k=self.settings.rrf_k
            )
        else:
            fused = rrf(rankings, k=self.settings.rrf_k)
        score_of = {m: {h.chunk_id: h.score for h in hits} for m, hits in lists.items()}
        passages = [
            RetrievedPassage(
                rank=i,
                chunk=self.store.chunks[self.store.row_of[f.chunk_id]],
                ranks={m: f.ranks[m] for m in rankings},
                rrf=f.score,
                scores={m: score_of[m].get(f.chunk_id) for m in rankings},
            )
            for i, f in enumerate(fused[:k], start=1)
        ]
        timings["fuse"] = _ms(t)
        timings["total"] = _ms(t0)
        return RetrievalResult(
            query=query,
            mode=mode,
            variant=self.store.variant,
            index_version=self.store.index_version,
            passages=passages,
            candidates={m: len(lists[m]) for m in rankings},
            timings_ms=timings,
        )


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0
