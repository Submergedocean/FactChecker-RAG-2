"""Sparse side: `rank_bm25.BM25Okapi` over `chunks.parquet` → `bm25_tokens`.

Documents and queries share one tokenizer, `claimcheck.textnorm.tokenize`:
lowercase, decimals kept (`3.7`), thousands separators stripped (`1,234.5` → `1234.5`), and a
canonical fiscal-year token (`fy2026`) emitted alongside the raw one, so `FY2025-26`, `2025-26`
and `FY26` all match. The index is rebuilt from the stored tokens at load time: there is no
pickle, and building it over the corpus takes well under a second.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from claimcheck.textnorm import tokenize

__all__ = ["BM25_PARAMS", "build_bm25", "query_tokens", "top_k", "tokenize"]

# rank_bm25's BM25Okapi defaults, passed explicitly; part of `index_version`.
BM25_PARAMS = {"k1": 1.5, "b": 0.75, "epsilon": 0.25}


def query_tokens(text: str) -> list[str]:
    """BM25 tokens of a query: the same tokenizer as the documents."""
    return tokenize(text)


def build_bm25(docs: Sequence[Sequence[str]]) -> BM25Okapi:
    """BM25Okapi (k1 1.5, b 0.75) over token lists, one per chunk row."""
    if not docs:
        raise ValueError("cannot build BM25 over an empty corpus")
    return BM25Okapi([list(d) for d in docs], **BM25_PARAMS)


def top_k(bm25: BM25Okapi, tokens: Sequence[str], k: int) -> list[tuple[int, float]]:
    """The k best rows as (row, score), best first. Rows that share no token with the query
    are left out (a shared token can still score 0: Okapi's idf is 0 for a term in exactly half
    the documents); equal scores are ordered by row, so the ranking is deterministic."""
    if not tokens or k <= 0:
        return []
    query = list(dict.fromkeys(tokens))
    scores = np.asarray(bm25.get_scores(list(tokens)), dtype=np.float64)
    shared = np.fromiter(
        (any(t in freqs for t in query) for freqs in bm25.doc_freqs),
        dtype=bool,
        count=len(bm25.doc_freqs),
    )
    hits = np.flatnonzero(shared)
    if hits.size == 0:
        return []
    order = hits[np.lexsort((hits, -scores[hits]))][:k]
    return [(int(i), float(scores[i])) for i in order]
