"""Sparse side: BM25Okapi over the stored chunk tokens, query tokenized the same way
(`claimcheck.textnorm.tokenize`: decimals kept, thousands separators stripped, `fy2026`
emitted for every fiscal-year form). Rows sharing no token with the query are not returned;
equal scores are ordered by row.

`PostingBM25` scores from posting lists built once from the store's `BM25Okapi` (its idf,
term frequencies, lengths, k1 and b). `BM25Okapi.get_scores` walks every document in Python
for each query token (about 100 ms per query at p95 over the corpus); the posting
lists touch only the rows that hold a token. Each row's score is the same float64 sum, term by
term in query order, so results equal `IndexStore.sparse_search` exactly (unit test and gate).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from claimcheck.index.bm25 import query_tokens
from claimcheck.index.store import IndexStore
from claimcheck.retrieval.dense import Hit, hits_of


class PostingBM25:
    """BM25Okapi scores from posting lists; same results as `claimcheck.index.bm25.top_k`."""

    def __init__(self, bm25) -> None:
        rows: dict[str, list[int]] = {}
        tfs: dict[str, list[int]] = {}
        for row, freqs in enumerate(bm25.doc_freqs):
            for term, tf in freqs.items():
                rows.setdefault(term, []).append(row)
                tfs.setdefault(term, []).append(tf)
        self.postings = {
            t: (np.asarray(rows[t], dtype=np.int64), np.asarray(tfs[t], dtype=np.int64))
            for t in rows
        }
        self.idf: dict[str, float] = bm25.idf
        self.k1 = bm25.k1
        self.n = bm25.corpus_size
        doc_len = np.array(bm25.doc_len)
        # As in BM25Okapi.get_scores: k1 * (1 - b + b * doc_len / avgdl), per row.
        self.norm = bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)

    def scores(self, tokens: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """(score of every row, rows that share a token with the query)."""
        score = np.zeros(self.n)
        for q in tokens:  # every token, repeats included, in order (as get_scores)
            post = self.postings.get(q)
            if post is None:
                continue
            rows, tf = post
            score[rows] += (self.idf.get(q) or 0) * (tf * (self.k1 + 1) / (tf + self.norm[rows]))
        posted = [self.postings[q][0] for q in dict.fromkeys(tokens) if q in self.postings]
        shared = np.unique(np.concatenate(posted)) if posted else np.zeros(0, dtype=np.int64)
        return score, shared

    def top_k(self, tokens: Sequence[str], k: int) -> list[tuple[int, float]]:
        """The k best rows as (row, score), best first, ties by row."""
        if not tokens or k <= 0:
            return []
        score, hits = self.scores(tokens)
        if hits.size == 0:
            return []
        order = hits[np.lexsort((hits, -score[hits]))][:k]
        return [(int(i), float(score[i])) for i in order]


class SparseRetriever:
    def __init__(self, store: IndexStore) -> None:
        self.store = store
        self.bm25 = PostingBM25(store.bm25)

    def tokens(self, query: str) -> list[str]:
        return query_tokens(query)

    def search(self, query: str, k: int) -> list[Hit]:
        return hits_of(self.store, self.bm25.top_k(self.tokens(query), k))
