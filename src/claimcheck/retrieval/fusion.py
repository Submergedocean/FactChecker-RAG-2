"""Fusion of ranked lists: Reciprocal Rank Fusion and a round-robin interleave.

Inputs are ranked lists of chunk ids, best first, one per method (`"bm25"`, `"dense"`); ranks
are 1-based. Both functions are pure and deterministic.

- `rrf`: `score(d) = Σ_m 1/(k + rank_m(d))` over the lists that hold `d` (k = 60); higher is
  better; equal scores are ordered by `chunk_id` (ascending).
- `interleave`: the fusion ablation (`hybrid_interleave`): take rank 1 of each list in
  the given method order, then rank 2, and so on, skipping ids already taken.

Each fused item carries every method's rank (None when the list does not hold it) and its RRF
score, so all modes report the same fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

RRF_K = 60


@dataclass(frozen=True)
class Fused:
    chunk_id: str
    score: float  # RRF score over all lists given
    ranks: dict[str, int | None]  # 1-based rank in each list, None if absent


def _rank_maps(rankings: Mapping[str, Sequence[str]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for method, ids in rankings.items():
        ranks: dict[str, int] = {}
        for i, cid in enumerate(ids, start=1):
            if cid in ranks:
                raise ValueError(f"{method}: {cid} appears twice")
            ranks[cid] = i
        out[method] = ranks
    return out


def rrf_score(ranks: Mapping[str, int | None], k: int = RRF_K) -> float:
    """Σ 1/(k + rank) over the methods that ranked the item."""
    return sum(1.0 / (k + r) for r in ranks.values() if r is not None)


def _fused(cid: str, maps: dict[str, dict[str, int]], k: int) -> Fused:
    ranks = {m: ranks_of.get(cid) for m, ranks_of in maps.items()}
    return Fused(chunk_id=cid, score=rrf_score(ranks, k), ranks=ranks)


def rrf(rankings: Mapping[str, Sequence[str]], *, k: int = RRF_K) -> list[Fused]:
    """Every id in any list, by RRF score (highest first), ties by `chunk_id`."""
    if k < 0:
        raise ValueError("RRF k must be non-negative")
    maps = _rank_maps(rankings)
    ids = {cid for ranks in maps.values() for cid in ranks}
    items = [_fused(cid, maps, k) for cid in ids]
    items.sort(key=lambda f: (-f.score, f.chunk_id))
    return items


def interleave(rankings: Mapping[str, Sequence[str]], *, k: int = RRF_K) -> list[Fused]:
    """Round-robin merge in the order of `rankings`, duplicates skipped (`score` is still RRF)."""
    maps = _rank_maps(rankings)
    seen: set[str] = set()
    out: list[Fused] = []
    depth = max((len(ids) for ids in rankings.values()), default=0)
    for i in range(depth):
        for ids in rankings.values():
            if i < len(ids) and ids[i] not in seen:
                seen.add(ids[i])
                out.append(_fused(ids[i], maps, k))
    return out
