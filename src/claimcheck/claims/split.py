"""`claimcheck claims split`: `fact_id` -> train / val / test, 70/15/15.

Splits are made over facts, never claims, so a claim, its REFUTED twin and its translations
always land in the same split (no fact leakage). Facts are stratified by their label profile
(the sorted set of labels among the fact's claims, e.g. `REFUTED+SUPPORTED` or `NEI`), and
within each stratum the facts are shuffled with a fixed seed and dealt out by largest
remainder, so each stratum is 70/15/15 up to rounding.

`splits.json` is generated once and then frozen: a later run keeps every existing assignment
and only places new facts, each into the split furthest below its target share in its
stratum. `--force` regenerates from scratch (only before any evaluation has used it).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from claimcheck.claims.schema import Claim
from claimcheck.schemas import Split

SPLIT_ORDER: tuple[Split, ...] = (Split.TRAIN, Split.VAL, Split.TEST)


def fact_strata(claims: Sequence[Claim]) -> dict[str, str]:
    """fact_id -> stratum key (sorted labels of the fact's claims, joined by '+')."""
    labels: dict[str, set[str]] = defaultdict(set)
    for c in claims:
        labels[c.fact_id].add(c.label.value)
    return {fact: "+".join(sorted(ls)) for fact, ls in labels.items()}


def _ratios(ratios: Mapping[str, float]) -> dict[Split, float]:
    out = {Split(k): float(v) for k, v in ratios.items()}
    total = sum(out.values())
    if set(out) != set(SPLIT_ORDER) or total <= 0:
        raise ValueError("split ratios must name train, val and test")
    return {s: out[s] / total for s in SPLIT_ORDER}


def _allocate(n: int, ratios: dict[Split, float]) -> dict[Split, int]:
    """Largest-remainder counts summing to n (ties go to the earlier split)."""
    exact = {s: n * ratios[s] for s in SPLIT_ORDER}
    counts = {s: int(exact[s]) for s in SPLIT_ORDER}
    left = n - sum(counts.values())
    by_remainder = sorted(
        SPLIT_ORDER, key=lambda s: (-(exact[s] - counts[s]), SPLIT_ORDER.index(s))
    )
    for s in by_remainder[:left]:
        counts[s] += 1
    return counts


def make_splits(
    claims: Sequence[Claim],
    ratios: Mapping[str, float],
    *,
    seed: int,
    existing: Mapping[str, Split] | None = None,
) -> dict[str, Split]:
    """The split of every fact in `claims`. Facts in `existing` keep their split; facts in
    `existing` that no longer have claims are kept too (the file is frozen)."""
    r = _ratios(ratios)
    strata = fact_strata(claims)
    out: dict[str, Split] = dict(existing or {})
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for fact, stratum in strata.items():
        by_stratum[stratum].append(fact)

    for stratum in sorted(by_stratum):
        facts = sorted(by_stratum[stratum])
        new = [f for f in facts if f not in out]
        if not new:
            continue
        rng = random.Random(f"{seed}:{stratum}")
        rng.shuffle(new)
        if len(new) == len(facts):  # a fresh stratum: deal out by largest remainder
            counts = _allocate(len(new), r)
            i = 0
            for s in SPLIT_ORDER:
                for f in new[i : i + counts[s]]:
                    out[f] = s
                i += counts[s]
            continue
        # A frozen stratum gaining facts: each goes where the stratum is furthest below target.
        have = Counter(out[f] for f in facts if f in out)
        for f in new:
            total = sum(have.values()) + 1
            s = max(SPLIT_ORDER, key=lambda s: (r[s] * total - have[s], -SPLIT_ORDER.index(s)))
            out[f] = s
            have[s] += 1
    return out


def leaked_facts(claims: Sequence[Claim], splits: Mapping[str, Split]) -> list[str]:
    """Facts whose claims would sit in more than one split (always empty for a fact-level
    mapping; kept as the check the gate runs on the materialised splits)."""
    seen: dict[str, set[Split]] = defaultdict(set)
    for c in claims:
        if c.fact_id in splits:
            seen[c.fact_id].add(splits[c.fact_id])
    return sorted(f for f, s in seen.items() if len(s) > 1)


def split_claims(claims: Sequence[Claim], splits: Mapping[str, Split]) -> dict[Split, list[Claim]]:
    """Claims grouped by split; claims whose fact has no split are left out."""
    out: dict[Split, list[Claim]] = {s: [] for s in SPLIT_ORDER}
    for c in claims:
        s = splits.get(c.fact_id)
        if s is not None:
            out[s].append(c)
    return out


def split_summary(claims: Sequence[Claim], splits: Mapping[str, Split]) -> list[str]:
    """One line per split: facts, claims and labels."""
    groups = split_claims(claims, splits)
    lines = []
    n_all = sum(len(v) for v in groups.values()) or 1
    for s in SPLIT_ORDER:
        cs = groups[s]
        labels = Counter(c.label.value for c in cs)
        facts = len({c.fact_id for c in cs})
        lines.append(
            f"{s.value:5}  facts {facts:4}  claims {len(cs):4} ({len(cs) / n_all:.0%})  "
            + ", ".join(f"{k} {v}" for k, v in sorted(labels.items()))
        )
    return lines
