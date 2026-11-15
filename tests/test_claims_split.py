"""Fact-level splits (70/15/15, stratified, frozen, no fact leakage)."""

from __future__ import annotations

from collections import Counter

from claimcheck.claims.split import (
    fact_strata,
    leaked_facts,
    make_splits,
    split_claims,
    split_summary,
)
from claimcheck.schemas import Split
from tests.claims_fixtures import claim

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def dataset(n_pairs: int = 160, n_nei: int = 80, translate_every: int = 3) -> list:
    """Facts with a SUPPORTED claim, a REFUTED twin and sometimes translations, plus NEI."""
    out, i = [], 0
    for f in range(1, n_pairs + 1):
        fact = f"F{f:04d}"
        i += 1
        sup = f"C{i:04d}"
        out.append(claim(i, fact_id=fact, claim_text=f"Fact {f} supported."))
        i += 1
        out.append(claim(i, fact_id=fact, label="REFUTED", claim_text=f"Fact {f} refuted."))
        if f % translate_every == 0:
            i += 1
            out.append(
                claim(i, fact_id=fact, language="ur", parallel_of=sup, claim_text=f"حقیقت {f}")
            )
    for f in range(n_pairs + 1, n_pairs + n_nei + 1):
        i += 1
        out.append(claim(i, fact_id=f"F{f:04d}", label="NEI", claim_text=f"Fact {f} absent."))
    return out


def test_no_fact_in_two_splits() -> None:
    claims = dataset()
    splits = make_splits(claims, RATIOS, seed=1)
    groups = split_claims(claims, splits)
    facts = {s: {c.fact_id for c in cs} for s, cs in groups.items()}
    assert not facts[Split.TRAIN] & facts[Split.VAL]
    assert not facts[Split.TRAIN] & facts[Split.TEST]
    assert not facts[Split.VAL] & facts[Split.TEST]
    assert set().union(*facts.values()) == {c.fact_id for c in claims}
    assert leaked_facts(claims, splits) == []
    # every claim of a fact (twin, translation) sits with its fact
    for s, cs in groups.items():
        assert all(splits[c.fact_id] is s for c in cs)


def test_ratios_and_stratification() -> None:
    claims = dataset()
    splits = make_splits(claims, RATIOS, seed=1)
    strata = fact_strata(claims)
    for stratum in set(strata.values()):
        facts = [f for f, s in strata.items() if s == stratum]
        counts = Counter(splits[f] for f in facts)
        n = len(facts)
        for split, ratio in RATIOS.items():
            assert abs(counts[Split(split)] - n * ratio) <= 1, (stratum, split, counts)
    assert set(strata.values()) == {"REFUTED+SUPPORTED", "NEI"}


def test_deterministic_and_seeded() -> None:
    claims = dataset()
    assert make_splits(claims, RATIOS, seed=1) == make_splits(
        list(reversed(claims)), RATIOS, seed=1
    )
    assert make_splits(claims, RATIOS, seed=1) != make_splits(claims, RATIOS, seed=2)


def test_frozen_extension_keeps_existing_assignments() -> None:
    claims = dataset(n_pairs=100, n_nei=40)
    first = make_splits(claims, RATIOS, seed=1)
    more = dataset(n_pairs=140, n_nei=60)
    # dataset() numbers facts in order, so the first 100 pair facts are unchanged; NEI ids shift
    second = make_splits(more, RATIOS, seed=1, existing=first)
    assert all(second[f] is s for f, s in first.items())
    assert set(second) == set(first) | {c.fact_id for c in more}
    pair_facts = [f for f, s in fact_strata(more).items() if s == "REFUTED+SUPPORTED"]
    counts = Counter(second[f] for f in pair_facts)
    assert abs(counts[Split.TRAIN] / len(pair_facts) - 0.70) < 0.05


def test_claims_of_unsplit_facts_are_left_out() -> None:
    claims = dataset(n_pairs=2, n_nei=0)
    groups = split_claims(claims, {"F0001": Split.TRAIN})
    assert {c.fact_id for c in groups[Split.TRAIN]} == {"F0001"}
    assert groups[Split.VAL] == groups[Split.TEST] == []


def test_summary_lines() -> None:
    claims = dataset(n_pairs=20, n_nei=10)
    lines = split_summary(claims, make_splits(claims, RATIOS, seed=3))
    assert [ln.split()[0] for ln in lines] == ["train", "val", "test"]
