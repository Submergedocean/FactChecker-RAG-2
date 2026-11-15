"""Cohen's kappa and the blind second-annotator sample."""

from __future__ import annotations

import pytest

from claimcheck.claims.kappa import cohen_kappa, kappa_of, kappa_sample
from tests.claims_fixtures import claim


def test_perfect_and_chance_agreement() -> None:
    labels = ["SUPPORTED", "REFUTED", "NEI"] * 4
    assert cohen_kappa(labels, labels).kappa == pytest.approx(1.0)
    r = cohen_kappa(["SUPPORTED"] * 4 + ["REFUTED"] * 4, ["SUPPORTED", "REFUTED"] * 4)
    assert r.observed == pytest.approx(0.5) and r.kappa == pytest.approx(0.0)


def test_known_value() -> None:
    # 2x2 textbook example: a=20, b=5, c=10, d=15 -> po=0.70, pe=0.50, kappa=0.40
    first = ["SUPPORTED"] * 25 + ["REFUTED"] * 25
    second = ["SUPPORTED"] * 20 + ["REFUTED"] * 5 + ["SUPPORTED"] * 10 + ["REFUTED"] * 15
    r = cohen_kappa(first, second)
    assert (r.observed, r.expected) == (pytest.approx(0.7), pytest.approx(0.5))
    assert r.kappa == pytest.approx(0.4)
    assert r.matrix["SUPPORTED"]["REFUTED"] == 5 and r.lines()[0].startswith("pairs 50")


def test_errors() -> None:
    with pytest.raises(ValueError):
        cohen_kappa([], [])
    with pytest.raises(ValueError):
        cohen_kappa(["NEI"], [])


def test_kappa_of_claims_uses_only_double_labelled() -> None:
    claims = [claim(1, second_label="SUPPORTED"), claim(2, second_label="NEI"), claim(3)]
    r = kappa_of(claims)
    assert r.n == 2 and r.observed == pytest.approx(0.5)


def test_sample_is_english_stratified_one_per_fact() -> None:
    claims = []
    for i in range(1, 41):  # 20 facts with a SUPPORTED and a REFUTED claim
        fact = f"F{(i + 1) // 2:04d}"
        label = "SUPPORTED" if i % 2 else "REFUTED"
        claims.append(claim(i, fact_id=fact, label=label, claim_text=f"Claim {i} text here."))
    for i in range(41, 51):
        claims.append(claim(i, label="NEI", claim_text=f"Claim {i} text here."))
    claims.append(
        claim(
            51, language="ur", parallel_of="C0001", fact_id="F0001", claim_text="اردو دعویٰ نمبر 51"
        )
    )
    claims.append(claim(52, second_label="NEI", claim_text="Already double labelled."))
    picked = kappa_sample(claims, 20, seed=7)
    assert all(c.language.value == "en" and c.second_label is None for c in picked)
    assert len({c.fact_id for c in picked}) == len(picked)
    labels = [c.label.value for c in picked]
    assert labels.count("NEI") == 4  # 10 of 50 eligible -> 20 %
    assert picked == kappa_sample(claims, 20, seed=7)
    assert [c.claim_id for c in picked] == sorted(c.claim_id for c in picked)
