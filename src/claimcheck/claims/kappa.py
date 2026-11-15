"""Inter-annotator agreement (proposal 7.2: Cohen's kappa on about 100 claims, target 0.70).

Workflow: `claims kappa-sample` writes a blind CSV (claim text, language, topic; no label, no
evidence) for the second annotator, stratified by the first label; `claims kappa-import`
stores their labels in `second_label`; `claims kappa` reports kappa over every claim that has
both labels.

The sample is English only: a translation carries its original's label by construction, so
including it would count one judgement twice.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from claimcheck.claims.schema import Claim, ClaimLanguage, Verdict

LABELS: tuple[str, ...] = tuple(v.value for v in Verdict)
SAMPLE_COLUMNS: tuple[str, ...] = (
    "claim_id",
    "claim_text",
    "language",
    "topic",
    "second_label",
    "second_annotator",
    "notes",
)


@dataclass(frozen=True)
class KappaResult:
    n: int
    observed: float  # share of agreement
    expected: float  # agreement expected by chance
    kappa: float
    matrix: dict[str, dict[str, int]]  # first label -> second label -> count

    def lines(self) -> list[str]:
        head = "first \\ second  " + "  ".join(f"{lab:>9}" for lab in LABELS)
        rows = [f"{a:>14}  " + "  ".join(f"{self.matrix[a][b]:>9}" for b in LABELS) for a in LABELS]
        return [
            f"pairs {self.n}  observed {self.observed:.3f}  chance {self.expected:.3f}  "
            f"kappa {self.kappa:.3f}",
            head,
            *rows,
        ]


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> KappaResult:
    """Cohen's kappa of two label sequences of equal length."""
    if len(first) != len(second):
        raise ValueError("label sequences differ in length")
    n = len(first)
    if n == 0:
        raise ValueError("no label pairs")
    matrix = {a: dict.fromkeys(LABELS, 0) for a in LABELS}
    for a, b in zip(first, second, strict=True):
        matrix[a][b] += 1
    observed = sum(matrix[a][a] for a in LABELS) / n
    p1 = Counter(first)
    p2 = Counter(second)
    expected = sum(p1[lab] * p2[lab] for lab in LABELS) / (n * n)
    if expected >= 1.0:  # both annotators used one label throughout
        kappa = 1.0 if observed == 1.0 else 0.0
    else:
        kappa = (observed - expected) / (1 - expected)
    return KappaResult(n, observed, expected, kappa, matrix)


def kappa_of(claims: Sequence[Claim]) -> KappaResult:
    pairs = [(c.label.value, c.second_label.value) for c in claims if c.second_label is not None]
    return cohen_kappa([a for a, _ in pairs], [b for _, b in pairs])


def kappa_sample(claims: Sequence[Claim], n: int, *, seed: int) -> list[Claim]:
    """Up to `n` English claims without a second label, stratified by label in proportion to
    the dataset (largest remainder), at most one claim per fact, in claim-id order."""
    pool = [c for c in claims if c.language is ClaimLanguage.EN and c.second_label is None]
    rng = random.Random(seed)
    by_label: dict[str, list[Claim]] = defaultdict(list)
    for c in sorted(pool, key=lambda c: c.claim_id):
        by_label[c.label.value].append(c)
    for group in by_label.values():
        rng.shuffle(group)
    total = len(pool)
    if total == 0 or n <= 0:
        return []
    n = min(n, total)
    exact = {lab: n * len(g) / total for lab, g in by_label.items()}
    counts = {lab: int(x) for lab, x in exact.items()}
    for lab in sorted(exact, key=lambda k: -(exact[k] - counts[k]))[: n - sum(counts.values())]:
        counts[lab] += 1
    picked: list[Claim] = []
    used_facts: set[str] = set()
    for lab in sorted(by_label):
        taken = 0
        for c in by_label[lab]:
            if taken >= counts[lab]:
                break
            if c.fact_id in used_facts:
                continue
            picked.append(c)
            used_facts.add(c.fact_id)
            taken += 1
    return sorted(picked, key=lambda c: c.claim_id)
