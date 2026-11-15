"""The BM25 tokenizer and the sparse ranking built on it."""

import pytest

from claimcheck.index.bm25 import build_bm25, query_tokens, tokenize, top_k


@pytest.mark.parametrize("text", ["FY2025-26", "2025-26", "FY26", "FY2026", "FY 2025-26"])
def test_fiscal_year_forms_yield_the_canonical_token(text: str) -> None:
    assert "fy2026" in tokenize(text)


@pytest.mark.parametrize(
    ("text", "raw"),
    [("FY2025-26", "fy2025-26"), ("2025-26", "2025-26"), ("FY26", "fy26")],
)
def test_the_raw_fiscal_year_token_is_kept_alongside(text: str, raw: str) -> None:
    assert tokenize(text) == [raw, "fy2026"]


def test_thousands_separators_are_stripped_and_decimals_kept() -> None:
    assert tokenize("1,234.5") == ["1234.5"]
    assert tokenize("Rs 12,34,567 crore and 3.7%") == [
        "rs",
        "1234567",
        "crore",
        "and",
        "3.7",
        "percent",
    ]


def test_lowercase_and_other_fiscal_years() -> None:
    assert tokenize("GDP Growth in FY2024-25") == ["gdp", "growth", "in", "fy2024-25", "fy2025"]


def test_queries_and_documents_share_the_tokenizer() -> None:
    assert query_tokens("Inflation FY26 was 4.5%") == tokenize("Inflation FY26 was 4.5%")


DOCS = [
    "Real GDP grew 2.7 percent in 2024-25 against 2.5 percent in 2023-24.",
    "Headline inflation averaged 4.5% in FY2025-26 (July-March).",
    "Exports reached US$ 1,234.5 million in FY26.",
    "The weather was pleasant.",
    "Large scale manufacturing declined.",
    "Remittances rose sharply.",
]


def test_any_fiscal_year_form_finds_the_same_documents() -> None:
    bm25 = build_bm25([tokenize(d) for d in DOCS])
    for q in ("FY2025-26", "2025-26", "FY26"):
        rows = [r for r, _ in top_k(bm25, query_tokens(q), 10)]
        assert set(rows) == {1, 2}, q


def test_a_grouped_number_matches_its_plain_form() -> None:
    bm25 = build_bm25([tokenize(d) for d in DOCS])
    assert top_k(bm25, query_tokens("1234.5"), 1)[0][0] == 2


def test_top_k_leaves_out_unmatched_rows_and_breaks_ties_by_row() -> None:
    bm25 = build_bm25([["a", "x"], ["b", "y"], ["a", "x"], ["c", "z"], ["d"], ["e"]])
    assert [r for r, _ in top_k(bm25, ["a"], 10)] == [0, 2]
    assert [r for r, _ in top_k(bm25, ["b", "a"], 10)] == [1, 0, 2]
    assert top_k(bm25, ["nothing"], 10) == []
    assert top_k(bm25, [], 10) == []
    assert len(top_k(bm25, ["a"], 1)) == 1


def test_a_shared_token_with_zero_idf_still_counts_as_a_match() -> None:
    bm25 = build_bm25([["a"], ["a"], ["b"], ["c"]])  # 'a' in half the documents: idf 0
    assert [r for r, _ in top_k(bm25, ["a"], 10)] == [0, 1]


def test_bm25_needs_documents() -> None:
    with pytest.raises(ValueError):
        build_bm25([])
