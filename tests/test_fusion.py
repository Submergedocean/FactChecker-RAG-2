"""RRF and round-robin fusion against hand-computed examples."""

import pytest

from claimcheck.retrieval.fusion import interleave, rrf, rrf_score


def test_rrf_matches_a_hand_computed_example() -> None:
    # bm25: a b c   dense: c a d   (k = 60)
    # a = 1/61 + 1/62 = 0.0325224...   c = 1/63 + 1/61 = 0.0322664...
    # b = 1/62        = 0.0161290...   d = 1/63        = 0.0158730...
    out = rrf({"bm25": ["a", "b", "c"], "dense": ["c", "a", "d"]})
    assert [f.chunk_id for f in out] == ["a", "c", "b", "d"]
    assert [f.score for f in out] == pytest.approx(
        [0.0325224749, 0.0322664585, 0.0161290323, 0.0158730159]
    )
    assert out[0].ranks == {"bm25": 1, "dense": 2}
    assert out[1].ranks == {"bm25": 3, "dense": 1}
    assert out[2].ranks == {"bm25": 2, "dense": None}
    assert out[3].ranks == {"bm25": None, "dense": 3}


def test_rrf_uses_the_given_k() -> None:
    out = rrf({"bm25": ["a"], "dense": ["b", "a"]}, k=0)
    # a = 1/1 + 1/2 = 1.5, b = 1/1
    assert [(f.chunk_id, f.score) for f in out] == [("a", 1.5), ("b", 1.0)]


def test_ties_are_broken_by_chunk_id_whatever_the_input_order() -> None:
    # x and y swap ranks: equal scores (1/61 + 1/62).
    one = rrf({"bm25": ["y", "x"], "dense": ["x", "y"]})
    two = rrf({"dense": ["y", "x"], "bm25": ["x", "y"]})
    assert [f.chunk_id for f in one] == [f.chunk_id for f in two] == ["x", "y"]
    assert one[0].score == one[1].score
    # Single-list items at the same rank tie too.
    assert [f.chunk_id for f in rrf({"bm25": ["b9"], "dense": ["a1"]})] == ["a1", "b9"]


def test_rrf_is_deterministic_and_keeps_every_candidate() -> None:
    lists = {
        "bm25": [f"{i:02d}" for i in range(50)],
        "dense": [f"{i:02d}" for i in range(99, 49, -1)],
    }
    first = rrf(lists)
    assert first == rrf(lists)
    assert len(first) == 100
    assert all(a.score >= b.score for a, b in zip(first, first[1:], strict=False))


def test_rrf_of_one_list_keeps_its_order() -> None:
    ids = ["c", "a", "b"]
    out = rrf({"dense": ids})
    assert [f.chunk_id for f in out] == ids
    assert out[0].score == pytest.approx(1 / 61)


def test_rrf_score_skips_absent_methods() -> None:
    assert rrf_score({"bm25": 2, "dense": None}) == pytest.approx(1 / 62)
    assert rrf_score({}) == 0.0


def test_a_duplicate_within_one_list_is_an_error() -> None:
    with pytest.raises(ValueError, match="twice"):
        rrf({"bm25": ["a", "a"]})


def test_empty_lists_fuse_to_nothing() -> None:
    assert rrf({"bm25": [], "dense": []}) == []
    assert interleave({"dense": [], "bm25": []}) == []


def test_interleave_is_round_robin_in_method_order_skipping_duplicates() -> None:
    out = interleave({"dense": ["c", "a", "d"], "bm25": ["a", "b", "c"]})
    # round 1: dense c, bm25 a; round 2: dense a (taken), bm25 b; round 3: dense d, bm25 c (taken)
    assert [f.chunk_id for f in out] == ["c", "a", "b", "d"]
    assert out[0].ranks == {"dense": 1, "bm25": 3}
    assert out[0].score == pytest.approx(1 / 61 + 1 / 63)


def test_interleave_handles_lists_of_different_lengths() -> None:
    out = interleave({"dense": ["a"], "bm25": ["b", "c", "d"]})
    assert [f.chunk_id for f in out] == ["a", "b", "c", "d"]
