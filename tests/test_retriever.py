"""The Retriever over a small index built offline (hashing embedder, no model)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from claimcheck.config import ChunkingSettings, RetrievalSettings
from claimcheck.index.build import build_index
from claimcheck.index.store import IndexStore
from claimcheck.ingest.chunk import write_chunks
from claimcheck.retrieval import cli as retrieval_cli
from claimcheck.retrieval.fusion import interleave, rrf
from claimcheck.retrieval.retriever import Mode, Retriever
from claimcheck.schemas import ChunkKind, EvidencePassage
from tests.test_index_store import EMB, TEXTS, HashEmbedder, make_chunks, paths_in

MORE = [
    "Table 2.4: Inflation | CPI percent | 2024-25 4.6 | 2025-26 4.5",
    "Agriculture grew 1.5 percent while services expanded.",
    "Workers' remittances rose to 38.0 billion dollars.",
]
SETTINGS = RetrievalSettings(top_k=3, bm25_k=4, dense_k=4, rrf_k=60)


@pytest.fixture(scope="module")
def retriever(tmp_path_factory: pytest.TempPathFactory) -> Retriever:
    p = paths_in(tmp_path_factory.mktemp("idx"))
    write_chunks(make_chunks(TEXTS + MORE), p.chunks)
    p.chunk_report.write_text(json.dumps({"chunking": {}}))
    emb = HashEmbedder()
    build_index("table_aware", p, emb, chunking=ChunkingSettings())
    store = IndexStore.load("table_aware", paths=p, embedding=EMB)
    return Retriever(store, emb, SETTINGS)


QUERY = "Headline inflation was 4.5 percent in FY26"


def test_hybrid_is_rrf_of_the_two_candidate_lists(retriever: Retriever) -> None:
    res = retriever.retrieve(QUERY)
    dense = [h.chunk_id for h in retriever.dense.search(QUERY, SETTINGS.dense_k)]
    sparse = [h.chunk_id for h in retriever.sparse.search(QUERY, SETTINGS.bm25_k)]
    want = rrf({"bm25": sparse, "dense": dense})[: SETTINGS.top_k]
    assert [p.chunk_id for p in res.passages] == [f.chunk_id for f in want]
    assert [p.rrf for p in res.passages] == [f.score for f in want]
    assert [p.rank for p in res.passages] == [1, 2, 3]
    assert res.mode is Mode.HYBRID
    assert res.candidates == {"bm25": len(sparse), "dense": len(dense)}
    for p in res.passages:
        assert list(p.ranks) == ["bm25", "dense"]
        assert p.ranks["dense"] == (dense.index(p.chunk_id) + 1 if p.chunk_id in dense else None)
        assert p.ranks["bm25"] == (sparse.index(p.chunk_id) + 1 if p.chunk_id in sparse else None)


def test_fiscal_year_forms_reach_the_same_passages(retriever: Retriever) -> None:
    ids = {
        q: [p.chunk_id for p in retriever.retrieve(f"inflation {q}", mode="sparse").passages]
        for q in ("FY2025-26", "2025-26", "FY26", "FY2026")
    }
    # Same passages for every form (order can differ: the raw token is kept too).
    assert len({frozenset(v) for v in ids.values()}) == 1
    texts = [p.chunk.text for p in retriever.retrieve("inflation FY26", mode="sparse").passages]
    assert any("2025-26" in t for t in texts)


def test_single_modes_report_only_their_method(retriever: Retriever) -> None:
    dense = retriever.retrieve(QUERY, mode="dense")
    assert [p.chunk_id for p in dense.passages] == [
        h.chunk_id for h in retriever.dense.search(QUERY, 3)
    ]
    assert all(list(p.ranks) == ["dense"] for p in dense.passages)
    assert [p.ranks["dense"] for p in dense.passages] == [1, 2, 3]
    assert "bm25" not in dense.timings_ms

    sparse = retriever.retrieve(QUERY, mode=Mode.SPARSE)
    assert all(list(p.ranks) == ["bm25"] for p in sparse.passages)
    assert [p.ranks["bm25"] for p in sparse.passages] == [1, 2, 3]
    assert "encode" not in sparse.timings_ms


def test_interleave_mode_is_round_robin_dense_first(retriever: Retriever) -> None:
    res = retriever.retrieve(QUERY, mode="interleave")
    dense = [h.chunk_id for h in retriever.dense.search(QUERY, SETTINGS.dense_k)]
    sparse = [h.chunk_id for h in retriever.sparse.search(QUERY, SETTINGS.bm25_k)]
    want = interleave({"dense": dense, "bm25": sparse})[: SETTINGS.top_k]
    assert [p.chunk_id for p in res.passages] == [f.chunk_id for f in want]
    assert res.passages[0].chunk_id == dense[0]
    assert list(res.passages[0].ranks) == ["bm25", "dense"]


def test_k_above_the_candidate_depth_widens_the_lists(retriever: Retriever) -> None:
    res = retriever.retrieve("percent", mode="dense", k=7)
    assert len(res.passages) == 7
    assert res.candidates == {"dense": 7}


def test_results_are_deterministic(retriever: Retriever) -> None:
    a = retriever.retrieve(QUERY)
    b = retriever.retrieve(QUERY)
    assert [(p.chunk_id, p.rrf, p.ranks) for p in a.passages] == [
        (p.chunk_id, p.rrf, p.ranks) for p in b.passages
    ]
    assert a.index_version == retriever.store.index_version


def test_scores_and_evidence_fields(retriever: Retriever) -> None:
    res = retriever.retrieve(QUERY)
    p = res.passages[0]
    if p.ranks["dense"] is not None:
        assert -1.0 <= p.scores["dense"] <= 1.0 + 1e-6
    ev = p.to_evidence(cited=True)
    assert isinstance(ev, EvidencePassage)
    assert ev.chunk_id == p.chunk.chunk_id and ev.rank == 1 and ev.cited
    assert ev.chapter == p.chunk.chapter_title and ev.kind is ChunkKind.TEXT
    assert ev.ranks == p.ranks and ev.rrf == p.rrf
    assert p.covers(p.chunk.doc_id, p.chunk.page_start)
    assert not p.covers(p.chunk.doc_id, p.chunk.page_end + 1)
    assert not p.covers("pes2425_ch01", p.chunk.page_start)
    assert res.timings_ms["total"] >= res.timings_ms["fuse"] >= 0


def test_a_query_sharing_no_token_still_gets_dense_results(retriever: Retriever) -> None:
    res = retriever.retrieve("zzzqqq", mode="hybrid")
    assert res.candidates["bm25"] == 0
    assert len(res.passages) == 3
    assert all(p.ranks["bm25"] is None for p in res.passages)


@pytest.mark.parametrize(
    ("query", "kwargs", "message"),
    [
        ("  ", {}, "empty query"),
        ("gdp", {"k": 0}, "at least 1"),
        ("gdp", {"mode": "x"}, "is not a valid Mode"),
    ],
)
def test_bad_arguments_are_refused(retriever: Retriever, query, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        retriever.retrieve(query, **kwargs)


def test_an_embedder_that_is_not_the_indexs_is_refused(retriever: Retriever) -> None:
    other = HashEmbedder()
    other.revision = "1" * 40
    with pytest.raises(ValueError, match="is not the index's"):
        Retriever(retriever.store, other)
    other = HashEmbedder()
    other.query_prefix = "query: "
    with pytest.raises(ValueError, match="query prefix"):
        Retriever(retriever.store, other)


# ------------------------------------------------------------------ CLI


def test_cli_prints_ranked_passages_with_both_ranks(
    retriever: Retriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Retriever, "from_settings", classmethod(lambda cls, v: retriever))
    runner = CliRunner()
    res = runner.invoke(retrieval_cli.app, ["retrieve", QUERY, "--mode", "hybrid", "-k", "3"])
    assert res.exit_code == 0, res.output
    lines = [ln for ln in res.output.splitlines() if ln.startswith("#")]
    assert len(lines) == 3
    assert all("bm25" in ln and "dense" in ln and "rrf" in ln for ln in lines)
    assert "candidates bm25" in res.output and "total" in res.output

    js = runner.invoke(retrieval_cli.app, ["retrieve", QUERY, "--json", "-k", "2"])
    assert js.exit_code == 0, js.output
    data = json.loads(js.output)
    assert len(data["passages"]) == 2 and data["mode"] == "hybrid"
    assert set(data["passages"][0]["ranks"]) == {"bm25", "dense"}


@pytest.mark.parametrize(
    "args", [["q", "--mode", "nope"], ["q", "--variant", "nope"], ["q", "-k", "0"], ["  "]]
)
def test_cli_refuses_bad_options(args: list[str]) -> None:
    res = CliRunner().invoke(retrieval_cli.app, ["retrieve", *args])
    assert res.exit_code == 2


def test_cli_reports_a_missing_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(cls, v):
        raise FileNotFoundError("faiss.index not found")

    monkeypatch.setattr(Retriever, "from_settings", classmethod(missing))
    res = CliRunner().invoke(retrieval_cli.app, ["retrieve", "gdp growth"])
    assert res.exit_code == 1
    assert "NOT LOADED" in res.output


# ------------------------------------------------------------------ posting-list BM25


@pytest.mark.parametrize(
    "query",
    [
        QUERY,
        "inflation inflation 2025-26 percent",  # repeated token
        "Remittances US$ 38 billion FY2026",
        "Table 7.1 exports Rs billion 1,234.5",
        "zzzqqq",
        "the percent",
        "",
    ],
)
def test_posting_bm25_equals_the_index_bm25_exactly(retriever: Retriever, query: str) -> None:
    tokens = retriever.sparse.tokens(query)
    for k in (1, 3, 50):
        assert retriever.sparse.bm25.top_k(tokens, k) == retriever.store.sparse_search(tokens, k)
    if tokens:
        scores, _ = retriever.sparse.bm25.scores(tokens)
        assert scores.tolist() == retriever.store.bm25.get_scores(tokens).tolist()
