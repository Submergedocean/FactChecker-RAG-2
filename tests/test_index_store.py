"""Index build and IndexStore.load checks, offline (a hashing embedder, no model)."""

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from claimcheck.config import ChunkingSettings, EmbeddingSettings
from claimcheck.index.build import build_index, index_version
from claimcheck.index.embed import EmbeddingCache, embed_passages
from claimcheck.index.store import (
    IndexMismatchError,
    IndexPaths,
    IndexStore,
    read_meta,
)
from claimcheck.ingest.chunk import write_chunks
from claimcheck.schemas import ChunkKind, ChunkRecord, Variant
from claimcheck.textnorm import tokenize

DIM = 384
REV = "0" * 40
EMB = EmbeddingSettings(model="fake/hash-embedder", revision=REV, dim=DIM)


class HashEmbedder:
    """Bag of hashed words, L2-normalised; counts the texts it embeds."""

    model = EMB.model
    revision = EMB.revision
    dim = DIM
    normalize = True
    query_prefix = EMB.query_prefix
    passage_prefix = EMB.passage_prefix

    def __init__(self) -> None:
        self.n_encoded = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v / (np.linalg.norm(v) or 1.0)

    def encode_passages(self, texts):
        self.n_encoded += len(texts)
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, DIM), "f4")

    def encode_queries(self, texts):
        return np.stack([self._vec(t) for t in texts])


TEXTS = [
    "Real GDP grew 2.7 percent in FY2025 led by agriculture.",
    "Headline inflation averaged 4.5 percent in 2025-26.",
    "Table 7.1: Exports | Rs billion | 1,234.5",
    "Remittances reached US$ 38 billion in FY26.",
    "The large scale manufacturing index declined.",
]


def make_chunks(texts: list[str] = TEXTS) -> list[ChunkRecord]:
    out = []
    for i, text in enumerate(texts):
        out.append(
            ChunkRecord(
                chunk_id=hashlib.sha1(text.encode()).hexdigest()[:16],
                doc_id="pes2526_ch01",
                edition="2025-26",
                chapter_no=1,
                chapter_title="Growth",
                page_start=i + 1,
                page_end=i + 1,
                kind=ChunkKind.TEXT,
                text=text,
                embed_text=f"PES 2025-26 · Growth\n{text}",
                bm25_tokens=tokenize(text),
                n_words=len(text.split()),
                n_tokens=len(text.split()) + 2,
            )
        )
    return out


def paths_in(root: Path, variant: str = "table_aware") -> IndexPaths:
    d = root / variant
    return IndexPaths(
        chunks=d / "chunks.parquet",
        faiss=d / "faiss.index",
        meta=d / "index_meta.json",
        chunk_report=d / "chunk_report.json",
    )


@pytest.fixture
def built(tmp_path: Path) -> tuple[IndexPaths, HashEmbedder]:
    p = paths_in(tmp_path)
    write_chunks(make_chunks(), p.chunks)
    p.chunk_report.write_text(json.dumps({"chunking": {"window_words": 150}}))
    emb = HashEmbedder()
    build_index("table_aware", p, emb, chunking=ChunkingSettings(), commit="abc")
    return p, emb


def load(p: IndexPaths, variant: str = "table_aware", emb: EmbeddingSettings = EMB):
    return IndexStore.load(variant, paths=p, embedding=emb)


def test_build_writes_matching_files_and_metadata(built) -> None:
    p, emb = built
    meta = read_meta(p.meta)
    assert meta.n_chunks == len(TEXTS) and meta.dim == DIM
    assert meta.embedding_revision == REV and meta.git_commit == "abc"
    assert meta.chunking == {"window_words": 150, "variant": "table_aware"}
    store = load(p)
    assert store.faiss.ntotal == len(store) == len(TEXTS)
    assert store.index_version == meta.index_version
    assert store.bm25.corpus_size == len(TEXTS)


def test_row_i_is_chunk_i_and_each_chunk_retrieves_itself(built) -> None:
    p, emb = built
    store = load(p)
    vecs = emb.encode_passages([c.embed_text for c in store.chunks])
    for i, vec in enumerate(vecs):
        assert np.allclose(store.vector(i), vec, atol=1e-6)
        assert store.dense_search(vec, 3)[0][0] == i


def test_sparse_search_uses_the_stored_tokens(built) -> None:
    store = load(built[0])
    hits = store.sparse_search(tokenize("FY2025-26 inflation"), 5)
    assert {store.chunks[r].text for r, _ in hits[:2]} == {TEXTS[1], TEXTS[3]}
    assert store.sparse_search(tokenize("1234.5"), 1)[0][0] == 2


def test_chunking_records_the_variant_built(tmp_path: Path) -> None:
    p = paths_in(tmp_path, "plain_text")
    write_chunks(make_chunks(), p.chunks)
    p.chunk_report.write_text(json.dumps({"chunking": {"variant": "table_aware"}}))
    res = build_index("plain_text", p, HashEmbedder(), chunking=ChunkingSettings())
    assert res.meta.chunking == {"variant": "plain_text"}


def test_index_version_is_deterministic(built, tmp_path: Path) -> None:
    p, emb = built
    first = read_meta(p.meta)
    build_index("table_aware", p, emb, chunking=ChunkingSettings())
    again = read_meta(p.meta)
    assert again.index_version == first.index_version
    assert again.index_version == index_version(Variant.TABLE_AWARE, emb, first.content_sha1)


def test_rebuild_reuses_cached_vectors(tmp_path: Path) -> None:
    p = paths_in(tmp_path)
    write_chunks(make_chunks(), p.chunks)
    emb = HashEmbedder()
    cache = EmbeddingCache(tmp_path / "cache" / "e.npz", DIM)
    first = build_index("table_aware", p, emb, chunking=ChunkingSettings(), cache=cache)
    assert first.n_embedded == len(TEXTS)
    write_chunks(make_chunks([*TEXTS, "A new paragraph on the fiscal deficit."]), p.chunks)
    cache2 = EmbeddingCache(tmp_path / "cache" / "e.npz", DIM)
    assert len(cache2) == len(TEXTS)
    second = build_index("table_aware", p, emb, chunking=ChunkingSettings(), cache=cache2)
    assert second.n_embedded == 1 and emb.n_encoded == len(TEXTS) + 1
    assert load(p).faiss.ntotal == len(TEXTS) + 1


def test_embed_passages_embeds_repeated_texts_once() -> None:
    emb = HashEmbedder()
    vecs, n = embed_passages(emb, ["a b", "c", "a b"])
    assert n == 2 and np.allclose(vecs[0], vecs[2])


# --------------------------------------------------------------------- refusals


def edit_meta(p: IndexPaths, **changes) -> None:
    data = json.loads(p.meta.read_text())
    data.update(changes)
    p.meta.write_text(json.dumps(data))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_chunks": 4}, "metadata says 4"),
        ({"dim": 768}, "dim is 768"),
        ({"variant": "plain_text"}, "variant plain_text"),
        ({"embedding_revision": "1" * 40}, "embedding_revision"),
        ({"embedding_model": "other/model"}, "embedding_model"),
        ({"query_prefix": ""}, "query_prefix"),
        ({"chunks_sha256": "0" * 64}, "chunks.parquet changed"),
        ({"faiss_sha256": "0" * 64}, "faiss.index changed"),
        ({"content_sha1": "0" * 40}, "chunk rows differ from those the index"),
    ],
)
def test_load_refuses_when_metadata_disagrees(built, changes, message) -> None:
    p, _ = built
    edit_meta(p, **changes)
    with pytest.raises(IndexMismatchError, match=re.escape(message)):
        load(p)


def test_load_refuses_changed_chunks(built) -> None:
    p, _ = built
    write_chunks(make_chunks(TEXTS[:4]), p.chunks)
    with pytest.raises(IndexMismatchError, match="chunks.parquet changed"):
        load(p)


def test_load_refuses_another_variants_files(built, tmp_path: Path) -> None:
    p, emb = built
    other = paths_in(tmp_path, "plain_text")
    write_chunks(make_chunks(TEXTS[:3]), other.chunks)
    build_index("plain_text", other, emb, chunking=ChunkingSettings())
    shutil.copy(other.faiss, p.faiss)
    with pytest.raises(IndexMismatchError, match="faiss.index changed"):
        load(p)
    with pytest.raises(IndexMismatchError, match="variant table_aware"):
        load(p, variant="plain_text")


def test_load_refuses_other_embedding_settings(built) -> None:
    p, _ = built
    with pytest.raises(IndexMismatchError, match="embedding_revision"):
        load(p, emb=EMB.model_copy(update={"revision": "f" * 40}))


def test_load_refuses_unreadable_metadata_and_missing_files(built) -> None:
    p, _ = built
    p.meta.write_text("{not json")
    with pytest.raises(IndexMismatchError, match="unreadable"):
        load(p)
    p.meta.unlink()
    with pytest.raises(FileNotFoundError):
        load(p)


def test_dense_search_checks_the_query_dimension(built) -> None:
    store = load(built[0])
    with pytest.raises(ValueError, match="dimension"):
        store.dense_search(np.zeros(10, dtype=np.float32), 1)


def test_macos_defaults_to_one_openmp_thread() -> None:
    """faiss and torch bundle two OpenMP runtimes on macOS (see index/_faiss.py)."""
    import os
    import sys

    import claimcheck.index  # noqa: F401
    from claimcheck.index._faiss import load_faiss

    assert load_faiss().__name__ == "faiss"
    if sys.platform == "darwin":
        assert os.environ.get("OMP_NUM_THREADS")


# ------------------------------------------- index_version covers retrieval and citations


def rebuild_with(p: IndexPaths, emb: HashEmbedder, cache: EmbeddingCache, change) -> tuple:
    chunks = make_chunks()
    chunks[0] = chunks[0].model_copy(update=change(chunks[0]))
    write_chunks(chunks, p.chunks)
    res = build_index("table_aware", p, emb, chunking=ChunkingSettings(), cache=cache)
    return res, load(p)


@pytest.fixture
def cached(tmp_path: Path):
    p = paths_in(tmp_path)
    write_chunks(make_chunks(), p.chunks)
    emb = HashEmbedder()
    cache = EmbeddingCache(tmp_path / "cache" / "e.npz", DIM)
    first = build_index("table_aware", p, emb, chunking=ChunkingSettings(), cache=cache)
    return p, emb, cache, first.meta.index_version


def test_a_bm25_token_change_gives_a_new_version(cached) -> None:
    p, emb, cache, v0 = cached
    assert load(p).sparse_search(["reviewprobe"], 5) == []
    res, store = rebuild_with(
        p, emb, cache, lambda c: {"bm25_tokens": [*c.bm25_tokens, "reviewprobe"]}
    )
    assert store.sparse_search(["reviewprobe"], 5)[0][0] == 0
    assert res.n_embedded == 0  # same embed_text: vectors reused
    assert res.meta.index_version != v0


@pytest.mark.parametrize(
    "change",
    [
        {"page_end": 2},
        {"table_id": "pes2526_ch01:p1:t1"},
        {"doc_id": "pes2425_ch01"},
        {"edition": "2024-25"},
        {"chapter_title": "Growth and Investment"},
    ],
)
def test_a_citation_metadata_change_gives_a_new_version(cached, change) -> None:
    p, emb, cache, v0 = cached
    res, store = rebuild_with(p, emb, cache, lambda c: change)
    assert res.n_embedded == 0
    assert res.meta.index_version != v0
    assert store.chunks[0].model_dump(include=set(change)) == change


def test_an_unchanged_rebuild_keeps_its_version(cached) -> None:
    p, emb, cache, v0 = cached
    res, _ = rebuild_with(p, emb, cache, lambda c: {})
    assert res.n_embedded == 0 and res.meta.index_version == v0
