"""The committed corpus manifest and, where present, the local raw PDFs."""

from __future__ import annotations

import re
from collections import Counter

import pytest

from claimcheck.config import REPO_ROOT
from claimcheck.ingest.download import CATALOGUE, count_pages, sha256_file
from claimcheck.schemas import DocKind, Manifest, ManifestEntry

MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"
RAW_DIR = REPO_ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def entries() -> list[ManifestEntry]:
    return Manifest.model_validate_json(MANIFEST_PATH.read_text(encoding="utf-8")).documents


def test_45_unique_documents(entries: list[ManifestEntry]) -> None:
    assert len(entries) == 45
    assert len({e.doc_id for e in entries}) == 45
    assert len({e.filename for e in entries}) == 45
    assert len({e.source_url for e in entries}) == 45


def test_editions_and_kinds(entries: list[ManifestEntry]) -> None:
    assert Counter(e.edition for e in entries) == {"2025-26": 22, "2024-25": 23}
    for edition, prefix in (("2025-26", "pes2526"), ("2024-25", "pes2425")):
        docs = [e for e in entries if e.edition == edition]
        assert all(e.doc_id.startswith(prefix + "_") for e in docs)
        chapters = [e for e in docs if e.kind == DocKind.CHAPTER]
        assert sorted(e.chapter_no or 0 for e in chapters) == list(range(1, 18))
        assert all(e.doc_id == f"{prefix}_ch{e.chapter_no:02d}" for e in chapters)
        assert all(e.chapter_no is None for e in docs if e.kind != DocKind.CHAPTER)
        counts = Counter(e.kind for e in docs)
        assert counts[DocKind.OVERVIEW] == counts[DocKind.HIGHLIGHTS] == 1
        assert counts[DocKind.INDICATORS] == 1
        assert counts[DocKind.ANNEX] == (2 if edition == "2025-26" else 3)


def test_no_complete_document(entries: list[ManifestEntry]) -> None:
    for e in entries:
        haystack = " ".join([e.doc_id, e.filename, e.title, e.source_url]).lower()
        assert "complete" not in haystack, e.doc_id


def test_chapter_titles_follow_the_survey_order(entries: list[ManifestEntry]) -> None:
    """Chapter 5/6 filenames are swapped in 2025-26; titles must still follow 2024-25."""
    title = {(e.edition, e.chapter_no): e.title for e in entries if e.kind == DocKind.CHAPTER}
    assert title[("2025-26", 5)] == title[("2024-25", 5)] == "Money and Credit"
    assert title[("2025-26", 6)] == title[("2024-25", 6)] == "Capital Market"


def test_checksums_and_pages_recorded(entries: list[ManifestEntry]) -> None:
    for e in entries:
        assert e.sha256 and re.fullmatch(r"[0-9a-f]{64}", e.sha256), e.doc_id
        assert e.n_pages is not None and e.n_pages >= 1, e.doc_id
    assert len({e.sha256 for e in entries}) == 45  # no two documents are the same file


def test_source_urls_are_encoded_and_on_the_expected_host(entries: list[ManifestEntry]) -> None:
    for e in entries:
        folder = "chapter_26" if e.edition == "2025-26" else "chapter_25"
        assert e.source_url.startswith(f"https://www.finance.gov.pk/survey/{folder}/"), e.doc_id
        assert " " not in e.source_url
        assert e.source_url.endswith(".pdf")


def test_manifest_matches_catalogue(entries: list[ManifestEntry]) -> None:
    """The manifest is generated from the catalogue; they must not drift apart."""
    expected = [
        (c.doc_id, c.edition, c.kind, c.chapter_no, c.title, c.source_url, c.filename)
        for c in CATALOGUE
    ]
    actual = [
        (e.doc_id, e.edition, e.kind, e.chapter_no, e.title, e.source_url, e.filename)
        for e in entries
    ]
    assert actual == expected


def test_raw_files_match_manifest(entries: list[ManifestEntry]) -> None:
    """Checksums and page counts of the local PDFs (skipped where data/raw/ is absent, e.g. CI)."""
    if not all((RAW_DIR / e.filename).exists() for e in entries):
        pytest.skip("data/raw/ is not fully present; run `claimcheck ingest download`")
    for e in entries:
        path = RAW_DIR / e.filename
        assert sha256_file(path) == e.sha256, f"{e.doc_id}: checksum mismatch"
        assert count_pages(path) == e.n_pages, f"{e.doc_id}: page count mismatch"
    on_disk = {p.name for p in RAW_DIR.glob("*.pdf")}
    assert on_disk == {e.filename for e in entries}, "unexpected PDFs in data/raw/"
