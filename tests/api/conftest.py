"""The real app over `FakeLLM` (scripted answers), a stub retriever,
a tiny generated PDF and a temporary SQLite file. No model, no corpus, no quota."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from claimcheck.api.deps import AppState
from claimcheck.api.main import create_app
from claimcheck.config import APISettings, LimitsSettings, PathsSettings, Settings
from claimcheck.db.session import make_engine
from claimcheck.pipeline.service import ClaimCheckService
from claimcheck.pipeline.sources import SourceIndex
from claimcheck.pipeline.store import VerificationStore
from claimcheck.schemas import DocKind, ManifestEntry
from tests.fakes import FakeLLM
from tests.test_guards import GDP_TEXT
from tests.test_service import CHUNKS, StubRetriever, sources

PREFIX = "/api/v1"
GDP_ON_PDF = "real GDP growth is recorded at 3.7 percent in FY 2026"
N_PAGES = 3
# Index order: two ch01 chunks next to each other, then ch07 (passage neighbours).
API_CHUNKS = [CHUNKS[0], CHUNKS[2], CHUNKS[1]]


class StubIndex:
    index_version = "test-index-v1"

    def __init__(self, chunks):  # noqa: ANN001
        self.chunks = tuple(chunks)
        self.row_of = {c.chunk_id: i for i, c in enumerate(self.chunks)}


def make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    for i in range(1, N_PAGES + 1):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i} of the Growth chapter.")
        if i == 3:
            page.insert_text((72, 100), "On the back of these positive developments,")
            page.insert_text((72, 114), GDP_ON_PDF + ".")
    doc.save(path)
    doc.close()


def make_sources(raw_dir: Path) -> SourceIndex:
    base = sources()
    docs = {}
    for doc_id, entry in base.documents.items():
        docs[doc_id] = entry.model_copy(update={"n_pages": N_PAGES})
    # In the manifest but not on disk (a host without raw PDFs).
    docs["pes2526_ch99"] = ManifestEntry(
        doc_id="pes2526_ch99",
        edition="2025-26",
        kind=DocKind.CHAPTER,
        chapter_no=99,
        title="Missing",
        source_url="https://example.org/ch99.pdf",
        filename="pes2526_ch99.pdf",
        n_pages=2,
    )
    make_pdf(raw_dir / "pes2526_ch01.pdf")
    return SourceIndex(docs, base.pages)


@dataclass
class Harness:
    client: TestClient
    state: AppState
    llm: FakeLLM
    settings: Settings
    tmp: Path
    extra: dict = field(default_factory=dict)

    def url(self, path: str) -> str:
        return PREFIX + path


def make_settings(tmp: Path, *, static_dir: Path | None = None, **limits: int) -> Settings:
    return Settings(
        env="test",
        paths=PathsSettings(
            data_dir=tmp,
            raw_dir=tmp / "raw",
            cache_dir=tmp / "cache",
            reports_dir=tmp / "reports",
            app_db=tmp / "app.db",
        ),
        limits=LimitsSettings(**limits),
        api=APISettings(static_dir=static_dir),
    )


def build_harness(
    tmp: Path,
    *,
    llm=None,
    static_dir: Path | None = None,
    **limits: int,  # noqa: ANN001
) -> Harness:
    (tmp / "raw").mkdir(parents=True, exist_ok=True)
    settings = make_settings(tmp, static_dir=static_dir, **limits)
    llm = llm if llm is not None else FakeLLM()
    retriever = StubRetriever(API_CHUNKS)
    retriever.store = StubIndex(API_CHUNKS)
    store = VerificationStore(make_engine(settings.paths.app_db))
    service = ClaimCheckService(
        retriever, llm, make_sources(tmp / "raw"), store=store, limits=settings.limits
    )
    state = AppState.build(settings, service)
    app = create_app(settings, state=state)
    client = TestClient(app, raise_server_exceptions=False)
    return Harness(client=client, state=state, llm=llm, settings=settings, tmp=tmp)


@pytest.fixture
def h(tmp_path: Path) -> Iterator[Harness]:
    harness = build_harness(tmp_path)
    with harness.client:
        yield harness


__all__ = ["GDP_TEXT", "Harness", "build_harness"]
