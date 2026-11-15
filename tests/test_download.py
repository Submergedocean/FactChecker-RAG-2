"""Download logic against a mocked HTTP server (no network, no real corpus)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pymupdf
import pytest

from claimcheck.config import IngestSettings
from claimcheck.ingest.download import (
    CATALOGUE,
    CatalogueEntry,
    RateLimiter,
    cut_from_complete,
    download_corpus,
    load_manifest,
)
from claimcheck.schemas import DocKind


def make_pdf(n_pages: int, tag: str = "x") -> bytes:
    doc = pymupdf.open()
    for i in range(n_pages):
        doc.new_page().insert_text((72, 72), f"{tag} page {i + 1}")
    data = doc.tobytes()
    doc.close()
    return data


def entry(doc_id: str, remote: str) -> CatalogueEntry:
    return CatalogueEntry(
        doc_id=doc_id,
        edition="2025-26",
        kind=DocKind.CHAPTER,
        chapter_no=1,
        title=doc_id,
        remote_path=remote,
    )


FAKE = [
    entry("t_a", "/survey/chapter_26/a file with spaces.pdf"),
    entry("t_b", "/survey/chapter_26/b.pdf"),
]
SETTINGS = IngestSettings(request_interval_s=0.0, max_retries=3, timeout_s=5.0)


class FakeServer:
    """Serves PDFs by path; `plan[path]` is a list of statuses to answer with first."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.plan: dict[str, list[int]] = {}
        self.bodies = {
            "/survey/chapter_26/a%20file%20with%20spaces.pdf": make_pdf(3, "a"),
            "/survey/chapter_26/b.pdf": make_pdf(2, "b"),
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode()
        self.requests.append(path)
        queued = self.plan.get(path, [])
        if queued:
            status = queued.pop(0)
            if status != 200:
                return httpx.Response(status)
        if path not in self.bodies:
            return httpx.Response(404)
        return httpx.Response(200, content=self.bodies[path])

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def run(server: FakeServer, tmp_path: Path, **kwargs):
    return download_corpus(
        tmp_path / "raw",
        tmp_path / "manifest.json",
        SETTINGS,
        catalogue=kwargs.pop("catalogue", FAKE),
        client=server.client(),
        sleep=lambda _s: None,
        **kwargs,
    )


def test_download_records_checksum_and_pages_and_encodes_spaces(tmp_path: Path) -> None:
    server = FakeServer()
    report = run(server, tmp_path)
    assert report.downloaded == ["t_a", "t_b"] and not report.skipped and not report.failed
    assert server.requests[0] == "/survey/chapter_26/a%20file%20with%20spaces.pdf"
    manifest = load_manifest(tmp_path / "manifest.json")
    assert manifest["t_a"].n_pages == 3 and manifest["t_b"].n_pages == 2
    assert all(m.sha256 and len(m.sha256) == 64 for m in manifest.values())
    assert (tmp_path / "raw" / "t_a.pdf").exists()
    assert not list((tmp_path / "raw").glob("*.part"))


def test_second_run_downloads_nothing(tmp_path: Path) -> None:
    run(FakeServer(), tmp_path)
    server = FakeServer()
    report = run(server, tmp_path)
    assert report.downloaded == [] and report.skipped == ["t_a", "t_b"]
    assert server.requests == []  # skipped files cost no requests


def test_corrupted_file_is_downloaded_again(tmp_path: Path) -> None:
    run(FakeServer(), tmp_path)
    (tmp_path / "raw" / "t_b.pdf").write_bytes(b"truncated")
    server = FakeServer()
    report = run(server, tmp_path)
    assert report.downloaded == ["t_b"] and report.skipped == ["t_a"]
    assert server.requests == ["/survey/chapter_26/b.pdf"]


def test_force_redownloads_everything(tmp_path: Path) -> None:
    run(FakeServer(), tmp_path)
    report = run(FakeServer(), tmp_path, force=True)
    assert report.downloaded == ["t_a", "t_b"] and not report.skipped


def test_transient_errors_are_retried(tmp_path: Path) -> None:
    server = FakeServer()
    server.plan["/survey/chapter_26/b.pdf"] = [503, 500, 200]
    report = run(server, tmp_path)
    assert report.failed == {} and "t_b" in report.downloaded
    assert server.requests.count("/survey/chapter_26/b.pdf") == 3


def test_gives_up_after_three_retries(tmp_path: Path) -> None:
    server = FakeServer()
    server.plan["/survey/chapter_26/b.pdf"] = [503] * 10
    report = run(server, tmp_path)
    assert "t_b" in report.failed and report.downloaded == ["t_a"]
    assert server.requests.count("/survey/chapter_26/b.pdf") == 4  # 1 try + 3 retries
    assert not (tmp_path / "raw" / "t_b.pdf").exists()


def test_dead_link_is_reported_without_retrying(tmp_path: Path) -> None:
    server = FakeServer()
    cat = [*FAKE, entry("t_dead", "/survey/chapter_26/gone.pdf")]
    report = run(server, tmp_path, catalogue=cat)
    assert list(report.failed) == ["t_dead"] and "404" in report.failed["t_dead"]
    assert server.requests.count("/survey/chapter_26/gone.pdf") == 1
    manifest = load_manifest(tmp_path / "manifest.json")
    assert manifest["t_dead"].sha256 is None and manifest["t_a"].sha256 is not None


def test_non_pdf_body_is_rejected(tmp_path: Path) -> None:
    server = FakeServer()
    server.bodies["/survey/chapter_26/b.pdf"] = b"<html>captive portal</html>"
    report = run(server, tmp_path)
    assert "t_b" in report.failed and "not a PDF" in report.failed["t_b"]
    assert not (tmp_path / "raw" / "t_b.pdf").exists()


def test_partial_run_keeps_other_manifest_entries(tmp_path: Path) -> None:
    cat = list(CATALOGUE[:2])
    server = FakeServer()
    server.bodies = {
        c.source_url.removeprefix("https://www.finance.gov.pk"): make_pdf(1, c.doc_id) for c in cat
    }
    run(server, tmp_path, catalogue=cat)
    run(server, tmp_path, catalogue=cat[:1])
    assert set(load_manifest(tmp_path / "manifest.json")) == {c.doc_id for c in cat}


def test_rate_limiter_spaces_requests() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(1.0, clock=lambda: now[0], sleep=sleep)
    limiter.wait()  # first request: no wait
    now[0] += 0.25
    limiter.wait()
    limiter.wait()
    assert slept == pytest.approx([0.75, 1.0])


def test_cut_from_complete(tmp_path: Path) -> None:
    complete = tmp_path / "complete.pdf"
    complete.write_bytes(make_pdf(10, "c"))
    dest = tmp_path / "out" / "cut.pdf"
    assert cut_from_complete(complete, dest, 4, 6) == 3
    with pymupdf.open(dest) as doc:
        assert len(doc) == 3 and "c page 4" in doc[0].get_text()
    with pytest.raises(ValueError):
        cut_from_complete(complete, dest, 9, 11)
