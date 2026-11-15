"""Corpus download: the 45 Pakistan Economic Survey PDFs.

`download_corpus` fetches every catalogue entry into `data/raw/`, one request per second with
retries, records `sha256` and `n_pages` in `data/manifest.json`, and is idempotent: a file whose
checksum matches the manifest is skipped without any network request.

The "Complete Document" PDFs duplicate every chapter and are deliberately not in the catalogue.
If a chapter link is dead, `cut_from_complete` slices it out of a complete PDF by page range.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import httpx

from claimcheck.config import IngestSettings
from claimcheck.schemas import DocKind, Manifest, ManifestEntry

log = logging.getLogger(__name__)

BASE_URL = "https://www.finance.gov.pk"
USER_AGENT = "claimcheck-corpus-downloader/0.1"
# Responses that will not change on retry: the link is dead (or refuses us).
DEAD_STATUSES = frozenset({400, 401, 403, 404, 410})


@dataclass(frozen=True)
class CatalogueEntry:
    """One corpus document and where it comes from."""

    doc_id: str
    edition: str
    kind: DocKind
    chapter_no: int | None
    title: str
    remote_path: str  # path on finance.gov.pk, as published (may contain spaces)

    @property
    def filename(self) -> str:
        return f"{self.doc_id}.pdf"

    @property
    def source_url(self) -> str:
        return BASE_URL + quote(self.remote_path, safe="/")


def _chapters(
    prefix: str, edition: str, folder: str, titles: dict[int, tuple[str, str]]
) -> list[CatalogueEntry]:
    return [
        CatalogueEntry(
            doc_id=f"{prefix}_ch{no:02d}",
            edition=edition,
            kind=DocKind.CHAPTER,
            chapter_no=no,
            title=title,
            remote_path=f"/survey/{folder}/{remote}",
        )
        for no, (title, remote) in sorted(titles.items())
    ]


def _extras(
    prefix: str,
    edition: str,
    folder: str,
    extras: Iterable[tuple[str, DocKind, str, str]],
) -> list[CatalogueEntry]:
    return [
        CatalogueEntry(
            doc_id=f"{prefix}_{slug}",
            edition=edition,
            kind=kind,
            chapter_no=None,
            title=title,
            remote_path=f"/survey/{folder}/{remote}",
        )
        for slug, kind, title, remote in extras
    ]


# 2025-26. The chapter 5/6 numbering in the filenames is swapped relative to 2024-25. Page 1
# of a chapter is an infographic with only a title, so the numbers below follow book order
# (printed page numbers): Money and Credit is chapter 5.
_PES2526_CHAPTERS = {
    1: ("Growth and Investment", "1_Growth_and_Investment.pdf"),
    2: ("Agriculture", "2_Agriculture.pdf"),
    3: ("Manufacturing and Mining", "3_Manufacturing and Mining.pdf"),
    4: ("Fiscal Development", "4_Fiscal Development.pdf"),
    5: ("Money and Credit", "6_Money and Credit.pdf"),
    6: ("Capital Market", "5_Capital Market.pdf"),
    7: ("Inflation", "7_Inflation.pdf"),
    8: ("Trade and Payments", "8_Trade and Payments.pdf"),
    9: ("Public Debt", "9_Public_Debt.pdf"),
    10: ("Education", "10_Education.pdf"),
    11: ("Health and Nutrition", "11_Health_and_Nutrition.pdf"),
    12: ("Population", "12_Population.pdf"),
    13: ("Transport and Communications", "13_Transport_and_Communications.pdf"),
    14: ("Energy", "14_Energy.pdf"),
    15: ("Information Technology", "15_Information_Technology.pdf"),
    16: ("Social Protection", "16_Social_Protection.pdf"),
    17: ("Climate Change", "17_Climate_Change.pdf"),
}
_PES2526_EXTRAS = [
    ("overview", DocKind.OVERVIEW, "Overview of the Economy", "Overview of the economy_26.pdf"),
    ("highlights", DocKind.HIGHLIGHTS, "Highlights", "Highlights_of_the_PES_2026.pdf"),
    ("annex_contingent", DocKind.ANNEX, "Contingent Liabilities", "Annex Contingent_26.pdf"),
    ("annex_tax", DocKind.ANNEX, "Tax Expenditure", "Tax_Expenditure_2026.pdf"),
    (
        "indicators",
        DocKind.INDICATORS,
        "Economic and Social Indicators",
        "Economic_and_Social_Indicators.pdf",
    ),
]

_PES2425_CHAPTERS = {
    1: ("Growth and Investment", "1_Growth_and_Investment.pdf"),
    2: ("Agriculture", "2_Agriculture.pdf"),
    3: ("Manufacturing and Mining", "3_Manufacturing_and_Mining.pdf"),
    4: ("Fiscal Development", "4_Fiscal_Development.pdf"),
    5: ("Money and Credit", "5_Money_and_Credit.pdf"),
    6: ("Capital Market", "6_Capital_Market.pdf"),
    7: ("Inflation", "7_Inflation.pdf"),
    8: ("Trade and Payments", "8_Trade_and_Payments.pdf"),
    9: ("Public Debt", "9_Public_Debt.pdf"),
    10: ("Education", "10_Education.pdf"),
    11: ("Health and Nutrition", "11_Health_and_Nutrition.pdf"),
    12: ("Population", "12_Population.pdf"),
    13: ("Transport and Communications", "13_Transport_and_Communication.pdf"),
    14: ("Energy", "14_Energy.pdf"),
    15: ("Information Technology", "15_Information_Technology.pdf"),
    16: ("Social Protection", "16_Social_Protection.pdf"),
    17: ("Climate Change", "17_Climate_Change.pdf"),
}
_PES2425_EXTRAS = [
    ("overview", DocKind.OVERVIEW, "Overview of the Economy", "overview_2024_25.pdf"),
    ("highlights", DocKind.HIGHLIGHTS, "Highlights", "Highlights.pdf"),
    (
        "annex_contingent",
        DocKind.ANNEX,
        "Contingent Liabilities",
        "annex_I_Contingent_Liabilities.pdf",
    ),
    ("annex_tax", DocKind.ANNEX, "Tax Expenditure", "annex_II_Tax_Expenditure.pdf"),
    (
        "annex_qna",
        DocKind.ANNEX,
        "Quarterly National Accounts",
        "annex_III_National_Accounts.pdf",
    ),
    ("indicators", DocKind.INDICATORS, "Economic and Social Indicators", "social_indicators.pdf"),
]

CATALOGUE: tuple[CatalogueEntry, ...] = tuple(
    _chapters("pes2526", "2025-26", "chapter_26", _PES2526_CHAPTERS)
    + _extras("pes2526", "2025-26", "chapter_26", _PES2526_EXTRAS)
    + _chapters("pes2425", "2024-25", "chapter_25", _PES2425_CHAPTERS)
    + _extras("pes2425", "2024-25", "chapter_25", _PES2425_EXTRAS)
)


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


class DeadLinkError(Exception):
    """The server answered with a status that retrying will not fix."""

    def __init__(self, url: str, status: int) -> None:
        super().__init__(f"HTTP {status} for {url}")
        self.url = url
        self.status = status


class DownloadError(Exception):
    """A document could not be fetched or failed validation after all retries."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def count_pages(path: Path) -> int:
    """Number of pages of a PDF; raises `DownloadError` if it is not a readable PDF."""
    import pymupdf  # imported lazily: keeps `claimcheck --help` fast

    if not path.read_bytes()[:5] == b"%PDF-":
        raise DownloadError(f"{path.name} is not a PDF (missing %PDF- header)")
    try:
        with pymupdf.open(path) as doc:
            n_pages = len(doc)
    except Exception as exc:  # pymupdf raises its own FileDataError family
        raise DownloadError(f"{path.name} cannot be opened as a PDF: {exc}") from exc
    if n_pages < 1:
        raise DownloadError(f"{path.name} has no pages")
    return n_pages


class RateLimiter:
    """Guarantees at least `interval_s` between the starts of consecutive requests."""

    def __init__(
        self,
        interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval_s = interval_s
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self.interval_s - (self._clock() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


@dataclass
class DownloadReport:
    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def n_ok(self) -> int:
        return len(self.downloaded) + len(self.skipped)


def _fetch(
    client: httpx.Client,
    url: str,
    dest: Path,
    *,
    limiter: RateLimiter,
    max_retries: int,
    backoff_s: float,
    sleep: Callable[[float], None],
) -> None:
    """Stream `url` into `dest`, retrying transient failures up to `max_retries` times."""
    last_error = "unknown error"
    for attempt in range(max_retries + 1):
        if attempt:
            sleep(backoff_s * 2 ** (attempt - 1))
        limiter.wait()
        try:
            with client.stream("GET", url) as response:
                if response.status_code in DEAD_STATUSES:
                    raise DeadLinkError(url, response.status_code)
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}"
                    log.warning(
                        "%s (attempt %d/%d): %s", url, attempt + 1, max_retries + 1, last_error
                    )
                    continue
                with dest.open("wb") as fh:
                    for block in response.iter_bytes(1 << 16):
                        fh.write(block)
            return
        except (httpx.TransportError, httpx.DecodingError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            log.warning("%s (attempt %d/%d): %s", url, attempt + 1, max_retries + 1, last_error)
    raise DownloadError(f"{url}: {last_error} after {max_retries + 1} attempts")


# --------------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    if not path.exists():
        return {}
    manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    return {entry.doc_id: entry for entry in manifest.documents}


def write_manifest(path: Path, entries: list[ManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(Manifest(documents=entries).model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _entry(cat: CatalogueEntry, sha256: str | None, n_pages: int | None) -> ManifestEntry:
    return ManifestEntry(
        doc_id=cat.doc_id,
        edition=cat.edition,
        kind=cat.kind,
        chapter_no=cat.chapter_no,
        title=cat.title,
        source_url=cat.source_url,
        filename=cat.filename,
        sha256=sha256,
        n_pages=n_pages,
    )


# --------------------------------------------------------------------------------------------
# Main entry points
# --------------------------------------------------------------------------------------------


def download_corpus(
    raw_dir: Path,
    manifest_path: Path,
    settings: IngestSettings,
    *,
    catalogue: Iterable[CatalogueEntry] = CATALOGUE,
    client: httpx.Client | None = None,
    force: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[str, str], None] | None = None,
) -> DownloadReport:
    """Download every catalogue entry that is missing or whose checksum no longer matches.

    The manifest is rewritten at the end (also on failure) so progress is never lost. Entries
    that fail keep `sha256`/`n_pages` as `None`; the report lists them with the reason.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    known = load_manifest(manifest_path)
    report = DownloadReport()
    entries: list[ManifestEntry] = []
    limiter = RateLimiter(settings.request_interval_s, sleep=sleep)
    own_client = client is None
    http = client or httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=settings.timeout_s,
        follow_redirects=True,
    )
    notify = on_progress or (lambda _doc_id, _status: None)
    try:
        for cat in catalogue:
            dest = raw_dir / cat.filename
            previous = known.get(cat.doc_id)
            entry = _entry(cat, None, None)
            try:
                if not force and dest.exists() and _matches(dest, previous):
                    assert previous is not None
                    entry = _entry(cat, previous.sha256, previous.n_pages)
                    report.skipped.append(cat.doc_id)
                    notify(cat.doc_id, "skipped")
                else:
                    tmp = dest.with_suffix(".pdf.part")
                    try:
                        _fetch(
                            http,
                            cat.source_url,
                            tmp,
                            limiter=limiter,
                            max_retries=settings.max_retries,
                            backoff_s=2.0,
                            sleep=sleep,
                        )
                        n_pages = count_pages(tmp)
                        sha = sha256_file(tmp)
                        tmp.replace(dest)
                    finally:
                        tmp.unlink(missing_ok=True)
                    entry = _entry(cat, sha, n_pages)
                    report.downloaded.append(cat.doc_id)
                    notify(cat.doc_id, "downloaded")
            except (DeadLinkError, DownloadError) as exc:
                log.error("DEAD/FAILED %s: %s", cat.doc_id, exc)
                report.failed[cat.doc_id] = str(exc)
                notify(cat.doc_id, "failed")
            entries.append(entry)
    finally:
        if own_client:
            http.close()
        # A partial run (e.g. --limit) must not forget documents it did not touch.
        merged = {**known, **{e.doc_id: e for e in entries}}
        order = {c.doc_id: i for i, c in enumerate(CATALOGUE)}
        final = sorted(merged.values(), key=lambda e: order.get(e.doc_id, len(order)))
        write_manifest(manifest_path, final)
    return report


def _matches(dest: Path, previous: ManifestEntry | None) -> bool:
    """True if the file on disk is the one the manifest describes."""
    if previous is None or previous.sha256 is None or previous.n_pages is None:
        return False
    return sha256_file(dest) == previous.sha256


def cut_from_complete(complete_pdf: Path, dest: Path, first_page: int, last_page: int) -> int:
    """Write pages `first_page`..`last_page` (1-based, inclusive) of `complete_pdf` to `dest`.

    Fallback for a dead chapter link. The complete PDF itself is never added to the manifest
    or to `data/raw/`; keep it outside the repository.
    """
    import pymupdf

    with pymupdf.open(complete_pdf) as src:
        if not 1 <= first_page <= last_page <= len(src):
            raise ValueError(f"page range {first_page}-{last_page} outside 1-{len(src)}")
        with pymupdf.open() as out:
            out.insert_pdf(src, from_page=first_page - 1, to_page=last_page - 1)
            dest.parent.mkdir(parents=True, exist_ok=True)
            out.save(dest)
    return last_page - first_page + 1
