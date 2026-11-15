"""PDF page images for `GET /pages/{doc_id}/{page}`.

Pages are rendered on demand with PyMuPDF, the quote highlighted where it is found, and cached
on disk as PNG under `paths.page_cache`, oldest files evicted past `api.page_cache_mb`.
Callers pass a manifest entry (never a user-supplied path) and a page already checked to be in
range.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from pathlib import Path

import pymupdf

from claimcheck.schemas import ManifestEntry

log = logging.getLogger(__name__)

RENDER_VERSION = "r1"
_TABLE_BARS = re.compile(r"[|]+|:?-{3,}:?")
_FALLBACK_WORDS = 8


class PageRenderer:
    def __init__(self, raw_dir: Path, cache_dir: Path, *, zoom: float, cap_mb: int) -> None:
        self.raw_dir = raw_dir.resolve()
        self.cache_dir = cache_dir
        self.zoom = zoom
        self.cap_bytes = cap_mb * 1024 * 1024
        self._lock = threading.Lock()  # PyMuPDF is not thread-safe; also guards eviction
        self._page_counts: dict[str, int | None] = {}

    def pdf_path(self, entry: ManifestEntry) -> Path | None:
        path = (self.raw_dir / entry.filename).resolve()
        if not path.is_relative_to(self.raw_dir) or not path.is_file():
            return None
        return path

    def page_count(self, entry: ManifestEntry) -> int | None:
        """Pages in the PDF: the manifest's count, else the file's; None if neither."""
        if entry.n_pages:
            return entry.n_pages
        if entry.doc_id not in self._page_counts:
            path = self.pdf_path(entry)
            count = None
            if path is not None:
                with self._lock, pymupdf.open(path) as doc:
                    count = doc.page_count
            self._page_counts[entry.doc_id] = count
        return self._page_counts[entry.doc_id]

    # ------------------------------------------------------------------------------

    def _key(self, entry: ManifestEntry, page: int, quote: str) -> str:
        parts = [RENDER_VERSION, entry.doc_id, entry.sha256 or "", str(page), str(self.zoom), quote]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]

    def render(self, entry: ManifestEntry, page: int, quote: str = "") -> tuple[bytes, bool]:
        """PNG bytes of a 1-based page and whether the quote was found and highlighted.
        Raises FileNotFoundError when the PDF is not on disk."""
        quote = " ".join(quote.split())
        key = self._key(entry, page, quote)
        for found in (True, False):
            cached = self.cache_dir / f"{key}-{int(found)}.png"
            if cached.is_file():
                try:
                    return cached.read_bytes(), found
                except OSError:  # evicted between the check and the read
                    break
        path = self.pdf_path(entry)
        if path is None:
            raise FileNotFoundError(entry.filename)
        with self._lock:
            png, found = self._render(path, page, quote)
            self._store(self.cache_dir / f"{key}-{int(found)}.png", png)
        return png, found

    def _render(self, path: Path, page: int, quote: str) -> tuple[bytes, bool]:
        with pymupdf.open(path) as doc:
            pg = doc[page - 1]
            found = False
            if quote:
                quads = self._search(pg, quote)
                if quads:
                    pg.add_highlight_annot(quads)
                    found = True
            pix = pg.get_pixmap(matrix=pymupdf.Matrix(self.zoom, self.zoom), annots=True)
            return pix.tobytes("png"), found

    @staticmethod
    def _search(pg: pymupdf.Page, quote: str) -> list:
        candidates = [quote]
        plain = " ".join(_TABLE_BARS.sub(" ", quote).split())
        if plain != quote:
            candidates.append(plain)
        words = plain.split()
        if len(words) > _FALLBACK_WORDS:
            candidates.append(" ".join(words[:_FALLBACK_WORDS]))
        for needle in candidates:
            if needle:
                quads = pg.search_for(needle, quads=True)
                if quads:
                    return quads
        return []

    def _store(self, target: Path, png: bytes) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(png)
            tmp.replace(target)
            self._evict()
        except OSError as exc:  # the cache is best-effort
            log.warning("page cache write failed: %s", exc)

    def _evict(self) -> None:
        files = []
        for f in self.cache_dir.glob("*.png"):
            try:
                st = f.stat()
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, f))
        total = sum(size for _, size, _ in files)
        for _, size, f in sorted(files, key=lambda x: x[0]):
            if total <= self.cap_bytes:
                break
            f.unlink(missing_ok=True)
            total -= size
