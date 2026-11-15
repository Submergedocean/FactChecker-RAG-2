"""Where a citation points: document URL, PDF page and printed page.

The model names a passage label only; everything here comes from the manifest, the chunk's
metadata and `pages.parquet`. A text chunk can span two pages: the cited page is the one whose
text holds the quote (or its start), else the chunk's first page. A table chunk's page is its
table's page.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from claimcheck.llm.prompts.verifier import chapter_label
from claimcheck.pipeline.guards import ValidCitation, match_key
from claimcheck.schemas import ChunkKind, ChunkRecord, Citation, Manifest, ManifestEntry

log = logging.getLogger(__name__)

_TABLE_PAGE_RE = re.compile(r":p(\d+):t\d+$")
_QUOTE_HEAD_CHARS = 40


@dataclass(frozen=True)
class PageInfo:
    printed_page: str | None
    key: str  # match_key of the page's body text


class SourceIndex:
    """Manifest entries and page texts, loaded once."""

    def __init__(
        self,
        documents: dict[str, ManifestEntry],
        pages: dict[tuple[str, int], PageInfo] | None = None,
    ) -> None:
        self.documents = documents
        self.pages = pages or {}

    @classmethod
    def load(cls, manifest_path: Path, pages_path: Path | None = None) -> SourceIndex:
        manifest = Manifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
        documents = {d.doc_id: d for d in manifest.documents}
        pages: dict[tuple[str, int], PageInfo] = {}
        if pages_path is not None and pages_path.exists():
            import pyarrow.parquet as pq

            table = pq.read_table(
                pages_path, columns=["doc_id", "page", "printed_page", "text_body"]
            )
            for row in table.to_pylist():
                pages[(row["doc_id"], int(row["page"]))] = PageInfo(
                    printed_page=row["printed_page"], key=match_key(row["text_body"] or "")
                )
        else:
            log.warning("pages file %s missing: citations use the chunk's first page", pages_path)
        return cls(documents, pages)

    # ------------------------------------------------------------------------------

    def page_of(self, chunk: ChunkRecord, quote: str) -> int:
        if chunk.kind is ChunkKind.TABLE and chunk.table_id:
            m = _TABLE_PAGE_RE.search(chunk.table_id)
            if m:
                return int(m.group(1))
        if chunk.page_start == chunk.page_end:
            return chunk.page_start
        q = match_key(quote)
        candidates = range(chunk.page_start, chunk.page_end + 1)
        for needle in (q, q[:_QUOTE_HEAD_CHARS]):
            for page in candidates:
                info = self.pages.get((chunk.doc_id, page))
                if info is not None and needle and needle in info.key:
                    return page
        return chunk.page_start

    def source_url(self, doc_id: str, page: int) -> str:
        entry = self.documents.get(doc_id)
        if entry is None:
            return ""
        return f"{entry.source_url}#page={page}"

    def citation(self, cit: ValidCitation) -> Citation:
        chunk = cit.chunk
        page = self.page_of(chunk, cit.quote)
        info = self.pages.get((chunk.doc_id, page))
        return Citation(
            chunk_id=chunk.chunk_id,
            quote=cit.quote,
            doc_id=chunk.doc_id,
            edition=chunk.edition,
            chapter=chapter_label(chunk),
            page=page,
            printed_page=info.printed_page if info else None,
            table_id=chunk.table_id,
            source_url=self.source_url(chunk.doc_id, page),
        )
