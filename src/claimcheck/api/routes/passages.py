"""`GET /passages/{chunk_id}`: one passage plus the previous and next chunk ids."""

from __future__ import annotations

import re

from fastapi import APIRouter

from claimcheck.api.deps import State
from claimcheck.api.errors import ERROR_RESPONSES, APIError
from claimcheck.llm.prompts.verifier import chapter_label
from claimcheck.schemas import PassageResponse

router = APIRouter(tags=["passages"])

CHUNK_ID_RE = re.compile(r"^[0-9a-f]{16}$")


@router.get("/passages/{chunk_id}", response_model=PassageResponse, responses=ERROR_RESPONSES)
def get_passage(chunk_id: str, state: State) -> PassageResponse:
    """Neighbours are the adjacent chunks of the same document in index order (reading
    order), or null at a document's edge."""
    row = state.row_of.get(chunk_id) if CHUNK_ID_RE.match(chunk_id) else None
    if row is None:
        raise APIError.not_found("Passage")
    chunks = state.chunks
    c = chunks[row]
    prev_id = chunks[row - 1].chunk_id if row > 0 and chunks[row - 1].doc_id == c.doc_id else None
    nxt = chunks[row + 1] if row + 1 < len(chunks) else None
    next_id = nxt.chunk_id if nxt is not None and nxt.doc_id == c.doc_id else None
    return PassageResponse(
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        edition=c.edition,
        chapter=chapter_label(c),
        page_start=c.page_start,
        page_end=c.page_end,
        kind=c.kind,
        table_id=c.table_id,
        text=c.text,
        source_url=state.sources.source_url(c.doc_id, c.page_start),
        prev_chunk_id=prev_id,
        next_chunk_id=next_id,
    )
