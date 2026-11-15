"""`GET /pages/{doc_id}/{page}?quote=`: PNG of a PDF page with the quote highlighted.

only a `doc_id` present in the manifest and an integer page in
range are accepted; the file name comes from the manifest, never from the request.
"""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Query, Response

from claimcheck.api.deps import PagesState
from claimcheck.api.errors import ERROR_RESPONSES, APIError

router = APIRouter(tags=["pages"])

PAGE_RE = re.compile(r"^[1-9][0-9]{0,4}$")


@router.get(
    "/pages/{doc_id}/{page}",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}, **ERROR_RESPONSES},
)
def get_page(
    doc_id: str,
    page: str,
    state: PagesState,
    quote: Annotated[str, Query(max_length=500)] = "",
) -> Response:
    """The header `X-Quote-Found` says whether the quote was found and highlighted."""
    entry = state.sources.documents.get(doc_id)
    if entry is None:
        raise APIError.not_found("Document")
    n = int(page) if PAGE_RE.match(page) else 0
    count = state.renderer.page_count(entry)
    if n < 1 or count is None or n > count:
        raise APIError.not_found("Page")
    try:
        png, found = state.renderer.render(entry, n, quote)
    except FileNotFoundError as exc:
        raise APIError.not_found("Page image") from exc
    return Response(
        png,
        media_type="image/png",
        headers={
            "X-Quote-Found": "true" if found else "false",
            "Cache-Control": "public, max-age=86400",
        },
    )
