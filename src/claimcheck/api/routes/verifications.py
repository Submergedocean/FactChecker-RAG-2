"""`GET /verifications/{id}` (permalink replay) and `POST /verifications/{id}/feedback`."""

from __future__ import annotations

import re

from fastapi import APIRouter, Response

from claimcheck.api.deps import FeedbackState, State
from claimcheck.api.errors import ERROR_RESPONSES, APIError
from claimcheck.db.models import Feedback, VerificationClaim, utc_now_iso
from claimcheck.db.session import make_session_factory, session_scope
from claimcheck.schemas import FeedbackRequest, VerificationResponse

router = APIRouter(tags=["verifications"])

ID_RE = re.compile(r"^ver_[0-9A-HJKMNP-TV-Z]{26}$")


def _check_id(verification_id: str) -> None:
    if not ID_RE.match(verification_id):
        raise APIError.not_found("Verification")


@router.get(
    "/verifications/{verification_id}",
    response_model=VerificationResponse,
    responses=ERROR_RESPONSES,
)
def get_verification(verification_id: str, state: State) -> VerificationResponse:
    """A stored result, exactly as first returned. Free hosts wipe disk on restart, so an old
    permalink may be gone (404)."""
    _check_id(verification_id)
    result = state.service.get(verification_id)
    if result is None:
        raise APIError.not_found("Verification")
    return result


@router.post(
    "/verifications/{verification_id}/feedback",
    status_code=204,
    response_class=Response,
    responses=ERROR_RESPONSES,
)
def post_feedback(
    verification_id: str,
    body: FeedbackRequest,
    state: FeedbackState,
) -> Response:
    """Thumbs up or down on one claim of a stored result, with an optional comment."""
    _check_id(verification_id)
    engine = state.engine
    if engine is None:
        raise APIError.not_found("Verification")
    comment = (body.comment or "").strip() or None
    with session_scope(make_session_factory(engine)) as s:
        if s.get(VerificationClaim, (verification_id, body.claim_index)) is None:
            raise APIError.not_found("Verification or claim")
        s.add(
            Feedback(
                verification_id=verification_id,
                claim_index=body.claim_index,
                rating=body.rating.value,
                comment=comment,
                created_at=utc_now_iso(),
            )
        )
    return Response(status_code=204)
