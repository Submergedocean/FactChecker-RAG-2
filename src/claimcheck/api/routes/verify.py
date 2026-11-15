"""`POST /verify`: verify a message of 10-500 characters."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from claimcheck.api.deps import AppState, VerifyState
from claimcheck.api.errors import ERROR_RESPONSES, APIError
from claimcheck.pipeline.guards import guard_input
from claimcheck.pipeline.store import input_hash
from claimcheck.schemas import ErrorCode, VerificationResponse, VerifyRequest

log = logging.getLogger(__name__)
router = APIRouter(tags=["verify"])

SLOT_TIMEOUT_S = 60.0


def _needs_llm(state: AppState, text: str) -> bool:
    """False when the service will answer from the result cache. Runs the service's own input
    guard first, so invalid input fails (422) without waiting for a slot. A peek only: the
    answer still comes from `ClaimCheckService.verify`, which looks the cache up again."""
    service = state.service
    clean = guard_input(text, service.limits.input_min_chars, service.limits.input_max_chars)
    if service.store is None:
        return True
    hit = service.store.lookup(input_hash(clean), service.index_version, service.pipeline_version)
    return hit is None


@router.post("/verify", response_model=VerificationResponse, responses=ERROR_RESPONSES)
def verify(body: VerifyRequest, state: VerifyState) -> VerificationResponse:
    """Extract up to 3 claims, retrieve evidence and verify each. A repeated message is served
    from the result cache (`meta.cached`)."""
    # At most `max_concurrent_llm` verifications call the LLM at once. Cached
    # results and invalid input never wait for a slot.
    if not _needs_llm(state, body.text):
        response = state.service.verify(body.text)
    else:
        if not state.llm_slots.acquire(timeout=SLOT_TIMEOUT_S):
            raise APIError(ErrorCode.LLM_UNAVAILABLE, "The checker is busy. Try again in a minute.")
        try:
            response = state.service.verify(body.text)
        finally:
            state.llm_slots.release()
    # Claim text is logged, client identity is not.
    log.info(
        "verify %s status=%s cached=%s claims=%d total_ms=%s text=%r",
        response.id,
        response.status.value,
        response.meta.cached,
        len(response.claims),
        response.meta.latency_ms.total,
        response.input.text,
    )
    return response
