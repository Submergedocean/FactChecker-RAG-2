"""`GET /meta` (versions, models, limits, example claims) and `GET /documents` (manifest)."""

from __future__ import annotations

from fastapi import APIRouter

from claimcheck.api.deps import State
from claimcheck.api.errors import ERROR_RESPONSES
from claimcheck.schemas import Manifest, MetaResponse

router = APIRouter(tags=["meta"])


@router.get("/meta", response_model=MetaResponse, responses=ERROR_RESPONSES)
def get_meta(state: State) -> MetaResponse:
    s = state.settings
    service = state.service
    providers = getattr(service, "_models", [])
    first = providers[0] if providers else None
    models = (
        {
            "extractor": f"{first['provider']}/{first['extractor']}",
            "verifier": f"{first['provider']}/{first['verifier']}",
            "fallback": ", ".join(p["provider"] for p in providers[1:]),
        }
        if first
        else {}
    )
    lim = s.limits
    return MetaResponse(
        index_version=service.index_version,
        pipeline_version=service.pipeline_version,
        edition=s.latest_edition,
        models=models,
        limits={
            "input_min_chars": lim.input_min_chars,
            "input_max_chars": lim.input_max_chars,
            "max_claims_per_message": lim.max_claims_per_message,
            "feedback_comment_max_chars": lim.feedback_comment_max_chars,
            "per_client_requests_per_minute": lim.per_client_requests_per_minute,
            "per_client_requests_per_day": lim.per_client_requests_per_day,
        },
        example_claims=list(s.api.example_claims),
    )


@router.get("/documents", response_model=Manifest, responses=ERROR_RESPONSES)
def get_documents(state: State) -> Manifest:
    """The corpus manifest, for the About page."""
    return Manifest(documents=list(state.sources.documents.values()))
