"""`GET /healthz` (liveness) and `GET /readyz` (index loaded, database open).

Mounted at the root (for the host's health check) and under the API prefix. A failed readiness
check answers 503 with the same body, so the probe can see which check failed.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from claimcheck.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def readyz(request: Request) -> HealthResponse | JSONResponse:
    state = getattr(request.app.state, "claimcheck", None)
    checks = state.checks() if state is not None else {"index": False, "database": False}
    body = HealthResponse(status="ok" if all(checks.values()) else "not_ready", checks=checks)
    if body.status != "ok":
        return JSONResponse(body.model_dump(mode="json"), status_code=503)
    return body
