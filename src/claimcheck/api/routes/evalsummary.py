"""`GET /eval/summary`: the contents of `reports/eval/summary.json`; 404 until written."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter

from claimcheck.api.deps import State
from claimcheck.api.errors import ERROR_RESPONSES, APIError

router = APIRouter(tags=["evaluation"])


@router.get("/eval/summary", response_model=dict[str, Any], responses=ERROR_RESPONSES)
def get_eval_summary(state: State) -> dict[str, Any]:
    path = state.settings.paths.eval_summary
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise APIError.not_found("Evaluation summary") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return data
