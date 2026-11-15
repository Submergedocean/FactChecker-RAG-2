"""The error envelope: every non-2xx answer is
`{"error": {"code": ..., "message": ..., "retry_after_s": ...}}`.

| HTTP | code                     | raised by                                              |
|------|--------------------------|--------------------------------------------------------|
| 422  | INPUT_INVALID            | request validation, `InputGuardError`                  |
| 404  | NOT_FOUND                | `APIError.not_found`, unknown routes                   |
| 429  | RATE_LIMITED             | the per-client limiter (`Retry-After` header set)      |
| 503  | DAILY_BUDGET_EXHAUSTED   | `BudgetExhausted`                                      |
| 503  | LLM_UNAVAILABLE          | `LLMUnavailable`                                       |
| 500  | INTERNAL                 | anything else; logged here, never a stack trace in the body |
"""

from __future__ import annotations

import logging
import math
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from claimcheck.llm.client import BudgetExhausted, LLMUnavailable
from claimcheck.schemas import ErrorBody, ErrorCode, ErrorEnvelope
from claimcheck.textnorm import InputGuardError

log = logging.getLogger(__name__)

MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.INPUT_INVALID: "The request is not valid.",
    ErrorCode.NOT_FOUND: "Not found.",
    ErrorCode.RATE_LIMITED: "Too many requests. Wait a moment and try again.",
    ErrorCode.DAILY_BUDGET_EXHAUSTED: (
        "The checker has used today's LLM budget. Results checked before are still available; "
        "new checks resume tomorrow (UTC)."
    ),
    ErrorCode.LLM_UNAVAILABLE: (
        "The language model providers did not answer. Try again in a minute."
    ),
    ErrorCode.INTERNAL: "Something went wrong on our side.",
}

STATUS: dict[ErrorCode, int] = {
    ErrorCode.INPUT_INVALID: 422,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.DAILY_BUDGET_EXHAUSTED: 503,
    ErrorCode.LLM_UNAVAILABLE: 503,
    ErrorCode.INTERNAL: 500,
}

# OpenAPI: the envelope for every documented error status (for openapi-typescript).
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status: {"model": ErrorEnvelope, "description": "Error envelope"}
    for status in sorted(set(STATUS.values()))
}


class APIError(Exception):
    def __init__(
        self, code: ErrorCode, message: str | None = None, *, retry_after_s: float | None = None
    ) -> None:
        super().__init__(message or MESSAGES[code])
        self.code = code
        self.message = message or MESSAGES[code]
        self.retry_after_s = None if retry_after_s is None else max(1, math.ceil(retry_after_s))

    @classmethod
    def not_found(cls, what: str = "") -> APIError:
        return cls(ErrorCode.NOT_FOUND, f"{what} not found." if what else None)

    @classmethod
    def invalid(cls, message: str) -> APIError:
        return cls(ErrorCode.INPUT_INVALID, message)


def error_response(
    code: ErrorCode, message: str | None = None, *, retry_after_s: int | None = None
) -> JSONResponse:
    body = ErrorEnvelope(
        error=ErrorBody(code=code, message=message or MESSAGES[code], retry_after_s=retry_after_s)
    )
    headers = {"Retry-After": str(retry_after_s)} if retry_after_s is not None else None
    return JSONResponse(body.model_dump(mode="json"), status_code=STATUS[code], headers=headers)


def _validation_message(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors()[:3]:
        loc = ".".join(str(x) for x in err.get("loc", ()) if x not in ("body", "query", "path"))
        msg = str(err.get("msg", "invalid"))
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or MESSAGES[ErrorCode.INPUT_INVALID]


async def _api_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, APIError)
    return error_response(exc.code, exc.message, retry_after_s=exc.retry_after_s)


async def _validation_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return error_response(ErrorCode.INPUT_INVALID, _validation_message(exc))


async def _input_guard(_request: Request, exc: Exception) -> JSONResponse:
    return error_response(ErrorCode.INPUT_INVALID, str(exc))


async def _budget(_request: Request, _exc: Exception) -> JSONResponse:
    log.warning("daily LLM budget exhausted")
    return error_response(ErrorCode.DAILY_BUDGET_EXHAUSTED)


async def _unavailable(_request: Request, exc: Exception) -> JSONResponse:
    log.warning("LLM unavailable: %s", exc)  # provider/model/reason only, no message text
    return error_response(ErrorCode.LLM_UNAVAILABLE)


async def _http_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    if exc.status_code == 404:
        return error_response(ErrorCode.NOT_FOUND)
    if exc.status_code == 405:
        # Not one of the contract's codes; keep the envelope shape.
        body = {"error": {"code": "METHOD_NOT_ALLOWED", "message": "Method not allowed."}}
        return JSONResponse(body, status_code=405, headers=getattr(exc, "headers", None))
    if exc.status_code == 422:
        return error_response(ErrorCode.INPUT_INVALID)
    if exc.status_code < 500:
        body = {"error": {"code": "BAD_REQUEST", "message": "Bad request."}}
        return JSONResponse(body, status_code=exc.status_code)
    return error_response(ErrorCode.INTERNAL)


async def _internal(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return error_response(ErrorCode.INTERNAL)


def install(app: FastAPI) -> None:
    app.add_exception_handler(APIError, _api_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(InputGuardError, _input_guard)
    app.add_exception_handler(BudgetExhausted, _budget)
    app.add_exception_handler(LLMUnavailable, _unavailable)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _internal)
