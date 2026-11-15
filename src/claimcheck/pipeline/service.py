"""`ClaimCheckService`: the single entry point for the API, the CLI and evaluation.

`verify(text)`:

1. Input guard (trim, NFC, Urdu digits, 10-500 characters); failures raise `InputGuardError`
   (API: 422 `INPUT_INVALID`) and are not stored.
2. Result cache: a stored result for the same input, index version and pipeline version is
   returned as is, with `meta.cached` true and no LLM call. `use_cache=False` runs afresh:
   no stored-result lookup and no LLM cache (`meta.cached` reports the result cache only).
3. Extraction: at most 3 English claims, or a rejection with a user-facing message.
4. Per claim: hybrid retrieval of the top 5 passages, the verifier, then the code guards.
5. The response is stored (when the service has a store) and returned.

LLM failures propagate as `BudgetExhausted` / `LLMUnavailable` (API: 503) and nothing is
stored. The pipeline version is a hash of the prompt versions, guard version, retrieval
settings and configured models, so a change to any of them bypasses old cached results.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from ulid import ULID

from claimcheck.config import LimitsSettings, Settings, get_settings
from claimcheck.llm.prompts import extractor as extractor_prompt
from claimcheck.llm.prompts import verifier as verifier_prompt
from claimcheck.llm.prompts.extractor import ExtractedClaim
from claimcheck.llm.prompts.verifier import chapter_label
from claimcheck.pipeline.extractor import Extraction, Extractor, JSONCompleter
from claimcheck.pipeline.guards import GUARDS_VERSION, REJECT_MESSAGES, guard_input
from claimcheck.pipeline.sources import SourceIndex
from claimcheck.pipeline.store import VerificationStore, input_hash
from claimcheck.pipeline.verifier import Verification, Verifier
from claimcheck.retrieval.retriever import Mode, RetrievalResult, Retriever
from claimcheck.schemas import (
    ClaimResult,
    InputInfo,
    LatencyMs,
    ResponseMeta,
    VerificationResponse,
    VerificationStatus,
)


@dataclass(frozen=True)
class ClaimTrace:
    """Everything behind one claim's result (evaluation and debugging)."""

    claim: ExtractedClaim
    retrieval: RetrievalResult
    verification: Verification


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


def new_verification_id() -> str:
    return f"ver_{ULID()}"


def _configured_models(llm: JSONCompleter) -> list[dict[str, str]]:
    providers = getattr(llm, "active_providers", None)
    if providers is None:  # FakeLLM wraps a real client
        providers = getattr(getattr(llm, "client", None), "active_providers", [])
    return [
        {"provider": p.name, "extractor": p.extractor_model, "verifier": p.verifier_model}
        for p in providers
    ]


class ClaimCheckService:
    def __init__(
        self,
        retriever: Retriever,
        llm: JSONCompleter,
        sources: SourceIndex,
        *,
        store: VerificationStore | None = None,
        limits: LimitsSettings | None = None,
        mode: Mode | str = Mode.HYBRID,
        top_k: int | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.sources = sources
        self.store = store
        self.limits = limits or LimitsSettings()
        self.mode = Mode(mode)
        self.top_k = top_k or retriever.settings.top_k
        self.extractor = Extractor(llm, max_claims=self.limits.max_claims_per_message)
        self.verifier = Verifier(llm)
        self._models = _configured_models(llm)
        self.pipeline_version = self._pipeline_version()

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        evaluation: bool = False,
        variant: str | None = None,
        mode: Mode | str = Mode.HYBRID,
        persist: bool | None = None,
        retriever: Retriever | None = None,
        llm: JSONCompleter | None = None,
    ) -> ClaimCheckService:
        """The app service (stores results in `data/app.db`) or, with `evaluation`, the
        evaluation service (evaluation LLM client, nothing stored unless `persist`)."""
        from claimcheck.db.session import make_engine
        from claimcheck.llm.client import LLMClient

        settings = settings or get_settings()
        retriever = retriever or Retriever.from_settings(variant)
        llm = llm or LLMClient.from_settings(settings, evaluation=evaluation)
        sources = SourceIndex.load(settings.paths.manifest, settings.paths.pages)
        persist = (not evaluation) if persist is None else persist
        store = VerificationStore(make_engine(settings.paths.app_db)) if persist else None
        return cls(retriever, llm, sources, store=store, limits=settings.limits, mode=mode)

    # ---------------------------------------------------------------------------

    @property
    def index_version(self) -> str:
        return self.retriever.store.index_version

    def _pipeline_version(self) -> str:
        parts = {
            "extractor_prompt": extractor_prompt.PROMPT_VERSION,
            "verifier_prompt": verifier_prompt.PROMPT_VERSION,
            "guards": GUARDS_VERSION,
            "retrieval": self.retriever.settings.model_dump(mode="json"),
            "mode": self.mode.value,
            "top_k": self.top_k,
            "max_claims": self.limits.max_claims_per_message,
            "models": self._models,
        }
        digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
        return f"p8-{digest[:12]}"

    def _default_models(self) -> dict[str, str]:
        if not self._models:
            return {"extractor": "", "verifier": ""}
        first = self._models[0]
        return {
            "extractor": f"{first['provider']}/{first['extractor']}",
            "verifier": f"{first['provider']}/{first['verifier']}",
        }

    # ---------------------------------------------------------------------------

    def get(self, verification_id: str) -> VerificationResponse | None:
        """A stored result (permalink), or None."""
        return None if self.store is None else self.store.get(verification_id)

    def verify(self, text: str, *, use_cache: bool = True) -> VerificationResponse:
        response, _ = self.verify_with_trace(text, use_cache=use_cache)
        return response

    def verify_with_trace(
        self, text: str, *, use_cache: bool = True
    ) -> tuple[VerificationResponse, list[ClaimTrace]]:
        t0 = time.perf_counter()
        clean = guard_input(text, self.limits.input_min_chars, self.limits.input_max_chars)
        key = input_hash(clean)

        if use_cache and self.store is not None:
            hit = self.store.lookup(key, self.index_version, self.pipeline_version)
            if hit is not None:
                meta = hit.meta.model_copy(
                    update={"cached": True, "latency_ms": LatencyMs(total=_ms(t0))}
                )
                return hit.model_copy(update={"meta": meta}), []

        t = time.perf_counter()
        extraction = self.extractor.extract(clean, use_cache=use_cache)
        extract_ms = _ms(t)
        models = self._default_models()
        models["extractor"] = f"{extraction.provider}/{extraction.model}"

        claims: list[ClaimResult] = []
        traces: list[ClaimTrace] = []
        retrieve_ms = verify_ms = 0
        verifier_models: list[str] = []
        if not extraction.rejected:
            for claim in extraction.claims:
                t = time.perf_counter()
                retrieval = self.retriever.retrieve(claim.claim_en, mode=self.mode, k=self.top_k)
                retrieve_ms += _ms(t)
                t = time.perf_counter()
                chunks = [p.chunk for p in retrieval.passages]
                verification = self.verifier.verify(claim, chunks, use_cache=use_cache)
                verify_ms += _ms(t)
                used = f"{verification.result.provider}/{verification.result.model}"
                if used not in verifier_models:
                    verifier_models.append(used)
                claims.append(self._claim_result(claim, retrieval, verification))
                traces.append(ClaimTrace(claim, retrieval, verification))
        if verifier_models:
            models["verifier"] = ", ".join(verifier_models)

        response = self._response(clean, extraction, claims, models)
        response.meta.latency_ms = LatencyMs(
            extract=extract_ms,
            retrieve=None if extraction.rejected else retrieve_ms,
            verify=None if extraction.rejected else verify_ms,
            total=_ms(t0),
        )
        if self.store is not None:
            self.store.save(response, key)
        return response, traces

    # ---------------------------------------------------------------------------

    def _claim_result(
        self, claim: ExtractedClaim, retrieval: RetrievalResult, verification: Verification
    ) -> ClaimResult:
        checked = verification.checked
        cited = {c.index for c in checked.citations}
        evidence = [
            p.to_evidence(cited=i in cited).model_copy(update={"chapter": chapter_label(p.chunk)})
            for i, p in enumerate(retrieval.passages)
        ]
        return ClaimResult(
            claim_en=claim.claim_en,
            verdict=checked.verdict,
            reasoning=checked.reasoning,
            flags=list(checked.flags),
            citations=[self.sources.citation(c) for c in checked.citations],
            evidence=evidence,
        )

    def _response(
        self,
        text: str,
        extraction: Extraction,
        claims: list[ClaimResult],
        models: dict[str, str],
    ) -> VerificationResponse:
        rejected = extraction.rejected
        return VerificationResponse(
            id=new_verification_id(),
            created_at=datetime.now(UTC),
            input=InputInfo(text=text, language=extraction.language),
            status=VerificationStatus.REJECTED if rejected else VerificationStatus.OK,
            reject_reason=extraction.reject_reason.value if rejected else None,
            message=REJECT_MESSAGES[extraction.reject_reason] if rejected else None,
            claims=claims,
            meta=ResponseMeta(
                index_version=self.index_version,
                pipeline_version=self.pipeline_version,
                models=models,
                cached=False,
            ),
        )
