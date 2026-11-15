"""Verification of one English claim against its retrieved passages.

The verifier model answers in JSON at temperature 0; `pipeline.guards.apply_guards` then
checks every citation in code and adjusts the verdict and flags.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from claimcheck.llm.client import LLMResult
from claimcheck.llm.prompts import verifier as prompt
from claimcheck.llm.prompts.extractor import ExtractedClaim
from claimcheck.llm.prompts.verifier import VerifierOutput
from claimcheck.pipeline.extractor import JSONCompleter
from claimcheck.pipeline.guards import CheckedVerdict, apply_guards
from claimcheck.schemas import ChunkRecord


@dataclass
class Verification:
    checked: CheckedVerdict
    raw: VerifierOutput
    result: LLMResult[VerifierOutput]

    @property
    def latency_ms(self) -> int:
        return 0 if self.result.cached else (self.result.latency_ms or 0)


class Verifier:
    def __init__(self, llm: JSONCompleter) -> None:
        self.llm = llm

    def verify(
        self, claim: ExtractedClaim, chunks: Sequence[ChunkRecord], *, use_cache: bool = True
    ) -> Verification:
        result = self.llm.complete_json(
            prompt.build_messages(claim.claim_en, chunks),
            VerifierOutput,
            role="verifier",
            prompt_version=prompt.PROMPT_VERSION,
            use_cache=use_cache,
        )
        checked = apply_guards(result.value, chunks, claim)
        return Verification(checked=checked, raw=result.value, result=result)
