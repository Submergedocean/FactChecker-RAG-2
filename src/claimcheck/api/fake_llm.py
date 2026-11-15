"""`CLAIMCHECK_FAKE_LLM=1`: canned LLM answers for UI work. No provider is called.

The answers go through the real pipeline (retrieval, citation check, guards, storage), so the
UI sees real passages and citations:

- Extractor: a message ending in "?" is rejected as a question; anything else is one claim,
  the message itself.
- Verifier: the verdict cycles SUPPORTED / REFUTED / NEI by a hash of the claim (stable, so a
  repeated claim gets the same answer); SUPPORTED and REFUTED quote the first words of P1.

Startup refuses this mode with `CLAIMCHECK_ENV=prod` (`security.check_startup`).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from pydantic import BaseModel

from claimcheck.llm.client import LLMResult, Messages, Role

PROVIDER = "fake"
_MESSAGE_RE = re.compile(r"Message \(JSON string\): (.*)\Z", re.DOTALL)
_CLAIM_RE = re.compile(r"Claim \(JSON string\): (.*?)\n\nPassages:", re.DOTALL)
_PASSAGE_RE = re.compile(r'<passage id="(P\d+)">[^\n]*\n(.*?)\n</passage>', re.DOTALL)
_VERDICTS = ("SUPPORTED", "REFUTED", "NEI")
QUOTE_WORDS = 12


@dataclass(frozen=True)
class _CannedProvider:
    name: str = PROVIDER
    extractor_model: str = "canned-extractor"
    verifier_model: str = "canned-verifier"


def _user_text(messages: Messages) -> str:
    return next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")


def _extract(user: str) -> dict[str, object]:
    m = _MESSAGE_RE.search(user)
    text = json.loads(m.group(1)) if m else user
    if text.rstrip().endswith("?"):
        return {"checkable": False, "reject_reason": "question", "language": "en", "claims": []}
    claim = {"claim_en": text, "indicator": None, "period": None, "value": None, "unit": None}
    return {"checkable": True, "reject_reason": None, "language": "en", "claims": [claim]}


def _verify(user: str) -> dict[str, object]:
    m = _CLAIM_RE.search(user)
    claim = json.loads(m.group(1)) if m else user
    verdict = _VERDICTS[int(hashlib.sha256(claim.encode()).hexdigest(), 16) % 3]
    passages = _PASSAGE_RE.findall(user)
    if verdict == "NEI" or not passages:
        return {
            "verdict": "NEI",
            "reasoning": "Canned answer (fake LLM): the passages are not compared.",
            "citations": [],
        }
    label, text = passages[0]
    words = text.replace("|", " ").split()
    quote = " ".join(words[:QUOTE_WORDS])
    return {
        "verdict": verdict,
        "reasoning": f"Canned answer (fake LLM): {verdict} citing the start of {label}.",
        "citations": [{"passage": label, "quote": quote}],
    }


class CannedLLM:
    """Implements the pipeline's `JSONCompleter` protocol with canned answers."""

    def __init__(self) -> None:
        self.active_providers = [_CannedProvider()]

    def complete_json[T: BaseModel](
        self,
        messages: Messages,
        schema: type[T],
        *,
        role: Role,
        prompt_version: str,
        use_cache: bool = True,
    ) -> LLMResult[T]:
        user = _user_text(messages)
        payload = _extract(user) if role == "extractor" else _verify(user)
        text = json.dumps(payload, ensure_ascii=False)
        provider = self.active_providers[0]
        return LLMResult(
            value=schema.model_validate(payload),
            text=text,
            provider=provider.name,
            model=provider.extractor_model if role == "extractor" else provider.verifier_model,
            cached=False,
            latency_ms=0,
        )
