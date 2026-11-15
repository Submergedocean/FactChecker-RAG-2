"""Stored verifications: permalinks and the result cache.

A result is stored once with its exact API payload. The cache is a lookup on
`ix_verification_lookup` (input hash, index version, pipeline version, newest first), so a
changed prompt, setting or index never serves an old answer.
"""

from __future__ import annotations

import hashlib
import json
import re

from sqlalchemy import Engine, select

from claimcheck.db.models import Verification, VerificationClaim
from claimcheck.db.session import make_session_factory, session_scope
from claimcheck.schemas import VerificationResponse


def input_hash(text: str) -> str:
    """sha256 of the guarded input with whitespace collapsed."""
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


class VerificationStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._sessions = make_session_factory(engine)

    def lookup(
        self, input_hash: str, index_version: str, pipeline_version: str
    ) -> VerificationResponse | None:
        stmt = (
            select(Verification.response_json)
            .where(
                Verification.input_hash == input_hash,
                Verification.index_version == index_version,
                Verification.pipeline_version == pipeline_version,
            )
            .order_by(Verification.created_at.desc())
            .limit(1)
        )
        with session_scope(self._sessions) as s:
            payload = s.execute(stmt).scalar_one_or_none()
        return None if payload is None else VerificationResponse.model_validate_json(payload)

    def get(self, verification_id: str) -> VerificationResponse | None:
        with session_scope(self._sessions) as s:
            row = s.get(Verification, verification_id)
            payload = row.response_json if row is not None else None
        return None if payload is None else VerificationResponse.model_validate_json(payload)

    def save(self, response: VerificationResponse, input_hash: str) -> None:
        with session_scope(self._sessions) as s:
            s.add(
                Verification(
                    id=response.id,
                    created_at=response.created_at.isoformat(),
                    input_text=response.input.text,
                    input_hash=input_hash,
                    language=response.input.language.value if response.input.language else None,
                    status=response.status.value,
                    reject_reason=response.reject_reason,
                    response_json=response.model_dump_json(),
                    index_version=response.meta.index_version,
                    pipeline_version=response.meta.pipeline_version,
                    latency_ms=response.meta.latency_ms.total,
                )
            )
            s.flush()  # parent row first: verification_claim has a foreign key on it
            for i, claim in enumerate(response.claims):
                s.add(
                    VerificationClaim(
                        verification_id=response.id,
                        claim_index=i,
                        claim_en=claim.claim_en,
                        verdict=claim.verdict.value,
                        flags=json.dumps(claim.flags),
                        n_citations=len(claim.citations),
                    )
                )
