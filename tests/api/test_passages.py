"""`GET /passages/{chunk_id}`."""

from __future__ import annotations

import pytest

from claimcheck.schemas import PassageResponse
from tests.api.conftest import API_CHUNKS, Harness
from tests.api.test_verify import assert_error


def test_passage_with_neighbours(h: Harness) -> None:
    first, second, other = API_CHUNKS
    body = h.client.get(h.url(f"/passages/{first.chunk_id}")).json()
    PassageResponse.model_validate(body)
    assert body["text"] == first.text
    assert body["chapter"] == "1 Growth and Investment"
    assert body["prev_chunk_id"] is None
    assert body["next_chunk_id"] == second.chunk_id
    assert body["source_url"].endswith(f"#page={first.page_start}")

    body = h.client.get(h.url(f"/passages/{second.chunk_id}")).json()
    assert body["prev_chunk_id"] == first.chunk_id
    assert body["next_chunk_id"] is None  # the next chunk is another document

    body = h.client.get(h.url(f"/passages/{other.chunk_id}")).json()
    assert body["kind"] == "table" and body["table_id"] == other.table_id
    assert body["prev_chunk_id"] is None and body["next_chunk_id"] is None


@pytest.mark.parametrize("cid", ["ffffffffffffffff", "zz", "0" * 17, "..%2F..%2Fetc"])
def test_passage_unknown_404(h: Harness, cid: str) -> None:
    assert_error(h.client.get(h.url(f"/passages/{cid}")), 404, "NOT_FOUND")
