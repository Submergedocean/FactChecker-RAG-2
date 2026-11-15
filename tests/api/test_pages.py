"""`GET /pages/{doc_id}/{page}` and its path-traversal guard."""

from __future__ import annotations

import pytest

from tests.api.conftest import GDP_ON_PDF, Harness
from tests.api.test_verify import assert_error

PNG = b"\x89PNG\r\n\x1a\n"


def test_page_png_with_highlight(h: Harness) -> None:
    resp = h.client.get(h.url("/pages/pes2526_ch01/3"), params={"quote": GDP_ON_PDF})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(PNG)
    assert resp.headers["X-Quote-Found"] == "true"


def test_page_without_or_with_missing_quote(h: Harness) -> None:
    resp = h.client.get(h.url("/pages/pes2526_ch01/1"))
    assert resp.status_code == 200 and resp.headers["X-Quote-Found"] == "false"
    resp = h.client.get(h.url("/pages/pes2526_ch01/1"), params={"quote": "not on this page"})
    assert resp.status_code == 200 and resp.headers["X-Quote-Found"] == "false"


def test_page_cached_on_disk(h: Harness) -> None:
    url = h.url("/pages/pes2526_ch01/3")
    first = h.client.get(url, params={"quote": GDP_ON_PDF}).content
    files = list(h.settings.paths.page_cache.glob("*.png"))
    assert len(files) == 1
    assert h.client.get(url, params={"quote": GDP_ON_PDF}).content == first


def test_page_cache_cap_evicts_oldest(h: Harness) -> None:
    h.state.renderer.cap_bytes = 1  # every new render evicts the older ones
    for p in (1, 2, 3):
        assert h.client.get(h.url(f"/pages/pes2526_ch01/{p}")).status_code == 200
    assert len(list(h.settings.paths.page_cache.glob("*.png"))) <= 1


@pytest.mark.parametrize(
    "path",
    [
        "/pages/pes9999_ch01/1",  # unknown doc_id
        "/pages/pes2526_ch01/0",  # page 0
        "/pages/pes2526_ch01/4",  # past the end (3 pages)
        "/pages/pes2526_ch01/-1",
        "/pages/pes2526_ch01/1.5",
        "/pages/pes2526_ch01/abc",
        "/pages/pes2526_ch01/99999999999999999999",
        "/pages/..%2F/1",
        "/pages/..%2F..%2Fpyproject.toml/1",
        "/pages/%2E%2E/1",
        "/pages/pes2526_ch01%2F..%2F..%2Fapp.db/1",
    ],
)
def test_page_rejects_unknown_and_traversal(h: Harness, path: str) -> None:
    assert_error(h.client.get(h.url(path)), 404, "NOT_FOUND")


def test_page_pdf_missing_on_disk_404(h: Harness) -> None:
    assert_error(h.client.get(h.url("/pages/pes2526_ch99/1")), 404, "NOT_FOUND")


def test_page_quote_too_long_422(h: Harness) -> None:
    resp = h.client.get(h.url("/pages/pes2526_ch01/1"), params={"quote": "q" * 501})
    assert_error(resp, 422, "INPUT_INVALID")


def test_pages_rate_limited(h: Harness) -> None:
    h.state.pages_limiter.per_minute = 2
    for _ in range(2):
        assert h.client.get(h.url("/pages/pes2526_ch01/1")).status_code == 200
    resp = h.client.get(h.url("/pages/pes2526_ch01/1"))
    assert_error(resp, 429, "RATE_LIMITED")
