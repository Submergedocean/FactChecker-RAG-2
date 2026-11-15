"""A tiny in-memory corpus and claim builders for the claims tests (no data needed)."""

from __future__ import annotations

import hashlib
from typing import Any

from claimcheck.claims.validate import ChunkView, Corpus, TableView
from claimcheck.schemas import (
    ChunkKind,
    ChunkRecord,
    Claim,
    DocKind,
    EvidenceRef,
    ManifestEntry,
)

INFLATION_PAGE = (
    "Chapter 7\nInflation\nDuring July-April FY 2026, average CPI inflation stood at 6.2 "
    "percent, compared with 4.7 percent in the same period last year. Food inflation eased."
)
INFLATION_PAGE_3 = "Core inflation in urban areas declined to 7.1 percent in April 2026."

CROPS_TABLE = TableView(
    table_id="pes2526_ch02:p5:t1",
    doc_id="pes2526_ch02",
    page=5,
    header=["Crops", "2024-25", "2025-26 P", "% Change"],
    labels=["Wheat", "Rice", "A MAJOR / Cotton", "B MINOR / Cotton"],
    cells=[
        ["Wheat", "31,806", "28,975", "-8.9"],
        ["Rice", "9,720", "9,323", "-4.1"],
        ["A MAJOR / Cotton", "5,550", "7,084", "27.6"],
        ["B MINOR / Cotton", "120", "130", "8.3"],
    ],
    lead=1,
)

TABLE_CHUNK_TEXT = (
    "TABLE 2.1 PRODUCTION OF CROPS\nUnits: 000 tonnes\n"
    "| Crops | 2024-25 | 2025-26 P | % Change |\n| --- | --- | --- | --- |\n"
    "| Wheat | 31,806 | 28,975 | -8.9 |\n| Rice | 9,720 | 9,323 | -4.1 |\n"
    "| A MAJOR / Cotton | 5,550 | 7,084 | 27.6 |"
)
PLAIN_TABLE_TEXT = (
    "TABLE 2.1 PRODUCTION OF CROPS 000 tonnes Crops 2024-25 2025-26 P % Change "
    "Wheat 31,806 28,975 -8.9 Rice 9,720 9,323 -4.1 A MAJOR Cotton 5,550 7,084 27.6"
)


def cid(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def make_corpus(*, plain_has_table: bool = True) -> Corpus:
    views = {
        "table_aware": [
            ChunkView(cid("ta-infl"), "pes2526_ch07", 2, 2, INFLATION_PAGE),
            ChunkView(cid("ta-core"), "pes2526_ch07", 3, 3, INFLATION_PAGE_3),
            ChunkView(cid("ta-crops"), "pes2526_ch02", 5, 5, TABLE_CHUNK_TEXT),
        ],
        "plain_text": [
            ChunkView(
                cid("pt-infl"), "pes2526_ch07", 2, 3, INFLATION_PAGE + " " + INFLATION_PAGE_3
            ),
        ],
    }
    if plain_has_table:
        views["plain_text"].append(
            ChunkView(cid("pt-crops"), "pes2526_ch02", 4, 5, PLAIN_TABLE_TEXT)
        )
    return Corpus(
        docs={"pes2526_ch07": 30, "pes2526_ch02": 38, "pes2526_overview": 15},
        pages={
            ("pes2526_ch07", 2): (INFLATION_PAGE, INFLATION_PAGE),
            ("pes2526_ch07", 3): (INFLATION_PAGE_3, INFLATION_PAGE_3),
            ("pes2526_ch02", 5): ("Wheat production fell.", PLAIN_TABLE_TEXT),
        },
        tables={CROPS_TABLE.table_id: CROPS_TABLE},
        chunk_views=views,
    )


def chunk_record(
    text: str,
    *,
    doc_id: str = "pes2526_ch07",
    page_start: int = 2,
    page_end: int | None = None,
    kind: ChunkKind = ChunkKind.TEXT,
    table_id: str | None = None,
    edition: str = "2025-26",
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=cid(f"{doc_id}:{page_start}:{text}"),
        doc_id=doc_id,
        edition=edition,
        chapter_no=7,
        chapter_title="Inflation",
        page_start=page_start,
        page_end=page_end or page_start,
        kind=kind,
        table_id=table_id,
        text=text,
        embed_text=text,
        bm25_tokens=[],
        n_words=len(text.split()),
        n_tokens=len(text.split()),
    )


def manifest_entry(doc_id: str = "pes2526_ch07", chapter_no: int | None = 7) -> ManifestEntry:
    return ManifestEntry(
        doc_id=doc_id,
        edition="2025-26",
        kind=DocKind.CHAPTER if chapter_no else DocKind.OVERVIEW,
        chapter_no=chapter_no,
        title="Inflation",
        source_url=f"https://example.org/{doc_id}.pdf",
        filename=f"{doc_id}.pdf",
    )


TEXT_EV = {
    "doc_id": "pes2526_ch07",
    "page": 2,
    "quote": "During July-April FY 2026, average CPI inflation stood at 6.2 percent",
}
TABLE_EV = {
    "doc_id": "pes2526_ch02",
    "page": 5,
    "table_id": "pes2526_ch02:p5:t1",
    "quote": "28,975",
    "row_label": "Wheat",
    "col_label": "2025-26 P",
}


def claim(n: int, **kw: Any) -> Claim:
    data: dict[str, Any] = {
        "claim_id": f"C{n:04d}",
        "fact_id": f"F{n:04d}",
        "claim_text": f"Average CPI inflation was {n}.5 percent in FY2026 in Pakistan.",
        "label": "SUPPORTED",
        "language": "en",
        "claim_type": "textual",
        "reasoning_type": "numeric",
        "topic": "inflation",
        "evidence": [TEXT_EV],
    }
    data.update(kw)
    if data["label"] == "NEI" and "evidence" not in kw:
        data["evidence"] = []
    data["evidence"] = [EvidenceRef.model_validate(e) for e in data["evidence"]]
    return Claim.model_validate(data)
