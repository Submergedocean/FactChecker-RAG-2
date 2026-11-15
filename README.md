# FactChecker-RAG

**Claim Checker**: retrieval-augmented fact checking of economic claims against the Pakistan
Economic Survey.

A user enters a claim about Pakistan's economy in English, Urdu or Roman Urdu. The system splits
the message into up to three atomic claims, retrieves the most relevant passages from the
survey, and returns a verdict for each claim (`SUPPORTED`, `REFUTED`, or `NEI`, not enough
information) with exact quotes and page-level citations.

## How it works

1. **Corpus.** The 2024-25 and 2025-26 editions of the Pakistan Economic Survey (45 PDFs) are
   extracted page by page. Tables are detected and extracted separately from prose, so they are
   never flattened into running text.
2. **Chunking.** Prose is cut into ~150-word windows with ~30 words of overlap; tables are cut
   between rows with their header repeated. Every chunk keeps its edition, chapter and pages.
3. **Retrieval.** Hybrid search: BM25 (keyword) and dense embeddings (`BAAI/bge-small-en-v1.5`,
   FAISS) are fused with Reciprocal Rank Fusion; the top 5 passages are kept.
4. **Extraction.** An LLM turns the message into at most three English claims, or rejects it
   (off-topic, a question, not a claim, too vague).
5. **Verification.** An LLM judges each claim against its passages at temperature 0 and must
   cite passages with verbatim quotes. Code checks every quote against the passage; a verdict
   without a valid citation is downgraded to `NEI`.
6. **API.** FastAPI serves verification, permalinks, passages, highlighted PDF page images,
   documents and metadata. Repeated claims are answered from a result cache.

## Requirements

- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 and all dependencies)
- About 5 GB of free disk
- API keys for the LLM providers (Groq, Gemini) — not needed in fake-LLM mode

## Setup

```bash
uv sync
cp .env.example .env      # then add GROQ_API_KEY and GEMINI_API_KEY
```

### Data

The corpus is not stored in the repository. Place the data archive's `data/raw/` (PDFs) and
`data/processed/` (extracted text and search index) inside `data/`, or rebuild them from the
sources listed in `data/manifest.json`:

```bash
uv run claimcheck ingest download   # PDFs into data/raw/
uv run claimcheck ingest extract    # pages
uv run claimcheck ingest tables     # tables
uv run claimcheck ingest chunk      # chunks
uv run claimcheck index build       # embeddings, FAISS and BM25
```

The first run downloads the embedding model (about 130 MB) from Hugging Face.

## Usage

Check a claim from the command line:

```bash
uv run claimcheck pipeline verify "Pakistan's real GDP grew by 3.7 percent in FY2026." --evidence
```

Run the API server:

```bash
uv run claimcheck api serve                         # http://127.0.0.1:8000
CLAIMCHECK_FAKE_LLM=1 uv run claimcheck api serve   # canned LLM answers, no API keys
```

- Interactive API docs: `/api/v1/docs` (development only)
- OpenAPI schema: `/api/v1/openapi.json`
- Health: `/healthz`, readiness: `/readyz`

## API

| Method and path | Purpose |
|-----------------|---------|
| `POST /api/v1/verify` | Verify a message (`{"text": "10–500 characters"}`) |
| `GET /api/v1/verifications/{id}` | A stored result (permalink) |
| `POST /api/v1/verifications/{id}/feedback` | Rate one claim's result |
| `GET /api/v1/passages/{chunk_id}` | A passage and its neighbours |
| `GET /api/v1/pages/{doc_id}/{page}?quote=` | PDF page image with the quote highlighted |
| `GET /api/v1/documents` | The documents in the corpus |
| `GET /api/v1/meta` | Versions, models, limits, example claims |
| `GET /api/v1/eval/summary` | Evaluation results, when available |

Errors share one envelope: `{"error": {"code", "message", "retry_after_s"}}`. Clients are
limited to 6 checks per minute and 60 per day.

## Frontend

The web app lives in `web/` (React, TypeScript, Vite). During development, run the Vite dev
server with `/api` proxied to `http://127.0.0.1:8000`. After `npm run build`, the API server
serves `web/dist/` at `/` on the same origin.

## Configuration

Settings are in `configs/default.yaml` and can be overridden with `CLAIMCHECK_*` environment
variables (nested keys use `__`, e.g. `CLAIMCHECK_RETRIEVAL__TOP_K=10`) or a `.env` file.

## Project layout

```
src/claimcheck/
  ingest/      PDF download, page and table extraction, cleaning, chunking
  index/       embeddings, FAISS and BM25 index
  retrieval/   dense, sparse and hybrid retrieval
  llm/         LLM client, prompts, cache and rate limits
  pipeline/    claim extraction, verification, guards, result store
  claims/      claims dataset tooling
  api/         FastAPI app
configs/       default settings
tests/         unit and API tests
```

## Development

```bash
uv run pytest -q      # unit and API tests (no data or API keys needed)
uv run ruff check .   # lint
```
