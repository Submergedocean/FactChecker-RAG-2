"""Application settings.

Precedence (highest first): environment variables (`CLAIMCHECK_*`, nested with `__`),
`.env`, the YAML file named by `CLAIMCHECK_CONFIG` (default `configs/default.yaml`),
then the defaults below. Secrets (API keys) come only from the environment.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

Variant = Literal["table_aware", "plain_text"]


class PathsSettings(BaseModel):
    data_dir: Path = REPO_ROOT / "data"
    manifest: Path = REPO_ROOT / "data" / "manifest.json"
    raw_dir: Path = REPO_ROOT / "data" / "raw"
    processed_dir: Path = REPO_ROOT / "data" / "processed"
    claims_dir: Path = REPO_ROOT / "data" / "claims"
    cache_dir: Path = REPO_ROOT / "data" / "cache"
    app_db: Path = REPO_ROOT / "data" / "app.db"
    eval_cache_db: Path = REPO_ROOT / "data" / "cache" / "eval_llm_cache.db"
    reports_dir: Path = REPO_ROOT / "reports"

    @property
    def pages(self) -> Path:
        """pages.parquet, written by `claimcheck ingest extract`."""
        return self.processed_dir / "pages.parquet"

    @property
    def extract_report(self) -> Path:
        """Run report of `ingest extract`: flagged pages, regions, headers and footers."""
        return self.processed_dir / "extract_report.json"

    @property
    def tables(self) -> Path:
        """tables.parquet, written by `claimcheck ingest tables`."""
        return self.processed_dir / "tables.parquet"

    @property
    def tables_report(self) -> Path:
        """Run report of `ingest tables`: flagged tables and the reason for each."""
        return self.processed_dir / "tables_report.json"

    def variant_dir(self, variant: Variant) -> Path:
        return self.processed_dir / variant

    def chunks(self, variant: Variant) -> Path:
        """chunks.parquet of one variant, written by `claimcheck ingest chunk`."""
        return self.variant_dir(variant) / "chunks.parquet"

    def chunk_report(self, variant: Variant) -> Path:
        """Run report of `ingest chunk`: counts, dropped duplicates, numeric-heavy text chunks."""
        return self.variant_dir(variant) / "chunk_report.json"

    def faiss_index(self, variant: Variant) -> Path:
        """faiss.index of one variant: row i is row i of chunks.parquet."""
        return self.variant_dir(variant) / "faiss.index"

    def index_meta(self, variant: Variant) -> Path:
        """index_meta.json of one variant, written by `claimcheck index build`."""
        return self.variant_dir(variant) / "index_meta.json"

    @property
    def page_cache(self) -> Path:
        """Rendered /pages PNGs, capped at `api.page_cache_mb`; never committed."""
        return self.cache_dir / "pages"

    @property
    def eval_summary(self) -> Path:
        """reports/eval/summary.json, written by the evaluation and served by GET /eval/summary."""
        return self.reports_dir / "eval" / "summary.json"

    @property
    def embedding_cache(self) -> Path:
        """Passage vectors keyed by text, reused by rebuilds; never committed."""
        return self.cache_dir / "embeddings"


class IngestSettings(BaseModel):
    request_interval_s: float = 1.0
    max_retries: int = 3
    timeout_s: float = 60.0
    # Extraction and cleaning
    header_footer_share: float = 0.6  # repeats on 60 %+ of a document's pages
    edge_band: float = 0.12  # top/bottom share of the page searched for headers and footers
    word_count_tolerance: float = 0.20  # pdfplumber vs PyMuPDF per-page word count
    workers: int = 0  # extraction processes; 0 = one per CPU core
    # A table with fewer filled cells or more ragged rows than this is
    # emitted as plain text (flag `fallback_plain`) instead of as cells.
    table_min_fill: float = 0.5  # share of non-empty value cells
    table_max_ragged: float = 0.25  # share of rows with under half the usual number of cells


class ChunkingSettings(BaseModel):
    variant: Variant = "table_aware"
    # Proposal 6.1: ~150-word sliding window with ~30-word overlap.
    window_words: int = 150
    overlap_words: int = 30
    min_words: int = 20
    max_tokens: int = 512  # embedding model input limit; every chunk must fit
    table_max_tokens: int = 350  # whole rows per table chunk, header repeated
    # A text chunk whose words are more than this share numbers is
    # listed in the chunk report (it may hold a table extraction did not find).
    numeric_share: float = 0.4


class EmbeddingSettings(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"  # proposal 6.2
    # Pinned (checked live on the Hub 2026-11-14: main = this commit, 2024-02-22).
    revision: str | None = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    dim: int = 384
    device: str = "cpu"  # CPU torch, also on macOS (reproducible vectors)
    batch_size: int = 32
    normalize: bool = True
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    passage_prefix: str = ""


class RetrievalSettings(BaseModel):
    top_k: int = 5  # proposal 6.3: K = 5
    bm25_k: int = 50
    dense_k: int = 50
    rrf_k: int = 60


class ProviderSettings(BaseModel):
    """One OpenAI-compatible LLM endpoint. The key is read from `api_key_env`."""

    name: str
    base_url: str
    api_key_env: str
    extractor_model: str
    verifier_model: str
    # Limits per model, as Groq and Gemini count them. Days are UTC dates in `llm_usage`
    # (providers reset on their own clocks, so the ledger is an approximation).
    requests_per_minute: int = 30
    requests_per_day: int = 1000
    tokens_per_day: int | None = None  # prompt + completion tokens; None = not enforced
    json_mode: bool = True  # send response_format {"type": "json_object"}
    # Extra request fields this endpoint understands (e.g. reasoning_effort); sent as given.
    extra_body: dict[str, Any] = Field(default_factory=dict)

    def model_for(self, role: Literal["extractor", "verifier"]) -> str:
        return self.extractor_model if role == "extractor" else self.verifier_model


class LLMSettings(BaseModel):
    """OpenAI-compatible providers tried in order, one cache, one ledger."""

    providers: list[ProviderSettings] = Field(default_factory=list)
    temperature: float = 0.0
    timeout_s: float = 30.0  # per request
    max_retries: int = 2  # retries after the first attempt, on 429, 5xx, timeouts
    backoff_base_s: float = 0.5  # full-jitter backoff: uniform(0, min(max, base * 2**n))
    backoff_max_s: float = 8.0
    # A provider asking to wait longer than this (Retry-After) counts as exhausted: next provider.
    max_retry_after_s: float = 20.0
    # Longest wait for room in a provider's requests-per-minute window before falling back.
    max_rate_wait_s: float = 5.0
    # Fallback to the next provider on exhaustion. Evaluation mode always turns it off, so an
    # evaluation run never mixes models.
    fallback: bool = True

    def api_key(self, provider: ProviderSettings) -> str | None:
        """The provider's key from the environment, else from the repo's `.env` (pydantic
        settings read `.env` only for `CLAIMCHECK_*` fields, so keys are looked up here).
        A variable set to "" in the environment means no key; `.env` is not consulted."""
        if provider.api_key_env in os.environ:  # set, even to "" (= no key), wins over .env
            return os.environ[provider.api_key_env] or None
        env_file = REPO_ROOT / ".env"
        if env_file.is_file():
            return dotenv_values(env_file).get(provider.api_key_env) or None
        return None


class LimitsSettings(BaseModel):
    input_min_chars: int = 10
    input_max_chars: int = 500
    feedback_comment_max_chars: int = 500
    max_claims_per_message: int = 3
    per_client_requests_per_minute: int = 6
    per_client_requests_per_day: int = 60
    daily_llm_request_budget: int = 600  # from the providers' free-tier limits
    # Cheaper endpoints get their own per-client buckets (writes and page rendering).
    per_client_feedback_per_minute: int = 20
    per_client_feedback_per_day: int = 200
    per_client_pages_per_minute: int = 60
    per_client_pages_per_day: int = 1000
    max_concurrent_llm: int = 4  # semaphore on concurrent verifications


class ClaimsSettings(BaseModel):
    """Proposal 7.2 targets and the drafting defaults."""

    target_total: int = 600  # "about 600"
    total_tolerance: float = 0.10  # final gate: within 10 % of the target
    label_mix: dict[str, float] = Field(
        default_factory=lambda: {"SUPPORTED": 0.40, "REFUTED": 0.40, "NEI": 0.20}
    )
    label_tolerance: float = 0.05  # "within 5 points of 40/40/20"
    tabular_share: float = 0.50  # "about half tabular"
    tabular_tolerance: float = 0.10
    english_share: float = 0.80  # 80/20 English to Urdu + Roman Urdu
    language_tolerance: float = 0.05
    split_ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15}
    )
    split_seed: int = 2026
    kappa_min: float = 0.70
    kappa_sample: int = 100
    pilot_size: int = 40
    # Drafting: share of facts that also get one Urdu and one Roman Urdu claim. With about
    # 180 drafted facts (2 English claims each) and 120 hand-written NEI claims, 1/3 gives
    # 60 + 60 translated claims = 20 % of 600.
    translate_share: float = 0.34
    draft_seed: int = 2026
    min_text_words: int = 40  # text chunks shorter than this are not sampled
    # Kinds of claim the drafter is asked for, by share of drafted facts (a hint: the model
    # writes what the passage supports). claim-draft-v1 wrote only `numeric` claims.
    reasoning_mix: dict[str, float] = Field(
        default_factory=lambda: {"numeric": 0.5, "trend": 0.3, "comparison": 0.2}
    )


class APISettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=list)
    static_dir: Path | None = REPO_ROOT / "web" / "dist"
    # The client is the Nth entry from the right of X-Forwarded-For, N = proxy hops
    # in front of the app (0 = no proxy: the socket peer). Measure it on the host.
    trusted_proxy_hops: int = 0
    # Origins allowed to frame the app (CSP frame-ancestors); the Space shows it in an iframe.
    frame_ancestors: list[str] = Field(default_factory=lambda: ["https://huggingface.co"])
    page_zoom: float = 1.5  # /pages PNG scale (1.0 = 72 dpi)
    page_cache_mb: int = 200  # on-disk cap of rendered pages
    example_claims: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLAIMCHECK_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    env: Literal["dev", "test", "prod"] = "dev"
    # CLAIMCHECK_FAKE_LLM=1: canned LLM answers for UI work; refused in prod.
    fake_llm: bool = False
    editions: list[str] = Field(default_factory=lambda: ["2024-25", "2025-26"])
    latest_edition: str = "2025-26"  # prevails when editions conflict
    paths: PathsSettings = Field(default_factory=PathsSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    limits: LimitsSettings = Field(default_factory=LimitsSettings)
    claims: ClaimsSettings = Field(default_factory=ClaimsSettings)
    api: APISettings = Field(default_factory=APISettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls),
            file_secret_settings,
        )


class _YamlSource(PydanticBaseSettingsSource):
    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        path = Path(os.environ.get("CLAIMCHECK_CONFIG", DEFAULT_CONFIG_PATH))
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
