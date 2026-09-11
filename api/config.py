from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and a .env file.

    All fields have defaults suitable for local development with Ollama and ChromaDB.
    Override any field via an environment variable or a .env file in the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # NCBI
    ncbi_api_key: str = ""

    # Vector store — ChromaDB, embedded, zero config
    chroma_persist_dir: str = "./data/chroma_db"

    # Embedding provider — miniml (default, local), bge (stronger, local),
    # medcpt (biomedical, local)
    embedding_provider: str = "miniml"

    # Reranker — cross-encoder second retrieval stage
    rerank_enabled: bool = True
    rerank_model: str = "ncbi/MedCPT-Cross-Encoder"
    rerank_pool: int = 30  # child candidates shortlisted before reranking

    # Hybrid retrieval — BM25 + dense → RRF fusion
    hybrid_search_enabled: bool = False

    # PHI/PII scrubbing — de-identify queries before any
    # cloud egress. "auto" scrubs only when a cloud provider is configured
    # (LLM_PROVIDER=anthropic/haiku/sonnet/openai); "on" always scrubs; "off"
    # never scrubs (not for clinical use). Runtime gate reads the env var
    # directly (pipeline.py style); this field keeps .env discoverable.
    phi_scrubbing: str = "auto"
    phi_spacy_model: str = "en_core_web_lg"

    # LLM provider — ollama (default), anthropic, haiku, sonnet, openai
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434/v1"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # CORS — restrict browser origins in production; comma-separated or JSON array
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Ingestion — the PubMed search string, and how many abstracts per run.
    # Multi-specialty deployments (docs/decisions/multi-specialty-corpus.md)
    # pick a specialty via the --specialty CLI flag on pipeline.py instead;
    # these are the fallback when no flag is passed.
    ingest_query: str = "oncology[Title/Abstract]"
    ingest_max_results: int = 500

    # API server
    log_level: str = "INFO"

    # Audit log — append-only JSONL record of every clinical query.
    # One immutable line per query: request_id, timestamp, post-scrub query, retrieved
    # PMIDs, model, truncated answer, guardrail results, confidence tier. Distinct from
    # app logs (debug, rotatable); this is the compliance/accountability trail.
    audit_log_path: str = "./data/audit.jsonl"
    audit_answer_max_chars: int = 500

    # Auth — comma-separated static API keys. Empty = auth disabled (safe for local dev).
    # Stored as a raw string so pydantic-settings doesn't try to JSON-decode it.
    # Parsed into a list inside verify_api_key (api/dependencies.py).
    api_keys: str = ""

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list) -> list[str]:
        """Accept both comma-separated string and JSON array for CORS_ORIGINS."""
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                import json

                return json.loads(v)
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @model_validator(mode="after")
    def check_required_keys(self) -> "Settings":
        """Raise ValueError if a required API key is missing for the configured provider."""
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai")
        needs_anthropic = self.llm_provider in ("anthropic", "haiku", "sonnet")
        if needs_anthropic and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set when LLM_PROVIDER=anthropic/haiku/sonnet"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings, parsed once at startup."""
    return Settings()
