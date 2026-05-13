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

    # Vector store — chroma (default, embedded, zero config) or qdrant (production)
    vector_store_backend: str = "chroma"
    chroma_persist_dir: str = "./data/chroma_db"
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # Embedding provider — miniml (default, local), medcpt (biomedical, local), openai (production)
    embedding_provider: str = "miniml"

    # LLM provider — ollama (default), anthropic, haiku, sonnet, openai
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434/v1"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # CORS — restrict browser origins in production; comma-separated or JSON array
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Corpus — specialty and time window
    pubmed_specialty: str = "oncology"
    pubmed_years_back: int = 10

    # Ingestion
    ingest_batch_size: int = 500

    # Observability — leave empty to disable (Phase 2)
    logfire_token: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # API server
    log_level: str = "INFO"

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
        needs_openai = self.llm_provider == "openai" or self.embedding_provider == "openai"
        if needs_openai and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set when LLM_PROVIDER=openai or EMBEDDING_PROVIDER=openai"
            )
        needs_anthropic = self.llm_provider in ("anthropic", "haiku", "sonnet")
        if needs_anthropic and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set when LLM_PROVIDER=anthropic/haiku/sonnet"
            )
        if self.vector_store_backend == "qdrant" and not self.qdrant_url:
            raise ValueError("QDRANT_URL must be set when VECTOR_STORE_BACKEND=qdrant")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings, parsed once at startup."""
    return Settings()
