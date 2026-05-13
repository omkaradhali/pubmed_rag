# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `haiku` and `sonnet` as first-class `LLM_PROVIDER` aliases for Anthropic models
- `get_model()` public accessor in `embed.py` — lazy loads the sentence-transformer on first call
- `Chunk` TypedDict in `types.py` for typed pipeline data
- `CORS_ORIGINS` env var to restrict browser access (defaults to localhost)
- `pytest-cov` with 80% coverage threshold enforced in CI
- `ruff format --check` step in CI
- Python 3.12 and 3.13 version matrix in CI
- GitHub issue templates and PR template
- `SECURITY.md` with responsible disclosure policy
- `Makefile` for common dev tasks
- PyPI classifiers, keywords, and project URLs in `pyproject.toml`
- `__version__` exported from `pubmed_rag` package; API version reads from package metadata

### Fixed
- `embed_chunks` no longer mutates the input list; returns new dicts instead
- `_run_incremental_update` no longer appends duplicate records to `abstracts.jsonl`
- `eval` optional dependencies now include `langchain-anthropic` and `langchain-community`

## [0.1.0] - 2026-04-13

Initial release.

### Added
- Full ingestion pipeline: PubMed E-utilities → chunking → sentence-transformer embeddings → ChromaDB
- Citation-enforced generation with pluggable LLM providers (Ollama, Anthropic, OpenAI)
- Pluggable vector store backend: ChromaDB (dev) / Qdrant (production)
- FastAPI backend with request ID tracing, JSON structured logging, and Swagger UI
- RAGAS evaluation suite with 20 clinical oncology questions
- Docker image with pre-baked `all-MiniLM-L6-v2` model weights
- GitHub Actions CI with lint and tests
- 7 Architecture Decision Records (ADR-033 through ADR-039)
- CITATION.cff for academic citation
