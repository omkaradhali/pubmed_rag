# pubmed-rag

[![CI](https://github.com/omkaradhali/pubmed_rag/actions/workflows/ci.yml/badge.svg)](https://github.com/omkaradhali/pubmed_rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)

A production-grade RAG pipeline for clinical literature. Fetch PubMed abstracts, embed them into a vector store, and answer natural language questions grounded in retrieved papers with inline citations.

**Built for:** clinical researchers, bioinformaticians, and developers learning biomedical RAG.

---

## Features

- **Full ingestion pipeline** — PubMed E-utilities → chunking → sentence-transformer embeddings → ChromaDB
- **Citation-enforced generation** — the LLM is instructed to cite every claim inline; hallucinated sources are structurally prevented
- **Pluggable providers** — swap the LLM (Ollama, Anthropic, OpenAI) and embedding model via a single env var
- **FastAPI backend** — structured JSON responses, request ID tracing, Swagger docs at `/docs`
- **RAGAS evaluation suite** — 20 clinical oncology questions covering mechanism, biomarker, prognosis, treatment, and epidemiology query types
- **Docker + CI** — ready-to-run Docker image and GitHub Actions workflow included
- **Production path** — swap ChromaDB → Qdrant and `all-MiniLM-L6-v2` → `text-embedding-3-small` with two env var changes

---

## Architecture

```
PubMed E-utilities
       │
       ▼
  ingest.py ──────────────────── abstracts.jsonl
       │                          (pmid, title, abstract, authors,
       │                           journal, year, doi, mesh_terms, ...)
       ▼
  chunk.py ───────────────────── chunks.jsonl
       │                          (1,000-char windows, 100-char overlap,
       │                           full source metadata preserved)
       ▼
  embed.py ───────────────────── 384-dim dense vectors
       │                          (all-MiniLM-L6-v2, L2-normalised)
       ▼
vectorstore.py ─────────────────  ChromaDB (default) / Qdrant (production)
                                          │
         query ───────────────────────────┘
                                          │
                                          ▼
                                retrieve.py  (cosine similarity, top-k)
                                          │
                                          ▼
                                generate.py  (citation-enforced prompt)
                                          │
                                          ▼
                                 answer + cited sources
```

The pipeline runs in two modes:

- **`incremental`** (default) — queries the pre-seeded vector store. Fast, ~1–3 sec, mostly LLM latency.
- **`full`** — wipes and rebuilds the corpus from scratch before querying. Use when corpus is stale.

---

## Quick Start

### Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- [Ollama](https://ollama.com) running locally (default LLM provider), **or** an `ANTHROPIC_API_KEY`

### Install

```bash
git clone https://github.com/omkaradhali/pubmed_rag
cd pubmed_rag
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env
# Edit .env — set NCBI_API_KEY (optional) and your chosen LLM provider key
```

### Seed the corpus and run a query

```bash
# Fetch 500 oncology abstracts, build the vector store, and answer a question
python -m pubmed_rag.pipeline "What is the mechanism of PD-1 checkpoint inhibition?" --mode full
```

### Run the API

```bash
uvicorn api.main:app --reload --port 8001
# Open http://localhost:8001/docs for the interactive Swagger UI
```

### Query via API

```bash
curl -s -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What biomarkers predict response to immunotherapy?"}' | jq .
```

### Step-by-step pipeline (CLI)

```bash
# 1. Fetch abstracts
python -m pubmed_rag.ingest --query "oncology[Title/Abstract]" --max-results 500 \
  --output data/abstracts.jsonl

# 2. Chunk
python -m pubmed_rag.chunk --input data/abstracts.jsonl --output data/chunks.jsonl

# 3. Embed
python -m pubmed_rag.embed --input data/chunks.jsonl --output data/embeddings.jsonl

# 4. Seed vector store
python -m pubmed_rag.vectorstore --input data/embeddings.jsonl

# 5. Query
python -m pubmed_rag.pipeline "What are the treatments for HER2-positive breast cancer?" --verbose
```

---

## Configuration

Copy `.env.example` to `.env`. All variables have sensible defaults for local development.

| Variable | Default | Description |
|---|---|---|
| `NCBI_API_KEY` | — | NCBI API key — optional, but raises rate limit from 3 to 10 req/s |
| `LLM_PROVIDER` | `ollama` | LLM backend: `ollama`, `anthropic`, or `openai` |
| `LLM_MODEL` | `llama3.1:8b` | Model name for the selected provider |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API endpoint |
| `VECTOR_STORE_BACKEND` | `chroma` | Vector store: `chroma` (local, zero config) or `qdrant` (production) |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB persistence directory |
| `QDRANT_URL` | — | Qdrant endpoint — required when backend is `qdrant` |
| `QDRANT_API_KEY` | — | Qdrant API key |
| `EMBEDDING_PROVIDER` | `miniml` | Embedding model: `miniml` (all-MiniLM, local), `medcpt` (biomedical, local), or `openai` |
| `PUBMED_SPECIALTY` | `oncology` | Corpus specialty — maps to a MeSH search string |
| `PUBMED_YEARS_BACK` | `10` | Years of PubMed literature to include in the corpus |
| `INGEST_QUERY` | `oncology[Title/Abstract]` | PubMed search string for corpus ingestion |
| `INGEST_MAX_RESULTS` | `500` | Maximum abstracts per ingestion run |
| `LOG_LEVEL` | `INFO` | API log level |

See `.env.example` for the full list including observability variables (Logfire, Langfuse).

---

## Evaluation

RAGAS baseline evaluated on a 500-abstract demo corpus using 20 clinical oncology questions:

| Metric | Score | Notes |
|---|---|---|
| Faithfulness | **0.84** | Strong — generation stays within retrieved context; citation enforcement works |
| Answer relevancy | 0.33 | Scales with corpus size |
| Context precision | 0.15 | Scales with corpus size |

**Faithfulness at 0.84** confirms the core architecture is sound: the LLM cites what it was given and refuses to answer from training data when context is absent.

Answer relevancy and context precision are low because a 500-abstract general corpus is unlikely to contain relevant documents for specific clinical questions. Both metrics improve substantially with a full specialty corpus (see ADR-035 for production corpus sizing).

Run the evaluation against your own deployment:

```bash
uv pip install -e ".[eval]"
ANTHROPIC_API_KEY=sk-... python eval/evaluate.py --output eval/results.csv
```

---

## Project Structure

```
pubmed_rag/
├── src/pubmed_rag/         # core pipeline library
│   ├── ingest.py           # PubMed E-utilities fetcher
│   ├── chunk.py            # RecursiveCharacterTextSplitter wrapper
│   ├── embed.py            # sentence-transformer embedder
│   ├── vectorstore.py      # ChromaDB / Qdrant abstraction
│   ├── retrieve.py         # cosine similarity retriever
│   ├── generate.py         # citation-enforced LLM caller
│   └── pipeline.py         # end-to-end orchestrator + dataclasses
├── api/                    # FastAPI application
│   ├── main.py             # app factory, middleware
│   ├── config.py           # Pydantic BaseSettings
│   ├── schemas.py          # request / response models
│   ├── logging_config.py   # JSON logging + request ID
│   └── routers/
│       ├── health.py       # GET /health
│       └── ask.py          # POST /ask
├── eval/
│   └── evaluate.py         # RAGAS evaluation script (20 questions, 3 metrics)
├── tests/                  # pytest unit tests (46 tests)
├── docs/decisions/         # architecture decision records (ADR-033–039)
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Docker

```bash
# Build (use --network=host if your firewall blocks Docker bridge outbound traffic)
docker build -t pubmed-rag .

# Run with docker compose
docker compose up
```

The Docker image pre-bakes the `all-MiniLM-L6-v2` model weights to avoid download latency at container start.

---

## Architecture Decision Records

Design decisions for the production deployment are documented in `docs/decisions/`:

| ADR | Decision |
|---|---|
| [ADR-033](docs/decisions/ADR-033-vector-store-backend.md) | Pluggable vector store: ChromaDB (dev) / Qdrant (prod) |
| [ADR-034](docs/decisions/ADR-034-embedding-provider.md) | Pluggable embedding: all-MiniLM (dev) / text-embedding-3-small (prod) |
| [ADR-035](docs/decisions/ADR-035-corpus-scope.md) | Corpus scope: oncology, 10 years, ~1.5M abstracts |
| [ADR-036](docs/decisions/ADR-036-pipeline-orchestration.md) | Pipeline scheduling: EventBridge + ECS Fargate |
| [ADR-037](docs/decisions/ADR-037-observability.md) | Observability: Logfire (app) + Langfuse (RAG tracing) |
| [ADR-038](docs/decisions/ADR-038-llm-production-default.md) | Production LLM: Claude Sonnet |
| [ADR-039](docs/decisions/ADR-039-multi-specialty-corpus.md) | Multi-specialty support via metadata filter |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Citation

If you use this software in your research, please cite it:

```bibtex
@software{adhali2026pubmedrag,
  author  = {Adhali, Omkar},
  title   = {pubmed-rag: A RAG Pipeline for Clinical Literature},
  url     = {https://github.com/omkaradhali/pubmed_rag},
  year    = {2026},
  license = {MIT}
}
```

---

## License

MIT © [Omkar Adhali](https://github.com/omkaradhali)
