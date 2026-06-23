# pubmed-rag

[![CI](https://github.com/omkaradhali/pubmed_rag/actions/workflows/ci.yml/badge.svg)](https://github.com/omkaradhali/pubmed_rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)

A production-grade RAG pipeline for clinical literature. Fetch PubMed abstracts, embed them into a vector store, and answer natural language questions grounded in retrieved papers with inline citations.

**Built for:** clinical researchers, bioinformaticians, and developers learning biomedical RAG.

---

## Features

- **Full ingestion pipeline** — PubMed E-utilities → parent-child chunking → sentence-transformer embeddings → ChromaDB
- **Two-stage retrieval** — bi-encoder dense retrieval shortlists candidates; `ncbi/MedCPT-Cross-Encoder` reranks for clinical relevance
- **Citation-enforced generation** — the LLM is instructed to cite every claim inline; hallucinated sources are structurally prevented
- **HL7 CDS Hooks integration** — `GET /cds-services` discovery + `POST /cds-services/pubmed-rag` patient-view hook; plug into any CDS Hooks-compatible EHR
- **Gradio demo UI** — browser-based interface at `demo/gradio_app.py`; no API tooling required
- **Pluggable providers** — swap the LLM (Ollama, Anthropic, OpenAI) and embedding model via a single env var
- **FastAPI backend** — structured JSON responses, request ID tracing, Swagger docs at `/docs`
- **Dual evaluation suite** — RAGAS (LLM-as-judge) + deterministic recall@k/MRR/nDCG against a 97-question labeled benchmark
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

v0.2 adds a two-stage retrieval layer: children are retrieved densely, reranked by `ncbi/MedCPT-Cross-Encoder`, then deduped to unique parents before generation. Retrieval metrics at N=97: **recall@20 = 0.97 · MRR = 0.95**.

The pipeline runs in two modes:

- **`incremental`** (default) — queries the pre-seeded vector store. Fast, ~1-3 sec, mostly LLM latency.
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

### Gradio demo UI

```bash
pip install gradio httpx
API_BASE_URL=http://localhost:8001 python demo/gradio_app.py
# Open http://localhost:7860
```

---

## HL7 CDS Hooks Integration

pubmed_rag implements the [HL7 CDS Hooks 1.0](https://cds-hooks.hl7.org/1.0/) specification. Any CDS Hooks-compatible EHR can subscribe to the pubmed-rag service and receive cited oncology evidence cards during the patient-view workflow.

### Discovery

```bash
curl http://localhost:8001/cds-services
```

```json
{
  "services": [{
    "hook": "patient-view",
    "title": "Oncology Evidence Search (pubmed_rag)",
    "description": "Search 35M+ PubMed oncology abstracts and receive a cited, LLM-synthesised evidence summary.",
    "id": "pubmed-rag",
    "prefetch": {}
  }]
}
```

### Query the service

```bash
curl -s -X POST http://localhost:8001/cds-services/pubmed-rag \
  -H "Content-Type: application/json" \
  -d '{
    "hookInstance": "example-001",
    "hook": "patient-view",
    "context": {
      "userId": "Practitioner/dr-smith",
      "patientId": "patient-42",
      "query": "What is the first-line treatment for HER2-positive metastatic breast cancer?"
    }
  }' | jq .
```

The service returns a CDS card with the synthesized answer, inline citations, and direct PubMed links:

```json
{
  "cards": [{
    "uuid": "...",
    "summary": "Evidence: The first-line treatment is dual HER2 blockade with trastuzumab...",
    "detail": "Full answer with [1][2][3] inline citations...\n\n**Sources**\n1. ...",
    "indicator": "info",
    "source": {
      "label": "pubmed_rag — Oncology Evidence Service",
      "url": "https://pubmed.ncbi.nlm.nih.gov"
    },
    "links": [
      { "label": "[1] PMID 42041395 — Post-Chemotherapy Antibody-Based...", "url": "https://pubmed.ncbi.nlm.nih.gov/42041395/", "type": "absolute" }
    ]
  }]
}
```

**EHR integration:** register `http://your-host:8001` as a CDS Hooks service base URL in your EHR's CDS Hooks configuration. The `context.query` field accepts the clinical question directly; in a full integration this can be synthesized from the patient's FHIR problem list or encounter diagnosis.

---

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

Evaluated on a 5,000-abstract oncology corpus across two metric families.

### RAGAS (LLM-as-judge, 20 clinical oncology questions)

| Metric | v0.1 (500 abstracts) | v0.2 (5,000 abstracts) |
|---|---|---|
| Faithfulness | 0.84 | **0.91** |
| Answer relevancy | 0.33 | **0.81** |
| Context precision | 0.15 | 0.53 |

Faithfulness of 0.91 confirms citation enforcement is working: 91% of answer statements are grounded in retrieved context.

### Deterministic retrieval metrics (97 labeled questions, zero variance)

Evaluated against a 97-question oncology benchmark with gold PubMed labels. These metrics use no LLM judge — results are identical across runs.

| Metric | Score |
|---|---|
| Recall@5 | 0.67 |
| Recall@10 | 0.82 |
| **Recall@20** | **0.97** |
| **MRR** | **0.95** |
| nDCG@20 | 0.90 |

Recall@20 of 0.97 means the correct abstract appears in the top 20 results for 97% of questions. MRR of 0.95 means the first relevant result is ranked first or second on average.

### Run the evaluation

```bash
# RAGAS (requires ANTHROPIC_API_KEY)
uv pip install -e ".[eval]"
python scripts/eval_v0_2.py --output eval/results.csv

# Deterministic retrieval metrics only (no LLM cost)
python scripts/eval_v0_2.py \
  --questions eval/questions.sample.jsonl \
  --output eval/results_retrieval.csv \
  --no-ragas
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
│       ├── ask.py          # POST /ask
│       └── cds_hooks.py    # GET /cds-services, POST /cds-services/pubmed-rag
├── demo/
│   └── gradio_app.py       # Gradio web UI (calls /ask, runs on port 7860)
├── eval/
│   ├── evaluate.py         # RAGAS evaluation (20 questions, 3 metrics)
│   ├── retrieval_metrics.py # recall@k, MRR, nDCG@k (deterministic, zero variance)
│   └── questions.sample.jsonl  # 15-question public eval sample
├── scripts/
│   └── eval_v0_2.py        # unified eval driver (RAGAS + deterministic, --questions flag)
├── tests/                  # pytest unit tests (74 tests)
├── docs/decisions/         # architecture decision records (ADR-033-035, 039)
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

Design decisions for the open-source architecture are documented in `docs/decisions/`:

| ADR | Decision |
|---|---|
| [ADR-033](docs/decisions/ADR-033-vector-store-backend.md) | Pluggable vector store: ChromaDB (dev) / Qdrant (prod) |
| [ADR-034](docs/decisions/ADR-034-embedding-provider.md) | Pluggable embedding: all-MiniLM (dev) / text-embedding-3-small (prod) |
| [ADR-035](docs/decisions/ADR-035-corpus-scope.md) | Corpus scope: oncology default, configurable depth |
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
