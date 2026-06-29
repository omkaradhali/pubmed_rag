# Known Limitations

This document captures the known constraints of pubmed_rag v0.2. Understanding these before deployment prevents surprises in production.

---

## Corpus

### Abstracts only — no full text
The pipeline ingests PubMed abstracts (typically 200–400 words), not full papers. Claims that require detailed methodology, supplementary data, or results tables cannot be fully supported. For clinical decision support contexts, this means complex statistical breakdowns or sub-group analyses from the methods section will not be retrievable.

**Planned mitigation:** v3.0 will add PMC full-text ingestion via the PubMed Central OAI-PMH endpoint.

### Corpus must be explicitly seeded
pubmed_rag does not auto-refresh. The vector store reflects the snapshot built at last ingestion. Run with `--mode full` or `reldate=N` to update.

### Default corpus is bounded
The default `INGEST_MAX_RESULTS=500` fetches 500 abstracts. Full PubMed oncology coverage requires hundreds of thousands of abstracts. Evaluation was performed on a 5,000-abstract corpus; retrieval quality will be lower on the 500-abstract default.

### English-centric retrieval
The default PubMed query does not filter by language. Non-English abstracts are ingested but the embedding model (`all-MiniLM-L6-v2`) and evaluation benchmark are English-only. Recall for non-English queries is untested.

---

## Retrieval

### ChromaDB in-process scaling limit
ChromaDB runs in the same process as the FastAPI server. Beyond ~100,000 chunks it can slow down substantially and memory usage grows linearly. The production path (ADR-033) is Qdrant, which runs as a separate service and handles millions of vectors efficiently.

### BM25 index is in-memory and rebuilt per process
The hybrid BM25 index (`rank_bm25`) is built from `parents.jsonl` at import time. It is not persisted and is rebuilt on every cold start. For a 5,000-abstract corpus this takes under one second; for larger corpora the startup cost grows linearly.

### Hybrid search is off by default
Dynamic hybrid BM25+RRF is disabled by default (`HYBRID_SEARCH_ENABLED=false`). On the 97-question evaluation set, the reranker alone (MRR=0.950) outperformed dense+hybrid+rerank (MRR=0.938). Enable hybrid only if your query workload is entity-heavy (drug names, gene symbols, trial IDs).

---

## Guardrails

### Topic relevance check is keyword-based
The input guardrail uses a fixed biomedical term set. A highly specific clinical query that uses no common biomedical keywords (e.g., a niche assay name not in the term list) may be passed through without validation. The permissive design (only reject when a blocklist pattern fires AND no biomedical signal exists) minimises false positives at the cost of some false negatives.

### Faithfulness check is lexical, not semantic
The output faithfulness guardrail computes unigram Jaccard overlap between a cited sentence and its source chunk. It catches outright hallucinations (zero shared tokens) but will not flag semantically faithful paraphrases that share few surface tokens. The Jaccard threshold of 0.05 is intentionally low — use RAGAS faithfulness (LLM-as-judge) for semantic coverage during evaluation.

---

## Evaluation

### RAGAS scores are nondeterministic at N=20
RAGAS uses an LLM as a judge. Scores vary by approximately ±0.05 at N=20 questions. The deterministic recall@k/MRR/nDCG metrics (97-question gold benchmark) are stable across runs and should be used for comparing pipeline configurations. Treat the RAGAS numbers in the README as indicative, not exact.

### Evaluation corpus ≠ deployment corpus
The 97-question labeled benchmark was generated from the same 5,000-abstract corpus used to build the vector store. Recall@k numbers will differ on a differently-composed or larger corpus.

---

## API and Security

### No authentication in v0.2
The `/ask` and `/cds-services` endpoints have no authentication. Do not expose the API publicly without completing the Day 29 auth + rate-limiting milestone (`slowapi` + static API key). See `SECURITY.md` for guidance.

### Rate limit: NCBI without an API key
Without a free NCBI API key, the ingestion pipeline is capped at 3 requests/second. Large corpus builds (`--mode full`, 5,000+ abstracts) will be throttled. Register at [https://www.ncbi.nlm.nih.gov/account/](https://www.ncbi.nlm.nih.gov/account/) and set `NCBI_API_KEY` to raise the limit to 10 req/s.

### Ollama must be running
When `LLM_PROVIDER=ollama` (the default), the Ollama service must be reachable at `OLLAMA_BASE_URL`. If Ollama is not running, all `/ask` calls will fail with a 500 error. Set `LLM_PROVIDER=anthropic` or `LLM_PROVIDER=openai` with the corresponding API key to remove the local dependency.

---

## Not Limitations — Common Misunderstandings

| Perceived issue | Reality |
|---|---|
| "No UI" | A Gradio demo UI exists at `demo/gradio_app.py`. The FastAPI backend intentionally has no built-in HTML UI — the Swagger docs at `/docs` serve as the primary interface. |
| "Only oncology" | The `INGEST_QUERY` env var accepts any valid PubMed search string. The oncology default is a starting point, not a constraint. |
| "Slow cold start" | The embedding model (~90MB) and BM25 index are loaded at startup. After that, query latency is ~1-3 sec (mostly LLM). Use `EMBEDDING_PROVIDER=miniml` to keep the model small. |
