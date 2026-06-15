# ADR-034: Pluggable Embedding Provider

**Date:** 2026-05-04
**Status:** Accepted

## Context

The project uses `all-MiniLM-L6-v2` (sentence-transformers) for development — it is free,
runs on CPU, and requires no API keys. However, its 256-token limit causes silent truncation
of PubMed structured abstracts (which commonly run 300-500 tokens), and its 384-dimensional
vectors have limited capacity to distinguish between semantically similar biomedical documents
at production corpus scale.

Three distinct use cases require different models:
1. OSS users running locally — need zero-cost, zero-configuration, CPU-friendly
2. Production with biomedical focus — need domain-specific quality, no per-call cost
3. Production prioritising retrieval quality — willing to accept API cost for best performance

## Decision

Implement a pluggable embedding provider controlled by `EMBEDDING_PROVIDER`:

| Value | Model | Dims | Token limit | Cost | Use case |
|---|---|---|---|---|---|
| `miniml` (default) | all-MiniLM-L6-v2 | 384 | 256 | Free, local | OSS / learning |
| `medcpt` | MedCPT (ncats) | 768 | 512 | Free, local | Self-hosted production |
| `openai` | text-embedding-3-small | 1,536 | 8,191 | $0.02/1M tokens | Production default |

`text-embedding-3-small` is the recommended production embedding because:
- 8,191-token limit covers entire PubMed abstracts — chunking at ingestion is eliminated
- One abstract = one vector = simpler pipeline and retrieval logic
- Cost for 10-year oncology corpus (~1.5M abstracts): ~$9 one-time, ~$1/year incremental
- Symmetric encoder (same model for indexing and queries) keeps the abstraction simple

MedCPT is a dual-encoder (separate query and article encoders). The interface's `embed_texts`
(ingestion) and `embed_query` (retrieval) methods naturally accommodate this — MedCPT routes
each to the appropriate encoder internally.

The provider exposes a `max_tokens` property. The pipeline uses this to decide whether
chunking is required (`miniml`/`medcpt`) or can be skipped (`openai`).

## Consequences

- Switching providers requires re-embedding the entire corpus — vectors from different models
  are incompatible even at the same dimensionality
- `EMBEDDING_PROVIDER`, model name, and dimensions are stored in corpus metadata so the
  vector store can detect a mismatch before a query is served with the wrong model
- `OPENAI_API_KEY` is required when using the `openai` provider
- Hybrid search sparse vectors (BM25 via FastEmbed) are separate from the dense provider and
  are only active when `VECTOR_STORE_BACKEND=qdrant`
