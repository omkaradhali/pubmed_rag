# ADR-037: Observability Stack — Logfire + Langfuse

**Date:** 2026-05-04
**Status:** Accepted

## Context

A production RAG system has two distinct observability concerns that require different tools:

1. **Application/infrastructure health** — Is the API up? Which endpoints are slow? Where are
   errors occurring? This is standard web service observability.

2. **RAG pipeline quality** — Why did this query return a poor answer? Which chunks were
   retrieved? What prompt was sent? What did the model do with the retrieved context? This
   requires LLM-aware tracing that standard APM tools do not provide.

These concerns do not overlap. A single tool cannot serve both well.

## Decision

Use **Logfire** for the application layer and **Langfuse** for the RAG layer.

**Logfire** (Pydantic's observability product):
- OpenTelemetry-based, integrates natively with FastAPI and Pydantic
- `logfire.instrument_fastapi(app)` auto-traces all routes — request lifecycle, latency, errors
- Already a natural fit since the project uses Pydantic BaseSettings throughout
- Captures: HTTP request/response spans, endpoint latency, exception tracing

**Langfuse** (open-source LLM observability):
- `@observe` decorator wraps pipeline functions — builds parent-child trace hierarchy automatically
- Captures per-query: query → retrieved chunks (with scores and metadata) → LLM prompt →
  completion → token counts → cost
- Essential for debugging retrieval quality ("why was this chunk returned instead of that one?")
- Self-hostable on Docker or available as Langfuse Cloud (free tier: 50K observations/month)

Deployment plan:
- Phase 2: Langfuse Cloud free tier (50K obs/month covers ~3K queries/day — sufficient for
  development and early production)
- Phase 3+: Self-host Langfuse on homelab if usage exceeds free tier or data residency matters

## Consequences

- `LOGFIRE_TOKEN` env var required (Logfire account)
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` env vars required
- `@observe` decorators added to `pipeline.py`, `retrieve.py`, `generate.py`
- Token counts and cost are captured from the Anthropic response `usage` object in `generate.py`
  and passed to Langfuse generation spans
- AWS CloudWatch handles infrastructure-level metrics (container CPU/memory) independently
  of Logfire — no overlap, both are needed
