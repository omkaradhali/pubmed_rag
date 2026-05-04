# ADR-033: Pluggable Vector Store Backend

**Date:** 2026-05-04
**Status:** Accepted

## Context

The project started with ChromaDB (embedded, in-process) as its vector store. ChromaDB is
easy to run locally with zero configuration, making it ideal for OSS users who clone the repo
and want to learn RAG patterns without standing up additional services.

However, ChromaDB is not designed for production workloads — it lacks multi-tenancy, has
limited horizontal scaling, and its embedded mode cannot be shared across multiple processes.
For a production deployment serving a 1.5M+ abstract oncology corpus, a purpose-built vector
database is required.

## Decision

Implement a pluggable vector store abstraction controlled by `VECTOR_STORE_BACKEND`:

- `VECTOR_STORE_BACKEND=chroma` (default) — ChromaDB embedded, zero extra services, right for
  local development and OSS users learning RAG
- `VECTOR_STORE_BACKEND=qdrant` — Qdrant, purpose-built vector DB, right for production

Both backends implement the same interface (`upsert`, `search`, `delete`, `dimensions`) so the
pipeline, retrieval, and API layers are backend-agnostic. Switching is a single env var change.

Qdrant was chosen as the production backend because:
- Native hybrid search support (dense + sparse vectors as named vectors)
- Written in Rust — performant and memory-efficient at scale
- Supports metadata filtering natively (needed for multi-specialty corpus)
- Qdrant Cloud free tier covers OSS demo deployments (~80K abstracts)
- Self-hosted on ECS + EBS covers production scale (~20GB for 1.5M abstracts)

## Consequences

- `vectorstore.py` becomes a dispatch layer with `ChromaBackend` and `QdrantBackend` classes
- Both backends must support metadata filtering (needed for specialty-scoped queries)
- `QDRANT_URL` and `QDRANT_API_KEY` env vars required when using Qdrant backend
- ChromaDB remains the default — no setup friction for OSS users
- pgvector (PostgreSQL) was considered but deferred; Omkar's existing Postgres expertise makes
  it a viable future backend if the project moves to a Postgres-centric stack
