# ADR-035: Corpus Scope — Focused Specialty, Configurable Depth

**Date:** 2026-05-04
**Status:** Accepted

## Context

"All of PubMed" (~37M abstracts) was considered as the corpus target. Embedding and serving all
of PubMed is technically feasible but costly at scale, and most of it is irrelevant to any single
clinical use case.

However, "all of PubMed" includes veterinary medicine, materials science, dentistry, and dozens
of fields unrelated to the system's primary use case. A general corpus degrades retrieval
precision because irrelevant documents occupy top-k slots. A focused specialty corpus is a
better product: faster retrieval, higher relevance, and a clearer value proposition.

## Decision

Default corpus: **oncology / cancer research, last 10 years** (MeSH: "Neoplasms", 2016–2026).

This is ~1.5M abstracts — large enough to cover the clinically relevant literature, small enough
to embed and serve cheaply. (Deployment-specific cost and storage sizing live with the private
production infra, not in this repo.)

Corpus depth is configurable via `PUBMED_YEARS_BACK` env var (default: 10). This drives the
`reldate` filter already implemented in `ingest.py`. Users can reduce scope:

```
PUBMED_YEARS_BACK=10   # 1.5M abstracts — default production
PUBMED_YEARS_BACK=5    # ~750K abstracts — lighter deployment
PUBMED_YEARS_BACK=2    # ~300K abstracts — OSS demo with real data
```

Specialty is configurable via `PUBMED_SPECIALTY` env var, backed by a `SPECIALTY_QUERIES`
mapping dict. This makes adding new specialties a one-line addition to the mapping.

Incremental updates use the `reldate` filter in `ingest.py` — fetch only the last N days of new
papers and upsert — so the corpus stays current without a full rebuild. Full re-ingestion is an
explicit opt-in via `mode=full`. How the incremental run is scheduled is a deployment concern.

## Consequences

- Specialty focus limits the system's scope intentionally — this is a feature, not a constraint
- Multi-specialty support (see ADR-039) requires adding `specialty` metadata through the stack
- `PUBMED_YEARS_BACK` affects initial setup time and storage cost linearly
- The 10-year default captures the majority of clinically relevant oncology literature
  (most guidelines reference evidence from the last 5–10 years)
