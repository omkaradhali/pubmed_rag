# ADR-039: Multi-Specialty Corpus Support

**Date:** 2026-05-04
**Status:** Accepted (design locked; implementation Phase 2)

## Context

The default corpus is oncology (MeSH: "Neoplasms"). Users may want to deploy pubmed_rag for
other specialties — pediatrics, veterinary medicine, cardiology, etc. The question is whether
this requires a new deployment or whether the existing infrastructure can serve multiple
specialties from a single deployment.

## Decision

Support multi-specialty via a `PUBMED_SPECIALTY` env var backed by a `SPECIALTY_QUERIES` dict.
Adding a new specialty is a one-line addition to the mapping:

```python
SPECIALTY_QUERIES = {
    "oncology":   '"Neoplasms"[MeSH]',
    "pediatrics": '"Pediatrics"[MeSH]',
    "veterinary": '"Veterinary Medicine"[MeSH]',
    "cardiology": '"Cardiovascular Diseases"[MeSH]',
}
```

Architecture: **single collection with specialty metadata filter** (not per-specialty collections).

Each chunk in the vector store carries a `specialty` metadata field set at ingest time.
At query time, the vector store filter restricts results to the requested specialty.
Qdrant's native filter conditions handle this efficiently even at large corpus sizes.

Files affected (all small, additive changes):
- `ingest.py` — `specialty` parameter → maps to MeSH search string via `SPECIALTY_QUERIES`
- `vectorstore.py` — add `specialty` to `_chunk_to_metadata()`, add filter support to `search()`
- `retrieve.py` — accept optional `specialty` filter, pass to backend
- `pipeline.py` — pass `specialty` through the chain
- `api/schemas.py` — add `specialty: str = "oncology"` to `AskRequest`
- `api/config.py` — add `PUBMED_SPECIALTY` env var

## Consequences

- New specialty = one line in `SPECIALTY_QUERIES` + a new ingestion run for that specialty
- Embedding cost for additional specialties:
  - Pediatrics (10yr, ~500K abstracts): ~$4.50 one-time
  - Veterinary (10yr, ~200K abstracts): ~$1.80 one-time
- Storage scales linearly — adding pediatrics + veterinary to the oncology corpus adds ~8.5GB
- Mixed-specialty queries are possible by omitting the filter (intentional design choice)
- Veterinary literature on PubMed overlaps with animal studies used in human medicine
  research — this is a known ambiguity that users of the `veterinary` specialty should be
  aware of
