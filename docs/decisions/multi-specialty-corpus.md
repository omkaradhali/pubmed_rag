# Multi-Specialty Corpus Support

**Date:** 2026-05-04 (design) · 2026-09-11 (implemented)
**Status:** Implemented

## Context

The default corpus is oncology (MeSH: "Neoplasms"). Users may want to deploy pubmed_rag for
other specialties — neuroscience, pediatrics, cardiology, etc. The question is whether this
requires a separate deployment per specialty or whether one deployment can serve several.

## Decision

Support multi-specialty via a `specialty` tag stamped onto every record at ingest time, carried
unchanged through chunking, embedding, and storage, and filtered on at query time.

`SPECIALTY_QUERIES` (in `ingest.py`) maps a short name to its Entrez search string:

```python
SPECIALTY_QUERIES = {
    "oncology": "Neoplasms[MeSH]",
    "neuroscience": "Nervous System Diseases[MeSH]",
}
```

Adding a specialty is a one-line addition to this mapping plus an ingestion run for it. Both
entries are deliberately broad MeSH *disease-category* headings, not narrow field-of-study
headings — "Neurosciences[MeSH]" (the field) returns a small fraction of the papers that
"Nervous System Diseases[MeSH]" (the disease category, same pattern as "Neoplasms[MeSH]") does,
since disease-category headings roll up thousands of subheadings hierarchically and index actual
research output, while field-of-study headings are narrow administrative tags.

Architecture: **one shared vector collection, filtered by specialty metadata** — not a separate
collection per specialty. Each chunk carries its `specialty` in ChromaDB metadata; `retrieve()`
takes an optional `specialty` parameter and applies it as a metadata filter on the dense
(ChromaDB `where`) path, and as a separate cached BM25 index scoped to that specialty's parents
on the hybrid path (`rank_bm25` has no native metadata filtering, so scoping it means indexing
only that specialty's texts).

Files touched: `ingest.py` (specialty → query mapping, record tagging), `chunk.py` (propagate the
tag through to every chunk), `vectorstore.py` (store it in ChromaDB metadata), `retrieve.py`
(filter on both retrieval lanes), `pipeline.py` (thread the parameter through end to end).

## Consequences

- A record with no specialty tag (ingested before this existed, or via a raw query string) has
  `specialty=""` and simply won't match any specialty filter — it's still returned when no filter
  is applied.
- `_run_full_ingest` wipes the whole corpus unconditionally, regardless of specialty — running it
  twice, once per specialty, destroys the first. Building a multi-specialty corpus means ingesting
  each specialty via the append-only incremental path, never the full-rebuild path, more than once.
- Mixed-specialty queries are possible by omitting the filter — this is the default.
- Storage scales linearly and additively: adding a specialty doesn't touch what's already indexed.
