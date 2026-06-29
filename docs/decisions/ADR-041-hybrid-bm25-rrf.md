# ADR-041: Dynamic hybrid BM25+dense+RRF with entity-based routing

**Date:** 2026-06-28
**Status:** Accepted

---

## Context

Dense retrieval (bi-encoder cosine similarity) is good at semantic matching but can miss exact-term queries: drug names, gene symbols (EGFR, BRCA1), mutation notation (V600E), clinical trial IDs (NCT...), and biomarkers. BM25 keyword search is strong at exact-term recall and complements dense retrieval.

The standard approach is to run both and merge with Reciprocal Rank Fusion (RRF).

However, ablation on the 97-question evaluation set showed that naive hybrid (always-on BM25+dense+RRF) degraded recall@20 from 0.970 (dense-only) to 0.796. The evaluation set is dominated by conceptual questions, for which BM25 adds noise.

---

## Options considered

| Approach | Recall@20 | MRR | Notes |
|---|---|---|---|
| Dense-only (MiniLM) | 0.970 | 0.616 | Production baseline before reranking |
| Dense + MedCPT reranker | 0.970 | **0.950** | Best MRR; production default |
| Naive hybrid (always BM25+RRF) | 0.796 | 0.827 | Large recall regression on conceptual questions |
| Dynamic hybrid + reranker | 0.903 | 0.938 | Recovered recall; MRR still below reranker-alone |

---

## Decision

Implement **dynamic hybrid** with two routing gates. BM25 activates only when at least one gate fires; otherwise the pipeline falls back to dense-only.

**Gate 1 — Entity detection**
Regex patterns detect named entities that benefit from exact-term recall:
- Drug names (common oncology agents, `-mab`, `-nib`, `-zumab` suffixes)
- Gene symbols (`[A-Z][A-Z0-9]{1,5}[0-9]` pattern, filtered against common words)
- Mutation notation (`V600E`, `exon 19`, `L858R`)
- Clinical trial IDs (`NCT\d{8}`)
- Biomarkers (`PD-L1`, `MSI-H`, `TMB-H`, `HER2`)

**Gate 2 — BM25 confidence threshold**
If the top BM25 score exceeds 90.0, BM25 found a strong keyword match regardless of entity detection. This threshold was calibrated on the 97Q corpus: non-entity queries top out at 81.7; the gap provides a clean decision boundary.

**RRF fusion (when hybrid activates)**
Score formula: `1/(60 + rank_dense) + 1/(60 + rank_bm25)`, where k=60 is the standard dampening constant. Results are fused and passed to the reranker.

**Production defaults**
```
RERANK_ENABLED=true           # MedCPT-Cross-Encoder, always on
HYBRID_SEARCH_ENABLED=false   # BM25+RRF, opt-in
```

The reranker alone (MRR=0.950) outperforms dense+hybrid+rerank (MRR=0.938) on the current evaluation set. Hybrid is available as an opt-in for entity-heavy workloads but is off by default.

---

## Implementation notes

- BM25 index (`rank_bm25`) is built in-memory from `parents.jsonl` at module import time. Not persisted to disk.
- `rrf_fuse(dense_results, bm25_results, k=60)` in `retrieve.py` handles fusion.
- `_detect_entities(query)` in `retrieve.py` implements Gate 1.
- `validate_collection_dimension()` in `vectorstore.py` was added to catch embedding-space mismatch at startup (caught a silent failure during the MedCPT embedding experiment).
- `post_fusion_rerank` parameter in `retrieve.py` controls whether the reranker runs after RRF or after dense-only.

---

## Consequences

- Dynamic routing recovers most of the recall regression from naive hybrid (0.796 → 0.903) without degrading conceptual question performance.
- The BM25 threshold (90.0) is calibrated on the current 5K-abstract oncology corpus. If the corpus composition changes substantially, recalibrate by inspecting the score distribution with `--debug-bm25`.
- The in-memory BM25 index does not survive process restarts without reloading `parents.jsonl`. For large corpora (v2.0+), replace with Atlas `$search` (Lucene) which persists natively and scales horizontally.
- Hybrid is the correct default for v2.0 (MongoDB Atlas) where the query workload may be more entity-heavy. The Day 23 implementation is the learning scaffold; Atlas `$rankFusion` replaces `rank_bm25` in v2.0.
