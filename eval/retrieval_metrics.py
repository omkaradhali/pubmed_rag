"""
retrieval_metrics.py — Deterministic retrieval quality metrics.

All functions accept retrieved_pmids (ordered list, best first) and
gold_pmids (the relevant PMIDs for a question) and return floats in [0, 1].

No LLM, no judge, no variance. Identical inputs always produce identical scores.
This supplements RAGAS (which uses an LLM judge and has ±0.05 run-to-run noise)
with noise-free ranking metrics that work at any N.

Public API:
    recall_at_k(retrieved, gold, k)         -> float
    mrr(retrieved, gold)                    -> float
    ndcg_at_k(retrieved, gold, k)           -> float
    score_question(retrieved, gold, ks)     -> dict[str, float]
"""

from __future__ import annotations

import math


def recall_at_k(retrieved_pmids: list[str], gold_pmids: list[str], k: int) -> float:
    """Fraction of gold PMIDs found in the top-k retrieved results.

    Returns 0.0 when gold_pmids is empty (unanswerable / not yet labeled).
    Returns 1.0 when all gold docs appear in the top-k.
    """
    if not gold_pmids:
        return 0.0
    top_k = set(retrieved_pmids[:k])
    return len(top_k & set(gold_pmids)) / len(gold_pmids)


def mrr(retrieved_pmids: list[str], gold_pmids: list[str]) -> float:
    """Reciprocal rank of the first relevant result.

    Scans retrieved_pmids in order and returns 1/rank at the first hit.
    Returns 0.0 if no gold PMID appears in the retrieved list.
    """
    if not gold_pmids:
        return 0.0
    gold_set = set(gold_pmids)
    for rank, pmid in enumerate(retrieved_pmids, 1):
        if pmid in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_pmids: list[str], gold_pmids: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance).

    DCG rewards finding relevant docs early (log-discounted by position).
    nDCG normalises by the ideal DCG (all gold docs at the top positions).
    Returns 0.0 when gold_pmids is empty or no gold doc is in the top-k.
    """
    if not gold_pmids:
        return 0.0
    gold_set = set(gold_pmids)
    dcg = sum(
        1.0 / math.log2(i + 2) for i, pmid in enumerate(retrieved_pmids[:k]) if pmid in gold_set
    )
    ideal_hits = min(len(gold_pmids), k)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def score_question(
    retrieved_pmids: list[str],
    gold_pmids: list[str],
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    """Compute all deterministic metrics for one question.

    Returns a flat dict with keys recall@k, ndcg@k (for each k in ks), and mrr.
    Questions with empty gold_pmids get 0.0 across all metrics.

    Args:
        retrieved_pmids: Ordered list of retrieved PMIDs (best first).
        gold_pmids:      Known-relevant PMIDs from the eval set.
        ks:              Cut-offs to evaluate at (default: 5, 10, 20).
    """
    result: dict[str, float] = {}
    for k in ks:
        result[f"recall@{k}"] = recall_at_k(retrieved_pmids, gold_pmids, k)
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved_pmids, gold_pmids, k)
    result["mrr"] = mrr(retrieved_pmids, gold_pmids)
    return result
