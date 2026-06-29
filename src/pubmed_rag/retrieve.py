"""
retrieve.py — Hybrid retrieval over the ChromaDB child-chunk store with
optional BM25 fusion and cross-encoder reranking (v0.2, D-042 + Day 18 + Day 23).

Dense-only pipeline (HYBRID_SEARCH_ENABLED=false, default):
  1. Embed the query with the bi-encoder from embed.py.
  2. Query ChromaDB (children only) for a pool of child hits.
  3. Filter by min_score and build candidate dicts, dense-sorted.
  4. (rerank on) Cross-encoder reranks the pool — reorders by joint relevance.
  5. Dedupe by parent_id, keeping the best-ranked child per parent.
  6. Resolve each surviving hit's parent text via parents.get_parent().

Hybrid pipeline (HYBRID_SEARCH_ENABLED=true):
  1-5. Same dense path but overfetches parents (no reranking in hybrid mode).
  6. BM25 path: tokenize query → score all parents → rank by BM25 score.
  7. RRF fusion: merge dense and BM25 parent lists via Reciprocal Rank Fusion
     (k=60). A parent appearing in both lists outscores one appearing in only
     one. Score units are incompatible across models so ranks are used, not
     raw scores.
  8. Return top-n fused parents.

Why hybrid: dense retrieval misses exact biomedical terms (drug names, gene
symbols, MeSH terms, trial IDs). BM25 catches these directly. RRF fuses both
signals without needing score normalisation. See Day 23 ablation results in
ablation/ABLATION.md.

The reported `score` field is cosine similarity for dense-only results; for
hybrid results it is the RRF score (not directly comparable to cosine).

List fields stored in ChromaDB as JSON strings (authors, publication_types,
mesh_terms) are deserialized back to Python lists before returning. In
parents.jsonl these fields are already native Python lists.

Public API:
    retrieve(query, n_results, min_score, rerank_enabled, hybrid_enabled) -> list[dict]
    rrf_fuse(dense_results, bm25_results, k, n_results) -> list[dict]
"""

import json
import logging
import os
import re

from rank_bm25 import BM25Okapi

from pubmed_rag.embed import MODEL_NAME, embed_query
from pubmed_rag.parents import get_all_parents, get_parent
from pubmed_rag.rerank import rerank
from pubmed_rag.vectorstore import get_collection

_logger = logging.getLogger(__name__)

_DEFAULT_N_RESULTS = 5
_DEFAULT_MIN_SCORE = 0.0

# Over-fetch factor (dense-only, rerank OFF): ask Chroma for
# OVERFETCH_MULTIPLIER × n_results child hits so dedup-by-parent still leaves
# ~n_results unique parents. 4× is generous for short PubMed abstracts where
# most parents have 1-3 children.
_OVERFETCH_MULTIPLIER = 4

# Candidate pool size (rerank ON): a larger fixed shortlist gives the cross-
# encoder room to promote a relevant-but-buried child that the bi-encoder
# ranked low. ~30 is the precision/latency sweet spot for short abstracts on
# CPU. Env-overridable.
_RERANK_POOL = int(os.getenv("RERANK_POOL", "30"))

# Hybrid over-fetch: how many unique parents to gather from each lane before
# RRF fusion. 4× n_results gives RRF enough candidates to promote BM25-only
# hits that dense retrieval ranked low.
_HYBRID_POOL_MULTIPLIER = 4

# Reranking on by default; set RERANK_ENABLED=false to fall back to dense-only.
_RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")

# Hybrid BM25+dense retrieval; off by default (dense-only is the safe default).
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() in ("1", "true", "yes")

# RRF constant k — standard value from the original RRF paper (Cormack 2009).
# Higher k reduces the weight advantage of top-ranked documents; 60 is the
# community default for RAG hybrid search.
_RRF_K = int(os.getenv("RRF_K", "60"))

# Fields stored as JSON strings in ChromaDB metadata — deserialized on retrieval.
_JSON_FIELDS = ("authors", "publication_types", "mesh_terms")

# BM25 score threshold (Strategy 1): only fuse BM25 results when the top BM25
# score exceeds this value. Low scores mean no useful term overlap — fusion
# would only hurt recall by displacing correct dense results.
BM25_SCORE_THRESHOLD = float(os.getenv("BM25_SCORE_THRESHOLD", "90.0"))

# Compiled regex patterns for biomedical entity detection (Strategy 2).
# When any pattern fires, BM25 is always fused regardless of the score threshold —
# exact-term queries for drugs/genes/trials are where BM25 adds the most value.
#
# Drug suffixes: monoclonal antibodies (-mab), kinase inhibitors (-nib),
# PARP inhibitors (-parib), CDK inhibitors (-ciclib).
_RE_DRUG = re.compile(r"\b\w+(?:mab|nib|parib|ciclib|lisib|rafenib)\b", re.IGNORECASE)
# Gene symbols: 3-6 uppercase letters (KRAS, BRAF, EGFR) or 2-6 uppercase
# letters followed by digits (TP53, BRCA1, HER2, CDK4, MLH1).
# Negative lookahead excludes pure Roman numerals (I, II, III, IV, V, VI...)
# which appear in cancer staging (stage III, stage IV) and would be false positives.
_RE_GENE = re.compile(r"\b(?![IVXLCDM]+\b)(?:[A-Z]{3,6}\d*|[A-Z]{2,6}\d+)\b")
# Mutation notation: single letter + digits + single letter (V600E, G12C, R175H).
_RE_MUTATION = re.compile(r"\b[A-Z]\d+[A-Z]\b")
# Clinical trial identifiers.
_RE_TRIAL = re.compile(r"\b(?:NCT\d+|KEYNOTE-\d+|CheckMate-\d+|IMpower\d+)\b", re.IGNORECASE)
# Common biomarker abbreviations in oncology.
_RE_BIOMARKER = re.compile(r"\b(?:MSI-[HL]|dMMR|pMMR|TMB|HER2|PD-L1|CTLA-4|PD-1|CDK4|CDK6)\b")

# BM25 module-level cache — built lazily on first hybrid query, then reused.
# Invalidated whenever clear_cache() is called on the parents module.
_bm25_index: BM25Okapi | None = None
_bm25_parents: list[dict] = []


def _has_biomedical_entity(query: str) -> bool:
    """Return True if the query contains a known biomedical exact-term entity.

    Checks for drug name suffixes, gene symbols, mutation notation, trial IDs,
    and common biomarker abbreviations. A match means BM25 is likely to add
    signal even when its score is below BM25_SCORE_THRESHOLD.
    """
    return bool(
        _RE_DRUG.search(query)
        or _RE_GENE.search(query)
        or _RE_MUTATION.search(query)
        or _RE_TRIAL.search(query)
        or _RE_BIOMARKER.search(query)
    )


def _should_use_bm25(query: str, bm25_top_score: float) -> bool:
    """Decide whether to fuse BM25 results into the final ranking.

    Two-gate decision (Strategy 1 + Strategy 2):
      Gate 1 — entity fast-path: if the query contains a drug name, gene
        symbol, mutation, trial ID, or biomarker, always fuse regardless of
        score. These are exactly the exact-term queries where BM25 excels.
      Gate 2 — score threshold: if BM25's top score exceeds BM25_SCORE_THRESHOLD
        the query has strong term overlap with the corpus, so fusion is likely
        to help even without a recognised entity pattern.

    Returns False when BM25 has no useful signal (low score, no entity) to
    avoid polluting the dense results with irrelevant term-matched parents.
    """
    if _has_biomedical_entity(query):
        return True
    return bm25_top_score > BM25_SCORE_THRESHOLD


def _get_bm25_index() -> tuple["BM25Okapi", list[dict]]:
    """Lazy-build and cache the BM25 index over all parent texts.

    Called on the first hybrid query; subsequent calls return the cached index.
    Tokenisation is whitespace + lowercase — fast and sufficient for biomedical
    term matching where exact spelling matters (drug names, gene symbols, IDs).
    """
    global _bm25_index, _bm25_parents
    if _bm25_index is not None:
        return _bm25_index, _bm25_parents

    _bm25_parents = get_all_parents()
    if not _bm25_parents:
        _logger.warning("No parents loaded — BM25 index will be empty.")
        _bm25_index = BM25Okapi([[]])
        return _bm25_index, _bm25_parents

    tokenized = [p["text"].lower().split() for p in _bm25_parents]
    _bm25_index = BM25Okapi(tokenized)
    _logger.info("BM25 index built over %d parents.", len(_bm25_parents))
    return _bm25_index, _bm25_parents


def _bm25_retrieve(query: str, n_results: int) -> list[dict]:
    """Rank parents by BM25 score and return the top-n as result dicts.

    BM25 operates at parent level (not child level) because:
      1. Parents are longer (~1200 chars) — more terms for exact matching.
      2. The dense path already returns parent-level results, so both lists
         share the same granularity going into RRF fusion.
      3. Avoids a second dedup step after BM25 ranking.

    Results that score zero (no query term overlap) are excluded so they don't
    pollute the RRF fusion with signal-less entries.
    """
    index, parents = _get_bm25_index()
    if not parents:
        return []

    scores = index.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            break
        parent = parents[idx]
        # authors/mesh_terms/publication_types are native lists in parents.jsonl
        results.append(
            {
                "text": parent.get("text", ""),
                "child_text": parent.get("text", ""),
                "pmid": parent.get("pmid", ""),
                "title": parent.get("title", ""),
                "year": parent.get("year", ""),
                "doi": parent.get("doi", ""),
                "doi_url": parent.get("doi_url", ""),
                "pmc_id": parent.get("pmc_id", ""),
                "pmc_url": parent.get("pmc_url", ""),
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{parent.get('pmid', '')}/",
                "journal": parent.get("journal", ""),
                "authors": parent.get("authors", []),
                "publication_types": parent.get("publication_types", []),
                "mesh_terms": parent.get("mesh_terms", []),
                "chunk_id": parent.get("chunk_id", ""),
                "parent_id": parent.get("chunk_id", ""),
                "chunk_index": 0,
                "chunk_total": 1,
                "score": round(float(scores[idx]), 4),
            }
        )
    return results


def rrf_fuse(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    n_results: int = 5,
) -> list[dict]:
    """Merge two parent-level ranked lists via Reciprocal Rank Fusion.

    Each document's RRF contribution from a list is 1 / (k + rank), where
    rank is 0-indexed. Documents appearing in both lists accumulate
    contributions from each — naturally promoting documents with consistent
    cross-lane relevance. k=60 is the standard value from Cormack (2009).

    Dense metadata takes precedence when a parent appears in both lists
    (dense results carry richer child_text attribution). The `score` field
    on returned dicts is the RRF score, not a cosine similarity.

    Args:
        dense_results: Parent dicts ranked by cosine (or rerank) score.
        bm25_results:  Parent dicts ranked by BM25 score.
        k:             RRF constant (default 60).
        n_results:     How many fused results to return.

    Returns:
        Fused list of parent dicts, sorted by descending RRF score.
    """
    rrf_scores: dict[str, float] = {}
    result_map: dict[str, dict] = {}

    for rank, result in enumerate(dense_results):
        pid = result.get("parent_id") or result.get("chunk_id", "")
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        result_map[pid] = result

    for rank, result in enumerate(bm25_results):
        pid = result.get("parent_id") or result.get("chunk_id", "")
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        if pid not in result_map:
            result_map[pid] = result

    top_pids = sorted(rrf_scores, key=lambda p: rrf_scores[p], reverse=True)[:n_results]
    return [{**result_map[pid], "score": round(rrf_scores[pid], 6)} for pid in top_pids]


def _build_child_result(child_text: str, meta: dict, score: float) -> dict:
    """
    Build a candidate result dict from one ChromaDB child hit.

    `text` is set to the child text as a placeholder — _dedup_and_resolve
    overwrites it with the parent text. List metadata fields stored as JSON
    strings are deserialized back to Python lists here.
    """
    # Fall back to chunk_id when parent_id is missing — guards against legacy
    # rows ingested before D-042 (none should exist after a re-seed).
    parent_id = meta.get("parent_id") or meta.get("chunk_id", "")

    result = {
        "text": child_text,  # placeholder — replaced with parent text on resolve
        "child_text": child_text,
        # Identifiers
        "pmid": meta.get("pmid", ""),
        "title": meta.get("title", ""),
        "year": meta.get("year", ""),
        # Links — derived here so callers never reconstruct them
        "doi": meta.get("doi", ""),
        "doi_url": meta.get("doi_url", ""),
        "pmc_id": meta.get("pmc_id", ""),
        "pmc_url": meta.get("pmc_url", ""),
        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{meta.get('pmid', '')}/",
        # Bibliographic
        "journal": meta.get("journal", ""),
        # Chunk identifiers (v0.2)
        "chunk_id": meta.get("chunk_id", ""),
        "parent_id": parent_id,
        # Chunk position (within parent)
        "chunk_index": meta.get("chunk_index", 0),
        "chunk_total": meta.get("chunk_total", 1),
        # Stage-1 cosine similarity (kept even after rerank reorders)
        "score": round(score, 4),
    }

    for field in _JSON_FIELDS:
        raw_val = meta.get(field, "[]")
        result[field] = json.loads(raw_val) if isinstance(raw_val, str) else raw_val

    return result


def _dedup_and_resolve(candidates: list[dict], n_results: int) -> list[dict]:
    """
    Collapse candidates to unique parents and swap in parent text.

    Iterates in the candidates' current order (rerank order when reranking ran,
    else dense order), keeps the first child seen per parent_id, replaces `text`
    with the resolved parent text, and stops at n_results. Drops the internal
    rerank_score key so the returned schema is stable across both paths.

    If a parent is missing from the sidecar store (parents.jsonl out of sync
    with the collection), falls back to the child text rather than crashing.
    """
    seen_parents: set[str] = set()
    results: list[dict] = []

    for candidate in candidates:
        parent_id = candidate["parent_id"]
        if not parent_id or parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        result = {**candidate}
        result.pop("rerank_score", None)

        try:
            result["text"] = get_parent(parent_id)["text"]
        except KeyError:
            _logger.warning(
                "Parent %s not found in sidecar — falling back to child text.", parent_id
            )
            result["text"] = candidate["child_text"]

        results.append(result)
        if len(results) >= n_results:
            break

    return results


def retrieve(
    query: str,
    n_results: int = _DEFAULT_N_RESULTS,
    min_score: float = _DEFAULT_MIN_SCORE,
    rerank_enabled: bool | None = None,
    hybrid_enabled: bool | None = None,
) -> list[dict]:
    """
    Find the most relevant parents to a query.

    Dense-only (default): bi-encoder shortlists child chunks → optional cross-
    encoder rerank → dedup to unique parents → return.

    Hybrid (HYBRID_SEARCH_ENABLED=true): dense path overfetches parents (no
    reranking) → BM25 ranks all parents by keyword score → RRF fuses both
    ranked lists → return top-n fused parents. Reranking is not applied in
    hybrid mode; it can be layered in a future session.

    Args:
        query:          Natural language query string.
        n_results:      Max number of UNIQUE PARENTS to return (default: 5).
        min_score:      Minimum stage-1 cosine score to keep a child (default:
                        0.0). Applied before reranking. Use ~0.5 to drop weak
                        matches; 0.0 keeps the full pool.
        rerank_enabled: Override the RERANK_ENABLED env default. Ignored when
                        hybrid_enabled is True.
        hybrid_enabled: Override the HYBRID_SEARCH_ENABLED env default. When
                        True, fuses dense + BM25 via RRF and skips reranking.

    Returns:
        List of result dicts, one per unique parent. Keys:
            text              (str)       — PARENT text (full context for the LLM)
            child_text        (str)       — child fragment that matched the query
            pmid              (str)       — PubMed ID
            title             (str)       — article title
            year              (str)       — publication year
            doi               (str)       — raw DOI string, or ""
            doi_url           (str)       — https://doi.org/{doi}, or ""
            pmc_id            (str)       — PubMed Central ID, or ""
            pmc_url           (str)       — PMC full-text link, or ""
            pubmed_url        (str)       — https://pubmed.ncbi.nlm.nih.gov/{pmid}/
            journal           (str)       — full journal title, or ""
            authors           (list[str]) — ["LastName Initials", ...], or []
            publication_types (list[str]) — ["Journal Article", ...], or []
            mesh_terms        (list[str]) — NLM MeSH descriptors, or []
            chunk_id          (str)       — child's stable ID, e.g. "12345_p0_c2"
            parent_id         (str)       — parent's stable ID, e.g. "12345_p0"
            chunk_index       (int)       — child position within its parent
            chunk_total       (int)       — total children produced from this parent
            score             (float)     — stage-1 cosine similarity in [0, 1]
    """
    if hybrid_enabled is None:
        hybrid_enabled = HYBRID_SEARCH_ENABLED
    if rerank_enabled is None:
        rerank_enabled = _RERANK_ENABLED

    # When hybrid is active, the cross-encoder runs AFTER RRF fusion rather than
    # before it. This lets both retrieval lanes (dense + BM25) contribute
    # unbiased candidates to fusion, then the reranker makes the final quality
    # decision over the merged set.
    post_fusion_rerank = hybrid_enabled and rerank_enabled
    if hybrid_enabled:
        rerank_enabled = False  # suppress Stage 2 pre-fusion reranking

    _logger.info("Embedding query with %s...", MODEL_NAME)
    query_vector = embed_query(query)

    collection = get_collection()

    # Pool size: hybrid overfetches parents from the dense lane so RRF has a
    # large enough candidate set from both lanes to fuse meaningfully.
    if hybrid_enabled:
        pool = n_results * _HYBRID_POOL_MULTIPLIER * 3  # child pool → dedup → ~4× parents
    elif rerank_enabled:
        pool = _RERANK_POOL
    else:
        pool = max(n_results * _OVERFETCH_MULTIPLIER, n_results)

    _logger.info("Querying ChromaDB for top %d child hits (request=%d)...", pool, n_results)

    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=pool,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB wraps results in an extra list (one per query). We send one
    # query at a time, so unpack index 0.
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    # Stage 1: dense candidates above min_score, in cosine order.
    candidates = [
        _build_child_result(child_text, meta, 1.0 - distance)
        for child_text, meta, distance in zip(documents, metadatas, distances)
        if 1.0 - distance >= min_score
    ]

    # Stage 2: cross-encoder reranks the pool (dense-only mode only).
    if rerank_enabled and candidates:
        _logger.info("Reranking %d candidates with cross-encoder...", len(candidates))
        candidates = rerank(query, candidates, text_key="child_text")

    # Hybrid path: dynamically decide whether BM25 adds signal before fusing,
    # then optionally rerank the merged results.
    if hybrid_enabled:
        dense_parents = _dedup_and_resolve(candidates, n_results * _HYBRID_POOL_MULTIPLIER)
        bm25_parents = _bm25_retrieve(query, n_results * _HYBRID_POOL_MULTIPLIER)
        bm25_top_score = bm25_parents[0]["score"] if bm25_parents else 0.0

        if _should_use_bm25(query, bm25_top_score):
            # Overfetch fused candidates so the reranker has a larger pool to
            # promote from when hybrid+rerank is active.
            fuse_n = n_results * 2 if post_fusion_rerank else n_results
            fused = rrf_fuse(dense_parents, bm25_parents, k=_RRF_K, n_results=fuse_n)
            if post_fusion_rerank and fused:
                _logger.info("Post-fusion reranking %d fused results...", len(fused))
                fused = rerank(query, fused, text_key="child_text")
            results = fused[:n_results]
            _logger.info(
                "Hybrid RRF%s: BM25 score=%.2f entity=%s → %d results.",
                "+rerank" if post_fusion_rerank else "",
                bm25_top_score,
                _has_biomedical_entity(query),
                len(results),
            )
        else:
            # BM25 suppressed — dense path with optional reranking at child level.
            if post_fusion_rerank:
                reranked = rerank(query, candidates, text_key="child_text")
                results = _dedup_and_resolve(reranked, n_results)
            else:
                results = dense_parents[:n_results]
            _logger.info(
                "Hybrid BM25 suppressed (score=%.2f, no entity) → dense%s.",
                bm25_top_score,
                "+rerank" if post_fusion_rerank else "",
            )
        return results

    # Dense-only path (default).
    results = _dedup_and_resolve(candidates, n_results)

    _logger.info(
        "Returned %d unique-parent results (min_score=%.2f, rerank=%s, examined %d child hits).",
        len(results),
        min_score,
        rerank_enabled,
        len(documents),
    )

    return results


# CLI entrypoint
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Semantic search over PubMed abstracts.")

    parser.add_argument("query", help="Natural language query string.")

    parser.add_argument(
        "--n",
        type=int,
        default=_DEFAULT_N_RESULTS,
        help=f"Number of unique parents to return (default: {_DEFAULT_N_RESULTS})",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=_DEFAULT_MIN_SCORE,
        help=f"Minimum similarity score to include (default: {_DEFAULT_MIN_SCORE})",
    )

    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable the cross-encoder reranker (dense-only retrieval).",
    )

    args = parser.parse_args()

    results = retrieve(
        args.query,
        n_results=args.n,
        min_score=args.min_score,
        rerank_enabled=not args.no_rerank,
    )

    print(f"\n-- Top {len(results)} results for: '{args.query}' --\n")

    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']}  pmid={r['pmid']}  year={r['year']}")
        print(f"    {r['title']}")
        print(f"    Authors: {r['authors']}")
        print(f"    Journal: {r['journal']}")
        print(f"    PubMed:  {r['pubmed_url']}")
        print(f"    Matched child:  {r['child_text'][:200]}...")
        print(f"    Parent context: {r['text'][:200]}...")
        print()

    print("-- Raw JSON --")
    print(json.dumps(results, indent=2))
