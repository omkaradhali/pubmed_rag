"""
retrieve.py — Two-stage retrieval over the ChromaDB child-chunk store with
cross-encoder reranking and parent-text resolution (v0.2, see D-042 + Day 18).

Pipeline:
  1. Embed the query with the same bi-encoder used in embed.py.
  2. Query ChromaDB (children only — D-042) for a POOL of child hits.
  3. Filter by min_score (cosine) and build candidate dicts, dense-sorted.
  4. (rerank on) Cross-encoder reranks the pool — reorders by joint relevance.
  5. Dedupe by parent_id, keeping the best-ranked child per parent.
  6. Resolve each surviving hit's parent text via parents.get_parent().
  7. Return result dicts where `text` = parent text (what the LLM sees) and
     `child_text` = the matched child fragment (citation traceability).

Why two stages: the bi-encoder (stage 1) is cheap but encodes query and chunk
independently, so its ordering is coarse. The cross-encoder (stage 4, rerank.py)
reads [query, child] jointly and is far better at ranking — but only affordable
on a shortlist. This directly targets RAGAS context_precision, an order-
dependent metric. Reranking operates on CHILDREN (sharp signal); parents are
resolved afterward so the LLM still gets full context. See rerank.py.

The reported `score` field is always the stage-1 cosine similarity, even when
reranking has reordered the results — it stays comparable across queries and
independent of the cross-encoder's unbounded logit scale.

List fields stored in ChromaDB as JSON strings (authors, publication_types,
mesh_terms) are deserialized back to Python lists before returning.

Public API:
    retrieve(query, n_results, min_score, rerank_enabled) -> list[dict]
"""

import json
import logging
import os

from pubmed_rag.embed import MODEL_NAME, get_model
from pubmed_rag.parents import get_parent
from pubmed_rag.rerank import rerank
from pubmed_rag.vectorstore import get_collection

_logger = logging.getLogger(__name__)

_DEFAULT_N_RESULTS = 5
_DEFAULT_MIN_SCORE = 0.0

# Over-fetch factor (rerank OFF): ask Chroma for OVERFETCH_MULTIPLIER × n_results
# child hits so dedup-by-parent still leaves ~n_results unique parents. 4× is
# generous for short PubMed abstracts where most parents have 1–3 children.
_OVERFETCH_MULTIPLIER = 4

# Candidate pool size (rerank ON): a larger fixed shortlist gives the cross-
# encoder room to promote a relevant-but-buried child that the bi-encoder
# ranked low. ~30 is the precision/latency sweet spot for short abstracts on
# CPU (research: retrieve top-20–30 → rerank → top-5). Env-overridable.
_RERANK_POOL = int(os.getenv("RERANK_POOL", "30"))

# Reranking on by default; set RERANK_ENABLED=false to fall back to dense-only.
_RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")

# Fields stored as JSON strings in ChromaDB metadata — deserialized on retrieval.
_JSON_FIELDS = ("authors", "publication_types", "mesh_terms")


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
) -> list[dict]:
    """
    Find the most relevant parents to a query via dense shortlist + rerank.

    Stage 1 (bi-encoder) shortlists a pool of child chunks; stage 2 (cross-
    encoder, optional) reorders them; results are then deduped to unique
    PARENTS — each parent represented by its best-ranked child — and the parent
    text is returned as `text`.

    Args:
        query:          Natural language query string.
        n_results:      Max number of UNIQUE PARENTS to return (default: 5).
        min_score:      Minimum stage-1 cosine score to keep a child (default:
                        0.0). Applied before reranking. Use ~0.5 to drop weak
                        matches; 0.0 keeps the full pool.
        rerank_enabled: Override the RERANK_ENABLED env default. True runs the
                        cross-encoder second stage; False is dense-only.

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
    if rerank_enabled is None:
        rerank_enabled = _RERANK_ENABLED

    _logger.info("Embedding query with %s...", MODEL_NAME)
    query_vector = get_model().encode([query]).tolist()[0]

    collection = get_collection()

    # A larger pool when reranking gives the cross-encoder room to reorder;
    # otherwise just over-fetch enough for dedup to yield ~n_results parents.
    pool = _RERANK_POOL if rerank_enabled else max(n_results * _OVERFETCH_MULTIPLIER, n_results)
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

    # Stage 2: cross-encoder reranks the pool (reorders by joint relevance).
    if rerank_enabled and candidates:
        _logger.info("Reranking %d candidates with cross-encoder...", len(candidates))
        candidates = rerank(query, candidates, text_key="child_text")

    # Collapse to unique parents and resolve parent text.
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
