"""
retrieve.py — Semantic retrieval over the ChromaDB child-chunk store with
parent-text resolution (v0.2, see D-042).

Pipeline:
  1. Embed the query with the same model used in embed.py.
  2. Query the ChromaDB collection (children only — D-042) for top-k child hits.
  3. Dedupe by parent_id, keeping the highest-scoring child per parent.
  4. Resolve each surviving hit's parent text via parents.get_parent().
  5. Return a list of result dicts where `text` = parent text (what the LLM
     sees) and `child_text` = the matched child fragment (kept for citation
     traceability and debugging).

ChromaDB returns distances (lower = more similar). This module converts them
to similarity scores (higher = more similar) via: score = 1 - distance.

List fields stored in ChromaDB as JSON strings (authors, publication_types,
mesh_terms) are deserialized back to Python lists before returning.

Note on dedup: the spec calls for retrieving more children than needed so
the dedup step has room to collapse same-parent hits. We over-fetch by a
configurable factor (default 4×) so n_results returns approximately n_results
unique parents after dedup.

Public API:
    retrieve(query, n_results, min_score) -> list[dict]
"""

import json
import logging

from pubmed_rag.embed import MODEL_NAME, get_model
from pubmed_rag.parents import get_parent
from pubmed_rag.vectorstore import get_collection

_logger = logging.getLogger(__name__)

_DEFAULT_N_RESULTS = 5
_DEFAULT_MIN_SCORE = 0.0

# Over-fetch factor: we ask Chroma for OVERFETCH_MULTIPLIER × n_results child
# hits so that, after deduplication by parent_id, we still have ~n_results
# unique parents to return. 4× is a generous default for short PubMed
# abstracts where most parents have 1–3 children. Tune for PMC full-text.
_OVERFETCH_MULTIPLIER = 4

# Fields stored as JSON strings in ChromaDB metadata — deserialized on retrieval.
_JSON_FIELDS = ("authors", "publication_types", "mesh_terms")


# Core retrieval
def retrieve(
    query: str,
    n_results: int = _DEFAULT_N_RESULTS,
    min_score: float = _DEFAULT_MIN_SCORE,
) -> list[dict]:
    """
    Find the most semantically similar parents to a query string.

    Searches over child chunks (the embedded targets) and returns up to
    n_results unique PARENTS — each parent represented by the highest-scoring
    child that matched it. The returned `text` field is the parent text;
    `child_text` is the child fragment that triggered the match.

    Args:
        query:     Natural language query string.
        n_results: Maximum number of UNIQUE PARENTS to return (default: 5).
        min_score: Minimum cosine similarity score to include (default: 0.0).
                   Applied to the child score, not the parent. Use ~0.5 to
                   filter weak matches; 0.0 returns all top-k.

    Returns:
        List of result dicts, sorted by score descending. One dict per unique
        parent. Each dict has:
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
            score             (float)     — cosine similarity in [0, 1] (child score)
    """
    _logger.info("Embedding query with %s...", MODEL_NAME)

    query_vector = get_model().encode([query]).tolist()[0]

    collection = get_collection()

    # Over-fetch so dedup-by-parent still leaves enough unique parents.
    overfetch = max(n_results * _OVERFETCH_MULTIPLIER, n_results)
    _logger.info("Querying ChromaDB for top %d child hits (request=%d)...", overfetch, n_results)

    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=overfetch,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB wraps results in an extra list (one per query).
    # We always send one query at a time, so unpack index 0.
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    # Dedup by parent_id, keep best child per parent
    # Chroma returns results sorted by distance ascending (best first), so
    # the first occurrence of each parent_id is the best one.
    seen_parents: set[str] = set()
    results: list[dict] = []

    for child_text, meta, distance in zip(documents, metadatas, distances):
        score = 1.0 - distance

        if score < min_score:
            continue

        # Fall back to chunk_id when parent_id is missing — protects against
        # legacy rows ingested before D-042 (none should exist after re-seed,
        # but the guard costs nothing).
        parent_id = meta.get("parent_id") or meta.get("chunk_id", "")

        if not parent_id or parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        # Resolve parent text from the sidecar store. If the parent is missing
        # (e.g. parents.jsonl out of sync with the collection), fall back to
        # the child text and log — better to return a partial result than
        # crash the whole query.
        try:
            parent_doc = get_parent(parent_id)
            parent_text = parent_doc["text"]
        except KeyError:
            _logger.warning(
                "Parent %s not found in sidecar — falling back to child text.", parent_id
            )
            parent_text = child_text

        result = {
            "text": parent_text,
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
            # Similarity
            "score": round(score, 4),
        }

        # Deserialize list fields stored as JSON strings in ChromaDB
        for field in _JSON_FIELDS:
            raw_val = meta.get(field, "[]")
            result[field] = json.loads(raw_val) if isinstance(raw_val, str) else raw_val

        results.append(result)

        if len(results) >= n_results:
            break

    _logger.info(
        "Returned %d unique-parent results (min_score=%.2f, examined %d child hits).",
        len(results),
        min_score,
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

    args = parser.parse_args()

    results = retrieve(args.query, n_results=args.n, min_score=args.min_score)

    print(f"\n── Top {len(results)} results for: '{args.query}' ──\n")

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
