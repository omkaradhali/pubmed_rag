"""
retrieve.py — Semantic retrieval over the ChromaDB vector store.

Embeds a query string with the same model used in embed.py, searches the
ChromaDB collection using HNSW cosine similarity, and returns the top-k
matching chunks with their full source metadata and similarity scores.

ChromaDB returns distances (lower = more similar). This module converts them
to similarity scores (higher = more similar) via: score = 1 - distance.

List fields stored in ChromaDB as JSON strings (authors, publication_types,
mesh_terms) are deserialized back to Python lists before returning.

Public API:
    retrieve(query, n_results, min_score) -> list[dict]
"""

import json
import logging

from pubmed_rag.embed import _MODEL_NAME, _model
from pubmed_rag.vectorstore import get_collection

_logger = logging.getLogger(__name__)

_DEFAULT_N_RESULTS = 5
_DEFAULT_MIN_SCORE = 0.0

# Fields stored as JSON strings in ChromaDB metadata — deserialized on retrieval.
_JSON_FIELDS = ("authors", "publication_types", "mesh_terms")


# ── Core retrieval ─────────────────────────────────────────────────────────────


def retrieve(
    query: str,
    n_results: int = _DEFAULT_N_RESULTS,
    min_score: float = _DEFAULT_MIN_SCORE,
) -> list[dict]:
    """
    Find the most semantically similar chunks to a query string.

    Embeds the query with the same model used to embed the corpus, queries
    ChromaDB for the nearest neighbors, and returns results above min_score.

    Args:
        query:     Natural language query string.
        n_results: Maximum number of results to return (default: 5).
        min_score: Minimum cosine similarity score to include (default: 0.0).
                   Use ~0.5 to filter weak matches; 0.0 returns all top-k.

    Returns:
        List of result dicts, sorted by score descending. Each dict has:
            text              (str)       — chunk text (abstract excerpt)
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
            chunk_index       (int)       — 0-based position within the abstract
            chunk_total       (int)       — total chunks from this abstract
            score             (float)     — cosine similarity in [0, 1]
    """
    _logger.info("Embedding query with %s...", _MODEL_NAME)

    query_vector = _model.encode([query]).tolist()[0]

    collection = get_collection()

    _logger.info("Querying ChromaDB for top %d results...", n_results)

    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB wraps results in an extra list (one per query).
    # We always send one query at a time, so unpack index 0.
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    results = []

    for text, meta, distance in zip(documents, metadatas, distances):
        score = 1.0 - distance

        if score < min_score:
            continue

        result = {
            "text": text,
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
            # Chunk position
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

    _logger.info("Returned %d results (min_score=%.2f).", len(results), min_score)

    return results


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Semantic search over PubMed abstracts.")

    parser.add_argument("query", help="Natural language query string.")

    parser.add_argument(
        "--n",
        type=int,
        default=_DEFAULT_N_RESULTS,
        help=f"Number of results to return (default: {_DEFAULT_N_RESULTS})",
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
        print(f"    {r['text'][:200]}...")
        print()

    print("── Raw JSON ──")
    print(json.dumps(results, indent=2))
