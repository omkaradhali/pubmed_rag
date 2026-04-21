"""
retrieve.py — Semantic retrieval over the ChromaDB vector store.

Embeds a query string with the same model used in embed.py, searches the
ChromaDB collection using HNSW cosine similarity, and returns the top-k
matching chunks with their source metadata and similarity scores.

ChromaDB returns distances (lower = more similar). This module converts them
to similarity scores (higher = more similar) via: score = 1 - distance.

Public API:
    retrieve(query, n_results, min_score) -> list[dict]
"""

import logging

from pubmed_rag.embed import _MODEL_NAME, _model
from pubmed_rag.vectorstore import get_collection

_logger = logging.getLogger(__name__)

_DEFAULT_N_RESULTS = 5
_DEFAULT_MIN_SCORE = 0.0


# Core retrieval


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
            text   (str)   — chunk text
            pmid   (str)   — PubMed ID
            title  (str)   — article title
            year   (str)   — publication year
            chunk_index (int) — position of chunk within the abstract
            chunk_total (int) — total chunks from that abstract
            score  (float) — cosine similarity score in [0, 1]
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

        results.append(
            {
                "text": text,
                "pmid": meta["pmid"],
                "title": meta["title"],
                "year": meta["year"],
                "chunk_index": meta["chunk_index"],
                "chunk_total": meta["chunk_total"],
                "score": round(score, 4),
            }
        )

    _logger.info("Returned %d results (min_score=%.2f).", len(results), min_score)

    return results


# CLI entrypoint

if __name__ == "__main__":
    import argparse
    import json

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
        print(f"    {r['text'][:200]}...")
        print()

    print("── Raw JSON ──")
    print(json.dumps(results, indent=2))
