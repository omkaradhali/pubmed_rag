"""
embed.py — Embed text chunks into dense vectors using sentence-transformers.

The embedding model is selected by the EMBEDDING_PROVIDER env var (ADR-034):
  miniml   — all-MiniLM-L6-v2, 384-dim, default (symmetric)
  bge      — BAAI/bge-large-en-v1.5, 1024-dim (asymmetric, query prefix)
  medcpt   — ncbi/MedCPT-Article-Encoder, 768-dim (asymmetric, separate query
             model ncbi/MedCPT-Query-Encoder — D-043, locked 2026-06-28)

Passages (chunk text) are embedded with get_model(); queries go through
embed_query(), which uses get_query_model(). For miniml/bge these return the
same model (query differentiated by prefix); for medcpt they are two separate
HuggingFace models loaded as independent singletons — the article encoder and
the query encoder were trained jointly by NCBI on 255M PubMed click-throughs.

Switching providers requires re-embedding the whole corpus and rebuilding the
vector store (different model → different vector space → incompatible collection).

Public API:
    EMBEDDING_PROVIDER                    — active provider (env-selected)
    MODEL_NAME                            — resolved HF article-encoder model id
    get_model()            -> SentenceTransformer  # lazy article-encoder singleton
    get_query_model()      -> SentenceTransformer  # lazy query-encoder singleton
    embed_query(query)     -> list[float]           # query vector (provider-aware)
    embed_chunks(chunks)   -> list[dict]            # NEW dicts with embedding (children)
    save_embeddings(chunks, path) -> None           # write chunks to JSONL
"""

import json
import logging
import os

from sentence_transformers import SentenceTransformer

_logger = logging.getLogger(__name__)

# Embedding provider dispatch (EMBEDDING_PROVIDER env var, see ADR-034 + D-043).
#
# Each entry: (article_model_id, query_prefix, query_model_id | None).
#
# query/passage SYMMETRY rules:
#   miniml  — SYMMETRIC: same model, no prefix for either side.
#   bge     — ASYMMETRIC via PREFIX: same model, instruction prefix on query only.
#             Omitting the prefix silently degrades retrieval (Day-19 ablation).
#   medcpt  — ASYMMETRIC via SEPARATE MODEL: ncbi/MedCPT-Article-Encoder encodes
#             passages; ncbi/MedCPT-Query-Encoder encodes queries. Both trained
#             jointly by NCBI on 255M PubMed click-throughs (D-043). No prefix
#             needed — the separate model handles the query/passage distinction.
_PROVIDERS: dict[str, tuple[str, str, str | None]] = {
    "miniml": ("all-MiniLM-L6-v2", "", None),
    "bge": (
        "BAAI/bge-large-en-v1.5",
        "Represent this sentence for searching relevant passages: ",
        None,
    ),
    "medcpt": ("ncbi/MedCPT-Article-Encoder", "", "ncbi/MedCPT-Query-Encoder"),
}

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "miniml").lower()

if EMBEDDING_PROVIDER not in _PROVIDERS:
    raise ValueError(
        f"EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r} is not supported here. "
        f"Choose one of: {', '.join(_PROVIDERS)}."
    )

MODEL_NAME, _QUERY_PREFIX, _QUERY_MODEL_NAME = _PROVIDERS[EMBEDDING_PROVIDER]

# Article-encoder singleton — loaded on first get_model() call, never at import time.
_model: SentenceTransformer | None = None
# Query-encoder singleton — only used when _QUERY_MODEL_NAME is not None (medcpt).
_query_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the article-encoder singleton, loading it on first call.

    Used for embedding corpus chunks. Which model loads is set by
    EMBEDDING_PROVIDER (default: miniml). The corpus and queries must use
    compatible models — re-seeding and query-time retrieval must run with the
    same EMBEDDING_PROVIDER value.
    """
    global _model
    if _model is None:
        _logger.info("Loading article encoder: %s (provider=%s)", MODEL_NAME, EMBEDDING_PROVIDER)
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_query_model() -> SentenceTransformer:
    """Return the query-encoder singleton, loading it on first call.

    For miniml and bge this is the same model as get_model() (query
    differentiation is handled by the prefix in embed_query). For medcpt it is
    a separate ncbi/MedCPT-Query-Encoder model — the two encoders were trained
    jointly so their vector spaces are compatible.
    """
    global _query_model
    if _QUERY_MODEL_NAME is None:
        return get_model()
    if _query_model is None:
        _logger.info(
            "Loading query encoder: %s (provider=%s)", _QUERY_MODEL_NAME, EMBEDDING_PROVIDER
        )
        _query_model = SentenceTransformer(_QUERY_MODEL_NAME)
    return _query_model


def embed_query(query: str) -> list[float]:
    """Embed one query string using the provider's query encoder.

    For miniml/bge: uses the shared model with an optional instruction prefix.
    For medcpt: uses the separate MedCPT-Query-Encoder (no prefix needed).
    Returns a plain list of floats (L2-normalised). retrieve.py calls this
    and remains provider-agnostic.
    """
    text = _QUERY_PREFIX + query
    return get_query_model().encode([text], normalize_embeddings=True).tolist()[0]


# Core logic
def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed child chunks and return new dicts with "embedding" added.

    Only chunks with chunk_role == "child" are embedded (D-042 sub-decision 4).
    Parents pass through the role filter and are not returned — embedding them
    would pollute the vector index. Pipeline orchestration persists parents to
    parents.jsonl via parents.save_parents() before calling this function.

    For backward-compatibility with v0.1 chunk dicts (which had no chunk_role
    field), chunks missing the role key are treated as children — this keeps
    older test fixtures and ad-hoc scripts working.

    Does not mutate the input list — each returned dict is a shallow copy of
    the original with an additional "embedding" key.

    Passes all texts to the model in one call so it can batch them internally
    (default batch size: 32). Much faster than embedding one chunk at a time.

    Each returned dict has a new key:
        embedding (list[float]) — dense vector; dimension depends on the provider
                                  (miniml=384, bge=1024, medcpt=768)

    Args:
        chunks: List of chunk dicts from chunk.py (must have a "text" key).

    Returns:
        New list of dicts — one per CHILD input chunk — each with "embedding" added.
        Parents in the input are dropped.
    """
    children = [c for c in chunks if c.get("chunk_role", "child") == "child"]
    n_skipped = len(chunks) - len(children)

    if not children:
        _logger.warning("No child chunks to embed (input had %d items).", len(chunks))
        return []

    if n_skipped:
        _logger.info("Skipping %d parents — only children are embedded.", n_skipped)

    model = get_model()
    texts = [chunk["text"] for chunk in children]

    _logger.info("Embedding %d child chunks with %s...", len(texts), MODEL_NAME)

    # encode() returns a numpy array of shape (N, dim). normalize_embeddings keeps
    # vectors unit-length (required for BGE's cosine usage; harmless for miniml).
    # tolist() converts to plain Python floats — required for JSON serialisation.
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True).tolist()

    _logger.info("Done. Each vector has %d dimensions.", len(vectors[0]))

    return [{**chunk, "embedding": vector} for chunk, vector in zip(children, vectors)]


# Persistence
def save_embeddings(chunks: list[dict], path: str | os.PathLike) -> None:
    """
    Write embedded chunks to a JSONL file.

    Args:
        chunks: List of chunk dicts that already have an "embedding" key.
        path:   Destination file path. Created or overwritten.
    """
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    _logger.info("Saved %d embedded chunks to %s", len(chunks), path)


# CLI entrypoint
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Embed PubMed chunks with sentence-transformers.")

    parser.add_argument(
        "--input",
        default="data/chunks.jsonl",
        help="Input JSONL file of chunks (default: data/chunks.jsonl)",
    )

    parser.add_argument(
        "--output",
        default="data/embeddings.jsonl",
        help="Output JSONL file with embeddings attached (default: data/embeddings.jsonl)",
    )

    args = parser.parse_args()

    chunks = []

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    _logger.info("Loaded %d chunks from %s", len(chunks), args.input)

    embedded = embed_chunks(chunks)

    save_embeddings(embedded, args.output)

    first = embedded[0]

    _logger.info(
        "Sample — pmid: %s | vector shape: (%d,) | first 5 values: %s",
        first["pmid"],
        len(first["embedding"]),
        first["embedding"][:5],
    )
