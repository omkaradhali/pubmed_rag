"""
embed.py — Embed text chunks into dense vectors using sentence-transformers.

The embedding model is selected by the EMBEDDING_PROVIDER env var (ADR-034):
miniml (all-MiniLM-L6-v2, 384-dim, default) or bge (BAAI/bge-large-en-v1.5,
1024-dim). It loads lazily on first call to get_model() — not at import time —
so importing the module triggers no download and is test-safe.

Passages (chunk text) are embedded with no prefix; queries go through
embed_query(), which applies the provider's query instruction prefix (BGE is
asymmetric — see below). All vectors are L2-normalised, so cosine similarity
equals the dot product — used by retrieve.py.

Switching providers requires re-embedding the whole corpus and rebuilding the
vector store (different model → different, often larger vectors → new collection).

Public API:
    EMBEDDING_PROVIDER                    — active provider (env-selected)
    MODEL_NAME                            — resolved HF model id
    get_model()            -> SentenceTransformer  # lazy singleton accessor
    embed_query(query)     -> list[float]           # query vector (with prefix)
    embed_chunks(chunks)   -> list[dict]            # NEW dicts with embedding (children)
    save_embeddings(chunks, path) -> None           # write chunks to JSONL
"""

import json
import logging
import os

from sentence_transformers import SentenceTransformer

_logger = logging.getLogger(__name__)

# Embedding provider dispatch (EMBEDDING_PROVIDER env var, see ADR-034).
#
# Both providers are sentence-transformers bi-encoders. The correctness-critical
# difference is query/passage SYMMETRY:
#   * miniml (all-MiniLM-L6-v2) — SYMMETRIC: query and passage encoded the same,
#     no prefix.
#   * bge (BAAI/bge-large-en-v1.5) — ASYMMETRIC: expects a short instruction
#     prepended to the QUERY only. Omitting it silently degrades retrieval;
#     passages get no prefix.
#
# Each entry: (hf_model_id, query_instruction_prefix).
_PROVIDERS: dict[str, tuple[str, str]] = {
    "miniml": ("all-MiniLM-L6-v2", ""),
    "bge": (
        "BAAI/bge-large-en-v1.5",
        "Represent this sentence for searching relevant passages: ",
    ),
}

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "miniml").lower()

if EMBEDDING_PROVIDER not in _PROVIDERS:
    raise ValueError(
        f"EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r} is not supported here. "
        f"Choose one of: {', '.join(_PROVIDERS)}."
    )

MODEL_NAME, _QUERY_PREFIX = _PROVIDERS[EMBEDDING_PROVIDER]

# Loaded on the first call to get_model(). Never loaded at import time.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the shared embedding model, initialising it on first call.

    Which model loads is set by EMBEDDING_PROVIDER (default: miniml). The corpus
    and the query must be embedded by the SAME model — vectors from different
    models are geometrically incompatible — so re-seeding and query-time
    retrieval must run with the same EMBEDDING_PROVIDER.
    """
    global _model
    if _model is None:
        _logger.info("Loading embedding model: %s (provider=%s)", MODEL_NAME, EMBEDDING_PROVIDER)
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_query(query: str) -> list[float]:
    """Embed one query string, applying the provider's query instruction prefix.

    Asymmetric models (BGE) need a prefix on the query but not on passages;
    symmetric models (miniml) use an empty prefix. Centralising it here keeps
    retrieve.py provider-agnostic. Returns a plain list of floats (L2-normalised).
    """
    text = _QUERY_PREFIX + query
    return get_model().encode([text], normalize_embeddings=True).tolist()[0]


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
                                  (miniml=384, bge=1024)

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
