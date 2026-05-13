"""
embed.py — Embed text chunks into dense vectors using sentence-transformers.

The model is loaded lazily on first call to get_model() — not at import time.
This avoids a ~90MB HuggingFace download whenever the module is imported, and
makes the module safe to import in tests without triggering network calls.

Each chunk gets a 384-float vector attached under the key "embedding". Vectors
are L2-normalised by the model, so cosine similarity equals the dot product —
used by retrieve.py.

Public API:
    MODEL_NAME                            — name of the default embedding model
    get_model()            -> SentenceTransformer  # lazy singleton accessor
    embed_chunks(chunks)   -> list[dict]            # returns NEW dicts with embedding
    save_embeddings(chunks, path) -> None           # write chunks to JSONL
"""

import json
import logging
import os

from sentence_transformers import SentenceTransformer

_logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

# Loaded on the first call to get_model(). Never loaded at import time.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the shared embedding model, initialising it on first call."""
    global _model
    if _model is None:
        _logger.info("Loading sentence-transformer model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ── Core embedding logic ───────────────────────────────────────────────────────


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed a list of chunk dicts and return new dicts with "embedding" added.

    Does not mutate the input list — each returned dict is a shallow copy of
    the original with an additional "embedding" key. Callers can safely reuse
    the original list after this call.

    Passes all texts to the model in one call so it can batch them internally
    (default batch size: 32). Much faster than embedding one chunk at a time.

    Each returned dict has a new key:
        embedding (list[float]) — 384-dimensional dense vector

    Args:
        chunks: List of chunk dicts from chunk.py (must have a "text" key).

    Returns:
        New list of dicts — one per input chunk — each with "embedding" added.
    """
    model = get_model()
    texts = [chunk["text"] for chunk in chunks]

    _logger.info("Embedding %d chunks with %s...", len(texts), MODEL_NAME)

    # encode() returns a numpy array of shape (N, 384).
    # tolist() converts to plain Python floats — required for JSON serialisation.
    vectors = model.encode(texts, show_progress_bar=True).tolist()

    _logger.info("Done. Each vector has %d dimensions.", len(vectors[0]))

    return [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)]


# ── Persistence ────────────────────────────────────────────────────────────────


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


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

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
