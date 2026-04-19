"""
embed.py — Embed text chunks into dense vectors using sentence-transformers.

Loads all-MiniLM-L6-v2 once at module level. Each chunk gets a 384-float
vector attached under the key "embedding". Vectors are L2-normalised by the
model, so cosine similarity equals the dot product — used by retrieve.py.

Public API:
    embed_chunks(chunks)           -> list[dict]   # attach embeddings in-place
    save_embeddings(chunks, path)  -> None         # write chunks to JSONL
"""

import json
import logging
import os

from sentence_transformers import SentenceTransformer

_logger = logging.getLogger(__name__)

# Loaded once at import time — ~1 s startup cost, zero cost on every call after.
_MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(_MODEL_NAME)


# ── Core embedding logic ───────────────────────────────────────────────────────


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed a list of chunk dicts and attach the vector to each.

    Passes all texts to the model in one call so it can batch them internally
    (default batch size: 32). Much faster than embedding one chunk at a time.

    Each chunk dict gets a new key:
        embedding (list[float]) — 384-dimensional dense vector

    Args:
        chunks: List of chunk dicts from chunk.py (must have a "text" key).

    Returns:
        The same list with "embedding" added to each dict.
    """

    texts = [chunk["text"] for chunk in chunks]

    _logger.info("Embedding %d chunks with %s...", len(texts), _MODEL_NAME)

    # encode() returns a numpy array of shape (N, 384).
    # tolist() converts to plain Python floats — required for JSON serialisation.
    vectors = _model.encode(texts, show_progress_bar=True).tolist()

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    _logger.info("Done. Each vector has %d dimensions.", len(vectors[0]))

    return chunks


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

    # Load chunks
    chunks = []

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    _logger.info("Loaded %d chunks from %s", len(chunks), args.input)

    # Embed and save
    embed_chunks(chunks)

    save_embeddings(chunks, args.output)

    # Sanity check — print the first embedding's shape and a preview
    first = chunks[0]

    _logger.info(
        "Sample — pmid: %s | vector shape: (%d,) | first 5 values: %s",
        first["pmid"],
        len(first["embedding"]),
        first["embedding"][:5],
    )
