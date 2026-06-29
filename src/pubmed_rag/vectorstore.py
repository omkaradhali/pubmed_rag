"""
vectorstore.py — Seed and query a ChromaDB collection from embeddings.jsonl.

Loads embedded child chunks from JSONL and upserts them into a persistent
ChromaDB collection keyed by chunk_id (e.g. "41980200_p0_c2", D-042). Seeding
is idempotent — running it twice overwrites existing IDs in place.

Only CHILD chunks live in ChromaDB. Parents are persisted by parents.py to a
sidecar JSONL and resolved at retrieval time. The vectorstore enforces this
invariant defensively — any non-child input is dropped with a warning.

ChromaDB metadata only supports scalar types (str, int, float, bool).
List fields (authors, publication_types, mesh_terms) are stored as JSON
strings and deserialized in retrieve.py when results are returned.

Public API:
    get_collection()                        -> chromadb.Collection   # open or create
    validate_collection_dimension()         -> None                  # raises on mismatch
    seed_collection(path)                   -> int                   # upsert from JSONL
    upsert_chunks(chunks)                   -> int                   # upsert in-memory chunks
"""

import json
import logging
import math
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv

load_dotenv()

_logger = logging.getLogger(__name__)

COLLECTION_NAME = "pubmed_abstracts"
_CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

# Number of chunks sent to ChromaDB per upsert call.
# Keeps memory flat when the JSONL grows beyond a few thousand chunks.
_UPSERT_BATCH_SIZE = 500

# Expected vector dimensions per embedding provider. Used by
# validate_collection_dimension() to catch reseed/provider mismatches before
# a query fails mid-run with an opaque ChromaDB error.
_PROVIDER_DIMS: dict[str, int] = {
    "miniml": 384,
    "bge": 1024,
    "medcpt": 768,
}


# Collection access
def get_collection() -> chromadb.Collection:
    """
    Open (or create) the persistent ChromaDB collection.

    Uses cosine similarity space so query scores are in [-1, 1] with
    1.0 = identical. Our vectors are L2-normalised by all-MiniLM-L6-v2,
    so cosine equals dot product.

    Returns:
        The 'pubmed_abstracts' collection.
    """
    Path(_CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=_CHROMA_PERSIST_DIR)

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def validate_collection_dimension() -> None:
    """Verify ChromaDB vector dimension matches the active EMBEDDING_PROVIDER.

    Fetches one embedding from the collection and compares its length against
    the expected dimension for the current EMBEDDING_PROVIDER env var. Raises
    ValueError with a clear reseed command if they don't match.

    Call this at the start of any eval or retrieval script to catch provider/
    reseed mismatches before spending time on model loading and question evaluation.

    Skips silently when the collection is empty (nothing to validate).
    """
    # Import lazily to avoid making embed a hard dependency of vectorstore at
    # module load time.
    from pubmed_rag.embed import EMBEDDING_PROVIDER  # noqa: PLC0415

    expected_dim = _PROVIDER_DIMS.get(EMBEDDING_PROVIDER)
    if expected_dim is None:
        _logger.warning(
            "Unknown EMBEDDING_PROVIDER=%r — skipping dimension check.", EMBEDDING_PROVIDER
        )
        return

    collection = get_collection()
    if collection.count() == 0:
        return  # nothing seeded yet; no mismatch possible

    result = collection.get(limit=1, include=["embeddings"])
    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return

    actual_dim = len(embeddings[0])
    if actual_dim != expected_dim:
        raise ValueError(
            f"ChromaDB dimension mismatch: collection has {actual_dim}-dim vectors "
            f"but EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r} expects {expected_dim}-dim.\n"
            f"Reseed with the correct provider:\n"
            f"  EMBEDDING_PROVIDER={EMBEDDING_PROVIDER} "
            f".venv/bin/python scripts/reseed_v0_2.py"
        )

    _logger.debug(
        "Dimension check passed: EMBEDDING_PROVIDER=%s, dim=%d.", EMBEDDING_PROVIDER, actual_dim
    )


# Metadata helpers
def _chunk_to_metadata(c: dict) -> dict:
    """
    Convert an embedded child chunk dict to a ChromaDB-safe metadata dict.

    ChromaDB only accepts scalar values (str, int, float, bool).
    List fields are serialized with json.dumps and deserialized in retrieve.py.
    Scalar fields fall back to "" so metadata is always complete.

    v0.2 (D-042) adds chunk_id, chunk_role, parent_id so retrieve.py can
    look up the parent text and dedup hits by parent_id.
    """
    return {
        # Scalar identifiers
        "pmid": c.get("pmid", ""),
        "title": c.get("title", ""),
        "year": c.get("year", ""),
        "doi": c.get("doi", ""),
        "doi_url": c.get("doi_url", ""),
        "pmc_id": c.get("pmc_id", ""),
        "pmc_url": c.get("pmc_url", ""),
        "journal": c.get("journal", ""),
        # Parent-child schema (v0.2). chunk_role is always "child" in ChromaDB
        # but stored explicitly to keep metadata self-describing for debugging.
        "chunk_id": c.get("chunk_id", ""),
        "chunk_role": c.get("chunk_role", "child"),
        "parent_id": c.get("parent_id", ""),
        # List fields serialized as JSON strings
        "authors": json.dumps(c.get("authors", [])),
        "publication_types": json.dumps(c.get("publication_types", [])),
        "mesh_terms": json.dumps(c.get("mesh_terms", [])),
        # Chunk position (within the parent — v0.2)
        "chunk_index": c.get("chunk_index", 0),
        "chunk_total": c.get("chunk_total", 1),
    }


# Seeding
def upsert_chunks(chunks: list[dict]) -> int:
    """
    Upsert a list of in-memory embedded child chunks into ChromaDB.

    Handles batching and metadata serialization. Used by pipeline.py for
    incremental updates where chunks are already in memory (no JSONL write).

    Defensive filter: any non-child chunks are dropped with a warning. The
    embed step should have removed them already (D-042 sub-decision 4) — this
    guard catches mistakes by callers that bypass embed_chunks.

    Args:
        chunks: List of embedded chunk dicts (output of embed.embed_chunks).

    Returns:
        Total number of chunks upserted.
    """
    if not chunks:
        return 0

    children = [c for c in chunks if c.get("chunk_role", "child") == "child"]
    n_skipped = len(chunks) - len(children)
    if n_skipped:
        _logger.warning(
            "Dropping %d non-child chunks from upsert — only children belong in ChromaDB.",
            n_skipped,
        )

    if not children:
        return 0

    collection = get_collection()
    n_batches = math.ceil(len(children) / _UPSERT_BATCH_SIZE)
    total = 0

    for i in range(0, len(children), _UPSERT_BATCH_SIZE):
        batch = children[i : i + _UPSERT_BATCH_SIZE]

        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[_chunk_to_metadata(c) for c in batch],
        )

        total += len(batch)
        _logger.info(
            "Upserted batch %d/%d (%d chunks total)",
            i // _UPSERT_BATCH_SIZE + 1,
            n_batches,
            total,
        )

    return total


def seed_collection(path: str | os.PathLike = "data/embeddings.jsonl") -> int:
    """
    Load embedded chunks from JSONL and upsert into ChromaDB.

    Args:
        path: Path to embeddings.jsonl produced by embed.py.

    Returns:
        Total number of chunks upserted.
    """
    chunks = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    _logger.info("Loaded %d chunks from %s", len(chunks), path)

    total = upsert_chunks(chunks)

    collection = get_collection()
    _logger.info(
        "Done. Collection '%s' has %d documents at %s",
        COLLECTION_NAME,
        collection.count(),
        _CHROMA_PERSIST_DIR,
    )

    return total


# CLI entrypoint
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Seed ChromaDB from embeddings.jsonl.")

    parser.add_argument(
        "--input",
        default="data/embeddings.jsonl",
        help="Input JSONL with embeddings (default: data/embeddings.jsonl)",
    )

    args = parser.parse_args()

    n = seed_collection(args.input)
    _logger.info("Seeded %d chunks into '%s' at %s", n, COLLECTION_NAME, _CHROMA_PERSIST_DIR)
