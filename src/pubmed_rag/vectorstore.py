"""
vectorstore.py — Seed and query a ChromaDB collection from embeddings.jsonl.

Loads embedded chunks from JSONL and upserts them into a persistent ChromaDB
collection keyed by "{pmid}_chunk_{chunk_index}". Seeding is idempotent —
running it twice is safe, existing IDs are overwritten not duplicated.

Public API:
    get_collection()          -> chromadb.Collection   # open or create
    seed_collection(path)     -> int                   # upsert all chunks
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


#  Seeding


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

    collection = get_collection()

    n_batches = math.ceil(len(chunks) / _UPSERT_BATCH_SIZE)

    total = 0

    for i in range(0, len(chunks), _UPSERT_BATCH_SIZE):
        batch = chunks[i : i + _UPSERT_BATCH_SIZE]

        collection.upsert(
            ids=[f"{c['pmid']}_chunk_{c['chunk_index']}" for c in batch],
            embeddings=[c["embedding"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "pmid": c["pmid"],
                    "title": c["title"],
                    "year": c["year"],
                    "chunk_index": c["chunk_index"],
                    "chunk_total": c["chunk_total"],
                }
                for c in batch
            ],
        )

        total += len(batch)

        _logger.info(
            "Upserted batch %d/%d (%d chunks total)",
            i // _UPSERT_BATCH_SIZE + 1,
            n_batches,
            total,
        )

    _logger.info(
        "Done. Collection '%s' has %d documents at %s",
        COLLECTION_NAME,
        collection.count(),
        _CHROMA_PERSIST_DIR,
    )

    return total


#  CLI entrypoint

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
