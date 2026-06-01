"""
One-shot re-seed for v0.2 parent-child evaluation.

Reuses the EXISTING data/abstracts.jsonl so the corpus is byte-identical to
v0.1 — only the chunking, indexing, and parent-storage layers change. This is
what gives us an apples-to-apples RAGAS comparison.

Wipes:
  data/chroma_db, data/embeddings.jsonl, data/parents.jsonl

Does NOT touch:
  data/abstracts.jsonl  (the corpus snapshot)
  data/chunks.jsonl     (v0.1 artifact — left for reference)

Run:
  .venv/bin/python scripts/reseed_v0_2.py
"""

import logging
import shutil
from pathlib import Path

from pubmed_rag.chunk import load_and_chunk, split_parents_children
from pubmed_rag.embed import embed_chunks, save_embeddings
from pubmed_rag.parents import save_parents
from pubmed_rag.vectorstore import seed_collection

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("reseed")

ABSTRACTS = Path("data/abstracts.jsonl")
CHROMA = Path("data/chroma_db")
EMBEDDINGS = Path("data/embeddings.jsonl")
PARENTS = Path("data/parents.jsonl")


def main() -> None:
    if not ABSTRACTS.exists():
        raise FileNotFoundError(f"{ABSTRACTS} missing — corpus snapshot required.")

    _logger.info("Wiping v0.1 state...")
    if CHROMA.exists():
        shutil.rmtree(CHROMA)
        _logger.info("  removed %s", CHROMA)
    if EMBEDDINGS.exists():
        EMBEDDINGS.unlink()
        _logger.info("  removed %s", EMBEDDINGS)
    if PARENTS.exists():
        PARENTS.unlink()
        _logger.info("  removed %s", PARENTS)

    _logger.info("Chunking from %s with v0.2 parent-child shape...", ABSTRACTS)
    chunks = load_and_chunk(ABSTRACTS)

    parents, children = split_parents_children(chunks)
    _logger.info("  parents=%d  children=%d  total=%d", len(parents), len(children), len(chunks))

    _logger.info("Saving parents → %s...", PARENTS)
    save_parents(parents, PARENTS)

    _logger.info("Embedding %d children...", len(children))
    embedded = embed_chunks(children)
    save_embeddings(embedded, EMBEDDINGS)

    _logger.info("Seeding ChromaDB...")
    n_seeded = seed_collection(EMBEDDINGS)
    _logger.info("Done — %d children in ChromaDB, %d parents in sidecar.", n_seeded, len(parents))


if __name__ == "__main__":
    main()
