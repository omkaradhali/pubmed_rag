"""
chunk.py — Split PubMed abstract records into overlapping text chunks.

Each chunk preserves the full metadata of its parent record so any retrieved
chunk can be traced back to its source paper.

Public API:
    chunk_record(record, splitter)  -> list[dict]   # chunk a single record
    chunk_records(records, ...)     -> list[dict]   # chunk a list of records
    load_and_chunk(path, ...)       -> list[dict]   # load JSONL + chunk in one call
"""

import json
import logging
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

_logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# all-MiniLM-L6-v2 has a 256-token hard limit (~1500-2000 chars for medical text).
# 1000 chars per chunk (~125-165 tokens) aligns with one structured abstract section.
# 100-char overlap (10%) ensures boundary sentences carry over between chunks.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100


# ── Core chunking logic ────────────────────────────────────────────────────────


def chunk_record(record: dict[str, str], splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    """
    Split a single abstract record into overlapping chunks.

    Prepends the title to the abstract text before splitting so every chunk
    carries subject context — important for short chunks that would otherwise
    lose all topic signal.

    Each returned chunk dict has:
        pmid        (str) — inherited from parent record
        title       (str) — inherited from parent record
        year        (str) — inherited from parent record
        text        (str) — the chunk text (title-prefixed abstract fragment)
        chunk_index (int) — position of this chunk within the record (0-based)
        chunk_total (int) — total number of chunks for this record

    Args:
        record:   A single record dict from ingest.py: {pmid, title, abstract, year}
        splitter: A configured RecursiveCharacterTextSplitter instance.

    Returns:
        List of chunk dicts. Returns [] if abstract is empty.
    """

    # Metadata for all chunks
    pmid = record["pmid"]
    title = record["title"]
    year = record["year"]

    abstract = record["abstract"]

    if not abstract:
        return []

    text = f"{title}\n\n{abstract}"
    fragments = splitter.split_text(text)

    return [
        {
            "pmid": pmid,
            "title": title,
            "year": year,
            "text": fragment,
            "chunk_index": i,
            "chunk_total": len(fragments),
        }
        for i, fragment in enumerate(fragments)
    ]


def chunk_records(
    records: list[dict[str, str]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split a list of abstract records into chunks.

    Creates a single shared splitter instance (avoids re-initialising on every
    record) then delegates to chunk_record for each record.

    Args:
        records:       List of record dicts from ingest.py or load_and_chunk.
        chunk_size:    Max characters per chunk (default: 1000).
        chunk_overlap: Characters shared between adjacent chunks (default: 100).

    Returns:
        Flat list of all chunk dicts across all records.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    _logger.info(
        "Chunking %d records (chunk_size=%d, overlap=%d)...",
        len(records),
        chunk_size,
        chunk_overlap,
    )

    chunks = []
    for record in records:
        chunks.extend(chunk_record(record, splitter))

    _logger.info("Produced %d chunks from %d records", len(chunks), len(records))

    return chunks


def load_and_chunk(
    path: str | os.PathLike,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Load records from a JSONL file and return chunks ready for embedding.

    Convenience wrapper: reads abstracts.jsonl line by line (memory-efficient),
    then passes all records to chunk_records.

    Args:
        path:          Path to a JSONL file produced by ingest.save_to_jsonl.
        chunk_size:    Forwarded to chunk_records.
        chunk_overlap: Forwarded to chunk_records.

    Returns:
        Flat list of chunk dicts.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return chunk_records(records, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Chunk PubMed abstracts from a JSONL file.")
    parser.add_argument(
        "--input",
        default="data/abstracts.jsonl",
        help="Input JSONL file (default: data/abstracts.jsonl)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Max characters per chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap between adjacent chunks (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    args = parser.parse_args()

    chunks = load_and_chunk(
        args.input, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )

    _logger.info("Total chunks: %d", len(chunks))

    if chunks:
        first = chunks[0]
        _logger.info(
            "First chunk — pmid: %s, chunk %d/%d\n%s",
            first["pmid"],
            first["chunk_index"] + 1,
            first["chunk_total"],
            first["text"][:200],
        )
