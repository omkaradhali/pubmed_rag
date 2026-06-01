"""
chunk.py — Split PubMed abstract records into parent + child chunks (v0.2).

v0.2 parent-child chunking (see D-042):
  * Parent chunks (~1200 chars, paragraph-aware) carry the surrounding context
    the LLM needs at generation time. They are NOT embedded.
  * Child chunks (~300 chars, sentence-aware, ~30 char overlap inside the
    parent) are the targets of vector search. ONLY children are embedded.

Both roles are emitted in a single flat list — uniform shape downstream.
Parents come first per record, then their children. embed.py filters the
list by chunk_role before embedding.

ID scheme (stable, human-readable, ChromaDB-safe):
  Parent:  {pmid}_p{parent_index}            e.g. "41980200_p0"
  Child:   {pmid}_p{parent_index}_c{child_index}   e.g. "41980200_p0_c2"

Invariants (enforced by emit, checked downstream):
  chunk_role == "parent"  →  parent_id is None
  chunk_role == "child"   →  parent_id == its parent's chunk_id

Public API:
    chunk_record(record, parent_splitter, child_splitter) -> list[dict]
    chunk_records(records, ...)                            -> list[dict]
    load_and_chunk(path, ...)                              -> list[dict]
    split_parents_children(chunks)                         -> tuple[list, list]
"""

import json
import logging
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

_logger = logging.getLogger(__name__)

# Constants

# v0.2 parent-child sizes. Char-count (not token-count) approximation per D-042
# sub-decision 1: matches v0.1, simpler, well-calibrated for English biomedical
# text. TODO: revisit with a real tokenizer when PMC full-text lands.
#
# The real char/token ratio for all-MiniLM-L6-v2 (BERT WordPiece) on medical
# text is ~4 chars/token — long medical words break into many subword tokens
# (e.g. "pembrolizumab" → ~5 tokens), not few. Earlier 6:1 estimates were
# too generous and would have overflowed MiniLM's 256-token cap.
#
# Parent target ~1800 tokens ≈ ~7200 chars. Cap, not target — short abstracts
# stay whole. Parents are NEVER embedded (they're sidecar-stored and resolved
# at retrieve time), so the 256-token embedding cap doesn't apply here.
#
# Child target ~250 tokens ≈ ~1000 chars. Stays safely under all-MiniLM-L6-v2's
# 256-token cap. Matches v0.1's proven calibration. Smaller children produce
# sharper, more discriminative embeddings — measured as context_precision
# improvement in RAGAS, not absolute cosine score.
DEFAULT_PARENT_CHUNK_SIZE = 7200
DEFAULT_PARENT_CHUNK_OVERLAP = 0  # parents don't overlap — they tile cleanly

DEFAULT_CHILD_CHUNK_SIZE = 1000
DEFAULT_CHILD_CHUNK_OVERLAP = 100  # ~10% overlap, child-internal only


# Splitter construction
def _build_parent_splitter(
    chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_PARENT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Parent splitter: paragraph > line > sentence boundaries."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _build_child_splitter(
    chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Child splitter: sentence > word > char (within parent text only)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[". ", " ", ""],
    )


# Core chunking logic


def _bibliographic_fields(record: dict) -> dict:
    """
    Extract the bibliographic + link-out fields carried on every chunk.

    Pulled into a helper so parents and children get identical metadata —
    the only fields that differ between roles are chunk_id, chunk_role,
    parent_id, text, chunk_index, chunk_total.
    """
    return {
        "pmid": record["pmid"],
        "title": record["title"],
        "year": record["year"],
        # Link-out fields (D-031) — .get() with empty defaults keeps this
        # backward-compatible with older abstracts.jsonl files.
        "doi": record.get("doi", ""),
        "doi_url": record.get("doi_url", ""),
        "pmc_id": record.get("pmc_id", ""),
        "pmc_url": record.get("pmc_url", ""),
        "authors": record.get("authors", []),
        "journal": record.get("journal", ""),
        "publication_types": record.get("publication_types", []),
        "mesh_terms": record.get("mesh_terms", []),
    }


def chunk_record(
    record: dict,
    parent_splitter: RecursiveCharacterTextSplitter,
    child_splitter: RecursiveCharacterTextSplitter,
) -> list[dict]:
    """
    Split a single abstract record into parent + child chunks.

    Strategy:
      1. Prepend title to abstract (preserves topic signal — same as v0.1).
      2. Split into parents using paragraph-aware boundaries.
      3. Split each parent into children using sentence-aware boundaries.
      4. Always emit at least one parent per non-empty record (D-042 sub-2).
      5. Return parents and children in a flat list, parents first, then all
         children for that parent, then the next parent's group, etc.

    For PubMed abstracts (~250 words) most records produce 1 parent + 1–3
    children. PMC full-text (future) will produce multiple parents per
    record with many children each.

    Returned chunk dicts share all bibliographic fields. The role-specific
    fields are:

        chunk_id    (str)        — stable unique ID, see module docstring
        chunk_role  ("parent" | "child")
        parent_id   (str | None) — None for parents, parent's chunk_id for children
        text        (str)        — full parent text OR child fragment
        chunk_index (int)        — position within its role group, 0-based
        chunk_total (int)        — total in its role group

    Args:
        record:          A record from ingest.py (must have "pmid" and "abstract").
        parent_splitter: Configured RecursiveCharacterTextSplitter for parents.
        child_splitter:  Configured RecursiveCharacterTextSplitter for children.

    Returns:
        Flat list of chunk dicts (parents + children). [] if abstract is empty.
    """
    abstract = record.get("abstract", "")
    if not abstract:
        return []

    pmid = record["pmid"]
    title = record["title"]
    bib = _bibliographic_fields(record)

    # Title-prefixed body — same convention as v0.1 so embeddings carry
    # subject context even for short fragments.
    body = f"{title}\n\n{abstract}"

    parent_texts = parent_splitter.split_text(body)

    # Defensive: if for any reason the splitter returns nothing (e.g. body is
    # only whitespace), still emit a single parent with the raw body so the
    # "always emit a parent" invariant holds. This should not happen in
    # practice — included for robustness.
    if not parent_texts:
        parent_texts = [body]

    parent_total = len(parent_texts)
    out: list[dict] = []

    for p_idx, parent_text in enumerate(parent_texts):
        parent_chunk_id = f"{pmid}_p{p_idx}"

        parent_chunk = {
            **bib,
            "chunk_id": parent_chunk_id,
            "chunk_role": "parent",
            "parent_id": None,
            "text": parent_text,
            "chunk_index": p_idx,
            "chunk_total": parent_total,
        }
        out.append(parent_chunk)

        # Children — split THIS parent's text only, so children never bleed
        # across parent boundaries (D-042 sub-2: parent-child invariant).
        child_texts = child_splitter.split_text(parent_text)

        # If a parent is shorter than child_chunk_size the child splitter
        # may return a single fragment equal to the parent. That is fine and
        # in fact required by D-042 sub-2 (always emit a parent row, always
        # emit at least one child under it).
        if not child_texts:
            child_texts = [parent_text]

        child_total = len(child_texts)
        for c_idx, child_text in enumerate(child_texts):
            child_chunk = {
                **bib,
                "chunk_id": f"{parent_chunk_id}_c{c_idx}",
                "chunk_role": "child",
                "parent_id": parent_chunk_id,
                "text": child_text,
                "chunk_index": c_idx,
                "chunk_total": child_total,
            }
            out.append(child_chunk)

    return out


def chunk_records(
    records: list[dict],
    parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    parent_chunk_overlap: int = DEFAULT_PARENT_CHUNK_OVERLAP,
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    child_chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split a list of abstract records into parent + child chunks.

    Builds a single shared splitter per role then delegates to chunk_record
    for each record.

    Args:
        records:               Records from ingest.py or load_and_chunk.
        parent_chunk_size:     Max chars per parent (default: 7200).
        parent_chunk_overlap:  Chars shared between adjacent parents (default: 0).
        child_chunk_size:      Max chars per child (default: 1000).
        child_chunk_overlap:   Chars shared between adjacent children (default: 100).

    Returns:
        Flat list of chunk dicts across all records, mixing parents and children.
    """
    parent_splitter = _build_parent_splitter(parent_chunk_size, parent_chunk_overlap)
    child_splitter = _build_child_splitter(child_chunk_size, child_chunk_overlap)

    _logger.info(
        "Chunking %d records (parent=%d/%d, child=%d/%d)...",
        len(records),
        parent_chunk_size,
        parent_chunk_overlap,
        child_chunk_size,
        child_chunk_overlap,
    )

    chunks: list[dict] = []
    for record in records:
        chunks.extend(chunk_record(record, parent_splitter, child_splitter))

    n_parents = sum(1 for c in chunks if c["chunk_role"] == "parent")
    n_children = sum(1 for c in chunks if c["chunk_role"] == "child")
    _logger.info(
        "Produced %d chunks from %d records (%d parents, %d children).",
        len(chunks),
        len(records),
        n_parents,
        n_children,
    )

    return chunks


def split_parents_children(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Partition a flat chunk list into (parents, children) by chunk_role.

    Used by pipeline.py to persist parents to the sidecar JSONL and pass only
    children to embed.py. Preserves input order within each role.
    """
    parents = [c for c in chunks if c["chunk_role"] == "parent"]
    children = [c for c in chunks if c["chunk_role"] == "child"]
    return parents, children


def load_and_chunk(
    path: str | os.PathLike,
    parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    parent_chunk_overlap: int = DEFAULT_PARENT_CHUNK_OVERLAP,
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    child_chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Load records from a JSONL file and return parent + child chunks.

    Convenience wrapper: reads abstracts.jsonl line by line (memory-efficient),
    then passes all records to chunk_records.
    """
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return chunk_records(
        records,
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
    )


# CLI entrypoint
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Chunk PubMed abstracts into parent + child chunks (v0.2).",
    )
    parser.add_argument(
        "--input",
        default="data/abstracts.jsonl",
        help="Input JSONL file (default: data/abstracts.jsonl)",
    )
    parser.add_argument(
        "--parent-chunk-size",
        type=int,
        default=DEFAULT_PARENT_CHUNK_SIZE,
        help=f"Max chars per parent (default: {DEFAULT_PARENT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--parent-chunk-overlap",
        type=int,
        default=DEFAULT_PARENT_CHUNK_OVERLAP,
        help=f"Overlap between adjacent parents (default: {DEFAULT_PARENT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--child-chunk-size",
        type=int,
        default=DEFAULT_CHILD_CHUNK_SIZE,
        help=f"Max chars per child (default: {DEFAULT_CHILD_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--child-chunk-overlap",
        type=int,
        default=DEFAULT_CHILD_CHUNK_OVERLAP,
        help=f"Overlap between adjacent children (default: {DEFAULT_CHILD_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save chunks to this JSONL file (default: print summary only)",
    )
    args = parser.parse_args()

    chunks = load_and_chunk(
        args.input,
        parent_chunk_size=args.parent_chunk_size,
        parent_chunk_overlap=args.parent_chunk_overlap,
        child_chunk_size=args.child_chunk_size,
        child_chunk_overlap=args.child_chunk_overlap,
    )

    _logger.info("Total chunks: %d", len(chunks))

    if chunks:
        first_parent = next((c for c in chunks if c["chunk_role"] == "parent"), None)
        first_child = next((c for c in chunks if c["chunk_role"] == "child"), None)
        if first_parent:
            _logger.info(
                "First parent — id=%s pmid=%s parent %d/%d\n%s",
                first_parent["chunk_id"],
                first_parent["pmid"],
                first_parent["chunk_index"] + 1,
                first_parent["chunk_total"],
                first_parent["text"][:200],
            )
        if first_child:
            _logger.info(
                "First child — id=%s parent_id=%s child %d/%d\n%s",
                first_child["chunk_id"],
                first_child["parent_id"],
                first_child["chunk_index"] + 1,
                first_child["chunk_total"],
                first_child["text"][:200],
            )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")
        _logger.info("Saved %d chunks to %s", len(chunks), args.output)
