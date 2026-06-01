"""
types.py — Shared type definitions for the pubmed_rag pipeline.

Importing this module has no side effects and no external dependencies.
"""

from typing import Literal, NotRequired, Required, TypedDict

# Parent/child chunking (v0.2, see D-042)
#
# Parent chunks (~1200 tokens, paragraph/section-aware) are stored in a sidecar
# JSONL on disk and looked up at retrieval time to give the LLM full context.
# Child chunks (~300 tokens) are what gets embedded and indexed in ChromaDB —
# they are the sharp targets for vector search. After a child is retrieved, its
# parent text is what the generator sees.
#
# Invariants (enforced in chunk.py + embed.py):
#   chunk_role == "parent"  →  parent_id is None  and  no "embedding" key
#   chunk_role == "child"   →  parent_id is not None  and  embedding present after embed step

ChunkRole = Literal["parent", "child"]


class Chunk(TypedDict, total=False):
    """
    A text chunk with source metadata. Produced by chunk.py.

    Required fields are always present after the chunk step. The `embedding`
    field is absent until embed.py runs (and only ever appears on children).
    All other optional fields depend on what PubMed returned for the source.
    """

    # Required: always present after chunking
    pmid: Required[str]
    title: Required[str]
    year: Required[str]
    text: Required[str]
    chunk_index: Required[int]
    chunk_total: Required[int]

    # v0.2 parent-child schema (D-042)
    chunk_id: Required[str]  # stable unique ID, see ID scheme below
    chunk_role: Required[ChunkRole]  # "parent" or "child"
    parent_id: Required[str | None]  # None for parents; parent chunk_id for children

    # Optional bibliographic fields
    doi: str
    doi_url: str
    pmc_id: str
    pmc_url: str
    authors: list[str]
    journal: str
    publication_types: list[str]
    mesh_terms: list[str]

    # Added by embed.py (children only)
    embedding: NotRequired[list[float]]


class ParentDoc(TypedDict):
    """
    A parent chunk persisted in data/parents.jsonl.

    Identical shape to a Chunk with chunk_role == "parent", but with the
    embedding field intentionally absent (parents are never embedded) and
    declared as a closed TypedDict for type-checker friendliness in the
    parent store (parents.py).
    """

    chunk_id: str
    chunk_role: ChunkRole  # always "parent" for ParentDoc
    parent_id: None  # always None for parents
    pmid: str
    title: str
    year: str
    text: str
    chunk_index: int
    chunk_total: int
    doi: str
    doi_url: str
    pmc_id: str
    pmc_url: str
    authors: list[str]
    journal: str
    publication_types: list[str]
    mesh_terms: list[str]
