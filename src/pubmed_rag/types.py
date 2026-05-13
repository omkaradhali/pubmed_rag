"""
types.py — Shared type definitions for the pubmed_rag pipeline.

Importing this module has no side effects and no external dependencies.
"""

from typing import NotRequired, Required, TypedDict


class Chunk(TypedDict, total=False):
    """
    A text chunk with source metadata. Produced by chunk.py.

    Required fields are always present after the chunk step. The `embedding`
    field is absent until embed.py runs; all other optional fields depend on
    what PubMed returned for the source article.
    """

    # ── Required: always present after chunking ────────────────────────────
    pmid: Required[str]
    title: Required[str]
    year: Required[str]
    text: Required[str]
    chunk_index: Required[int]
    chunk_total: Required[int]

    # ── Optional bibliographic fields ──────────────────────────────────────
    doi: str
    doi_url: str
    pmc_id: str
    pmc_url: str
    authors: list[str]
    journal: str
    publication_types: list[str]
    mesh_terms: list[str]

    # ── Added by embed.py ──────────────────────────────────────────────────
    embedding: NotRequired[list[float]]
