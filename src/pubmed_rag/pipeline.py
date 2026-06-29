"""
pipeline.py — End-to-end RAG pipeline orchestrator.

Wires ingest → chunk → embed → seed → retrieve → generate into a single
callable. Two modes:

    full        Rebuild the corpus from scratch. Existing chroma_db and
                data files are wiped before ingesting. Use --reldate N to
                limit the ingest window to the last N days.

    incremental Skip the expensive rebuild and query the existing collection.
                With --reldate N, fetches and appends only recent abstracts
                before querying — avoids a full reseed.

Public API:
    run_pipeline(query, mode, reldate, n_results, min_score, verbose) -> str
    run_pipeline_structured(query, mode, reldate, n_results, min_score) -> PipelineResult
    format_pipeline_output(result, verbose) -> str

Data classes:
    SourceChunk   — one retrieved chunk with full source metadata
    PipelineResult — complete structured output
"""

import datetime
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from pubmed_rag.chunk import chunk_records, load_and_chunk, split_parents_children
from pubmed_rag.embed import embed_chunks, save_embeddings
from pubmed_rag.generate import generate_answer
from pubmed_rag.guardrails import run_input_guardrails, run_output_guardrails
from pubmed_rag.ingest import ingest, save_to_jsonl
from pubmed_rag.parents import append_parents, save_parents
from pubmed_rag.retrieve import retrieve
from pubmed_rag.vectorstore import get_collection, seed_collection, upsert_chunks

load_dotenv()

_logger = logging.getLogger(__name__)

# Defaults (overridable via env vars)

ABSTRACTS_PATH = Path(os.getenv("ABSTRACTS_PATH", "data/abstracts.jsonl"))
EMBEDDINGS_PATH = Path(os.getenv("EMBEDDINGS_PATH", "data/embeddings.jsonl"))
PARENTS_PATH = Path(os.getenv("PARENTS_PATH", "data/parents.jsonl"))
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db"))
INGEST_QUERY = os.getenv("INGEST_QUERY", "oncology[Title/Abstract]")
INGEST_MAX_RESULTS = int(os.getenv("INGEST_MAX_RESULTS", "500"))

_SEP = "─" * 62

# Phrases that suggest the LLM couldn't find enough context
_LOW_COVERAGE_PHRASES = (
    "does not address",
    "not sufficient",
    "cannot answer",
    "no relevant",
    "additional literature",
    "not contain",
    "does not provide",
)


# Data classes


@dataclass
class SourceChunk:
    """
    One retrieved chunk with full source metadata.

    All fields are available for UI rendering — display whichever subset
    makes sense for the context (e.g. hide mesh_terms in a compact card,
    show them in a detail drawer).
    """

    number: int  # citation number matching [N] in the answer text
    pmid: str
    title: str
    authors: list[str]
    journal: str
    year: str
    publication_types: list[str]  # e.g. ["Journal Article", "Randomized Controlled Trial"]
    mesh_terms: list[str]  # NLM MeSH descriptors; empty for recently indexed articles
    doi: str  # raw DOI string, e.g. "10.1038/s41591-024-01234-5", or ""
    doi_url: str  # clickable https://doi.org/{doi}, or ""
    pmc_id: str  # e.g. "PMC11234567", or "" if no free full text
    pmc_url: str  # clickable PMC link, or ""
    pubmed_url: str  # always set: https://pubmed.ncbi.nlm.nih.gov/{pmid}/
    score: float  # cosine similarity [0, 1]; higher = more relevant
    chunk_index: int  # 0-based position of this chunk within its abstract
    chunk_total: int  # total chunks produced from this abstract
    text: str  # the actual chunk text (abstract excerpt used as LLM context)


@dataclass
class PipelineResult:
    """
    Complete structured output from the RAG pipeline.

    Design intent: use this dataclass directly in the FastAPI response model.
    All fields are populated on every run — the UI picks which subset to
    render based on context (compact card vs. full detail view).
    """

    # Query
    query: str

    # LLM output
    answer: str  # cited answer text with inline [N] references

    # Sources
    # One entry per retrieved chunk, numbered to match [N] in answer.
    # Ordered by relevance (highest score first).
    sources: list[SourceChunk] = field(default_factory=list)

    # LLM provenance
    llm_provider: str = ""  # e.g. "anthropic", "ollama", "openai"
    llm_model: str = ""  # e.g. "claude-haiku-4-5-20251001", "llama3.1:8b"

    # Retrieval stats
    n_chunks_retrieved: int = 0  # actual chunks returned (≤ n_chunks_requested)
    n_chunks_requested: int = 0  # n_results parameter passed to retrieve()

    # Corpus stats
    n_docs_in_corpus: int = 0  # total chunk count in ChromaDB collection
    corpus_updated_at: str = ""  # ISO date of abstracts.jsonl last modification

    # Confidence
    avg_score: float = 0.0
    min_score_retrieved: float = 0.0
    max_score_retrieved: float = 0.0
    # "High" (avg ≥ 0.70) | "Medium" (0.50-0.70) | "Low" (< 0.50) | "None" (no chunks)
    confidence_tier: str = "None"

    # Coverage
    # Set when the LLM explicitly indicates the corpus lacks sufficient context.
    coverage_note: str | None = None

    # Output guardrail warnings (empty list = all checks passed).
    # Each entry is a dict with keys: code, reason, detail.
    guardrail_flags: list[dict] = field(default_factory=list)


# Output formatting


def format_pipeline_output(result: PipelineResult, verbose: bool = False) -> str:
    """
    Render a PipelineResult as a human-readable string for CLI output.

    Args:
        result:  Structured pipeline output.
        verbose: If True, include MeSH terms and abstract excerpt per source.

    Returns:
        Multi-line formatted string ready for print().
    """
    lines: list[str] = []

    # Query
    lines.append(f"Query: {result.query}")
    lines.append("")

    # Answer
    lines.append(result.answer)

    # Sources
    if result.sources:
        lines.append("")
        lines.append(_SEP)
        lines.append("Sources")
        lines.append(_SEP)

        for src in result.sources:
            lines.append(f"[{src.number}] {src.title}")

            # Authors — first 3 then et al.
            if len(src.authors) > 3:
                author_str = ", ".join(src.authors[:3]) + " et al."
            elif src.authors:
                author_str = ", ".join(src.authors)
            else:
                author_str = "Authors not available"
            lines.append(f"    Authors:  {author_str}")

            # Journal + year
            if src.journal:
                lines.append(f"    Journal:  {src.journal}, {src.year}")
            else:
                lines.append(f"    Year:     {src.year}")

            # Publication types
            if src.publication_types:
                lines.append(f"    Type:     {', '.join(src.publication_types)}")

            # Score + chunk position
            lines.append(
                f"    Score:    {src.score:.4f}"
                f"  |  Chunk {src.chunk_index + 1} of {src.chunk_total}"
            )

            # Links — always show PubMed; DOI and PMC only when present
            lines.append(f"    PubMed:   {src.pubmed_url}")
            if src.doi_url:
                lines.append(f"    DOI:      {src.doi_url}")
            if src.pmc_url:
                lines.append(f"    PMC:      {src.pmc_url}")

            # Verbose extras
            if verbose:
                if src.mesh_terms:
                    mesh_preview = ", ".join(src.mesh_terms[:6])
                    suffix = " ..." if len(src.mesh_terms) > 6 else ""
                    lines.append(f"    MeSH:     {mesh_preview}{suffix}")
                lines.append(f"    Excerpt:  {src.text[:150]}...")

            lines.append("")

    # Footer
    lines.append(_SEP)
    lines.append(f"Generated by:  {result.llm_provider} / {result.llm_model}")
    lines.append(
        f"Corpus:        {result.n_docs_in_corpus:,} chunks  |  Updated: {result.corpus_updated_at}"
    )
    lines.append(
        f"Confidence:    {result.confidence_tier}"
        f"  (avg score: {result.avg_score:.4f}"
        f"  |  {result.n_chunks_retrieved}/{result.n_chunks_requested} chunks retrieved)"
    )
    if result.coverage_note:
        lines.append(f"⚠  {result.coverage_note}")
    lines.append(_SEP)

    return "\n".join(lines)


# Public API


def run_pipeline_structured(
    query: str,
    mode: str = "incremental",
    reldate: int | None = None,
    n_results: int = 5,
    min_score: float = 0.0,
) -> PipelineResult:
    """
    Run the RAG pipeline and return a fully structured PipelineResult.

    Primary entry point for the FastAPI layer. Serialize the returned
    dataclass directly into the API response model.

    Args:
        query:     Question to answer.
        mode:      "full" rebuilds corpus from scratch; "incremental" uses existing collection.
        reldate:   Restrict ingest to abstracts indexed in the last N days.
        n_results: Number of chunks to retrieve (default: 5).
        min_score: Minimum cosine similarity threshold (default: 0.0).

    Returns:
        PipelineResult with all fields populated.
    """
    # Input guardrails — raises GuardrailError on failure; caller handles it.
    run_input_guardrails(query)

    if mode == "full":
        _logger.info("Mode: full — rebuilding corpus from scratch.")
        _run_full_ingest(reldate=reldate)
    elif reldate is not None:
        _logger.info("Mode: incremental — appending last %d days of abstracts.", reldate)
        _run_incremental_update(reldate=reldate)
    else:
        _logger.info("Mode: incremental — querying existing collection.")

    _logger.info("Retrieving top-%d chunks for: %r", n_results, query)
    chunks = retrieve(query, n_results=n_results, min_score=min_score)

    answer = (
        generate_answer(query, chunks)
        if chunks
        else "No relevant abstracts found in the current corpus for this query."
    )
    _logger.info("Generated answer (%d chars).", len(answer))

    # Output guardrails — advisory warnings, never block the response.
    guardrail_flags = [
        asdict(r)
        for r in run_output_guardrails(answer, chunks)
        if not r.passed
    ]

    # Build sources
    sources = [
        SourceChunk(
            number=i + 1,
            pmid=c["pmid"],
            title=c["title"],
            authors=c.get("authors", []),
            journal=c.get("journal", ""),
            year=c["year"],
            publication_types=c.get("publication_types", []),
            mesh_terms=c.get("mesh_terms", []),
            doi=c.get("doi", ""),
            doi_url=c.get("doi_url", ""),
            pmc_id=c.get("pmc_id", ""),
            pmc_url=c.get("pmc_url", ""),
            pubmed_url=c.get("pubmed_url", f"https://pubmed.ncbi.nlm.nih.gov/{c['pmid']}/"),
            score=c["score"],
            chunk_index=c["chunk_index"],
            chunk_total=c["chunk_total"],
            text=c["text"],
        )
        for i, c in enumerate(chunks)
    ]

    # Corpus stats
    collection = get_collection()
    n_docs = collection.count()

    corpus_updated_at = "unknown"
    if ABSTRACTS_PATH.exists():
        mtime = ABSTRACTS_PATH.stat().st_mtime
        corpus_updated_at = datetime.date.fromtimestamp(mtime).isoformat()

    # Confidence
    scores = [c["score"] for c in chunks]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    confidence_tier = (
        "None"
        if not scores
        else "High"
        if avg_score >= 0.70
        else "Medium"
        if avg_score >= 0.50
        else "Low"
    )

    # Coverage note
    coverage_note = None
    if any(phrase in answer.lower() for phrase in _LOW_COVERAGE_PHRASES):
        coverage_note = (
            "Answer may be incomplete — corpus lacks sufficient coverage for this query."
        )

    # LLM provenance
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")
    llm_model = os.getenv("LLM_MODEL", "unknown")

    return PipelineResult(
        query=query,
        answer=answer,
        sources=sources,
        llm_provider=llm_provider,
        llm_model=llm_model,
        n_chunks_retrieved=len(chunks),
        n_chunks_requested=n_results,
        n_docs_in_corpus=n_docs,
        corpus_updated_at=corpus_updated_at,
        avg_score=round(avg_score, 4),
        min_score_retrieved=round(min(scores), 4) if scores else 0.0,
        max_score_retrieved=round(max(scores), 4) if scores else 0.0,
        confidence_tier=confidence_tier,
        coverage_note=coverage_note,
        guardrail_flags=guardrail_flags,
    )


def run_pipeline(
    query: str,
    mode: str = "incremental",
    reldate: int | None = None,
    n_results: int = 5,
    min_score: float = 0.0,
    verbose: bool = False,
) -> str:
    """
    Run the RAG pipeline and return a formatted string for CLI output.

    Thin wrapper around run_pipeline_structured + format_pipeline_output.
    For programmatic use (e.g. FastAPI), call run_pipeline_structured() directly.

    Args:
        query:     Question to answer.
        mode:      "full" or "incremental" (default: "incremental").
        reldate:   Restrict ingest to last N days (optional).
        n_results: Chunks to retrieve (default: 5).
        min_score: Minimum similarity threshold (default: 0.0).
        verbose:   Include MeSH terms and abstract excerpts in source listings.

    Returns:
        Formatted multi-line string with answer, sources, and metadata footer.
    """
    result = run_pipeline_structured(
        query=query,
        mode=mode,
        reldate=reldate,
        n_results=n_results,
        min_score=min_score,
    )
    return format_pipeline_output(result, verbose=verbose)


# Private helpers


def _run_full_ingest(reldate: int | None = None) -> None:
    """Wipe existing corpus and rebuild: ingest → chunk → save parents → embed children → seed."""
    for path in (ABSTRACTS_PATH, EMBEDDINGS_PATH, PARENTS_PATH):
        if path.exists():
            path.unlink()
            _logger.info("Removed stale %s", path)

    # Wipe chroma_db so the collection starts clean — upsert alone won't remove
    # chunks from PMIDs that no longer exist in the new corpus. v0.2 also
    # changes the ID format (chunk_id, D-042) so legacy rows must go.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        _logger.info("Wiped chroma_db at %s", CHROMA_DIR)

    _logger.info("Ingesting up to %d abstracts (reldate=%s)...", INGEST_MAX_RESULTS, reldate)
    records = ingest(INGEST_QUERY, max_results=INGEST_MAX_RESULTS, reldate=reldate)
    _logger.info("Fetched %d records.", len(records))

    if not records:
        _logger.warning("Ingest returned 0 records — corpus is empty.")
        return

    save_to_jsonl(records, ABSTRACTS_PATH)

    chunks = load_and_chunk(ABSTRACTS_PATH)
    parents, children = split_parents_children(chunks)
    _logger.info("Produced %d parents + %d children.", len(parents), len(children))

    # Parents go to the sidecar JSONL — they're never embedded (D-042 sub-4).
    save_parents(parents, PARENTS_PATH)

    # Children are embedded and indexed in ChromaDB.
    embedded = embed_chunks(children)
    save_embeddings(embedded, EMBEDDINGS_PATH)
    _logger.info("Embedded %d children → %s", len(embedded), EMBEDDINGS_PATH)

    n_seeded = seed_collection(EMBEDDINGS_PATH)
    _logger.info("Seeded %d children into ChromaDB.", n_seeded)


def _run_incremental_update(reldate: int) -> None:
    """
    Fetch abstracts from the last N days and upsert into the existing collection.

    Only the new records are chunked and embedded — existing corpus is untouched.
    ChromaDB upsert is idempotent: re-ingesting a PMID already in the collection
    overwrites its children cleanly (same chunk_id values, updated metadata).
    Parents are appended to parents.jsonl — duplicate chunk_id entries from a
    re-ingest will be re-read by parents.load_parents but the dict-keyed cache
    naturally collapses them to the latest.

    Records are not appended to abstracts.jsonl — that file is the full-ingest
    snapshot and would accumulate duplicates across repeated incremental runs.
    ChromaDB + parents.jsonl are the authoritative state; abstracts.jsonl is
    only written during a full rebuild.
    """
    new_records = ingest(INGEST_QUERY, max_results=INGEST_MAX_RESULTS, reldate=reldate)

    if not new_records:
        _logger.info("No new abstracts in the last %d days.", reldate)
        return

    _logger.info("Fetched %d new records.", len(new_records))

    new_chunks = chunk_records(new_records)
    parents, children = split_parents_children(new_chunks)
    _logger.info("Produced %d new parents + %d new children.", len(parents), len(children))

    append_parents(parents, PARENTS_PATH)

    new_embedded = embed_chunks(children)
    n_upserted = upsert_chunks(new_embedded)
    _logger.info("Upserted %d new children into ChromaDB.", n_upserted)


# CLI entrypoint

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Run the pubmed_rag pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python -m pubmed_rag.pipeline "What is CAR-T cell therapy?"\n'
            '  python -m pubmed_rag.pipeline "PD-L1 immunotherapy" --mode full\n'
            '  python -m pubmed_rag.pipeline "BRCA1 mutations" --reldate 30\n'
            '  python -m pubmed_rag.pipeline "immunotherapy" --verbose\n'
        ),
    )
    parser.add_argument("query", help="Question to answer.")
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="incremental",
        help="'full' rebuilds corpus from scratch; 'incremental' uses existing"
        " collection (default: incremental)",
    )
    parser.add_argument(
        "--reldate",
        type=int,
        default=None,
        metavar="N",
        help="Restrict ingest to abstracts indexed in the last N days.",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of chunks to retrieve as context (default: 5).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum cosine similarity threshold (default: 0.0).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include MeSH terms and abstract excerpts in source listings.",
    )
    args = parser.parse_args()

    output = run_pipeline(
        query=args.query,
        mode=args.mode,
        reldate=args.reldate,
        n_results=args.n_results,
        min_score=args.min_score,
        verbose=args.verbose,
    )

    print("\n" + output)
