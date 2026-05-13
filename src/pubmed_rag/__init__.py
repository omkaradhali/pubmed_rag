"""
pubmed_rag — A RAG pipeline for biomedical literature.

Public API:
    run_pipeline(query, ...)             -> str             # CLI-friendly answer string
    run_pipeline_structured(query, ...)  -> PipelineResult  # structured output for API/UI
    PipelineResult                       — complete structured output dataclass
    SourceChunk                          — one retrieved source with full metadata
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pubmed-rag")
except PackageNotFoundError:
    __version__ = "dev"

from pubmed_rag.pipeline import PipelineResult, SourceChunk, run_pipeline, run_pipeline_structured

__all__ = [
    "__version__",
    "run_pipeline",
    "run_pipeline_structured",
    "PipelineResult",
    "SourceChunk",
]
