from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GuardrailFlagResponse(BaseModel):
    code: str = Field(description="Machine-readable guardrail code, e.g. MISSING_CITATIONS.")
    reason: str = Field(description="Human-readable explanation of why the check failed.")
    detail: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured detail, e.g. out-of-range citation numbers or low-overlap pairs.",
    )


class HealthResponse(BaseModel):
    status: str = Field(description='Service status. "ok" when the API is running.')
    version: str = Field(description="API version string, e.g. 0.1.0.")


class AskRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What are the treatments for HER2-positive metastatic breast cancer?",
                "mode": "incremental",
                "n_results": 5,
                "min_score": 0.0,
                "reldate": None,
            }
        }
    )

    query: str = Field(
        description="The clinical question to answer.",
        examples=["What are the treatments for HER2-positive metastatic breast cancer?"],
    )
    mode: Literal["incremental", "full"] = Field(
        default="incremental",
        description=(
            "Pipeline mode. "
            '"incremental" (default) — queries the existing corpus, fast (~1-3 sec). '
            '"full" — wipes and rebuilds the corpus from scratch before querying, slow (~2-5 min). '
            "Use incremental for all normal queries."
        ),
    )
    n_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Number of source chunks to retrieve from the vector store. "
            "Higher values give the LLM more context but increase token usage and latency."
        ),
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity threshold [0.0-1.0]. "
            "Chunks scoring below this value are excluded from the context. "
            "0.0 returns all top-k chunks regardless of relevance score."
        ),
    )
    reldate: int | None = Field(
        default=None,
        description=(
            "Only applies when mode=incremental. "
            "If set, fetches abstracts published in the last N days and upserts them "
            "into the corpus before querying. Leave null to skip corpus update."
        ),
    )


class SourceChunkResponse(BaseModel):
    number: int = Field(
        description="Citation number matching the [N] inline references in the answer text."
    )
    pmid: str = Field(description="PubMed ID of the source article.")
    title: str = Field(description="Full title of the source article.")
    authors: list[str] = Field(
        description='Author list in "LastName Initials" format, e.g. ["Smith J", "Lee A"]. '
        "Empty list if not available."
    )
    journal: str = Field(description="Journal name. Empty string if not available.")
    year: str = Field(description="Publication year, e.g. 2024.")
    score: float = Field(
        description="Cosine similarity between this chunk and the query [0.0-1.0]. "
        "Higher means more relevant."
    )
    pubmed_url: str = Field(
        description="Direct PubMed link. Always set: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    )
    doi_url: str = Field(
        description="Clickable DOI link, e.g. https://doi.org/10.1038/... "
        "Empty string if no DOI available."
    )
    pmc_url: str = Field(
        description="PubMed Central full-text link. Empty string if no free full text is available."
    )
    chunk_index: int = Field(
        description="0-based position of this chunk within its abstract. "
        "0 means this is the first (or only) chunk from this article."
    )
    chunk_total: int = Field(
        description="Total number of chunks produced from this abstract. "
        "chunk_index + 1 == chunk_total means this is the last chunk."
    )
    text: str = Field(
        description="The abstract excerpt that was used as LLM context for this source."
    )


class AskResponse(BaseModel):
    query: str = Field(description="The original question as submitted.")
    answer: str = Field(
        description="LLM-generated answer grounded in the retrieved abstracts. "
        "Inline [N] markers reference the corresponding entry in sources."
    )
    sources: list[SourceChunkResponse] = Field(
        description="Retrieved source chunks ordered by relevance score (highest first). "
        "Each [N] in the answer maps to sources[N-1]."
    )
    llm_provider: str = Field(
        description='LLM backend used to generate the answer: "anthropic", "ollama", or "openai".'
    )
    llm_model: str = Field(
        description='Specific model name, e.g. "claude-haiku-4-5-20251001" or "llama3.1:8b".'
    )
    n_chunks_retrieved: int = Field(
        description="Actual number of chunks returned by the vector store. "
        "May be less than n_chunks_requested if the corpus is small."
    )
    n_chunks_requested: int = Field(description="The n_results value from the request.")
    n_docs_in_corpus: int = Field(
        description="Total number of chunks currently indexed in the ChromaDB collection."
    )
    corpus_updated_at: str = Field(
        description="ISO date of the last corpus update, derived from abstracts.jsonl mtime."
    )
    avg_score: float = Field(
        description="Average cosine similarity of all retrieved chunks [0.0-1.0]. "
        "Drives the confidence_tier calculation."
    )
    confidence_tier: str = Field(
        description=(
            "Retrieval confidence derived from avg_score. "
            '"High" — avg ≥ 0.70. '
            '"Medium" — avg 0.50-0.69. '
            '"Low" — avg < 0.50. '
            '"None" — no chunks were retrieved.'
        )
    )
    coverage_note: str | None = Field(
        description="Set when the LLM signals the corpus lacks sufficient context for the query "
        "(e.g. answer contains phrases like 'does not address' or 'cannot answer'). "
        "Null when the answer is fully grounded."
    )
    guardrail_flags: list[GuardrailFlagResponse] = Field(
        default_factory=list,
        description=(
            "Output guardrail warnings. Empty when all checks pass. "
            "Non-empty entries indicate citation or faithfulness issues with the generated answer. "
            "The answer is still returned — these are advisory flags, not hard blocks."
        ),
    )
