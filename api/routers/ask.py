import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from api.dependencies import verify_api_key
from api.limiter import limiter
from api.schemas import AskRequest, AskResponse, GuardrailFlagResponse, SourceChunkResponse
from pubmed_rag.guardrails import GuardrailError
from pubmed_rag.pipeline import run_pipeline_structured

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ask"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a clinical question",
    responses={
        200: {"description": "Answer generated successfully with cited sources."},
        401: {"description": "Invalid or missing X-API-Key header (when API_KEYS is configured)."},
        422: {"description": "Input guardrail rejected query (off-topic or injection detected)."},
        429: {"description": "Rate limit exceeded — 10 requests per hour per IP."},
        500: {"description": "Pipeline error — embedding, retrieval, or LLM call failed."},
    },
)
@limiter.limit("10/hour")
async def ask(
    request: Request,
    body: AskRequest,
    _: None = Depends(verify_api_key),
) -> AskResponse:
    """
    Submit a clinical question and receive a cited answer grounded in PubMed abstracts.

    **How it works:**
    1. The query is embedded using the same model that indexed the corpus.
    2. A cosine similarity search retrieves the top `n_results` chunks from ChromaDB.
    3. Retrieved chunks are injected into an LLM prompt with citation enforcement.
    4. The LLM generates an answer with inline [N] references — each [N] maps to `sources[N-1]`.

    **Choosing a mode:**
    - `incremental` (default) — queries the existing pre-seeded corpus. Fast (~1-3 sec,
      mostly LLM latency). Use this for all normal queries.
    - `full` — wipes the ChromaDB collection, re-ingests from PubMed, re-embeds, and
      rebuilds the index before querying. Slow (~2-5 min). Use only to refresh a stale corpus.

    **Refreshing the corpus without a full rebuild:**
    Set `reldate=30` (or any number of days) with `mode=incremental` to fetch and upsert
    only the abstracts published in the last N days — faster than a full rebuild.

    **Interpreting confidence_tier:**
    - `High` — avg cosine similarity ≥ 0.70. Strong corpus coverage for this query.
    - `Medium` — avg 0.50-0.69. Reasonable match; answer may miss some nuance.
    - `Low` — avg < 0.50. Weak match; treat the answer with caution.
    - `None` — no chunks retrieved. Answer will state the corpus has no relevant content.

    **coverage_note:**
    When set, the LLM detected it could not fully answer from the available context.
    Consider running with `mode=full` or `reldate=N` to refresh the corpus.
    """
    logger.info("received query", extra={"query": body.query, "mode": body.mode})

    try:
        result = await asyncio.to_thread(
            run_pipeline_structured,
            query=body.query,
            mode=body.mode,
            reldate=body.reldate,
            n_results=body.n_results,
            min_score=body.min_score,
        )
    except GuardrailError as exc:
        logger.warning(
            "input guardrail rejected query",
            extra={"code": exc.result.code, "reason": exc.result.reason},
        )
        raise HTTPException(
            status_code=422,
            detail={"code": str(exc.result.code), "reason": exc.result.reason},
        ) from exc
    except Exception as exc:
        logger.exception("pipeline error")
        raise HTTPException(status_code=500, detail="Pipeline error") from exc

    return AskResponse(
        query=result.query,
        answer=result.answer,
        sources=[
            SourceChunkResponse(
                number=s.number,
                pmid=s.pmid,
                title=s.title,
                authors=s.authors,
                journal=s.journal,
                year=s.year,
                score=s.score,
                pubmed_url=s.pubmed_url,
                doi_url=s.doi_url,
                pmc_url=s.pmc_url,
                chunk_index=s.chunk_index,
                chunk_total=s.chunk_total,
                text=s.text,
            )
            for s in result.sources
        ],
        llm_provider=result.llm_provider,
        llm_model=result.llm_model,
        n_chunks_retrieved=result.n_chunks_retrieved,
        n_chunks_requested=result.n_chunks_requested,
        n_docs_in_corpus=result.n_docs_in_corpus,
        corpus_updated_at=result.corpus_updated_at,
        avg_score=result.avg_score,
        confidence_tier=result.confidence_tier,
        coverage_note=result.coverage_note,
        guardrail_flags=[
            GuardrailFlagResponse(
                code=f["code"],
                reason=f["reason"],
                detail=f.get("detail", {}),
            )
            for f in result.guardrail_flags
        ],
    )
