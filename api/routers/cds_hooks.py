"""
cds_hooks.py — HL7 CDS Hooks integration for pubmed_rag.

Implements the CDS Hooks 1.0 specification:
  https://cds-hooks.hl7.org/1.0/

Endpoints:
  GET  /cds-services              — service discovery
  POST /cds-services/pubmed-rag   — oncology evidence service

The pubmed-rag service accepts a clinical question via the context.query
extension field (present when an EHR or clinician populates it) and returns
CDS cards containing the RAG-generated answer plus one link card per cited
PubMed source.

In a full EHR integration the query would be synthesised from the patient's
active problem list, encounter diagnosis, or medication list. The context.query
extension makes the integration point explicit without requiring FHIR read
access for demonstration purposes.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter
from pydantic import BaseModel, Field

from pubmed_rag.pipeline import run_pipeline_structured

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cds-services", tags=["CDS Hooks"])

_SERVICE_ID = "pubmed-rag"
_SOURCE_LABEL = "pubmed_rag — Oncology Evidence Service"
_SOURCE_URL = "https://pubmed.ncbi.nlm.nih.gov"
_SUMMARY_LIMIT = 140  # CDS Hooks spec recommends ≤140 chars for summary


# ── CDS Hooks request schema ──────────────────────────────────────────────────


class CDSContext(BaseModel):
    """
    CDS Hooks hook context.

    Standard fields (patient-view hook) plus the pubmed-rag query extension.
    All fields are optional because different hooks populate different fields.
    """

    userId: str | None = Field(default=None, description="FHIR Practitioner resource ID.")
    patientId: str | None = Field(default=None, description="FHIR Patient resource ID.")
    encounterId: str | None = Field(default=None, description="FHIR Encounter resource ID.")
    query: str | None = Field(
        default=None,
        description=(
            "pubmed-rag extension: the clinical question to answer. "
            "In a full EHR integration this is synthesised from the patient context; "
            "for direct API use it is provided by the clinician explicitly."
        ),
    )

    model_config = {"extra": "allow"}  # accept any additional EHR context fields


class CDSHookRequest(BaseModel):
    """Incoming CDS Hooks 1.0 request."""

    hookInstance: str = Field(description="UUID identifying this specific invocation.")
    hook: str = Field(description='Hook identifier, e.g. "patient-view".')
    fhirServer: str | None = Field(
        default=None, description="Base URL of the calling EHR's FHIR server."
    )
    context: CDSContext = Field(description="Hook-specific context data.")
    prefetch: dict | None = Field(default=None, description="Pre-fetched FHIR resources.")

    model_config = {"extra": "allow"}


# ── CDS Hooks response schema ─────────────────────────────────────────────────


class CDSLink(BaseModel):
    """A URL link attached to a CDS card."""

    label: str = Field(description="Human-readable link label.")
    url: str = Field(description="Absolute URL.")
    type: str = Field(default="absolute", description='"absolute" or "smart".')


class CDSSource(BaseModel):
    """Origin of a CDS card."""

    label: str = Field(description="Short display label for the source.")
    url: str | None = Field(default=None, description="Optional link to source documentation.")


class CDSCard(BaseModel):
    """
    A single CDS Hooks card.

    Cards appear in the EHR UI as actionable, dismissable panels. Each card
    has a short summary (≤140 chars), optional markdown detail, an indicator
    (info / warning / critical), a source attribution, and optional links.
    """

    uuid: str = Field(description="Unique identifier for this card instance.")
    summary: str = Field(description="One-line card title (≤140 chars recommended).")
    detail: str | None = Field(
        default=None,
        description="Markdown-formatted extended content (full answer + attribution).",
    )
    indicator: str = Field(
        default="info",
        description='"info" | "warning" | "critical". Drives EHR visual treatment.',
    )
    source: CDSSource
    links: list[CDSLink] = Field(default_factory=list)


class CDSHookResponse(BaseModel):
    """CDS Hooks 1.0 service response."""

    cards: list[CDSCard] = Field(description="Cards to display in the EHR.")


# ── Discovery endpoint ────────────────────────────────────────────────────────


class CDSService(BaseModel):
    """One entry in the service discovery manifest."""

    hook: str
    title: str
    description: str
    id: str
    prefetch: dict = Field(default_factory=dict)


class CDSServicesResponse(BaseModel):
    services: list[CDSService]


@router.get(
    "",
    response_model=CDSServicesResponse,
    summary="CDS service discovery",
    description=(
        "Returns the list of CDS Hooks services provided by this API. "
        "EHR systems call this endpoint on startup to discover available services."
    ),
)
def get_cds_services() -> CDSServicesResponse:
    return CDSServicesResponse(
        services=[
            CDSService(
                hook="patient-view",
                title="Oncology Evidence Search (pubmed_rag)",
                description=(
                    "Search 35M+ PubMed oncology abstracts and receive a cited, "
                    "LLM-synthesised evidence summary for the current clinical question. "
                    "Provide the clinical question in context.query."
                ),
                id=_SERVICE_ID,
            )
        ]
    )


# ── Service endpoint ──────────────────────────────────────────────────────────


@router.post(
    f"/{_SERVICE_ID}",
    response_model=CDSHookResponse,
    summary="Oncology evidence search",
    description=(
        "Accepts a CDS Hooks patient-view request and returns evidence cards. "
        "Set context.query to the clinical question; the service retrieves relevant "
        "PubMed abstracts via RAG and returns an answer with per-source citation links."
    ),
)
async def pubmed_rag_service(request: CDSHookRequest) -> CDSHookResponse:
    query = (request.context.query or "").strip()

    if not query:
        _logger.info(
            "CDS Hook invoked without query (hookInstance=%s, patientId=%s)",
            request.hookInstance,
            request.context.patientId,
        )
        return CDSHookResponse(
            cards=[
                CDSCard(
                    uuid=str(uuid.uuid4()),
                    summary="Specify a clinical question to search PubMed evidence",
                    detail=(
                        "Provide a clinical question in `context.query` to receive an "
                        "evidence summary from 35M+ PubMed oncology abstracts.\n\n"
                        "**Example:** "
                        "*'What is the first-line treatment for HER2-positive "
                        "metastatic breast cancer?'*"
                    ),
                    indicator="info",
                    source=CDSSource(label=_SOURCE_LABEL, url=_SOURCE_URL),
                )
            ]
        )

    _logger.info("CDS Hook query (hookInstance=%s): %s", request.hookInstance, query[:80])

    try:
        result = await asyncio.to_thread(
            run_pipeline_structured, query=query, mode="incremental", n_results=5
        )
    except Exception as exc:
        _logger.error("Pipeline error for CDS query %r: %s", query[:60], exc)
        return CDSHookResponse(
            cards=[
                CDSCard(
                    uuid=str(uuid.uuid4()),
                    summary="Evidence search temporarily unavailable",
                    detail=f"The pubmed_rag service encountered an error: {exc}",
                    indicator="warning",
                    source=CDSSource(label=_SOURCE_LABEL, url=_SOURCE_URL),
                )
            ]
        )

    # Build the summary from the first sentence of the answer (≤140 chars)
    first_sentence = result.answer.split(".")[0].strip()
    summary = (
        first_sentence[:_SUMMARY_LIMIT] + "…"
        if len(first_sentence) > _SUMMARY_LIMIT
        else first_sentence
    )

    # Build markdown detail: full answer + source attribution list
    source_list = "\n".join(
        f"{i}. **{src.title}** — {src.journal} ({src.year}) · [PubMed {src.pmid}]({src.pubmed_url})"
        for i, src in enumerate(result.sources, 1)
    )
    detail = f"{result.answer}\n\n---\n\n**Sources**\n\n{source_list}"

    # One link per cited source so the clinician can open the abstract directly
    links = [
        CDSLink(
            label=f"[{i}] PMID {src.pmid} — {src.title[:60]}{'…' if len(src.title) > 60 else ''}",
            url=src.pubmed_url,
        )
        for i, src in enumerate(result.sources, 1)
    ]

    card = CDSCard(
        uuid=str(uuid.uuid4()),
        summary=f"Evidence: {summary}",
        detail=detail,
        indicator="info",
        source=CDSSource(label=_SOURCE_LABEL, url=_SOURCE_URL),
        links=links,
    )

    _logger.info(
        "CDS Hook response: %d sources, confidence=%s",
        len(result.sources),
        result.confidence_tier,
    )
    return CDSHookResponse(cards=[card])
