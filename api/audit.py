"""Immutable audit trail for clinical queries (Session D, Tier 1).

Each audited exit path of ``POST /ask`` produces one append-only JSONL record —
whether the query succeeded, was rejected by an input guardrail, or errored inside
the pipeline. (Requests rejected *before* the handler runs — auth 401, rate-limit
429, request-body 422 — are not audited here; those are covered by the app logs.)
This is distinct from the application logs in ``logging_config.py``: those exist for
debugging and may be sampled, rotated, or dropped. An audit record is an
accountability primitive. If a clinician ever acts on an answer, this file lets you
reconstruct exactly what the system surfaced on that date: the query, the PMIDs it
retrieved, the model that answered, and which guardrails fired.

Design constraints:
- **Append-only.** One JSON object per line, never mutated in place.
- **Never breaks the request.** A failed write logs an error and returns; it must
  not turn a clinician's query into a 500. (A stricter fail-closed policy —
  refuse the query if it cannot be audited — is deferred to v2.0.)
- **Thread-safe within one process.** The caller offloads the blocking write via
  ``asyncio.to_thread``, so two requests can write concurrently; the module lock
  serializes them. This assumes a single-process deployment (the Dockerfile runs
  one uvicorn worker). Multi-worker/multi-process safety would need an OS file
  lock (``fcntl.flock``) — deferred to v2.0 alongside log rotation.
"""

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Serializes concurrent appends from writer threads (asyncio.to_thread workers).
# Single-process only — see the module docstring for the multi-worker caveat.
_write_lock = threading.Lock()


def _restricted_opener(path: str, flags: int) -> int:
    """Open (creating with mode 0o600) so a freshly created audit file is not
    world-readable. Audit records may contain the raw query on rejected/error
    paths, so keep the on-disk PHI surface owner-only."""
    return os.open(path, flags, 0o600)


def _truncate(text: str | None, max_chars: int) -> str | None:
    """Cap a long field to at most ``max_chars`` characters (ellipsis included).

    Returns None unchanged. The ellipsis replaces the final character rather than
    extending past the ceiling, so ``len(result) <= max_chars`` always holds.
    """
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    return text[: max_chars - 1] + "…"


def build_audit_record(
    *,
    request_id: str,
    query: str,
    status: str,
    result: Any | None = None,
    guardrail_results: list[dict] | None = None,
    answer_max_chars: int = 500,
) -> dict:
    """Assemble one audit record. Pure function — no I/O.

    Args:
        request_id: Per-request trace ID (from the request_id ContextVar).
        query: The query as known at the call site. Overridden by the pipeline's
            post-scrub query when ``result`` is provided.
        status: One of ``"success"``, ``"guardrail_rejected"``, ``"error"``.
        result: The structured pipeline result on the success path. When present,
            the PMIDs, model, answer, guardrail flags, confidence tier, and the
            post-scrub query are read from it.
        guardrail_results: Guardrail flags for a rejected/errored path where no
            full ``result`` exists.
        answer_max_chars: Truncation ceiling for the stored answer.

    Note:
        On the success path ``query`` is the post-scrub query (PHI removed before
        any cloud egress). On the guardrail/error paths the pipeline raised before
        scrubbing, so the raw query is recorded — acceptable because the audit
        file is a local sink, never sent to a cloud provider.
    """
    record: dict[str, Any] = {
        "request_id": request_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "status": status,
        "query": query,
        "retrieved_pmids": [],
        "llm_provider": None,
        "llm_model": None,
        "answer": None,
        "guardrail_results": guardrail_results or [],
        "confidence_tier": None,
    }

    if result is not None:
        record["query"] = result.query
        record["retrieved_pmids"] = [s.pmid for s in result.sources]
        record["llm_provider"] = result.llm_provider
        record["llm_model"] = result.llm_model
        record["answer"] = _truncate(result.answer, answer_max_chars)
        record["guardrail_results"] = result.guardrail_flags
        record["confidence_tier"] = result.confidence_tier

    return record


def write_audit_record(record: dict, path: str) -> None:
    """Append one record as a JSON line to the audit log.

    Creates the parent directory on first write. Never raises: any failure is
    logged and swallowed so auditing cannot break the request path.
    """
    try:
        audit_path = Path(path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with (
            _write_lock,
            open(audit_path, "a", encoding="utf-8", opener=_restricted_opener) as f,
        ):
            f.write(line + "\n")
    except Exception:
        # Auditing is best-effort at the HTTP layer; a broken sink must not 500 a
        # clinical query. Surface it loudly in the app logs instead.
        logger.exception(
            "failed to write audit record",
            extra={"request_id": record.get("request_id")},
        )
