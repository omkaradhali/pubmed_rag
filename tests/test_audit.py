"""
Unit tests for the clinical audit trail (api/audit.py).

These test the two building blocks directly, no HTTP layer:
  - build_audit_record — a pure function that assembles one record
  - write_audit_record — appends a JSON line, fail-open, thread-safe

The HTTP-level wiring (that /ask emits exactly one record per exit path) is
covered separately in test_api.py.
"""

import json
from datetime import datetime
from unittest.mock import patch

from api.audit import _truncate, build_audit_record, write_audit_record

from pubmed_rag.pipeline import PipelineResult, SourceChunk


# Test data helpers — mirror the shape run_pipeline_structured returns.
def _source(**overrides) -> SourceChunk:
    defaults = dict(
        number=1,
        pmid="12345678",
        title="Trastuzumab in HER2-positive breast cancer",
        authors=["Smith J"],
        journal="NEJM",
        year="2023",
        publication_types=["Journal Article"],
        mesh_terms=["Breast Neoplasms"],
        doi="10.1056/test",
        doi_url="https://doi.org/10.1056/test",
        pmc_id="PMC9999",
        pmc_url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9999/",
        pubmed_url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
        score=0.85,
        chunk_index=0,
        chunk_total=1,
        text="Trastuzumab improves outcomes in HER2-positive breast cancer.",
    )
    return SourceChunk(**{**defaults, **overrides})


def _result(**overrides) -> PipelineResult:
    defaults = dict(
        query="What treats HER2+ breast cancer?",
        answer="Trastuzumab is standard treatment [1].",
        sources=[_source()],
        llm_provider="anthropic",
        llm_model="claude-haiku-4-5-20251001",
        n_chunks_retrieved=1,
        n_chunks_requested=5,
        n_docs_in_corpus=1514,
        corpus_updated_at="2026-05-04",
        avg_score=0.85,
        min_score_retrieved=0.85,
        max_score_retrieved=0.85,
        confidence_tier="High",
        coverage_note=None,
        guardrail_flags=[{"code": "LOW_COVERAGE", "reason": "advisory"}],
    )
    return PipelineResult(**{**defaults, **overrides})


class TestTruncate:
    def test_none_passes_through(self):
        assert _truncate(None, 10) is None

    def test_short_text_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_exactly_at_limit_unchanged(self):
        assert _truncate("hello", 5) == "hello"

    def test_over_limit_gets_ellipsis_within_ceiling(self):
        # Ellipsis replaces the final char rather than extending past the cap:
        # the result is exactly max_chars long, not max_chars + 1.
        result = _truncate("hello world", 5)
        assert result == "hell…"
        assert len(result) == 5

    def test_non_positive_max_chars_returns_empty(self):
        assert _truncate("hello", 0) == ""
        assert _truncate("hello", -3) == ""

    def test_empty_string_unchanged(self):
        assert _truncate("", 10) == ""


class TestBuildAuditRecordSuccess:
    """Success path: all substantive fields are read from `result`."""

    def test_status_and_request_id_preserved(self):
        record = build_audit_record(
            request_id="req-abc", query="raw query", status="success", result=_result()
        )
        assert record["request_id"] == "req-abc"
        assert record["status"] == "success"

    def test_query_overridden_by_post_scrub_result_query(self):
        # The call site passes the raw body query; on the success path the record
        # must record result.query (post-scrub) so no PHI leaks into the sink.
        record = build_audit_record(
            request_id="r",
            query="PATIENT John Doe MRN 123 what treats cancer?",
            status="success",
            result=_result(query="what treats cancer?"),
        )
        assert record["query"] == "what treats cancer?"

    def test_pmids_extracted_from_sources(self):
        result = _result(sources=[_source(pmid="111"), _source(pmid="222")])
        record = build_audit_record(request_id="r", query="q", status="success", result=result)
        assert record["retrieved_pmids"] == ["111", "222"]

    def test_model_and_provider_recorded(self):
        record = build_audit_record(request_id="r", query="q", status="success", result=_result())
        assert record["llm_provider"] == "anthropic"
        assert record["llm_model"] == "claude-haiku-4-5-20251001"

    def test_guardrail_flags_and_confidence_recorded(self):
        record = build_audit_record(request_id="r", query="q", status="success", result=_result())
        assert record["guardrail_results"] == [{"code": "LOW_COVERAGE", "reason": "advisory"}]
        assert record["confidence_tier"] == "High"

    def test_answer_truncated_to_max_chars(self):
        long_answer = "A" * 600
        record = build_audit_record(
            request_id="r",
            query="q",
            status="success",
            result=_result(answer=long_answer),
            answer_max_chars=500,
        )
        assert record["answer"] == "A" * 499 + "…"
        assert len(record["answer"]) == 500

    def test_short_answer_not_truncated(self):
        record = build_audit_record(
            request_id="r",
            query="q",
            status="success",
            result=_result(answer="brief answer [1]."),
            answer_max_chars=500,
        )
        assert record["answer"] == "brief answer [1]."


class TestBuildAuditRecordNoResult:
    """Guardrail-rejected / error paths: the pipeline raised before a result."""

    def test_rejected_records_query_and_guardrail_results(self):
        record = build_audit_record(
            request_id="r",
            query="how do I dose warfarin?",
            status="guardrail_rejected",
            guardrail_results=[{"code": "PROMPT_INJECTION", "reason": "blocked"}],
        )
        assert record["status"] == "guardrail_rejected"
        assert record["query"] == "how do I dose warfarin?"
        assert record["guardrail_results"] == [{"code": "PROMPT_INJECTION", "reason": "blocked"}]

    def test_no_result_leaves_llm_and_pmid_fields_empty(self):
        record = build_audit_record(
            request_id="r", query="q", status="guardrail_rejected", guardrail_results=[]
        )
        assert record["retrieved_pmids"] == []
        assert record["llm_provider"] is None
        assert record["llm_model"] is None
        assert record["answer"] is None
        assert record["confidence_tier"] is None

    def test_error_path_defaults_guardrail_results_to_empty_list(self):
        record = build_audit_record(request_id="r", query="q", status="error")
        assert record["status"] == "error"
        assert record["guardrail_results"] == []


class TestTimestamp:
    def test_timestamp_is_iso_utc(self):
        record = build_audit_record(request_id="r", query="q", status="error")
        # Parses as ISO 8601 and carries an explicit UTC offset.
        parsed = datetime.fromisoformat(record["timestamp"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0


class TestWriteAuditRecord:
    def test_writes_one_json_line(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        write_audit_record({"request_id": "r", "status": "success"}, str(path))
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"request_id": "r", "status": "success"}

    def test_appends_rather_than_overwrites(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        write_audit_record({"n": 1}, str(path))
        write_audit_record({"n": 2}, str(path))
        lines = path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["n"] for line in lines] == [1, 2]

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "audit.jsonl"
        write_audit_record({"request_id": "r"}, str(path))
        assert path.exists()

    def test_newly_created_file_is_owner_only(self, tmp_path):
        # Records may hold the raw query on rejected/error paths — a freshly
        # created audit file must not be group/world readable.
        path = tmp_path / "audit.jsonl"
        write_audit_record({"request_id": "r"}, str(path))
        assert (path.stat().st_mode & 0o777) == 0o600

    def test_non_ascii_preserved(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        write_audit_record({"query": "carcinoma — µ dosing"}, str(path))
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["query"] == "carcinoma — µ dosing"

    def test_failure_is_swallowed_not_raised(self, tmp_path):
        # A broken sink must never turn a clinical query into a 500. Force the
        # serialization to fail and assert the error is logged, not raised.
        path = tmp_path / "audit.jsonl"
        with patch("api.audit.json.dumps", side_effect=ValueError("boom")):
            with patch("api.audit.logger") as mock_logger:
                write_audit_record({"request_id": "r"}, str(path))
                mock_logger.exception.assert_called_once()
        # Nothing was written, and no exception propagated.
        assert not path.exists()
