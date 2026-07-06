"""
Unit tests for pipeline.py orchestration logic.

Focuses on _generate_grounded_answer — the generate -> verify -> retry -> hard-
block control flow. generate_answer and run_output_guardrails are mocked, so no
LLM calls or vector store access occur (the sentence-transformers import is
stubbed in conftest.py).
"""

from unittest.mock import patch

from pubmed_rag.guardrails import GuardrailCode, GuardrailResult
from pubmed_rag.pipeline import SAFE_FALLBACK_ANSWER, _generate_grounded_answer

_CHUNKS = [{"text": "EGFR mutations are common in NSCLC.", "score": 0.8}]


def _missing_citations() -> GuardrailResult:
    return GuardrailResult(passed=False, code=GuardrailCode.MISSING_CITATIONS, reason="no [N]")


def _passing() -> GuardrailResult:
    return GuardrailResult(passed=True)


def _low_overlap(n_pairs: int) -> GuardrailResult:
    return GuardrailResult(
        passed=False,
        code=GuardrailCode.LOW_CITATION_OVERLAP,
        detail={"low_overlap_pairs": [{"source_n": i} for i in range(n_pairs)]},
    )


class TestGenerateGroundedAnswer:
    def test_clean_answer_returned_without_retry(self):
        with (
            patch("pubmed_rag.pipeline.generate_answer", return_value="Grounded [1].") as gen,
            patch("pubmed_rag.pipeline.run_output_guardrails", return_value=[_passing()]),
        ):
            answer, flags, blocked = _generate_grounded_answer("q", _CHUNKS)

        assert answer == "Grounded [1]."
        assert blocked is False
        assert flags == []
        gen.assert_called_once()  # no retry

    def test_retry_recovers_on_second_attempt(self):
        # First attempt hard-blocks, retry produces a clean answer.
        with (
            patch(
                "pubmed_rag.pipeline.generate_answer",
                side_effect=["Uncited answer.", "Fixed answer [1]."],
            ) as gen,
            patch(
                "pubmed_rag.pipeline.run_output_guardrails",
                side_effect=[[_missing_citations()], [_passing()]],
            ),
        ):
            answer, flags, blocked = _generate_grounded_answer("q", _CHUNKS)

        assert answer == "Fixed answer [1]."
        assert blocked is False
        assert flags == []
        assert gen.call_count == 2
        # The retry carries the correction instruction.
        assert gen.call_args_list[1].kwargs.get("correction")

    def test_persistent_failure_returns_safe_fallback(self):
        with (
            patch(
                "pubmed_rag.pipeline.generate_answer",
                side_effect=["Uncited one.", "Still uncited."],
            ) as gen,
            patch(
                "pubmed_rag.pipeline.run_output_guardrails",
                side_effect=[[_missing_citations()], [_missing_citations()]],
            ),
        ):
            answer, flags, blocked = _generate_grounded_answer("q", _CHUNKS)

        assert answer == SAFE_FALLBACK_ANSWER
        assert blocked is True
        assert gen.call_count == 2  # exactly one retry, no infinite loop
        # The blocking failure is still reported in the flags.
        assert any(f["code"] == GuardrailCode.MISSING_CITATIONS for f in flags)

    def test_advisory_only_failure_does_not_trigger_retry(self):
        # A single low-overlap pair is advisory, not a hard block — the answer
        # is returned as-is with the flag attached, no retry.
        with (
            patch("pubmed_rag.pipeline.generate_answer", return_value="Answer [1].") as gen,
            patch("pubmed_rag.pipeline.run_output_guardrails", return_value=[_low_overlap(1)]),
        ):
            answer, flags, blocked = _generate_grounded_answer("q", _CHUNKS)

        assert answer == "Answer [1]."
        assert blocked is False
        gen.assert_called_once()  # no retry for advisory failures
        assert any(f["code"] == GuardrailCode.LOW_CITATION_OVERLAP for f in flags)
