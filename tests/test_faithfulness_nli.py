"""
Unit tests for faithfulness_nli.py.

The NLI model is never loaded here: contradiction_scores is patched (or the
autouse stub in conftest keeps it at 0.0), so no transformers weights download.
Tests cover find_contradictions logic and contradiction-label resolution.
"""

from unittest.mock import patch

import pytest

from pubmed_rag import faithfulness_nli as fn

# Captured at import, before conftest's autouse stub patches the attribute — lets
# us exercise the real function's empty-input guard without loading the model.
_REAL_CONTRADICTION_SCORES = fn.contradiction_scores

_CHUNKS = [
    {"text": "Pembrolizumab is effective in MSI-H colorectal cancer."},
    {"text": "The patient presented with tachycardia on admission."},
]


class _Config:
    """Minimal stand-in for a transformers model config."""

    def __init__(self, id2label):
        self.id2label = id2label


class TestResolveContradictionIndex:
    def test_finds_contradiction_first(self):
        cfg = _Config({0: "contradiction", 1: "entailment", 2: "neutral"})
        assert fn._resolve_contradiction_index(cfg) == 0

    def test_finds_contradiction_last(self):
        cfg = _Config({0: "entailment", 1: "neutral", 2: "contradiction"})
        assert fn._resolve_contradiction_index(cfg) == 2

    def test_matches_case_insensitively_and_prefix(self):
        # Some checkpoints label it "CONTRADICTION" or "contradict".
        cfg = _Config({0: "ENTAILMENT", 1: "CONTRADICTION"})
        assert fn._resolve_contradiction_index(cfg) == 1

    def test_raises_when_no_contradiction_label(self):
        cfg = _Config({0: "entailment", 1: "neutral"})
        with pytest.raises(ValueError, match="no 'contradiction' label"):
            fn._resolve_contradiction_index(cfg)


class TestFindContradictions:
    def test_flags_scores_at_or_above_threshold(self):
        answer = "Pembrolizumab is not effective [1]. The patient had bradycardia [2]."
        with patch.object(fn, "contradiction_scores", return_value=[0.95, 0.91]) as scorer:
            flagged = fn.find_contradictions(answer, _CHUNKS, threshold=0.8)

        assert {f["source_n"] for f in flagged} == {1, 2}
        assert all(f["contradiction"] >= 0.8 for f in flagged)
        # premise = source chunk, hypothesis = cited sentence (NLI direction).
        pairs = scorer.call_args[0][0]
        assert pairs[0][0] == _CHUNKS[0]["text"]
        assert "[1]" in pairs[0][1]

    def test_does_not_flag_below_threshold(self):
        answer = "Pembrolizumab is effective [1]."
        with patch.object(fn, "contradiction_scores", return_value=[0.4]):
            assert fn.find_contradictions(answer, _CHUNKS, threshold=0.8) == []

    def test_threshold_boundary_is_inclusive(self):
        answer = "Claim [1]."
        with patch.object(fn, "contradiction_scores", return_value=[0.8]):
            flagged = fn.find_contradictions(answer, _CHUNKS, threshold=0.8)
        assert len(flagged) == 1

    def test_out_of_range_citation_skipped_without_scoring(self):
        answer = "Some claim [9]."
        with patch.object(fn, "contradiction_scores", return_value=[0.99]) as scorer:
            assert fn.find_contradictions(answer, _CHUNKS, threshold=0.8) == []
        scorer.assert_not_called()  # no valid pairs -> scorer never invoked

    def test_no_citation_returns_empty(self):
        with patch.object(fn, "contradiction_scores", return_value=[0.99]) as scorer:
            assert fn.find_contradictions("A sentence with no citation.", _CHUNKS) == []
        scorer.assert_not_called()

    def test_empty_chunks_returns_empty(self):
        assert fn.find_contradictions("Claim [1].", [], threshold=0.8) == []

    def test_duplicate_citation_in_sentence_scored_once(self):
        answer = "Claim referencing [1] and again [1]."
        with patch.object(fn, "contradiction_scores", return_value=[0.95]) as scorer:
            flagged = fn.find_contradictions(answer, _CHUNKS, threshold=0.8)
        assert len(scorer.call_args[0][0]) == 1  # deduped to a single pair
        assert len(flagged) == 1

    def test_empty_source_text_skipped(self):
        chunks = [{"text": ""}]
        with patch.object(fn, "contradiction_scores", return_value=[0.99]) as scorer:
            assert fn.find_contradictions("Claim [1].", chunks, threshold=0.8) == []
        scorer.assert_not_called()


def test_contradiction_scores_empty_input_short_circuits():
    # The real function must return [] for empty input without loading the model.
    assert _REAL_CONTRADICTION_SCORES([]) == []
