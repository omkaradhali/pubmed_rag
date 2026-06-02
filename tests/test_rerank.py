"""
Unit tests for rerank.py — cross-encoder reranking logic.

The model (ncbi/MedCPT-Cross-Encoder) is never loaded here. score_pairs() is
patched so these tests exercise the pure reordering logic: score attachment,
descending sort, top_k truncation, text_key selection, and empty handling.
The one score_pairs test uses its empty-input early return, which needs no model.
"""

from unittest.mock import patch

from pubmed_rag.rerank import rerank, score_pairs


def _cand(child_text: str, parent_id: str) -> dict:
    """Minimal candidate dict — only the fields rerank touches."""
    return {"child_text": child_text, "parent_id": parent_id, "score": 0.5}


class TestRerank:
    def test_reorders_by_descending_score(self):
        candidates = [
            _cand("low", "p_low"),
            _cand("high", "p_high"),
            _cand("mid", "p_mid"),
        ]
        # score_pairs returns logits aligned to the input passage order.
        with patch("pubmed_rag.rerank.score_pairs", return_value=[-2.0, 9.0, 1.0]):
            ranked = rerank("q", candidates)
        assert [c["parent_id"] for c in ranked] == ["p_high", "p_mid", "p_low"]

    def test_attaches_rerank_score(self):
        candidates = [_cand("a", "p_a"), _cand("b", "p_b")]
        with patch("pubmed_rag.rerank.score_pairs", return_value=[3.0, 8.0]):
            ranked = rerank("q", candidates)
        assert ranked[0]["rerank_score"] == 8.0
        assert ranked[1]["rerank_score"] == 3.0

    def test_does_not_mutate_inputs(self):
        candidates = [_cand("a", "p_a")]
        with patch("pubmed_rag.rerank.score_pairs", return_value=[5.0]):
            rerank("q", candidates)
        assert "rerank_score" not in candidates[0]

    def test_respects_top_k(self):
        candidates = [_cand(str(i), f"p{i}") for i in range(5)]
        with patch("pubmed_rag.rerank.score_pairs", return_value=[1.0, 5.0, 2.0, 4.0, 3.0]):
            ranked = rerank("q", candidates, top_k=2)
        assert len(ranked) == 2
        assert [c["parent_id"] for c in ranked] == ["p1", "p3"]

    def test_scores_the_text_key_field(self):
        candidates = [{"body": "passage text", "parent_id": "p0"}]
        with patch("pubmed_rag.rerank.score_pairs", return_value=[1.0]) as mock_score:
            rerank("my query", candidates, text_key="body")
        # The configured field is what gets scored, paired with the query.
        mock_score.assert_called_once_with("my query", ["passage text"])

    def test_empty_candidates_returns_empty(self):
        assert rerank("q", []) == []


class TestScorePairs:
    def test_empty_passages_returns_empty_without_model(self):
        # Early return — must not touch the model/tokenizer.
        assert score_pairs("q", []) == []
