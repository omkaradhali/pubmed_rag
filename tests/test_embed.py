"""
Unit tests for embed.py — provider dispatch, query prefix, normalization.

sentence_transformers is mocked session-wide in conftest.py, so no model loads.
These tests patch get_model and _QUERY_PREFIX to exercise the pure logic:
the asymmetric query-prefix application and the children-only + normalized
passage embedding.
"""

from unittest.mock import MagicMock, patch

from pubmed_rag.embed import embed_chunks, embed_query


def test_embed_query_applies_provider_prefix():
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2]]
    with (
        patch("pubmed_rag.embed.get_model", return_value=mock_model),
        patch("pubmed_rag.embed._QUERY_PREFIX", "PREFIX: "),
    ):
        vec = embed_query("my query")
    args, kwargs = mock_model.encode.call_args
    assert args[0] == ["PREFIX: my query"]  # prefix prepended to the query
    assert kwargs.get("normalize_embeddings") is True
    assert vec == [0.1, 0.2]


def test_embed_query_no_prefix_for_symmetric_provider():
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [[1.0]]
    with (
        patch("pubmed_rag.embed.get_model", return_value=mock_model),
        patch("pubmed_rag.embed._QUERY_PREFIX", ""),
    ):
        embed_query("q")
    args, _ = mock_model.encode.call_args
    assert args[0] == ["q"]  # empty prefix → query unchanged


def test_embed_chunks_embeds_children_only_and_normalizes():
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]
    chunks = [
        {"text": "parent text", "chunk_role": "parent"},
        {"text": "child text", "chunk_role": "child"},
    ]
    with patch("pubmed_rag.embed.get_model", return_value=mock_model):
        out = embed_chunks(chunks)
    assert len(out) == 1  # parent dropped
    assert out[0]["chunk_role"] == "child"
    assert out[0]["embedding"] == [0.1, 0.2, 0.3]
    # passages embedded without a prefix and normalized
    args, kwargs = mock_model.encode.call_args
    assert args[0] == ["child text"]
    assert kwargs.get("normalize_embeddings") is True
