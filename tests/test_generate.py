"""
Unit tests for generate.py.

Tests cover format_context() output shape and content, generate_answer()
provider routing via match statement, and edge cases (empty chunks,
unknown provider, haiku/sonnet aliases).

All LLM provider calls are mocked — no real API calls made.
"""

from unittest.mock import patch

import pytest

from pubmed_rag.generate import format_context, generate_answer

# Fixtures


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "pmid": "11111111",
            "title": "Pembrolizumab in MSI-H CRC",
            "year": "2023",
            "text": "Patients showed improved PFS with pembrolizumab.",
            "score": 0.85,
        },
        {
            "pmid": "22222222",
            "title": "Checkpoint inhibitors in dMMR tumors",
            "year": "2022",
            "text": "Durable responses observed in mismatch repair deficient tumors.",
            "score": 0.72,
        },
    ]


# format_context()
def test_format_context_numbering(sample_chunks):
    result = format_context(sample_chunks)
    assert "[1]" in result
    assert "[2]" in result


def test_format_context_includes_pmid(sample_chunks):
    result = format_context(sample_chunks)
    assert "11111111" in result
    assert "22222222" in result


def test_format_context_includes_title(sample_chunks):
    result = format_context(sample_chunks)
    assert "Pembrolizumab in MSI-H CRC" in result
    assert "Checkpoint inhibitors in dMMR tumors" in result


def test_format_context_includes_year(sample_chunks):
    result = format_context(sample_chunks)
    assert "2023" in result
    assert "2022" in result


def test_format_context_includes_text(sample_chunks):
    result = format_context(sample_chunks)
    assert "improved PFS" in result
    assert "Durable responses" in result


def test_format_context_empty_chunks():
    result = format_context([])
    assert result == ""


def test_format_context_single_chunk(sample_chunks):
    result = format_context(sample_chunks[:1])
    assert "[1]" in result
    assert "[2]" not in result


# generate_answer() — edge casess
def test_generate_answer_empty_chunks_returns_fallback():
    result = generate_answer("any question", [])
    assert "does not address" in result.lower()


def test_generate_answer_unknown_provider_raises(monkeypatch, sample_chunks):
    monkeypatch.setenv("LLM_PROVIDER", "not_a_real_provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        generate_answer("any question", sample_chunks)


# generate_answer() — provider routing (mocked)
#
# The match statement in generate_answer calls the private _call_* functions
# directly, so we patch them by name rather than via a dispatch dict.


def test_generate_answer_routes_to_anthropic(monkeypatch, sample_chunks):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with patch(
        "pubmed_rag.generate._call_anthropic", return_value="Mocked anthropic answer [1]."
    ) as mock:
        result = generate_answer("Does pembrolizumab help?", sample_chunks)
        assert result == "Mocked anthropic answer [1]."
        prompt = mock.call_args[0][0]
        assert "Does pembrolizumab help?" in prompt


def test_generate_answer_routes_to_ollama(monkeypatch, sample_chunks):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    with patch(
        "pubmed_rag.generate._call_ollama", return_value="Mocked ollama answer [1]."
    ) as mock:
        result = generate_answer("Does pembrolizumab help?", sample_chunks)
        assert result == "Mocked ollama answer [1]."
        mock.assert_called_once()


def test_generate_answer_routes_to_openai(monkeypatch, sample_chunks):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with patch(
        "pubmed_rag.generate._call_openai", return_value="Mocked openai answer [1]."
    ) as mock:
        result = generate_answer("Does pembrolizumab help?", sample_chunks)
        assert result == "Mocked openai answer [1]."
        mock.assert_called_once()


def test_generate_answer_routes_haiku_to_anthropic(monkeypatch, sample_chunks):
    monkeypatch.setenv("LLM_PROVIDER", "haiku")
    with patch("pubmed_rag.generate._call_anthropic", return_value="haiku answer [1].") as mock:
        result = generate_answer("test", sample_chunks)
        assert result == "haiku answer [1]."
        assert mock.call_args[1]["default_model"] == "claude-haiku-4-5-20251001"


def test_generate_answer_routes_sonnet_to_anthropic(monkeypatch, sample_chunks):
    monkeypatch.setenv("LLM_PROVIDER", "sonnet")
    with patch("pubmed_rag.generate._call_anthropic", return_value="sonnet answer [1].") as mock:
        result = generate_answer("test", sample_chunks)
        assert result == "sonnet answer [1]."
        assert mock.call_args[1]["default_model"] == "claude-sonnet-4-6"


def test_generate_answer_prompt_contains_context(monkeypatch, sample_chunks):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with patch("pubmed_rag.generate._call_anthropic", return_value="answer") as mock:
        generate_answer("test query", sample_chunks)
        prompt = mock.call_args[0][0]
        assert "11111111" in prompt
        assert "Pembrolizumab in MSI-H CRC" in prompt
