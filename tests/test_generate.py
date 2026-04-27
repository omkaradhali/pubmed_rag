"""
Unit tests for generate.py.

Tests cover format_context() output shape and content, generate_answer()
provider routing, and edge cases (empty chunks, unknown provider).
All LLM provider calls are mocked — no real API calls made.
"""

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


# generate_answer() — edge cases


def test_generate_answer_empty_chunks_returns_fallback():
    result = generate_answer("any question", [])
    assert "does not address" in result.lower()


def test_generate_answer_unknown_provider_raises(monkeypatch, sample_chunks):
    monkeypatch.setenv("LLM_PROVIDER", "not_a_real_provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        generate_answer("any question", sample_chunks)


# generate_answer() — provider routing (mocked)
#
# _PROVIDERS is a module-level dict built at import time, so monkeypatch.setattr
# on the function names won't update it. Patch the dict entries directly via
# monkeypatch.setitem instead.


def test_generate_answer_routes_to_anthropic(monkeypatch, sample_chunks):
    import pubmed_rag.generate as gen_module

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    called_with = {}

    def mock_call_anthropic(prompt: str) -> str:
        called_with["prompt"] = prompt
        return "Mocked anthropic answer [1]."

    monkeypatch.setitem(gen_module._PROVIDERS, "anthropic", mock_call_anthropic)

    result = generate_answer("Does pembrolizumab help?", sample_chunks)

    assert result == "Mocked anthropic answer [1]."
    assert "Does pembrolizumab help?" in called_with["prompt"]


def test_generate_answer_routes_to_ollama(monkeypatch, sample_chunks):
    import pubmed_rag.generate as gen_module

    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    called_with = {}

    def mock_call_ollama(prompt: str) -> str:
        called_with["prompt"] = prompt
        return "Mocked ollama answer [1]."

    monkeypatch.setitem(gen_module._PROVIDERS, "ollama", mock_call_ollama)

    result = generate_answer("Does pembrolizumab help?", sample_chunks)

    assert result == "Mocked ollama answer [1]."


def test_generate_answer_routes_to_openai(monkeypatch, sample_chunks):
    import pubmed_rag.generate as gen_module

    monkeypatch.setenv("LLM_PROVIDER", "openai")

    def mock_call_openai(prompt: str) -> str:
        return "Mocked openai answer [1]."

    monkeypatch.setitem(gen_module._PROVIDERS, "openai", mock_call_openai)

    result = generate_answer("Does pembrolizumab help?", sample_chunks)

    assert result == "Mocked openai answer [1]."


def test_generate_answer_prompt_contains_context(monkeypatch, sample_chunks):
    import pubmed_rag.generate as gen_module

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    captured = {}

    def mock_call_anthropic(prompt: str) -> str:
        captured["prompt"] = prompt
        return "answer"

    monkeypatch.setitem(gen_module._PROVIDERS, "anthropic", mock_call_anthropic)

    generate_answer("test query", sample_chunks)

    assert "11111111" in captured["prompt"]
    assert "Pembrolizumab in MSI-H CRC" in captured["prompt"]
