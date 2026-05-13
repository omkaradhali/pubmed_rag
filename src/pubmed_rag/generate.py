"""
generate.py — LLM answer generation with citation enforcement.

Takes a user query and a list of retrieved chunks (from retrieve.py), formats
them into a citation-enforced prompt, and calls the configured LLM provider to
produce a grounded answer with inline [number] citations.

LLM provider is selected via the LLM_PROVIDER environment variable:
    ollama     — local Ollama instance (default, no API cost)
    anthropic  — Anthropic API, uses LLM_MODEL or claude-haiku-4-5-20251001
    haiku      — Anthropic claude-haiku-4-5-20251001 (ANTHROPIC_API_KEY required)
    sonnet     — Anthropic claude-sonnet-4-6 (ANTHROPIC_API_KEY required)
    openai     — OpenAI API (OPENAI_API_KEY required)

For Anthropic aliases (anthropic / haiku / sonnet), LLM_MODEL overrides the
per-alias default when set explicitly.

Public API:
    format_context(chunks)          -> str
    generate_answer(query, chunks)  -> str
"""

import logging
import os

_logger = logging.getLogger(__name__)

_DEFAULT_PROVIDER = "ollama"
_DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Per-alias default model for Anthropic-backed providers.
# LLM_MODEL env var overrides these when explicitly set.
_ANTHROPIC_PROVIDER_DEFAULTS: dict[str, str] = {
    "anthropic": _DEFAULT_ANTHROPIC_MODEL,
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
}

_SYSTEM_PROMPT = (
    "You are a clinical literature assistant. "
    "Answer questions using ONLY the PubMed abstracts provided in the context. "
    "Cite every claim with the source number in square brackets, e.g. [1] or [2]. "
    "If the context does not contain enough information to answer the question, "
    "respond with: 'The retrieved literature does not address this question directly.' "
    "Do not use any knowledge outside the provided context."
)


# ── Context formatting ─────────────────────────────────────────────────────────


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Each chunk becomes a labelled passage with PMID, title, year, and text.
    The numbers correspond to inline citations in the generated answer.

    Args:
        chunks: List of result dicts from retrieve.py. Each must have keys:
                pmid, title, year, text.

    Returns:
        Formatted string with numbered passages, e.g.:
            [1] PMID 12345678 | Some Title | 2023
                "...chunk text..."

            [2] PMID 87654321 | Another Title | 2022
                "...chunk text..."
    """
    lines = []

    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[{i}] PMID {chunk['pmid']} | {chunk['title']} | {chunk['year']}")
        lines.append(f'    "{chunk["text"]}"')
        lines.append("")

    return "\n".join(lines).strip()


# ── Prompt assembly ────────────────────────────────────────────────────────────


def build_prompt(query: str, context: str) -> str:
    """
    Assemble the full user-turn prompt from a query and formatted context block.

    Args:
        query:   The user's natural language question.
        context: Formatted context string from format_context().

    Returns:
        Complete user message string to send to the LLM.
    """
    return (
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer the question using only the above context. "
        "Cite each claim with the source number in square brackets, e.g. [1]."
    )


# ── Provider-specific callers ──────────────────────────────────────────────────


def _call_ollama(prompt: str) -> str:
    """Call a local Ollama instance via its OpenAI-compatible API."""
    from openai import OpenAI  # type: ignore[import-untyped]

    model = os.getenv("LLM_MODEL", _DEFAULT_OLLAMA_MODEL)
    base_url = os.getenv("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE_URL)

    client = OpenAI(api_key="ollama", base_url=base_url)

    _logger.info("Calling Ollama model=%s ...", model)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


def _call_anthropic(prompt: str, *, default_model: str = _DEFAULT_ANTHROPIC_MODEL) -> str:
    """
    Call the Anthropic API (requires ANTHROPIC_API_KEY).

    Args:
        prompt:        User prompt string.
        default_model: Model used when LLM_MODEL env var is not set.
                       Allows provider aliases (haiku, sonnet) to carry
                       distinct per-alias defaults.
    """
    import anthropic  # type: ignore[import-untyped]

    model = os.getenv("LLM_MODEL") or default_model
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise OSError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)

    _logger.info("Calling Anthropic model=%s ...", model)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


def _call_openai(prompt: str) -> str:
    """Call the OpenAI API (requires OPENAI_API_KEY)."""
    from openai import OpenAI  # type: ignore[import-untyped]

    model = os.getenv("LLM_MODEL", _DEFAULT_OPENAI_MODEL)
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise OSError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    _logger.info("Calling OpenAI model=%s ...", model)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


# ── Main entry point ───────────────────────────────────────────────────────────


def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Generate a citation-grounded answer for a query given retrieved chunks.

    Formats the chunks into a numbered context block, builds a citation-enforced
    prompt, and dispatches to the provider configured in LLM_PROVIDER.

    Args:
        query:  Natural language question from the user.
        chunks: Retrieved chunks from retrieve.py (list of dicts with pmid,
                title, year, text keys).

    Returns:
        LLM-generated answer string with inline [number] citations.

    Raises:
        ValueError: If LLM_PROVIDER is not one of the supported values.
        OSError:    If the required API key env var is missing.
    """
    if not chunks:
        return "The retrieved literature does not address this question directly."

    provider = os.getenv("LLM_PROVIDER", _DEFAULT_PROVIDER)

    context = format_context(chunks)
    prompt = build_prompt(query, context)

    _logger.info("Generating answer via provider=%s ...", provider)

    match provider:
        case "ollama":
            answer = _call_ollama(prompt)
        case "anthropic" | "haiku" | "sonnet":
            answer = _call_anthropic(prompt, default_model=_ANTHROPIC_PROVIDER_DEFAULTS[provider])
        case "openai":
            answer = _call_openai(prompt)
        case _:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{provider}'. "
                f"Valid options: ollama, anthropic, haiku, sonnet, openai"
            )

    _logger.info("Generation complete.")

    return answer


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from pubmed_rag.retrieve import retrieve

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Generate a citation-grounded answer from PubMed abstracts."
    )
    parser.add_argument("query", help="Natural language question.")
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.3,
        help="Minimum retrieval score to include (default: 0.3)",
    )

    args = parser.parse_args()

    chunks = retrieve(args.query, n_results=args.n, min_score=args.min_score)

    print(f"\n── Retrieved {len(chunks)} chunks ──\n")

    answer = generate_answer(args.query, chunks)

    print("── Answer ──\n")
    print(answer)
    print()
