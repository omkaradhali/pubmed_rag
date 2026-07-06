"""
Session-wide test fixtures and patches.

Injects a MagicMock for sentence_transformers before any test file is imported.
embed.py instantiates SentenceTransformer at module level, which would trigger a
~90MB model download in CI. Patching here, in the conftest loaded first by pytest,
intercepts that call so no download occurs regardless of test order.
"""

import sys
from unittest.mock import MagicMock

import pytest

if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = MagicMock()


@pytest.fixture(autouse=True)
def _stub_nli_scorer(monkeypatch):
    """
    Keep the NLI faithfulness check from loading the DeBERTa model in tests.

    check_nli_faithfulness -> faithfulness_nli.find_contradictions ->
    contradiction_scores would otherwise download ~280MB of transformers weights
    on any answer with valid citations. By default the scorer reports no
    contradiction (0.0). Tests that exercise the NLI path patch
    contradiction_scores or find_contradictions explicitly to override this.
    """
    from pubmed_rag import faithfulness_nli

    monkeypatch.setattr(
        faithfulness_nli, "contradiction_scores", lambda pairs: [0.0] * len(pairs)
    )
