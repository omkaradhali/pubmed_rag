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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_phi: exercise the real phi.scrub_phi (opt out of the identity stub)",
    )


@pytest.fixture(autouse=True)
def _stub_phi_scrubber(request, monkeypatch):
    """
    Keep PHI scrubbing from building the Presidio/spaCy engine in tests.

    pipeline runs scrub_phi() on every query; building the analyzer would
    download a ~500MB spaCy model. By default we swap scrub_phi for the identity
    so unrelated tests never touch Presidio. Tests marked `real_phi` opt out and
    call the real function (they stub get_analyzer with a fake instead).
    """
    if "real_phi" in request.keywords:
        return
    from pubmed_rag import phi

    monkeypatch.setattr(phi, "scrub_phi", lambda text: text)


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

    monkeypatch.setattr(faithfulness_nli, "contradiction_scores", lambda pairs: [0.0] * len(pairs))
