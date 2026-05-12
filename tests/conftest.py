"""
Session-wide test fixtures and patches.

Injects a MagicMock for sentence_transformers before any test file is imported.
embed.py instantiates SentenceTransformer at module level, which would trigger a
~90MB model download in CI. Patching here, in the conftest loaded first by pytest,
intercepts that call so no download occurs regardless of test order.
"""

import sys
from unittest.mock import MagicMock

if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = MagicMock()
