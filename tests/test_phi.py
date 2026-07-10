"""
Tests for phi.py — PHI/PII scrubbing gate and anonymization wiring.

Two groups:
  * Gate logic (cloud_egress_configured / should_scrub) — pure env-var reads, no
    Presidio needed, always run.
  * Scrub behavior (scrub_phi) — marked `real_phi` to opt out of the conftest
    identity stub. These fake the analyzer (so no spaCy model download) but use
    the REAL Presidio anonymizer, which needs no model. They importorskip
    presidio so the suite still runs before deps are synced.

The fake analyzer validates our analyze->anonymize->tag wiring, NOT Presidio's
own NER/context logic. True MRN context-anchoring and eponym false-positive rates
are validated by a real-model spot check outside CI (see the design brief).
"""

import logging

import pytest

from pubmed_rag import phi

# Gate logic


@pytest.mark.parametrize(
    ("llm", "embedding", "expected"),
    [
        ("ollama", "miniml", False),  # fully local — no egress
        ("ollama", "medcpt", False),  # local embeddings too
        ("anthropic", "miniml", True),  # cloud LLM
        ("haiku", "miniml", True),
        ("sonnet", "miniml", True),
        ("openai", "miniml", True),
        ("ollama", "openai", True),  # THE LEAK: local LLM, remote embedding
        ("ANTHROPIC", "MINIML", True),  # case-insensitive
        # Fail-safe allowlist: any provider not known-local counts as cloud.
        ("gemini", "miniml", True),
        ("bedrock", "miniml", True),
        ("ollama", "cohere", True),
        ("", "", True),  # blanked out -> not provably local -> scrub
    ],
)
def test_cloud_egress_configured(monkeypatch, llm, embedding, expected):
    monkeypatch.setenv("LLM_PROVIDER", llm)
    monkeypatch.setenv("EMBEDDING_PROVIDER", embedding)
    assert phi.cloud_egress_configured() is expected


def test_cloud_egress_defaults_to_local(monkeypatch):
    # Unset both — defaults (ollama + miniml) must read as no egress.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    assert phi.cloud_egress_configured() is False


def test_should_scrub_off_overrides_cloud(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("PHI_SCRUBBING", "off")
    assert phi.should_scrub() is False


def test_should_scrub_on_overrides_local(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "miniml")
    monkeypatch.setenv("PHI_SCRUBBING", "on")
    assert phi.should_scrub() is True


def test_should_scrub_auto_follows_egress(monkeypatch):
    monkeypatch.setenv("PHI_SCRUBBING", "auto")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert phi.should_scrub() is True
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "miniml")
    assert phi.should_scrub() is False


def test_should_scrub_unknown_value_warns_and_autos(monkeypatch, caplog):
    monkeypatch.setenv("PHI_SCRUBBING", "yes-please")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with caplog.at_level(logging.WARNING):
        assert phi.should_scrub() is True
    assert "yes-please" in caplog.text


def test_threshold_invariant_warns_outside_band(monkeypatch, caplog):
    # Above the context floor -> genuine MRNs would leak -> must warn.
    monkeypatch.setattr(phi, "_ANALYZE_SCORE_THRESHOLD", 0.6)
    with caplog.at_level(logging.WARNING):
        phi._validate_threshold_invariant()
    assert "safe band" in caplog.text


def test_threshold_invariant_silent_in_band(monkeypatch, caplog):
    monkeypatch.setattr(phi, "_ANALYZE_SCORE_THRESHOLD", 0.35)
    with caplog.at_level(logging.WARNING):
        phi._validate_threshold_invariant()
    assert "safe band" not in caplog.text


# Scrub behavior (real anonymizer, fake analyzer)


@pytest.fixture
def fake_analyzer():
    """Return a factory that builds a fake AnalyzerEngine from (entity, substring) spans."""
    RecognizerResult = pytest.importorskip("presidio_analyzer").RecognizerResult

    class FakeAnalyzer:
        def __init__(self, spans):
            self._spans = spans

        def analyze(self, text, entities, language, **kwargs):
            results = []
            for entity_type, substring in self._spans:
                if entity_type not in entities:
                    continue
                start = text.index(substring)
                results.append(
                    RecognizerResult(
                        entity_type=entity_type,
                        start=start,
                        end=start + len(substring),
                        score=0.9,
                    )
                )
            return results

    return FakeAnalyzer


@pytest.mark.real_phi
def test_scrub_phi_noop_on_local_stack(monkeypatch):
    monkeypatch.setattr(phi, "should_scrub", lambda: False)
    # get_analyzer must never be reached on the no-op path.
    monkeypatch.setattr(phi, "get_analyzer", lambda: pytest.fail("analyzer built on no-op path"))
    text = "my patient John Doe, MRN 4471902, stage IV NSCLC"
    assert phi.scrub_phi(text) == text


@pytest.mark.real_phi
@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_scrub_phi_empty_returns_as_is(monkeypatch, text):
    monkeypatch.setattr(phi, "should_scrub", lambda: True)
    monkeypatch.setattr(phi, "get_analyzer", lambda: pytest.fail("analyzer built on empty input"))
    assert phi.scrub_phi(text) == text


@pytest.mark.real_phi
def test_scrub_phi_replaces_entities(monkeypatch, fake_analyzer):
    pytest.importorskip("presidio_anonymizer")
    phi.reset_engine_cache()
    monkeypatch.setattr(phi, "should_scrub", lambda: True)
    monkeypatch.setattr(
        phi,
        "get_analyzer",
        lambda: fake_analyzer(
            [("PERSON", "John Doe"), (phi._DATE_ENTITY, "05/12/1960"), ("MRN", "4471902")]
        ),
    )

    text = "patient John Doe, DOB 05/12/1960, MRN 4471902, stage IV NSCLC"
    scrubbed = phi.scrub_phi(text)

    assert "<PERSON>" in scrubbed
    assert "<DATE_TIME>" in scrubbed  # _DATE_ENTITY renders as <DATE_TIME>
    assert "<MRN>" in scrubbed
    # Raw PHI values must be gone.
    assert "John Doe" not in scrubbed
    assert "05/12/1960" not in scrubbed
    assert "4471902" not in scrubbed
    # Non-PHI clinical content is preserved.
    assert "stage IV NSCLC" in scrubbed


@pytest.mark.real_phi
def test_scrub_phi_no_detections_returns_original(monkeypatch, fake_analyzer):
    pytest.importorskip("presidio_anonymizer")
    monkeypatch.setattr(phi, "should_scrub", lambda: True)
    monkeypatch.setattr(phi, "get_analyzer", lambda: fake_analyzer([]))
    text = "does pembrolizumab improve survival in stage IV NSCLC"
    assert phi.scrub_phi(text) == text


@pytest.mark.real_phi
def test_mrn_recognizer_is_context_anchored(monkeypatch):
    """
    Probe the real analyzer's MRN recognizer in isolation (entities=["MRN"]),
    skipped in CI where the model isn't installed.

    Locks the fix for the MRN false-positive: a bare digit run with no cue word
    must NOT match MRN, but a digit run next to an MRN cue must. Probing MRN
    alone keeps this focused; end-to-end date/corpus behavior is covered by
    test_scrub_phi_real_model_corpus_and_phi.
    """
    import importlib.util

    pytest.importorskip("presidio_analyzer")
    if importlib.util.find_spec("en_core_web_sm") is None:
        pytest.skip("en_core_web_sm not installed")

    monkeypatch.setattr(phi, "_SPACY_MODEL", "en_core_web_sm")
    phi.reset_engine_cache()
    try:
        analyzer = phi.get_analyzer()

        def mrn_hits(text):
            return analyzer.analyze(
                text=text,
                entities=["MRN"],
                language="en",
                score_threshold=phi._ANALYZE_SCORE_THRESHOLD,
            )

        # No cue word -> a bare number (e.g. a PMID) must not be flagged as MRN.
        assert mrn_hits("See PMID 4471902 for data") == []
        # An MRN cue word nearby -> the context enhancer lifts it over the line.
        assert any(h.entity_type == "MRN" for h in mrn_hits("patient MRN 4471902 recorded"))
    finally:
        # Don't leave the real spaCy engine cached for later tests.
        phi.reset_engine_cache()


@pytest.mark.real_phi
def test_scrub_phi_real_model_corpus_and_phi(monkeypatch):
    """
    End-to-end against real Presidio + spaCy (skipped in CI without the model).

    Locks the two guarantees the custom date recognizer buys us: corpus-critical
    tokens (PMIDs, publication years, durations, counts) survive, while real PHI
    (DOB, MRN with a cue, phone, name) is scrubbed.
    """
    import importlib.util

    pytest.importorskip("presidio_analyzer")
    if importlib.util.find_spec("en_core_web_sm") is None:
        pytest.skip("en_core_web_sm not installed")

    monkeypatch.setattr(phi, "_SPACY_MODEL", "en_core_web_sm")
    monkeypatch.setattr(phi, "should_scrub", lambda: True)
    phi.reset_engine_cache()
    try:
        # Corpus tokens preserved (incl. month+year with no day).
        preserved = (
            "PMID 4471902, the 2019 KEYNOTE trial published May 2020, "
            "OS at 5 years in 4471 patients"
        )
        assert phi.scrub_phi(preserved) == preserved

        # PHI scrubbed: name, month-first DOB, MRN (postfix cue), phone.
        out = phi.scrub_phi("John Doe DOB 05/12/1960, 8891234 MRN, call 212-555-0199")
        assert "05/12/1960" not in out
        assert "8891234" not in out
        assert "212-555-0199" not in out
        assert "John Doe" not in out

        # Date variants the panel flagged: year-first and month-name-no-day.
        assert "1960/05/12" not in phi.scrub_phi("born 1960/05/12")
        assert "May 12" not in phi.scrub_phi("admitted May 12 with fever")

        # EHR-paste identifiers (IP + URL recognizers).
        assert "10.0.0.5" not in phi.scrub_phi("host 10.0.0.5 logged the result")
        assert "hospital.com" not in phi.scrub_phi("see https://portal.hospital.com/patient/9")

        # Singleton is idempotent (double-checked lock returns the same engine).
        assert phi.get_analyzer() is phi.get_analyzer()
    finally:
        phi.reset_engine_cache()


@pytest.mark.real_phi
def test_scrub_phi_logs_counts_not_values(monkeypatch, fake_analyzer, caplog):
    pytest.importorskip("presidio_anonymizer")
    phi.reset_engine_cache()
    monkeypatch.setattr(phi, "should_scrub", lambda: True)
    monkeypatch.setattr(phi, "get_analyzer", lambda: fake_analyzer([("PERSON", "John Doe")]))
    with caplog.at_level(logging.INFO):
        phi.scrub_phi("patient John Doe has NSCLC")
    # The log records the entity TYPE and count, never the raw value.
    assert "PERSON" in caplog.text
    assert "John Doe" not in caplog.text
