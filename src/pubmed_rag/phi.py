"""
phi.py — PHI/PII scrubbing before any query text leaves the server.

Tier 1 pre-release blocker (Session C). Clinicians habitually paste patient
context into free-text boxes ("my patient John Doe, DOB 05/12/1960, MRN
4471902, stage IV NSCLC..."). If a cloud egress path is configured — the LLM
provider is anthropic/haiku/sonnet/openai, OR EMBEDDING_PROVIDER=openai — that
text leaves the server and becomes a HIPAA violation. LLM_PROVIDER=ollama with
EMBEDDING_PROVIDER=miniml/medcpt is fully local and needs no scrubbing.

Scrubbing runs ONCE at pipeline entry (run_pipeline_structured), after the input
guardrails and before retrieval, so the same de-identified query flows to the
embedder, the generation prompt, AND the Session D audit log. See the design
brief in claude-brain: projects/pubmed-rag/session-c-phi-scrubbing.md.

Engine: Microsoft Presidio (presidio-analyzer + presidio-anonymizer), backed by
a spaCy pipeline for the PERSON/LOCATION NER recognizers. The analyzer is a heavy
import (it loads a ~500MB spaCy model), so we lazy-load a singleton exactly like
faithfulness_nli.get_nli_model: presidio + spacy are imported inside the loader
and the engine is built on the first scrub call. Importing this module stays
cheap and side-effect-free; unit tests patch scrub_phi (autouse identity stub in
conftest) or get_analyzer so no model download occurs in CI.

Anonymization is deterministic replacement: each detected span becomes an
entity-type tag, e.g. `<PERSON>`, `<DATE_TIME>`, `<US_SSN>`, `<MRN>`. No
reversible tokens and no per-run randomness — the same input always yields the
same output, which the Session D audit log relies on.

NEVER log raw query text or detected PHI values. Log only aggregate counts by
entity type at INFO.

Public API:
    PHI_ENTITIES                 — entity types scrubbed
    cloud_egress_configured()    — True when a provider would send text off-box
    should_scrub()               — cloud egress AND not force-disabled (PHI_SCRUBBING)
    get_analyzer()               — lazy Presidio AnalyzerEngine singleton
    get_anonymizer()             — lazy Presidio AnonymizerEngine singleton
    scrub_phi(text)              — de-identified text (unchanged if !should_scrub)
    reset_engine_cache()         — drop singletons (tests)
"""

import logging
import os
import threading
from collections import Counter

_logger = logging.getLogger(__name__)

# Providers known to run ON the server (no egress). This is an ALLOWLIST, not a
# denylist: any provider NOT listed here (a future gemini/azure/bedrock/cohere,
# or a typo) is treated as cloud and scrubbed. Fail safe — never leak because a
# new provider wasn't anticipated.
_LOCAL_LLM_PROVIDERS = frozenset({"ollama"})
_LOCAL_EMBEDDING_PROVIDERS = frozenset({"miniml", "medcpt"})

# Custom entity for real calendar dates. We deliberately do NOT request the
# built-in DATE_TIME: spaCy's NER tags durations ("5 years") and Presidio's date
# recognizer grabs bare numbers (PMIDs, publication years) as DATE_TIME, which is
# corpus-fatal for a PubMed tool. Requesting a custom entity that only our
# date-format regex emits (see get_analyzer) naturally excludes those built-ins,
# since analyze() filters results to the requested entity set. Anonymized output
# is still tagged <DATE_TIME> (see _ENTITY_TAGS). Rationale: HIPAA Safe Harbor
# removes date elements smaller than a year, so keeping standalone years is
# compliant AND preserves literature references like "KEYNOTE-189 (2021)".
_DATE_ENTITY = "PHI_DATE"

# Entity types Presidio scrubs. US_SSN / PHONE_NUMBER / EMAIL_ADDRESS / PERSON /
# LOCATION / IP_ADDRESS / URL are built-in Presidio recognizers. MRN and
# _DATE_ENTITY are custom pattern recognizers registered in get_analyzer().
# DATES (with a day) cover DOB. IP_ADDRESS / URL catch identifiers that show up
# when a clinician pastes a block straight out of an EHR.
PHI_ENTITIES: tuple[str, ...] = (
    "PERSON",
    _DATE_ENTITY,
    "US_SSN",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "LOCATION",
    "IP_ADDRESS",
    "URL",
    "MRN",
)

# Output tag per entity. Everything maps to <ENTITY> except the custom date
# entity, which reads as the familiar <DATE_TIME> in scrubbed text.
_ENTITY_TAGS: dict[str, str] = {entity: f"<{entity}>" for entity in PHI_ENTITIES}
_ENTITY_TAGS[_DATE_ENTITY] = "<DATE_TIME>"

# Month names (full and common abbreviations) for the calendar-date patterns.
_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)

# Calendar-date regexes. Each requires at least a day+month (or a numeric triple),
# so a standalone year ("2021") and a duration ("5 years") never match — only real
# dates like a DOB do. Numeric two-component forms ("10/14", "5/10") are
# deliberately NOT matched: they collide with dosing and ratios (documented gap).
_DATE_REGEXES: tuple[str, ...] = (
    r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",  # month-first: 05/12/1960, 5-12-60, 05.12.1960
    r"\b\d{4}[/.-]\d{1,2}[/.-]\d{1,2}\b",  # year-first: 1960-05-12, 1960/05/12, 1960.05.12
    # Month name + day, year optional: "March 5, 2019", "born May 12". A trailing
    # bare year is kept only when preceded by a day, so "May 2020" (month+year,
    # no day) is left alone as a likely publication reference.
    rf"(?i)\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{2,4}})?\b",
    rf"(?i)\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:,?\s+\d{{2,4}})?\b",  # "5 March 2019", "12 May"
)

# spaCy model backing the Presidio NER recognizers. en_core_web_lg is Presidio's
# recommended default (best PERSON/LOCATION recall); en_core_web_sm is lighter if
# the container image size matters. Override via PHI_SPACY_MODEL.
_SPACY_MODEL = os.getenv("PHI_SPACY_MODEL", "en_core_web_lg")

# Minimum acceptance score for a detection to be scrubbed. Presidio's default is
# 0, which would accept the MRN pattern on any bare number (a PMID, a dosage).
# We instead give the MRN pattern a base score BELOW this floor and rely on the
# context enhancer to boost it over the line only when an MRN cue word is nearby
# (see get_analyzer). Built-in recognizers (PERSON, DATE_TIME, PHONE_NUMBER, ...)
# score comfortably above this, so the floor does not suppress them.
_ANALYZE_SCORE_THRESHOLD = float(os.getenv("PHI_SCORE_THRESHOLD", "0.35"))

# Base score for the MRN bare-digit pattern. Deliberately below
# _ANALYZE_SCORE_THRESHOLD so a bare digit run is dropped; Presidio's context
# enhancer adds ~0.35 (and floors to _CONTEXT_SIMILARITY_FLOOR) when an MRN cue
# word sits nearby, lifting a prefix-cued MRN over the line.
_MRN_BASE_SCORE = 0.01

# Presidio's LemmaContextAwareEnhancer floors a context-boosted match to this
# score (its min_score_with_context_similarity default). A prefix-cued MRN thus
# lands at exactly this value, so the acceptance threshold must not exceed it or
# genuine MRNs would leak. Enforced by _validate_threshold_invariant().
_CONTEXT_SIMILARITY_FLOOR = 0.4


def _validate_threshold_invariant() -> None:
    """
    Warn (do not crash) if PHI_SCORE_THRESHOLD is set outside the safe band.

    The MRN scheme only works when _MRN_BASE_SCORE < threshold <=
    _CONTEXT_SIMILARITY_FLOOR: below the base score, bare numbers (PMIDs) get
    scrubbed; above the context floor, genuine cue-anchored MRNs leak. A misset
    env var is a silent PHI risk, so surface it loudly at import.
    """
    if not (_MRN_BASE_SCORE < _ANALYZE_SCORE_THRESHOLD <= _CONTEXT_SIMILARITY_FLOOR):
        _logger.warning(
            "PHI_SCORE_THRESHOLD=%.3f is outside the safe band (%.3f, %.3f]: "
            "MRNs may leak or PMIDs may be over-scrubbed.",
            _ANALYZE_SCORE_THRESHOLD,
            _MRN_BASE_SCORE,
            _CONTEXT_SIMILARITY_FLOOR,
        )


_validate_threshold_invariant()

# Cached singletons — None until the first scrub call. _engine_lock guards their
# construction so concurrent cold-start requests (FastAPI runs the pipeline in a
# thread pool) don't build duplicate heavy engines — mirrors the double-checked
# locking used for the BM25 index in retrieve.py.
_engine_lock = threading.Lock()
_analyzer = None
_anonymizer = None


def cloud_egress_configured() -> bool:
    """
    True when the configured providers would transmit query text off the server.

    Fail-safe allowlist: egress is assumed UNLESS both the LLM and the embedding
    provider are known-local (_LOCAL_LLM_PROVIDERS / _LOCAL_EMBEDDING_PROVIDERS).
    Any unrecognized provider — a future gemini/azure/bedrock/cohere or a typo —
    counts as cloud, so a query is never leaked because a new backend wasn't
    anticipated. Note the embedding path matters: the query is embedded remotely
    at retrieval, before generation, so an openai embedder egresses too.

    Reads os.getenv to match pipeline.py's config style (load_dotenv is already
    called there), with the same defaults used elsewhere in the codebase.
    """
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "miniml").strip().lower()
    llm_local = llm_provider in _LOCAL_LLM_PROVIDERS
    embedding_local = embedding_provider in _LOCAL_EMBEDDING_PROVIDERS
    return not (llm_local and embedding_local)


def should_scrub() -> bool:
    """
    True when scrubbing must run for this request.

    Default policy: scrub iff cloud_egress_configured(). The PHI_SCRUBBING env
    var overrides:
        "auto" (default) — follow cloud_egress_configured()
        "on"             — always scrub (defense in depth, e.g. shared logs)
        "off"            — never scrub (NOT for clinical use; document the risk)
    An unrecognized value is treated as "auto" and logged as a warning.
    """
    policy = os.getenv("PHI_SCRUBBING", "auto").strip().lower()
    if policy == "on":
        return True
    if policy == "off":
        return False
    if policy != "auto":
        _logger.warning("Unrecognized PHI_SCRUBBING=%r; falling back to 'auto'.", policy)
    return cloud_egress_configured()


def get_analyzer():
    """
    Return the lazily-built Presidio AnalyzerEngine singleton.

    presidio_analyzer and spacy are imported here, not at module top, so
    importing phi.py stays cheap and tests can stub without the heavy model
    download. The engine loads _SPACY_MODEL and registers a custom MRN
    recognizer.

    The MRN recognizer is context-anchored: a bare 6-10 digit run scores low, so
    on its own (a PMID, a dosage, a year) it stays below the acceptance
    threshold. Two paths lift a genuine MRN over the line: Presidio's context
    enhancer boosts a PREFIX cue ("MRN 4471902") via the single-word cue list,
    and a lookahead pattern catches a SUFFIX cue ("4471902 MRN") that the
    enhancer (which only looks backward) would miss.

    Built with double-checked locking: FastAPI runs the pipeline in a thread
    pool, so two cold-start requests can enter concurrently.
    """
    global _analyzer
    if _analyzer is not None:
        return _analyzer
    with _engine_lock:
        if _analyzer is not None:
            return _analyzer

        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        _logger.info("Building Presidio analyzer (spaCy model=%s, first call)...", _SPACY_MODEL)
        nlp_engine = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": _SPACY_MODEL}],
            }
        ).create_engine()

        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

        # Single-word cues: Presidio's context enhancer matches token lemmas, so
        # multi-word phrases ("medical record") never fire; "medical" + "record"
        # together still cover "medical record number".
        _MRN_CUES = "mrn|medical|record|chart"
        mrn_recognizer = PatternRecognizer(
            supported_entity="MRN",
            patterns=[
                # Prefix cue: bare number, lifted by the context enhancer.
                Pattern(name="mrn_digits", regex=r"\b\d{6,10}\b", score=_MRN_BASE_SCORE),
                # Suffix cue: number immediately followed by a cue word. The
                # lookahead keeps the span to just the digits (the cue stays in
                # the text). Scored above threshold so it fires on its own.
                Pattern(
                    name="mrn_digits_suffix_cue",
                    regex=rf"(?i)\b\d{{6,10}}\b(?=\s+(?:{_MRN_CUES})\b)",
                    score=0.85,
                ),
            ],
            context=_MRN_CUES.split("|"),
        )
        analyzer.registry.add_recognizer(mrn_recognizer)

        # Custom calendar-date recognizer under _DATE_ENTITY. Replaces the
        # built-in DATE_TIME (never requested) so durations and bare numbers are
        # left alone; only real dates (which include a day) are caught.
        date_recognizer = PatternRecognizer(
            supported_entity=_DATE_ENTITY,
            patterns=[
                Pattern(name=f"date_{i}", regex=regex, score=0.6)
                for i, regex in enumerate(_DATE_REGEXES)
            ],
        )
        analyzer.registry.add_recognizer(date_recognizer)

        _analyzer = analyzer
    return _analyzer


def get_anonymizer():
    """Return the lazily-built Presidio AnonymizerEngine singleton (no model needed)."""
    global _anonymizer
    if _anonymizer is not None:
        return _anonymizer
    with _engine_lock:
        if _anonymizer is None:
            from presidio_anonymizer import AnonymizerEngine

            _anonymizer = AnonymizerEngine()
    return _anonymizer


def scrub_phi(text: str) -> str:
    """
    Return `text` with PHI/PII replaced by entity-type tags.

    No-op fast path: if not should_scrub(), return text unchanged (local-only
    stack — nothing leaves the box, and scrubbing would needlessly degrade
    retrieval on legitimately medical queries). Empty/whitespace text is also
    returned as-is.

    Otherwise: analyze over PHI_ENTITIES, then anonymize each detected span to
    `<{entity_type}>`. Logs only the per-type span counts — never the raw text
    or the detected values.
    """
    if not text or not text.strip() or not should_scrub():
        return text

    from presidio_anonymizer.entities import OperatorConfig

    analyzer = get_analyzer()
    anonymizer = get_anonymizer()

    results = analyzer.analyze(
        text=text,
        entities=list(PHI_ENTITIES),
        language="en",
        score_threshold=_ANALYZE_SCORE_THRESHOLD,
    )
    if not results:
        return text

    operators = {
        entity: OperatorConfig("replace", {"new_value": tag})
        for entity, tag in _ENTITY_TAGS.items()
    }
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)

    counts = Counter(result.entity_type for result in results)
    _logger.info("Scrubbed PHI spans by type: %s", dict(counts))

    return anonymized.text


def reset_engine_cache() -> None:
    """Drop the cached engines. Test hook."""
    global _analyzer, _anonymizer
    _analyzer = None
    _anonymizer = None
