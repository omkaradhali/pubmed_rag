"""
guardrails.py — Input and output safety checks for the pubmed_rag pipeline.

Input guardrails run before retrieval + generation. They raise GuardrailError
on failure so the pipeline exits early without wasting compute.

    check_topic_relevance(query)          -> GuardrailResult
    check_injection(query)                -> GuardrailResult
    run_input_guardrails(query)           -> list[GuardrailResult]

Output guardrails run after generation. They return warning flags; the answer
is still returned to the caller with the flags attached.

    check_citations(answer, n_sources)    -> GuardrailResult
    check_faithfulness(answer, chunks)    -> GuardrailResult
    check_nli_faithfulness(answer, chunks)-> GuardrailResult
    run_output_guardrails(answer, chunks) -> list[GuardrailResult]
"""

import re
import string
from dataclasses import dataclass, field
from enum import StrEnum

from pubmed_rag import faithfulness_nli


class GuardrailCode(StrEnum):
    OFF_TOPIC = "OFF_TOPIC"
    INJECTION_SUSPECTED = "INJECTION_SUSPECTED"
    MISSING_CITATIONS = "MISSING_CITATIONS"
    CITATION_OUT_OF_RANGE = "CITATION_OUT_OF_RANGE"
    LOW_CITATION_OVERLAP = "LOW_CITATION_OVERLAP"
    CONTRADICTS_SOURCE = "CONTRADICTS_SOURCE"


@dataclass
class GuardrailResult:
    passed: bool
    code: GuardrailCode | None = None  # None when passed=True
    reason: str = ""
    detail: dict = field(default_factory=dict)


class GuardrailError(Exception):
    """Raised by run_input_guardrails when any input check fails."""

    def __init__(self, result: GuardrailResult) -> None:
        self.result = result
        super().__init__(result.reason)


# ── Signal sets (module-level so they compile once) ──────────────────────────

# A query passes the topic check if ANY of these appear as a whole word.
# Broad by design — false positives (rejecting valid clinical queries) are
# worse than false negatives (letting borderline queries through).
_BIOMEDICAL_TERMS: frozenset[str] = frozenset(
    {
        "cancer",
        "tumor",
        "tumour",
        "oncology",
        "oncologist",
        "therapy",
        "treatment",
        "clinical",
        "patient",
        "diagnosis",
        "mutation",
        "gene",
        "protein",
        "drug",
        "efficacy",
        "survival",
        "prognosis",
        "chemotherapy",
        "immunotherapy",
        "biomarker",
        "pathology",
        "pathologist",
        "biopsy",
        "carcinoma",
        "sarcoma",
        "lymphoma",
        "leukemia",
        "melanoma",
        "metastasis",
        "metastatic",
        "remission",
        "recurrence",
        "trial",
        "randomized",
        "cohort",
        "epidemiology",
        "inflammatory",
        "immune",
        "antibody",
        "antigen",
        "receptor",
        "inhibitor",
        "agonist",
        "antagonist",
        "radiation",
        "surgery",
        "resection",
        "histology",
        "chromosome",
        "genomics",
        "proteomics",
        "transcriptome",
        "pubmed",
        "abstract",
        "study",
        "evidence",
        "literature",
        "dose",
        "dosage",
        "adverse",
        "toxicity",
        "safety",
        "screening",
        "staging",
        "grading",
        "biomarkers",
        "expression",
        "amplification",
        "deletion",
        "translocation",
        "her2",
        "brca",
        "egfr",
        "pd-l1",
        "pd1",
        "ctla4",
        "msi",
        "tmb",
        "ngs",
        "wgs",
        "rna",
        "dna",
        "pathogenesis",
        "etiology",
        "aetiology",
        "comorbidity",
        "progression",
        "regression",
        "relapse",
        "refractory",
        "adjuvant",
        "neoadjuvant",
        "palliative",
        "curative",
        "response",
        "resistance",
        "sensitization",
        "synergy",
        "biopsy",
        "resection",
        "excision",
        "ablation",
    }
)

# Only applied when NO biomedical signal is found — reduces false positives.
_OFFTOPIC_PATTERNS: list[str] = [
    r"\bweather\b",
    r"\brecipe(s)?\b",
    r"\bsport(s)?\b",
    r"\bfootball\b",
    r"\bbasketball\b",
    r"\bsoccer\b",
    r"\bstock(s)?\b",
    r"\bcrypto(currency)?\b",
    r"\bwrite\b.{0,25}\b(poem|story|essay|script|email|letter)\b",
    r"\btranslate\b",
    r"\bsong(s)?\b",
    r"\bmovie(s)?\b",
    r"\bvideo\s+game(s)?\b",
    r"\bcook(ing)?\b",
    r"\bhoroscope\b",
    r"\bastrology\b",
    r"\bceleb(rity|rities)?\b",
]

# Injection attack signatures. Matched case-insensitively against the raw query.
# Patterns target complete phrases — single words like "ignore" are not matched
# to avoid false positives on queries like "ignoring confounders in survival
# analysis".
_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(previous|all|above|prior)\s+instructions?",
    r"forget\s+(everything|what|your|all)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if\b|a\b|an\b)",
    r"you\s+are\s+now\b",
    r"system\s+prompt",
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"\{\{",  # Jinja/template injection
    r"\$\{",  # shell variable expansion
]

# Zero-width, bidirectional-override, and BOM characters used to hide payloads.
# Built with chr() so code point coverage is explicit and auditable without
# embedding invisible characters in source — previous version used literal
# invisible chars that were invisible in editors and missing U+200E/200F/2060.
_INJECTION_UNICODE_RE = re.compile(
    "["
    + chr(0x200B)
    + "-"
    + chr(0x200F)  # zero-width space, ZWNJ, ZWJ, LRM, RLM
    + chr(0x202A)
    + "-"
    + chr(0x202E)  # LRE, RLE, PDF, LRO, RLO (bidi overrides)
    + chr(0x2060)  # word joiner
    + chr(0x2066)
    + "-"
    + chr(0x2069)  # LRI, RLI, FSI, PDI (bidi isolates)
    + chr(0xFEFF)  # BOM / zero-width no-break space
    + "]"
)

# Regex to extract all [N] citation markers from answer text.
# Capped at 5 digits — prevents adversarial inputs from triggering expensive
# int() conversion or hitting Python's integer string length limit.
_CITATION_RE = re.compile(r"\[(\d{1,5})\]")

# These phrases indicate the LLM declared it couldn't answer — citations are
# intentionally absent and the citation check should be skipped.
_NO_CONTEXT_PHRASES: tuple[str, ...] = (
    "does not address",
    "not sufficient",
    "cannot answer",
    "no relevant",
    "not contain",
    "does not provide",
)

# Common English words + clinical filler stripped before Jaccard computation.
# Keep medical nouns out of this list — "patient", "study", "result" are too
# frequent to contribute signal but DO appear in source text.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "their",
        "we",
        "our",
        "he",
        "she",
        "his",
        "her",
        "showed",
        "shown",
        "found",
        "suggest",
        "suggests",
        "reported",
        "also",
        "however",
        "therefore",
        "thus",
        "moreover",
        "although",
        "while",
        "when",
        "which",
        "who",
        "where",
        "how",
        "what",
        "than",
        "not",
        "no",
        "as",
        "such",
        "both",
        "between",
        "into",
        "through",
    }
)


# ── Input guardrails ─────────────────────────────────────────────────────────


def check_topic_relevance(query: str) -> GuardrailResult:
    """
    Verify the query is related to clinical or biomedical topics.

    Passes if any biomedical term from _BIOMEDICAL_TERMS is found as a whole
    word. Only applies the off-topic blocklist when no biomedical signal is
    present — avoids false positives on mixed queries.
    """
    words = query.strip().split()
    if len(words) < 3:
        return GuardrailResult(
            passed=False,
            code=GuardrailCode.OFF_TOPIC,
            reason=(
                "Query is too short to evaluate topic relevance. "
                "Please ask a specific clinical or biomedical question."
            ),
        )

    lowered = query.lower()

    has_biomedical = any(
        re.search(r"\b" + re.escape(term) + r"\b", lowered) for term in _BIOMEDICAL_TERMS
    )
    if has_biomedical:
        return GuardrailResult(passed=True)

    for pattern in _OFFTOPIC_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailResult(
                passed=False,
                code=GuardrailCode.OFF_TOPIC,
                reason=(
                    "Query does not appear to be related to clinical or biomedical topics. "
                    "Please ask a medical or scientific question."
                ),
            )

    return GuardrailResult(passed=True)


def check_injection(query: str) -> GuardrailResult:
    """
    Detect prompt injection attempts in the query.

    Checks for Unicode control characters first (hidden payload technique),
    then matches against known injection phrase patterns. The matched pattern
    is recorded in detail but the raw query is never echoed back to the caller.
    """
    if _INJECTION_UNICODE_RE.search(query):
        return GuardrailResult(
            passed=False,
            code=GuardrailCode.INJECTION_SUSPECTED,
            reason=(
                "Query contains patterns associated with prompt injection. "
                "Please rephrase your clinical question."
            ),
            detail={"matched_pattern": "suspicious_unicode"},
        )

    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return GuardrailResult(
                passed=False,
                code=GuardrailCode.INJECTION_SUSPECTED,
                reason=(
                    "Query contains patterns associated with prompt injection. "
                    "Please rephrase your clinical question."
                ),
                detail={"matched_pattern": pattern},
            )

    return GuardrailResult(passed=True)


def run_input_guardrails(query: str) -> list[GuardrailResult]:
    """
    Run all input guardrails in order. Raises GuardrailError on the first failure.

    Topic relevance runs before injection detection — off-topic queries don't
    need injection scanning (fail fast).

    Returns:
        List of passed GuardrailResult objects when all checks clear.
    Raises:
        GuardrailError: wrapping the failing GuardrailResult.
    """
    results: list[GuardrailResult] = []
    for check in (check_topic_relevance, check_injection):
        result = check(query)
        results.append(result)
        if not result.passed:
            raise GuardrailError(result)
    return results


# ── Output guardrails ────────────────────────────────────────────────────────


def check_citations(answer: str, n_sources: int) -> GuardrailResult:
    """
    Verify the generated answer contains valid inline [N] citations.

    Skipped entirely when n_sources == 0 or when the answer is the
    "does not address" fallback (citations are intentionally absent there).

    Checks:
      A) At least one [N] marker present.
      B) No citation number exceeds n_sources.
    """
    if n_sources == 0:
        return GuardrailResult(passed=True)

    lowered = answer.lower()
    if any(phrase in lowered for phrase in _NO_CONTEXT_PHRASES):
        return GuardrailResult(passed=True)

    cited = [int(n) for n in _CITATION_RE.findall(answer)]

    if not cited:
        return GuardrailResult(
            passed=False,
            code=GuardrailCode.MISSING_CITATIONS,
            reason=(
                f"Answer contains no inline [N] citations despite "
                f"{n_sources} source(s) being retrieved."
            ),
        )

    # [0] is invalid — sources are 1-indexed. Also catches numbers exceeding n_sources.
    out_of_range = sorted(set(n for n in cited if n < 1 or n > n_sources))
    if out_of_range:
        return GuardrailResult(
            passed=False,
            code=GuardrailCode.CITATION_OUT_OF_RANGE,
            reason=(
                f"Answer cites source(s) {out_of_range} but only "
                f"{n_sources} source(s) were retrieved."
            ),
            detail={"out_of_range": out_of_range, "n_sources": n_sources},
        )

    return GuardrailResult(passed=True)


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, split on whitespace, remove stopwords."""
    translator = str.maketrans("", "", string.punctuation)
    return {w for w in text.lower().translate(translator).split() if w and w not in _STOPWORDS}


def check_faithfulness(answer: str, chunks: list[dict]) -> GuardrailResult:
    """
    Lightweight lexical faithfulness check: cited sentences vs. source chunk text.

    For each sentence in the answer that contains a [N] citation, computes
    unigram Jaccard overlap between that sentence and the text of chunk N.
    Flags pairs where overlap < 0.05 — a very low bar that catches outright
    hallucinations without penalising paraphrasing.

    This is deterministic and cheap (no LLM call). It complements RAGAS
    faithfulness (LLM-as-judge) which runs at evaluation time.
    """
    if not chunks:
        return GuardrailResult(passed=True)

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    low_overlap_pairs: list[dict] = []

    for sentence in sentences:
        cited_nums = [int(n) for n in _CITATION_RE.findall(sentence)]
        if not cited_nums:
            continue

        sentence_tokens = _tokenize(sentence)
        if not sentence_tokens:
            continue

        for n in cited_nums:
            if n < 1 or n > len(chunks):
                continue
            chunk_tokens = _tokenize(chunks[n - 1].get("text", ""))
            if not chunk_tokens:
                continue

            union = sentence_tokens | chunk_tokens
            jaccard = len(sentence_tokens & chunk_tokens) / len(union)

            if jaccard < 0.05:
                low_overlap_pairs.append(
                    {
                        "sentence": sentence[:200],
                        "source_n": n,
                        "jaccard": round(jaccard, 4),
                    }
                )

    if low_overlap_pairs:
        return GuardrailResult(
            passed=False,
            code=GuardrailCode.LOW_CITATION_OVERLAP,
            reason="One or more cited claims have low lexical overlap with their source.",
            detail={"low_overlap_pairs": low_overlap_pairs},
        )

    return GuardrailResult(passed=True)


def check_nli_faithfulness(answer: str, chunks: list[dict]) -> GuardrailResult:
    """
    NLI faithfulness check: does any cited claim contradict its source?

    Second faithfulness pass after the lexical check_faithfulness. Where Jaccard
    overlap is blind to negation ("is effective" vs "is not effective") and
    entity swaps ("tachycardia" vs "bradycardia") — the tokens barely change —
    a natural-language-inference cross-encoder reads the two together and scores
    P(contradiction). Fails when any cited sentence is contradicted by chunk N
    above faithfulness_nli.CONTRADICTION_THRESHOLD.

    The model is lazy-loaded on first use (see faithfulness_nli); this returns a
    passing result immediately when there is nothing cited to check.
    """
    contradictions = faithfulness_nli.find_contradictions(answer, chunks)
    if not contradictions:
        return GuardrailResult(passed=True)

    return GuardrailResult(
        passed=False,
        code=GuardrailCode.CONTRADICTS_SOURCE,
        reason="One or more cited claims are contradicted by their cited source.",
        detail={"contradictions": contradictions},
    )


def run_output_guardrails(answer: str, chunks: list[dict]) -> list[GuardrailResult]:
    """
    Run all output guardrails. Returns all results (passed and failed).

    Unlike input guardrails, failures here do not raise — they are advisory
    warnings. The caller filters for passed=False and attaches those to the
    PipelineResult as guardrail_flags.
    """
    return [
        check_citations(answer, n_sources=len(chunks)),
        check_faithfulness(answer, chunks),
        check_nli_faithfulness(answer, chunks),
    ]


# Minimum number of low-overlap cited claims before LOW_CITATION_OVERLAP is
# treated as a systemic (hard-block) failure rather than an advisory warning.
# A single low-overlap sentence may be a short transition; multiple is systemic.
LOW_OVERLAP_HARD_BLOCK_THRESHOLD = 2


def is_hard_block(result: GuardrailResult) -> bool:
    """
    Decide whether an output guardrail failure must hard-block the answer.

    Hard blocks (vs. advisory flags) trigger the retry + safe-fallback path in
    pipeline._generate_grounded_answer — an ungrounded answer must never reach a
    clinical user:
      * MISSING_CITATIONS / CITATION_OUT_OF_RANGE — always a hard block.
      * CONTRADICTS_SOURCE — always a hard block. A single high-confidence NLI
        contradiction (e.g. a negated or entity-swapped claim) is severe, unlike
        the noisier lexical-overlap signal.
      * LOW_CITATION_OVERLAP — a hard block only when at least
        LOW_OVERLAP_HARD_BLOCK_THRESHOLD cited claims have low overlap.
    """
    if result.passed:
        return False
    if result.code in (
        GuardrailCode.MISSING_CITATIONS,
        GuardrailCode.CITATION_OUT_OF_RANGE,
        GuardrailCode.CONTRADICTS_SOURCE,
    ):
        return True
    if result.code == GuardrailCode.LOW_CITATION_OVERLAP:
        pairs = result.detail.get("low_overlap_pairs", [])
        return len(pairs) >= LOW_OVERLAP_HARD_BLOCK_THRESHOLD
    return False
