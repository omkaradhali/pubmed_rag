# ADR-040: Deterministic input/output guardrails

**Date:** 2026-06-29
**Status:** Accepted

---

## Context

The RAG pipeline has one safety layer at generation time: a system prompt that instructs the LLM to cite every claim and refuse off-topic answers. This is necessary but insufficient for two reasons:

1. The system prompt is LLM-side. It cannot prevent a malicious or off-topic query from consuming retrieval compute, and a sufficiently crafted prompt can override it.
2. LLMs are nondeterministic. Even with citation-enforcement instructions, the model can hallucinate a citation number, cite the wrong source, or paraphrase so loosely that the original text does not support the claim.

The options evaluated were:

| Approach | Cost | Determinism | Latency |
|---|---|---|---|
| LLM-as-judge for input validation | Per-call API cost | Nondeterministic | +1-3 sec per query |
| Semantic similarity for faithfulness | Embedding cost | Deterministic | +100-300ms |
| Pattern-based input checks | Zero | Deterministic | <1ms |
| Lexical overlap (Jaccard) for faithfulness | Zero | Deterministic | <5ms |

---

## Decision

Implement four deterministic guardrails that bracket the pipeline with no LLM cost and no nondeterminism. Guardrails live in `src/pubmed_rag/guardrails.py`.

### Input guardrails (run before retrieval)

**1. Topic relevance check (`check_topic_relevance`)**
- Checks for at least one biomedical term from a curated set as a whole-word match.
- Only applies a blocklist of off-topic patterns when no biomedical signal is present — avoids false positives on mixed queries.
- Rejects queries with fewer than 3 words.
- On failure: raises `GuardrailError(code=OFF_TOPIC)`.

**2. Injection detection (`check_injection`)**
- Matches against known injection phrase patterns (`re.IGNORECASE`).
- Checks for Unicode control characters (zero-width, bidirectional-override) used to hide payloads.
- Patterns target complete injection phrases, not individual trigger words — avoids flagging clinical queries that contain words like "ignore" (e.g., "ignoring confounders in survival analysis").
- On failure: raises `GuardrailError(code=INJECTION_SUSPECTED)`. Raw query is never echoed back in the reason string.

### Output guardrails (run after generation)

**3. Citation check (`check_citations`)**
- Verifies the answer contains at least one `[N]` marker when sources were retrieved and the answer is not the "does not address" fallback.
- Also checks that no citation number exceeds `n_sources`.
- On failure: returns `GuardrailResult(passed=False, code=MISSING_CITATIONS | CITATION_OUT_OF_RANGE)`.

**4. Faithfulness check (`check_faithfulness`)**
- Sentence-splits the answer, finds `[N]` markers in each sentence, and computes unigram Jaccard overlap between that sentence and the text of chunk N.
- Flags pairs where Jaccard < 0.05 — a very low bar that catches outright hallucinations without penalising paraphrasing.
- On failure: returns `GuardrailResult(passed=False, code=LOW_CITATION_OVERLAP)` with per-pair detail.

### Failure semantics

| Guardrail type | Failure action | API response |
|---|---|---|
| Input | Raises `GuardrailError` | HTTP 422 with `code` + `reason` |
| Output | Returns flagged `GuardrailResult` | Answer still returned; `guardrail_flags` populated |

Input failures are hard blocks (user error, fix the query). Output failures are advisory warnings (system observation, answer still goes out).

---

## Consequences

- All four checks are zero-cost and deterministic — no LLM call, no embedding, no network I/O.
- False positive risk on topic relevance is mitigated by the permissive design: only reject when both no biomedical signal AND a blocklist pattern fire.
- The faithfulness threshold (Jaccard < 0.05) catches only hallucinations, not paraphrases. For full semantic faithfulness coverage, use RAGAS at evaluation time.
- `GuardrailResult` and `GuardrailError` are exposed in `api/schemas.py` as `GuardrailFlagResponse` for API consumers.
- 53 unit tests cover all four checks, the orchestrators, and edge cases (false-positive traps, zero-source bypass, Unicode injection).
