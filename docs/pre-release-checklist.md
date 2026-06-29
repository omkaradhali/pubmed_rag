# Pre-Release Checklist — Clinical Production

This checklist must be completed before pubmed_rag v1.0 backs any UI used by
clinicians, pathologists, or oncologists in a real patient-care context.

Derived from the dual Antigravity + Codex code review conducted 2026-06-29.
Reviewed against: guardrails (PR #8), hybrid BM25+RRF (PR #7).

---

## Status key

| Symbol | Meaning |
|---|---|
| ✅ | Done — merged to main |
| 🔲 | Not started |
| 🚫 | Out of scope — documented, accepted risk |

---

## Tier 1 — Blockers (must ship before any clinical user touches this)

### Patient safety

- [ ] **Retry + hard block on citation failure**
  `check_citations` currently flags missing/invalid citations as an advisory warning — the answer still reaches the UI. For a clinical tool, a hallucinated answer with a footnote warning is not acceptable (automation bias). Implement:
  1. On `MISSING_CITATIONS` or `CITATION_OUT_OF_RANGE`, re-prompt the LLM once with an explicit correction instruction.
  2. If the retry also fails, return a hardcoded safe fallback: *"I was unable to produce a properly grounded answer for this question. Please consult primary sources directly."*
  3. Never surface an uncited generated answer in the clinical UI.
  **File:** `src/pubmed_rag/pipeline.py` — add retry loop after `generate_answer()`.

- [ ] **Hard block on LOW_CITATION_OVERLAP above a severity threshold**
  Currently `check_faithfulness` is advisory. Add a severity tier: if ≥ 2 low-overlap pairs are found, treat it as a hard block (same retry + fallback path as citation failure). A single low-overlap sentence may be a short transition sentence; multiple is a systemic faithfulness failure.
  **File:** `src/pubmed_rag/pipeline.py`.

- [ ] **Upgrade faithfulness check to catch negation and entity swaps**
  Jaccard unigram overlap gives near-identical scores for "drug is effective" vs "drug is NOT effective" and for "tachycardia" vs "bradycardia". Both are clinically dangerous hallucinations the current check misses entirely.
  Recommended fix: add a local NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`, runs in ~50ms on CPU) as a second faithfulness pass. Input: `(cited_sentence, source_chunk_text)`. Block if the model returns `contradiction` with confidence > 0.8.
  **File:** new `src/pubmed_rag/faithfulness_nli.py` + wire into `guardrails.py`.

### Data privacy and compliance

- [ ] **PHI/PII scrubbing before LLM dispatch**
  Clinicians habitually paste patient context into text boxes ("My patient John Doe, DOB 05/12/1960, stage IV NSCLC..."). If `LLM_PROVIDER` is set to `anthropic` or `openai`, that text leaves the server — a HIPAA violation.
  Required: integrate [Microsoft Presidio](https://microsoft.github.io/presidio/) as a scrubbing layer inside `generate.py` before the prompt is assembled. Scrub: names, dates of birth, MRNs, phone numbers, addresses, SSNs.
  Note: `LLM_PROVIDER=ollama` (local) is already safe — scrubbing is only mandatory for cloud providers.
  **File:** `src/pubmed_rag/generate.py` — add `scrub_phi(query)` call at entry.

- [ ] **Audit logging**
  Every clinical query must produce an immutable audit record. Required fields: `request_id`, `timestamp`, `query` (post-scrub), `retrieved_pmids`, `llm_provider`, `llm_model`, `answer` (truncated), `guardrail_results`, `confidence_tier`.
  Store in a structured append-only log (JSON Lines file or database table). Minimum retention: 7 years (HIPAA).
  **File:** `api/routers/ask.py` — emit audit log after `run_pipeline_structured()` completes.

### Infrastructure

- [ ] **BM25 index thread safety**
  `_get_bm25_index()` in `retrieve.py` uses a module-level global with no locking. Two concurrent requests on cold start can build competing indices with mismatched `_bm25_parents` lists. With multiple uvicorn workers this is a real race condition.
  Fix: add a `threading.Lock` around the initialization block and assign `_bm25_index` / `_bm25_parents` from local variables in one atomic step.
  **File:** `src/pubmed_rag/retrieve.py`.
  Note: hybrid search is off by default; this must be fixed before enabling it in production.

- [ ] **Auth + rate limiting** *(Day 29 — already planned)*
  No auth currently. Do not expose the API to any network without completing Day 29 (`slowapi` + static API key).
  **File:** `api/main.py`, `api/routers/ask.py`.

---

## Tier 2 — Strong-should-haves (ship in v1.0, not hard blockers but close)

- [ ] **Sentence splitter upgrade for faithfulness check**
  `re.split(r"(?<=[.!?])\s+", ...)` splits on abbreviations (`Dr.`, `vs.`, `No.`, `mo.`) and incorrectly detaches cited fragments from their sentences. This produces false `LOW_CITATION_OVERLAP` flags that erode clinician trust in the guardrail system.
  Fix: replace with `nltk.sent_tokenize()` (punkt tokenizer, handles common abbreviations).
  **File:** `src/pubmed_rag/guardrails.py` — `check_faithfulness()`.

- [ ] **`[0]` citation and digit cap** ✅ *(fixed 2026-06-29)*
  `check_citations` now rejects `[0]` and caps citation regex at 5 digits.

- [ ] **Unicode injection regex — explicit code points** ✅ *(fixed 2026-06-29)*
  `_INJECTION_UNICODE_RE` rewritten with `chr()` to cover U+200E, U+200F, U+2060 and be auditable in source.

- [ ] **Explicit cannot-answer guardrail**
  If `n_chunks_retrieved == 0` or `avg_score < 0.3`, skip LLM generation entirely and return the safe fallback string directly. Currently the pipeline still calls the LLM and relies on it saying "does not address" — that's nondeterministic.
  **File:** `src/pubmed_rag/pipeline.py` — add pre-generation score gate.

- [ ] **User feedback mechanism**
  Clinical UI must expose upvote / downvote / flag-for-review on every answer. Flagged answers should trigger an alert to the dev team. This is the only way to catch hallucinations that pass the guardrail stack.
  **File:** new API endpoint `POST /feedback` + UI component.

---

## Tier 3 — Deferred to v2.0 (documented, accepted risk for v1.0)

- 🚫 **Injection bypass via homoglyphs, language switching, indirect injection**
  The current regex stack catches common English-phrase injection patterns and Unicode control characters. Homoglyph attacks (Cyrillic "о" substituting for Latin "o"), non-English injection, and indirect injection via retrieved document content are not caught. For an internal clinical tool with vetted users this is an accepted risk. Document in `SECURITY.md`. Revisit for v2.0 if the tool becomes publicly accessible.

- 🚫 **Topic relevance keyword fragility**
  Terms like "drug", "study", "response", "safety" let through off-topic queries. The `< 3 words` rule blocks short valid queries. A classifier-based approach (embedding similarity to a biomedical prototype, or a fastText model) is significantly more robust but out of scope for v1.0. Document in `known-limitations.md` (already done).

- 🚫 **BM25 score threshold corpus drift**
  The `BM25_SCORE_THRESHOLD=90.0` gate will shift as the corpus grows. This is mitigated by hybrid being off by default. When hybrid is enabled for production (v2.0 Atlas migration), replace the score gate with Atlas `$rankFusion` and rely solely on Gate 1 (entity detection).

- 🚫 **`_RE_GENE` false positives on acronyms**
  The gene-symbol regex matches FDA, WHO, USA, DNA, COVID, ECOG. Cosmetic issue since hybrid is off. Rename to `_RE_UPPERCASE_ENTITY` and add a denylist at v2.0.

- 🚫 **`_RRF_K` env var not wired to callers**
  `RRF_K` is read but not passed to `rrf_fuse()` calls. Dead config until hybrid goes live.

- 🚫 **Type annotation tightening**
  `detail: dict` → `dict[str, Any]` in `GuardrailResult`. `chunks: list[dict]` → `list[dict[str, Any]]`. Low priority cosmetic improvement.

---

## How to use this checklist

1. Before opening any clinical pilot or beta, Tier 1 items must all be checked off.
2. Tier 2 items should be complete before onboarding more than 5 users.
3. Tier 3 items are tracked here so they are not forgotten — they are not currently blocking.
4. When each item is completed, mark it `✅`, add the PR number and date.

---

*Last updated: 2026-06-29 — post code review (Antigravity + Codex, high effort)*
