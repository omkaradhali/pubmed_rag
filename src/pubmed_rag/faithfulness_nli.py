"""
faithfulness_nli.py — NLI-based faithfulness check.

Second faithfulness pass after the lexical Jaccard check in guardrails.py.
Catches contradictions that share most of their tokens with the source and so
slip past unigram overlap — negation ("is effective" vs "is not effective") and
entity swaps ("tachycardia" vs "bradycardia"). Both are dangerous clinical
hallucinations the lexical check scores as near-identical.

For each cited sentence in an answer, a natural-language-inference cross-encoder
scores (premise = source chunk, hypothesis = cited sentence) and we report the
pairs the model labels CONTRADICTION above a confidence threshold.

Model: cross-encoder/nli-deberta-v3-small — DeBERTa-v3 fine-tuned on SNLI+MNLI,
3-way {contradiction, entailment, neutral}. ~140M params, CPU-feasible
(~50ms/pair). Override via the NLI_MODEL env var.

Lazy-loaded singleton like rerank.py: torch/transformers are imported inside the
loader and the weights load on the first scoring call, so importing this module
stays cheap and side-effect-free — unit tests patch contradiction_scores (or
get_nli_model) without downloading weights.

Public API:
    NLI_MODEL_NAME                                   — active model id (env-overridable)
    CONTRADICTION_THRESHOLD                          — default P(contradiction) block cutoff
    get_nli_model()                                  — (model, tokenizer) singleton
    contradiction_scores(pairs)                      — P(contradiction) per (premise, hypothesis)
    find_contradictions(answer, chunks, threshold)   — flagged cited claims
"""

import logging
import os
import re

_logger = logging.getLogger(__name__)

NLI_MODEL_NAME = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")

# DeBERTa-v3 caps at 512 tokens; a source chunk plus one sentence rarely exceeds
# it, but we truncate defensively (the premise is truncated first by the pair).
_MAX_LENGTH = 512

# P(contradiction) at or above this blocks the claim. 0.8 keeps the check
# conservative — only high-confidence contradictions trip it, so paraphrase and
# neutral elaboration are not punished.
CONTRADICTION_THRESHOLD = float(os.getenv("NLI_CONTRADICTION_THRESHOLD", "0.8"))

# Citation marker — mirrors guardrails._CITATION_RE. Duplicated (not imported)
# so this module never imports guardrails: guardrails imports THIS module, and a
# reverse import would create a cycle.
_CITATION_RE = re.compile(r"\[(\d{1,5})\]")

# Cached singletons — None until the first scoring call.
_model = None
_tokenizer = None
_contradiction_idx: int | None = None


def _resolve_contradiction_index(config) -> int:
    """
    Return the logit index of the CONTRADICTION label from the model config.

    Reading id2label instead of hardcoding an index guards against NLI models
    that order their three labels differently.
    """
    id2label = getattr(config, "id2label", None) or {}
    for idx, label in id2label.items():
        if str(label).lower().startswith("contradict"):
            return int(idx)
    raise ValueError(
        f"NLI model {NLI_MODEL_NAME} exposes no 'contradiction' label; id2label={id2label!r}"
    )


def get_nli_model():
    """
    Load and cache the NLI cross-encoder + tokenizer (lazy singleton).

    transformers/torch are imported here, not at module top, so importing
    faithfulness_nli.py stays cheap and tests can stub the scorer without the
    heavy dependency or a model download.
    """
    global _model, _tokenizer, _contradiction_idx
    if _model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _logger.info("Loading NLI model %s (first call)...", NLI_MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        _model.eval()  # inference only — disable dropout
        _contradiction_idx = _resolve_contradiction_index(_model.config)
    return _model, _tokenizer


def contradiction_scores(pairs: list[tuple[str, str]]) -> list[float]:
    """
    Return P(contradiction) for each (premise, hypothesis) pair.

    premise = source evidence, hypothesis = the cited claim; a high score means
    the source refutes the claim. Order matches the input. Returns [] for empty
    input. All pairs run in one batched forward pass under torch.no_grad().
    """
    if not pairs:
        return []

    import torch

    model, tokenizer = get_nli_model()
    premises = [premise for premise, _ in pairs]
    hypotheses = [hypothesis for _, hypothesis in pairs]

    encoded = tokenizer(
        premises,
        hypotheses,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=_MAX_LENGTH,
    )

    with torch.no_grad():
        logits = model(**encoded).logits  # (N, 3)
        probs = torch.softmax(logits, dim=1)

    return probs[:, _contradiction_idx].tolist()


def find_contradictions(
    answer: str,
    chunks: list[dict],
    threshold: float = CONTRADICTION_THRESHOLD,
) -> list[dict]:
    """
    Flag cited sentences the NLI model finds contradicted by their source.

    Splits the answer into sentences and, for each sentence carrying an [N]
    citation, pairs it with chunk N's text (premise) and scores P(contradiction).
    Returns one dict per (sentence, source) pair scoring at or above threshold.

    Citations out of range are skipped. Returns [] when there are no chunks or
    no cited sentences — the caller treats an empty list as "no contradiction".
    """
    if not chunks:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())

    pairs: list[tuple[str, str]] = []
    located: list[tuple[str, int]] = []  # (sentence, source_n), aligned with pairs
    for sentence in sentences:
        for n in sorted({int(x) for x in _CITATION_RE.findall(sentence)}):
            if n < 1 or n > len(chunks):
                continue
            source_text = chunks[n - 1].get("text", "")
            if not source_text:
                continue
            pairs.append((source_text, sentence))
            located.append((sentence, n))

    if not pairs:
        return []

    scores = contradiction_scores(pairs)

    flagged: list[dict] = []
    for (sentence, n), score in zip(located, scores):
        if score >= threshold:
            flagged.append(
                {
                    "sentence": sentence[:200],
                    "source_n": n,
                    "contradiction": round(float(score), 4),
                }
            )
    return flagged
