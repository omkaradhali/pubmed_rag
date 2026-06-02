"""
rerank.py — Cross-encoder reranking of retrieved children (v0.2, Day 18).

Second retrieval stage. The first stage (retrieve.py) uses the bi-encoder from
embed.py to cheaply shortlist a POOL of candidate child chunks. This module
re-scores each (query, child_text) pair with a cross-encoder and reorders the
pool by relevance. retrieve.py then dedupes by parent and expands the winners
to parent text for the LLM.

Why a cross-encoder beats the bi-encoder score it reorders:
  * The bi-encoder (embed.py) encodes the query and each chunk INDEPENDENTLY
    into vectors, then compares with cosine. It never sees the two together,
    so it can only measure generic semantic closeness.
  * A cross-encoder feeds [query, chunk] through the transformer TOGETHER, so
    every query token can attend to every chunk token. Far more discriminative
    for ranking — but O(pool) forward passes per query, too slow to run over
    the whole corpus. Hence two stages: cheap bi-encoder to shortlist, costly
    cross-encoder to order the shortlist.

This is the lever that most directly targets RAGAS context_precision, which is
an order-dependent metric (a relevant chunk buried below an irrelevant one
tanks the score). See the Day 18 research synthesis.

Model: ncbi/MedCPT-Cross-Encoder (default) — a PubMedBERT cross-encoder trained
on 255M PubMed search-log query-article pairs. Purpose-built for reranking
PubMed abstracts (~109M params, 512-token cap, CPU-feasible). The output is a
single relevance logit per pair; higher = more relevant. Override the model via
the RERANK_MODEL env var.

Lazy-loaded singleton like embed.py: the model (~400MB) loads on the first
scoring call, not at import, and transformers/torch are imported inside the
loader. Importing this module is therefore cheap and side-effect-free, so unit
tests can patch score_pairs/rerank without downloading weights.

Public API:
    RERANK_MODEL_NAME            — active cross-encoder id (env-overridable)
    get_reranker()               — (model, tokenizer) singleton
    score_pairs(query, passages) — one relevance logit per passage
    rerank(query, candidates, text_key, top_k) — reordered candidate dicts
"""

import logging
import os

_logger = logging.getLogger(__name__)

RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "ncbi/MedCPT-Cross-Encoder")

# MedCPT (and most BERT cross-encoders) cap at 512 tokens. Children are ~1000
# chars (~250 tokens) so truncation rarely fires, but we set it defensively.
_MAX_LENGTH = 512

# Cached singletons — None until the first scoring call.
_model = None
_tokenizer = None


def get_reranker():
    """
    Load and cache the cross-encoder model + tokenizer (lazy singleton).

    transformers/torch are imported here, not at module top, so importing
    rerank.py stays cheap and tests can stub score_pairs without the heavy
    dependency or a model download.
    """
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _logger.info("Loading reranker %s (first call)...", RERANK_MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_NAME)
        _model.eval()  # disable dropout — we only ever do inference
    return _model, _tokenizer


def score_pairs(query: str, passages: list[str]) -> list[float]:
    """
    Score each (query, passage) pair with the cross-encoder.

    Returns one relevance logit per passage, in the same order as the input.
    Higher = more relevant. Returns [] for an empty passage list.

    All pairs are tokenized and scored in a single batched forward pass under
    torch.no_grad() — no gradients, no autograd graph, lower memory.
    """
    if not passages:
        return []

    import torch

    model, tokenizer = get_reranker()
    pairs = [[query, passage] for passage in passages]

    encoded = tokenizer(
        pairs,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=_MAX_LENGTH,
    )

    with torch.no_grad():
        # logits shape is (N, 1) for a single-label regression head; squeeze
        # the label dim to get a flat (N,) of per-passage scores.
        logits = model(**encoded).logits.squeeze(dim=1)

    return logits.tolist()


def rerank(
    query: str,
    candidates: list[dict],
    text_key: str = "child_text",
    top_k: int | None = None,
) -> list[dict]:
    """
    Reorder candidate dicts by cross-encoder relevance to the query.

    Scores each candidate[text_key] jointly with the query, attaches the score
    as candidate["rerank_score"], and returns the candidates sorted by that
    score descending. Does not mutate the inputs — each returned dict is a
    shallow copy with the extra key.

    Args:
        query:      The natural-language query.
        candidates: First-stage result dicts to reorder.
        text_key:   Which field to score. Defaults to "child_text" — we rerank
                    the small, precise child fragments (sharp signal), not the
                    larger parent text (retrieve.py expands to parents after).
        top_k:      If given, return only the top_k after sorting; else all.

    Returns:
        Reordered list of candidate dicts (copies, with "rerank_score" added).
    """
    if not candidates:
        return []

    passages = [c[text_key] for c in candidates]
    scores = score_pairs(query, passages)

    ranked = [{**c, "rerank_score": round(float(s), 4)} for c, s in zip(candidates, scores)]
    ranked.sort(key=lambda c: c["rerank_score"], reverse=True)

    return ranked[:top_k] if top_k is not None else ranked
