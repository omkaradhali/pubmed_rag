"""
One-shot eval driver for the v0.2 RAGAS + deterministic retrieval re-run.

Combines two complementary metric families:

  RAGAS (LLM-as-judge, Haiku):
    faithfulness       — answer grounded in retrieved context
    answer_relevancy   — answer addresses the question
    context_precision  — relevant chunks ranked above irrelevant ones

  Deterministic retrieval metrics (zero variance, requires gold_pmids):
    recall@k           — fraction of gold PMIDs found in top-k (k=5,10,20)
    MRR                — reciprocal rank of first relevant result
    nDCG@k             — position-discounted ranking quality

Usage:
  # Smoke-test on the original 20 inline questions (no gold, RAGAS only):
  .venv/bin/python scripts/eval_v0_2.py --output eval/results_v0_2.csv

  # Full eval on the 110Q private set (RAGAS + retrieval metrics):
  .venv/bin/python scripts/eval_v0_2.py \
      --questions /path/to/questions.jsonl \
      --output eval/results_100q.csv

  # Deterministic-only (fast, no LLM judge cost):
  .venv/bin/python scripts/eval_v0_2.py \
      --questions /path/to/questions.jsonl \
      --output eval/results_100q.csv \
      --no-ragas

Environment:
  EVAL_QUESTIONS_PATH   optional default path for --questions
  RERANK_ENABLED        set to "false" for dense-only retrieval (default: true)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Make eval/ importable as a module
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from evaluate import (  # noqa: E402
    EVAL_QUESTIONS,
    build_dataset,
    build_evaluator_llm,
    run_single,
)
from langchain_community.embeddings import HuggingFaceEmbeddings as LCHFEmbeddings  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.metrics._answer_relevance import answer_relevancy  # noqa: E402
from ragas.metrics._context_precision import context_precision  # noqa: E402
from ragas.metrics._faithfulness import faithfulness  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402
from retrieval_metrics import score_question  # noqa: E402

from pubmed_rag.retrieve import retrieve  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("eval_v0_2")

_RETRIEVAL_N = 20  # top-k to retrieve when computing recall@k / MRR / nDCG
_RAGAS_N = 5  # top-k passed to the LLM for RAGAS scoring (unchanged)


# ── Question loading ──────────────────────────────────────────────────────────


def load_questions(path: Path) -> list[dict]:
    """Load questions from a JSONL file (one JSON object per line).

    Each object must have at minimum: question, ground_truth.
    Optional fields used here: gold_pmids, answerable, type, difficulty, cancer.
    """
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    _logger.info("Loaded %d questions from %s", len(questions), path)
    return questions


# ── RAGAS scoring ─────────────────────────────────────────────────────────────


def score_ragas(dataset, llm, timeout_s: int = 600):
    """Run RAGAS with the known-good timeout config (600s/2-workers).

    Saves the DataFrame from whatever column names the installed ragas version
    uses (0.1.x vs 0.4.x differ) and returns with legacy names so downstream
    code stays stable.
    """
    embeddings = LangchainEmbeddingsWrapper(
        LCHFEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    faithfulness.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings
    context_precision.llm = llm

    run_config = RunConfig(timeout=timeout_s, max_retries=3, max_workers=2)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        run_config=run_config,
    )
    df = result.to_pandas()
    # ragas 0.4.x renames columns; normalise back to legacy names
    df = df.rename(
        columns={
            "user_input": "question",
            "response": "answer",
            "retrieved_contexts": "contexts",
            "reference": "ground_truth",
        }
    )
    return df


# ── Retrieval metric computation ──────────────────────────────────────────────


def compute_retrieval_metrics(questions: list[dict]) -> dict[str, dict]:
    """Run retrieve() at n=20 for each answerable question and score vs gold.

    Returns a dict keyed by question text with retrieval metric dicts.
    Questions that are unanswerable or have no gold_pmids are skipped
    (their metrics default to 0.0 when merged into the final CSV).
    """
    results: dict[str, dict] = {}
    answerable = [q for q in questions if q.get("answerable", True) and q.get("gold_pmids")]
    _logger.info(
        "Computing retrieval metrics for %d/%d questions with gold labels...",
        len(answerable),
        len(questions),
    )
    for i, q in enumerate(answerable):
        _logger.info("Retrieval %d/%d: %s", i + 1, len(answerable), q["question"][:60])
        hits = retrieve(q["question"], n_results=_RETRIEVAL_N)
        retrieved_pmids = [h["pmid"] for h in hits]
        scores = score_question(retrieved_pmids, q["gold_pmids"])
        results[q["question"]] = scores
    return results


# ── Printing helpers ──────────────────────────────────────────────────────────


def print_ragas_summary(df) -> None:
    """Print per-question RAGAS scores + mean row."""
    q_col = next(
        (c for c in ("question", "user_input", "query") if c in df.columns),
        df.columns[0],
    )
    metric_cols = [
        c for c in ("faithfulness", "answer_relevancy", "context_precision") if c in df.columns
    ]
    col_w = 60
    sep = "-" * (col_w + 12 * len(metric_cols))
    header = f"{q_col:<{col_w}}  " + "  ".join(f"{m[:6]:>8}" for m in metric_cols)
    print("\n=== RAGAS (LLM-judge) ===")
    print(header)
    print(sep)
    for _, row in df.iterrows():
        q = str(row[q_col])[:col_w]
        vals = "  ".join(
            f"{row[m]:>8.3f}" if row[m] == row[m] else f"{'NaN':>8}" for m in metric_cols
        )
        print(f"{q:<{col_w}}  {vals}")
    print(sep)
    means = df[metric_cols].mean()
    means_str = "  ".join(f"{means[m]:>8.3f}" for m in metric_cols)
    print(f"{'MEAN':<{col_w}}  {means_str}")
    print()


def print_retrieval_summary(ret_scores: dict[str, dict]) -> None:
    """Print deterministic retrieval metric means."""
    if not ret_scores:
        return
    all_metrics: dict[str, list[float]] = {}
    for scores in ret_scores.values():
        for k, v in scores.items():
            all_metrics.setdefault(k, []).append(v)
    col_w = 60
    metric_keys = ["recall@5", "recall@10", "recall@20", "mrr", "ndcg@5", "ndcg@10", "ndcg@20"]
    metric_keys = [k for k in metric_keys if k in all_metrics]
    sep = "-" * (col_w + 10 * len(metric_keys))
    header = f"{'question':<{col_w}}  " + "  ".join(f"{m:>8}" for m in metric_keys)
    print("\n=== Deterministic retrieval metrics (zero variance) ===")
    print(header)
    print(sep)
    for question, scores in ret_scores.items():
        q = question[:col_w]
        vals = "  ".join(f"{scores.get(m, 0.0):>8.3f}" for m in metric_keys)
        print(f"{q:<{col_w}}  {vals}")
    print(sep)
    means_str = "  ".join(f"{sum(all_metrics[m]) / len(all_metrics[m]):>8.3f}" for m in metric_keys)
    print(f"{'MEAN':<{col_w}}  {means_str}")
    print(f"  (n={len(ret_scores)} questions with gold labels)\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=os.getenv("EVAL_QUESTIONS_PATH"),
        help="Path to questions.jsonl (overrides inline EVAL_QUESTIONS). "
        "Can also be set via EVAL_QUESTIONS_PATH env var.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results_v0_2.csv"),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="RAGAS per-job timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip RAGAS scoring — compute only deterministic retrieval metrics.",
    )
    args = parser.parse_args()

    # Guard: check ChromaDB dimension matches EMBEDDING_PROVIDER before loading
    # the model or running any questions — fails fast with a clear fix command.
    from pubmed_rag.vectorstore import validate_collection_dimension

    validate_collection_dimension()

    # Load questions from file or fall back to the 20 inline questions
    if args.questions:
        questions = load_questions(args.questions)
    else:
        questions = EVAL_QUESTIONS
        _logger.info("No --questions file provided; using %d inline questions.", len(questions))

    # ── Deterministic retrieval metrics (fast, no LLM cost) ──────────────────
    ret_scores = compute_retrieval_metrics(questions)

    if args.no_ragas:
        # Build a minimal DataFrame from retrieval scores only and save
        import pandas as pd

        rows = []
        for q in questions:
            row = {
                "question": q["question"],
                "type": q.get("type", ""),
                "difficulty": q.get("difficulty", ""),
                "cancer": q.get("cancer", ""),
                "answerable": q.get("answerable", True),
                "gold_pmids": json.dumps(q.get("gold_pmids", [])),
            }
            row.update(ret_scores.get(q["question"], {}))
            rows.append(row)
        df_ret = pd.DataFrame(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df_ret.to_csv(args.output, index=False)
        _logger.info("Saved → %s", args.output)
        print_retrieval_summary(ret_scores)
        return

    # ── RAGAS scoring ─────────────────────────────────────────────────────────
    llm = build_evaluator_llm()
    _logger.info(
        "v0.2 RAGAS eval — %d questions, judge: %s, timeout: %ds",
        len(questions),
        os.getenv("RAGAS_EVAL_MODEL", "claude-haiku-4-5-20251001"),
        args.timeout,
    )

    rows = []
    for i, item in enumerate(questions):
        _logger.info("Q %d/%d: %s", i + 1, len(questions), item["question"][:60])
        rows.append(run_single(item["question"], item.get("ground_truth", "")))

    dataset = build_dataset(rows)

    _logger.info("Scoring RAGAS (10-15 min with parent-context judging)...")
    df = score_ragas(dataset, llm, timeout_s=args.timeout)

    # Attach metadata columns from the questions file
    for col in ("type", "difficulty", "cancer", "answerable"):
        df[col] = [q.get(col, "") for q in questions]
    df["gold_pmids"] = [json.dumps(q.get("gold_pmids", [])) for q in questions]

    # Attach deterministic metric columns (NaN for questions without gold)
    metric_keys = ["recall@5", "recall@10", "recall@20", "mrr", "ndcg@5", "ndcg@10", "ndcg@20"]
    for mk in metric_keys:
        df[mk] = [ret_scores.get(q["question"], {}).get(mk, float("nan")) for q in questions]

    # SAVE FIRST — never let a print bug lose the run
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    _logger.info("Saved → %s", args.output)

    print_ragas_summary(df)
    print_retrieval_summary(ret_scores)


if __name__ == "__main__":
    main()
