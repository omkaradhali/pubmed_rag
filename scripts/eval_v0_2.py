"""
One-shot eval driver for the v0.2 RAGAS re-run.

Reuses run_single + build_dataset + score_dataset from eval/evaluate.py and
handles the two issues from the first attempt:

  1. RAGAS 0.4.x returns a DataFrame with `user_input` / `response` /
     `retrieved_contexts` / `reference` instead of the 0.1.x names — save
     CSV with whatever columns exist, then print defensively.
  2. Default RAGAS timeout is too short for parent-resolved contexts (parents
     are ~5× the chars of v0.1 children). Bump it via RunConfig.

CSV is saved BEFORE printing so a print bug never loses the run.

Run:
  .venv/bin/python scripts/eval_v0_2.py --output eval/results_v0_2.csv
"""

import argparse
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("eval_v0_2")


def score_with_timeout(dataset, llm, timeout_s: int = 600):
    """Same as eval.evaluate.score_dataset but with a longer per-job timeout.

    Defaults match the known-good config from cf83e19: timeout=600, workers=2.
    The faithfulness and context_precision metrics each make several sequential
    judge calls per question; at 120s/4-workers they time out and silently
    return NaN (observed 2026-06-02). 600s/2-workers is the safe envelope.
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
    return result.to_pandas()


def print_defensive(df) -> None:
    """Print results without assuming RAGAS column names — auto-detect."""
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
    print()
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("eval/results_v0_2.csv"))
    parser.add_argument("--timeout", type=int, default=600, help="RAGAS per-job timeout, seconds")
    args = parser.parse_args()

    llm = build_evaluator_llm()
    _logger.info(
        "v0.2 RAGAS eval — %d questions, judge: %s, timeout: %ds",
        len(EVAL_QUESTIONS),
        os.getenv("RAGAS_EVAL_MODEL", "claude-haiku-4-5-20251001"),
        args.timeout,
    )

    rows = []
    for i, item in enumerate(EVAL_QUESTIONS):
        _logger.info("Q %d/%d: %s", i + 1, len(EVAL_QUESTIONS), item["question"][:60])
        rows.append(run_single(item["question"], item["ground_truth"]))

    dataset = build_dataset(rows)

    _logger.info("Scoring (this can take 10-15 min with parent-context judging)...")
    df = score_with_timeout(dataset, llm, timeout_s=args.timeout)

    # SAVE FIRST — never let a print bug lose the run.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    _logger.info("Saved → %s", args.output)

    print_defensive(df)


if __name__ == "__main__":
    main()
