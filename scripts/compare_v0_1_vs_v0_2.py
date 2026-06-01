"""
Side-by-side comparison of v0.1 (flat chunks) vs v0.2 (parent-child) RAGAS results.

Reads two CSVs and emits:
  * Aggregate means with deltas
  * Per-question diff table with arrows (↑ ↓ →)
  * Class-A vs Class-B breakdown (questions where v0.1 retrieval worked vs failed)
  * Honest caveat about the ragas 0.1.21 → 0.4.3 version bump

Defensive about column names — RAGAS 0.1.x used `question`, 0.4.x uses
`user_input`. The script auto-detects.

Run:
  .venv/bin/python scripts/compare_v0_1_vs_v0_2.py \\
      --v01 eval/results.csv \\
      --v02 eval/results_v0_2.csv
"""

import argparse
from pathlib import Path

import pandas as pd

METRICS = ("faithfulness", "answer_relevancy", "context_precision")


def detect_q_col(df: pd.DataFrame) -> str:
    for c in ("question", "user_input", "query"):
        if c in df.columns:
            return c
    raise KeyError(f"No question column in {df.columns.tolist()}")


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    q_col = detect_q_col(df)
    df = df.rename(columns={q_col: "question"})
    # Normalise NaN handling — RAGAS NaN means "no claims to score"
    return df


def fmt_delta(old: float, new: float) -> str:
    if pd.isna(old) and pd.isna(new):
        return "  —  "
    if pd.isna(old):
        return f"  new={new:.3f}"
    if pd.isna(new):
        return f"  lost={old:.3f}"
    diff = new - old
    arrow = "↑" if diff > 0.005 else "↓" if diff < -0.005 else "→"
    return f"{arrow}{diff:+.3f}"


def print_aggregate(v01: pd.DataFrame, v02: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("AGGREGATE — v0.1 (flat 1000-char chunks) vs v0.2 (parent-child)")
    print("=" * 78)
    print(f"{'Metric':<22}  {'v0.1 mean':>12}  {'v0.2 mean':>12}  {'Δ':>12}")
    print("-" * 78)
    for m in METRICS:
        m1 = v01[m].mean(skipna=True)
        m2 = v02[m].mean(skipna=True)
        used1 = v01[m].notna().sum()
        used2 = v02[m].notna().sum()
        diff = m2 - m1
        arrow = "↑" if diff > 0.005 else "↓" if diff < -0.005 else "→"
        print(
            f"{m:<22}  {m1:>8.3f} ({used1:>2}/20)  {m2:>8.3f} ({used2:>2}/20)  {arrow}{diff:+.3f}"
        )
    print()


def print_per_question(v01: pd.DataFrame, v02: pd.DataFrame) -> None:
    print("=" * 100)
    print("PER-QUESTION DELTA  (Δ shown as new − old)")
    print("=" * 100)
    merged = v01[["question", *METRICS]].merge(
        v02[["question", *METRICS]],
        on="question",
        suffixes=("_v01", "_v02"),
    )
    col_w = 56
    header = f"{'Question':<{col_w}}  " + "  ".join(f"{m[:5]:>5}Δ" for m in METRICS)
    print()
    print(header)
    print("-" * (col_w + 6 * len(METRICS) + 8))
    for _, row in merged.iterrows():
        q = str(row["question"])[:col_w]
        deltas = "  ".join(fmt_delta(row[f"{m}_v01"], row[f"{m}_v02"]) for m in METRICS)
        print(f"{q:<{col_w}}  {deltas}")
    print()


def print_class_breakdown(v01: pd.DataFrame, v02: pd.DataFrame) -> None:
    """Bucket questions by whether v0.1 retrieval worked (context_precision > 0)."""
    print("=" * 78)
    print("CLASS BREAKDOWN — where the v0.2 gains landed")
    print("=" * 78)
    merged = (
        v01[["question", "context_precision"]]
        .merge(
            v02[["question", *METRICS]],
            on="question",
        )
        .merge(
            v01[METRICS].rename(columns={m: f"{m}_v01" for m in METRICS}),
            left_index=True,
            right_index=True,
        )
    )
    class_a = merged[merged["context_precision"] > 0.01]
    class_b = merged[merged["context_precision"] <= 0.01]

    for label, sub in (
        ("Class A (v0.1 retrieval worked)", class_a),
        ("Class B (v0.1 retrieval missed)", class_b),
    ):
        print(f"\n{label} — n={len(sub)}")
        for m in METRICS:
            old = sub[f"{m}_v01"].mean(skipna=True)
            new = sub[m].mean(skipna=True)
            diff = new - old if not (pd.isna(old) or pd.isna(new)) else float("nan")
            print(f"  {m:<22}  v0.1={old:>6.3f}  v0.2={new:>6.3f}  Δ={diff:+.3f}")
    print()


def print_caveats() -> None:
    print("=" * 78)
    print("CAVEATS")
    print("=" * 78)
    print("""
1. ragas version bump (0.1.21 → 0.4.3) between v0.1 and v0.2 runs. Metric
   internals changed; the comparison isn't purely about parent-child chunking.
2. Generator (Claude Haiku) is identical across both runs.
3. Corpus identical — both runs use the same data/abstracts.jsonl snapshot.
4. v0.2 contexts are PARENT text (~5x v0.1 chunk size) — this is the change
   under test. The judge sees more context per chunk.
5. v0.2 dedups by parent_id — fewer unique sources per question but each more
   informative.
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v01", type=Path, default=Path("eval/results.csv"))
    parser.add_argument("--v02", type=Path, default=Path("eval/results_v0_2.csv"))
    args = parser.parse_args()

    v01 = load(args.v01)
    v02 = load(args.v02)

    print_aggregate(v01, v02)
    print_per_question(v01, v02)
    print_class_breakdown(v01, v02)
    print_caveats()


if __name__ == "__main__":
    main()
