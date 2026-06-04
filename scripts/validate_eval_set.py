"""
validate_eval_set.py — integrity checks for the frozen evaluation set.

The frozen 110Q set is PRIVATE (claude-brain), so this script takes the path as
an argument / env var rather than hardcoding a repo location:

  EVAL_QUESTIONS_PATH=/mnt/.../eval100q/questions.jsonl \
      .venv/bin/python scripts/validate_eval_set.py

  # or explicitly, and validating anchored gold against the corpus:
  .venv/bin/python scripts/validate_eval_set.py \
      --questions /mnt/.../questions.jsonl --corpus data/abstracts.jsonl

Exits non-zero if any check fails. The same script validates the public sample
(eval/questions.sample.jsonl), which has no unanswerable rows — pass
--no-require-unanswerable for that case.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

TYPES = ("mechanism", "biomarker", "prognosis", "treatment", "epidemiology")
MODES = ("canonical", "anchored", "unanswerable")
REQUIRED_FIELDS = (
    "id",
    "type",
    "difficulty",
    "cancer",
    "question",
    "ground_truth",
    "gold_pmids",
    "answerable",
    "mode",
    "needs_pooling",
)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def validate(
    rows: list[dict], corpus_pmids: set[str] | None, require_unanswerable: int | None
) -> list[str]:
    """Return a list of error strings (empty == valid)."""
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(msg)

    # 1. Schema: required fields present, no duplicate ids.
    ids = [r.get("id") for r in rows]
    for dup, n in Counter(ids).items():
        if n > 1:
            err(f"duplicate id {dup!r} ({n}x)")
    for r in rows:
        missing = [f for f in REQUIRED_FIELDS if f not in r]
        if missing:
            err(f"{r.get('id', '?')}: missing fields {missing}")
        if r.get("type") not in TYPES:
            err(f"{r.get('id')}: bad type {r.get('type')!r}")
        if r.get("mode") not in MODES:
            err(f"{r.get('id')}: bad mode {r.get('mode')!r}")

    # 2. No duplicate question text (case-insensitive).
    for q, n in Counter(r["question"].strip().lower() for r in rows if r.get("question")).items():
        if n > 1:
            err(f"duplicate question ({n}x): {q[:70]}...")

    # 3. Non-empty ground_truth everywhere.
    for r in rows:
        if not str(r.get("ground_truth", "")).strip():
            err(f"{r.get('id')}: empty ground_truth")

    # 4. answerable <-> gold_pmids / mode consistency.
    for r in rows:
        rid, mode, answerable = r.get("id"), r.get("mode"), r.get("answerable")
        gold = r.get("gold_pmids", [])
        if mode == "unanswerable":
            if answerable is not False:
                err(f"{rid}: unanswerable must have answerable=false")
            if gold:
                err(f"{rid}: unanswerable must have empty gold_pmids")
        elif mode == "canonical":
            if answerable is not True:
                err(f"{rid}: canonical must be answerable=true")
            if gold and not r.get("needs_pooling"):
                err(f"{rid}: canonical with gold but needs_pooling=false")
            if not gold and not r.get("needs_pooling"):
                err(f"{rid}: canonical with no gold must have needs_pooling=true")
        elif mode == "anchored":
            if answerable is not True:
                err(f"{rid}: anchored must be answerable=true")
            if not gold:
                err(f"{rid}: anchored must have >=1 gold_pmid")
            if r.get("needs_pooling"):
                err(f"{rid}: anchored must have needs_pooling=false")

    # 5. Anchored gold PMIDs exist in the corpus (only when corpus provided).
    if corpus_pmids is not None:
        for r in rows:
            if r.get("mode") == "anchored":
                for pmid in r.get("gold_pmids", []):
                    if str(pmid) not in corpus_pmids:
                        err(f"{r.get('id')}: gold_pmid {pmid} not in corpus")

    # 6. Balance: 20 per type among the 100 answerable; unanswerable count.
    answerable = [r for r in rows if r.get("answerable")]
    if require_unanswerable is not None:
        n_unans = sum(1 for r in rows if not r.get("answerable"))
        if n_unans != require_unanswerable:
            err(f"expected {require_unanswerable} unanswerable, found {n_unans}")
        type_counts = Counter(r["type"] for r in answerable)
        for t in TYPES:
            if type_counts[t] != 20:
                err(f"type {t}: expected 20 answerable, found {type_counts[t]}")
        mode_counts = Counter(r["mode"] for r in answerable)
        if mode_counts["canonical"] != 70 or mode_counts["anchored"] != 30:
            err(
                f"expected 70 canonical / 30 anchored, found "
                f"{mode_counts['canonical']} / {mode_counts['anchored']}"
            )

    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    default_q = os.getenv("EVAL_QUESTIONS_PATH")
    ap.add_argument(
        "--questions",
        type=Path,
        default=Path(default_q) if default_q else None,
        required=default_q is None,
    )
    ap.add_argument("--corpus", type=Path, default=Path("data/abstracts.jsonl"))
    ap.add_argument(
        "--no-require-unanswerable",
        action="store_true",
        help="skip the 110Q balance checks (use for the public sample)",
    )
    args = ap.parse_args()

    rows = _load_jsonl(args.questions)
    corpus_pmids = None
    if args.corpus.exists():
        corpus_pmids = {
            str(json.loads(line)["pmid"]) for line in args.corpus.open() if line.strip()
        }
        print(f"corpus: {len(corpus_pmids)} PMIDs ({args.corpus})")
    else:
        print(f"corpus not found ({args.corpus}) — skipping gold-PMID existence check")

    require = None if args.no_require_unanswerable else 10
    errors = validate(rows, corpus_pmids, require)

    print(f"validated {len(rows)} questions from {args.questions}")
    if errors:
        print(f"\nFAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("OK — all checks passed.")


if __name__ == "__main__":
    main()
