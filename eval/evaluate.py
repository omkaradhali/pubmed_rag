"""
evaluate.py — RAGAS evaluation for pubmed_rag.

Runs 20 clinical oncology questions through the pipeline and scores them
across three RAGAS metrics:
    faithfulness       — does the answer stay within the retrieved context?
    answer_relevancy   — does the answer address the actual question?
    context_precision  — were the right chunks retrieved?

Prerequisites:
    uv pip install -e ".[eval]"        # installs ragas + datasets
    ANTHROPIC_API_KEY set in .env      # RAGAS uses Claude as its internal judge
    RAGAS_EVAL_MODEL (optional)        # defaults to claude-haiku-4-5-20251001

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --output eval/results.csv
"""

import argparse
import logging
import os
from pathlib import Path

from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings as LCHFEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._faithfulness import faithfulness

from pubmed_rag.pipeline import run_pipeline_structured

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger(__name__)

_DEFAULT_EVAL_MODEL = "claude-haiku-4-5-20251001"


# ── LLM builder ───────────────────────────────────────────────────────────────


def build_evaluator_llm():
    """
    Build the Claude LLM that RAGAS uses internally to judge answers.

    Reads ANTHROPIC_API_KEY from the environment (.env loaded by pipeline import).
    Model is controlled by RAGAS_EVAL_MODEL env var (default: claude-haiku-4-5-20251001).

    Returns:
        RAGAS-compatible LLM object to pass into each metric's constructor.
    """
    # llm_factory + instructor sends both temperature and top_p, which Anthropic rejects.
    # LangchainLLMWrapper + ChatAnthropic handles API parameters correctly.
    model = os.getenv("RAGAS_EVAL_MODEL", _DEFAULT_EVAL_MODEL)
    chat = ChatAnthropic(model=model, api_key=os.environ["ANTHROPIC_API_KEY"])
    return LangchainLLMWrapper(chat)


# ── Evaluation question set ────────────────────────────────────────────────────
# 20 questions spanning 5 types x multiple cancer types.
# ground_truth: short factual reference (1-2 sentences) — used by context_precision only.
# faithfulness + answer_relevancy are reference-free (no ground_truth needed).

EVAL_QUESTIONS: list[dict[str, str]] = [
    # ── Mechanism (4) ──────────────────────────────────────────────────────────
    {
        "question": "How does PD-1/PD-L1 checkpoint inhibition restore anti-tumor immunity?",
        "ground_truth": (
            "PD-1/PD-L1 checkpoint inhibitors block the interaction between PD-1 on T cells "
            "and PD-L1 on tumor cells, restoring cytotoxic T cell activity against the tumor."
        ),
    },
    {
        "question": "What is the mechanism of action of VEGF inhibitors in cancer treatment?",
        "ground_truth": (
            "VEGF inhibitors block vascular endothelial growth factor signaling, inhibiting "
            "tumor angiogenesis and starving tumors of the blood supply needed for growth."
        ),
    },
    {
        "question": "How does microsatellite instability-high (MSI-H) status affect immunotherapy response?",
        "ground_truth": (
            "MSI-H tumors have defective DNA mismatch repair, leading to high mutational burden "
            "and abundant neoantigens that enhance T cell recognition and response to checkpoint inhibitors."
        ),
    },
    {
        "question": "What is the role of BRCA1/BRCA2 mutations in homologous recombination deficiency?",
        "ground_truth": (
            "BRCA1/BRCA2 mutations impair homologous recombination DNA repair, creating genomic "
            "instability and sensitivity to PARP inhibitors."
        ),
    },
    # ── Biomarker (4) ──────────────────────────────────────────────────────────
    {
        "question": "Which biomarkers predict response to immune checkpoint inhibitors in solid tumors?",
        "ground_truth": (
            "PD-L1 expression, tumor mutational burden (TMB), and microsatellite instability-high "
            "(MSI-H) status are established predictive biomarkers for checkpoint inhibitor response."
        ),
    },
    {
        "question": "What is the role of HER2 overexpression in breast cancer treatment selection?",
        "ground_truth": (
            "HER2 overexpression or amplification identifies patients who benefit from HER2-targeted "
            "therapies such as trastuzumab, pertuzumab, and trastuzumab deruxtecan."
        ),
    },
    {
        "question": "How is EGFR mutation status used to guide treatment in non-small cell lung cancer?",
        "ground_truth": (
            "EGFR activating mutations such as exon 19 deletions and L858R predict response to "
            "EGFR tyrosine kinase inhibitors including erlotinib, gefitinib, and osimertinib."
        ),
    },
    {
        "question": "What is the significance of tumor mutational burden in cancer immunotherapy?",
        "ground_truth": (
            "High tumor mutational burden is associated with greater neoantigen load, enhanced "
            "T cell recognition, and improved response to immune checkpoint inhibitors."
        ),
    },
    # ── Prognosis (4) ──────────────────────────────────────────────────────────
    {
        "question": "What is the approximate 5-year survival rate for stage III colorectal cancer?",
        "ground_truth": (
            "The 5-year survival rate for stage III colorectal cancer ranges from approximately "
            "40% to 80% depending on substage, with stage IIIA having better outcomes than IIIC."
        ),
    },
    {
        "question": "What factors are associated with poor prognosis in pancreatic adenocarcinoma?",
        "ground_truth": (
            "Poor prognostic factors in pancreatic adenocarcinoma include advanced stage, "
            "lymph node involvement, elevated CA 19-9, and R1 resection margins."
        ),
    },
    {
        "question": "How does triple-negative breast cancer prognosis compare to other subtypes?",
        "ground_truth": (
            "Triple-negative breast cancer has a worse prognosis than hormone receptor-positive "
            "or HER2-positive subtypes due to higher early relapse rates and limited targeted therapies."
        ),
    },
    {
        "question": "Why does ovarian cancer have a high mortality rate despite being treatable?",
        "ground_truth": (
            "Ovarian cancer has high mortality primarily because most cases are diagnosed at "
            "advanced stage, when curative resection is often no longer possible."
        ),
    },
    # ── Treatment (5) ──────────────────────────────────────────────────────────
    {
        "question": (
            "What is the current first-line treatment for advanced non-small cell lung cancer "
            "without actionable mutations and high PD-L1 expression?"
        ),
        "ground_truth": (
            "For advanced NSCLC with PD-L1 >= 50% and no actionable mutations, "
            "pembrolizumab monotherapy is the standard first-line treatment."
        ),
    },
    {
        "question": "What chemotherapy regimens are used as first-line treatment for metastatic colorectal cancer?",
        "ground_truth": (
            "FOLFOX and FOLFIRI, often combined with bevacizumab or cetuximab, are standard "
            "first-line regimens for metastatic colorectal cancer."
        ),
    },
    {
        "question": "How are PARP inhibitors used in ovarian cancer treatment?",
        "ground_truth": (
            "PARP inhibitors such as olaparib, niraparib, and rucaparib are used as maintenance "
            "therapy in platinum-sensitive ovarian cancer, particularly in BRCA-mutated patients."
        ),
    },
    {
        "question": "What cancers are CAR-T cell therapies currently approved to treat?",
        "ground_truth": (
            "CAR-T cell therapies are approved for certain B-cell malignancies including "
            "large B-cell lymphoma and B-cell ALL, as well as multiple myeloma."
        ),
    },
    {
        "question": "What is the standard treatment for diffuse large B-cell lymphoma?",
        "ground_truth": (
            "R-CHOP (rituximab, cyclophosphamide, doxorubicin, vincristine, prednisone) is "
            "the standard first-line treatment, with CAR-T therapy for relapsed/refractory disease."
        ),
    },
    # ── Epidemiology / Other (3) ───────────────────────────────────────────────
    {
        "question": "What are the major risk factors for pancreatic cancer?",
        "ground_truth": (
            "Major risk factors for pancreatic cancer include smoking, obesity, type 2 diabetes, "
            "chronic pancreatitis, family history, and germline BRCA2 mutations."
        ),
    },
    {
        "question": "What is the global incidence of colorectal cancer?",
        "ground_truth": (
            "Colorectal cancer is the third most common cancer worldwide, with approximately "
            "1.9 million new cases diagnosed annually."
        ),
    },
    {
        "question": "How is liquid biopsy used in oncology?",
        "ground_truth": (
            "Liquid biopsy detects circulating tumor DNA in blood, enabling minimally invasive "
            "tumor genotyping, treatment response monitoring, and early detection of relapse."
        ),
    },
]


# ── Step 1: run_single ────────────────────────────────────────────────────────


def run_single(question: str, ground_truth: str) -> dict:
    """
    Run one question through the pipeline and return a RAGAS-ready row.

    Args:
        question:     Clinical question string.
        ground_truth: Short reference answer (used by context_precision only).

    Returns:
        Dict with keys: question, answer, contexts, ground_truth.
        `contexts` is list[str] — one entry per retrieved chunk's raw text.
    """
    result = run_pipeline_structured(query=question, mode="incremental", n_results=5)
    return {
        "question": question,
        "answer": result.answer,
        "contexts": [src.text for src in result.sources],
        "ground_truth": ground_truth,
    }


# ── Step 2: build_dataset ─────────────────────────────────────────────────────


def build_dataset(rows: list[dict]) -> Dataset:
    """
    Convert a list of rows into a HuggingFace Dataset for RAGAS.

    Args:
        rows: List of dicts from run_single() — keys:
              question, answer, contexts (list[str]), ground_truth.

    Returns:
        HuggingFace Dataset ready to pass to evaluate().
    """
    return Dataset.from_list(rows)


# ── Step 3: score_dataset ─────────────────────────────────────────────────────


def score_dataset(dataset: Dataset, llm):
    """
    Run RAGAS metrics and return a pandas DataFrame of per-question scores.

    Uses Claude as the internal LLM judge (passed in via `llm`).
    Each metric is a float in [0, 1]. Higher is better for all three.

    Args:
        dataset: HuggingFace Dataset from build_dataset().
        llm:     RAGAS-compatible LLM from build_evaluator_llm().

    Returns:
        pandas DataFrame with columns:
        question, faithfulness, answer_relevancy, context_precision.
    """
    # RAGAS 0.1.21 API: set llm + embeddings directly on each metric singleton.
    # AnswerRelevancy also needs an embedding model to compute question similarity.
    embeddings = LangchainEmbeddingsWrapper(
        LCHFEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    faithfulness.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings
    context_precision.llm = llm

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    return result.to_pandas()


# ── Step 4: print_results ─────────────────────────────────────────────────────


def print_results(df) -> None:
    """
    Print a formatted summary table to stdout.

    Shows per-question scores + a mean row at the bottom.
    Rows where faithfulness < 0.6 are flagged with a warning marker.

    Args:
        df: DataFrame from score_dataset().
    """
    col_w = 62
    sep = "-" * (col_w + 42)

    print(f"\n{'Question':<{col_w}}  {'Faith':>6}  {'Relev':>6}  {'Prec':>6}")
    print(sep)

    for _, row in df.iterrows():
        q = str(row["question"])[:col_w]
        faith = row.get("faithfulness", float("nan"))
        relev = row.get("answer_relevancy", float("nan"))
        prec = row.get("context_precision", float("nan"))
        flag = "  !" if faith < 0.6 else ""
        print(f"{q:<{col_w}}  {faith:>6.2f}  {relev:>6.2f}  {prec:>6.2f}{flag}")

    means = df[["faithfulness", "answer_relevancy", "context_precision"]].mean()
    print(sep)
    print(
        f"{'MEAN':<{col_w}}  {means['faithfulness']:>6.2f}  "
        f"{means['answer_relevancy']:>6.2f}  {means['context_precision']:>6.2f}"
    )
    print()


# ── Step 5: save_csv ──────────────────────────────────────────────────────────


def save_csv(df, path: Path) -> None:
    """
    Save the full results DataFrame to a CSV file.

    Args:
        df:   DataFrame from score_dataset().
        path: Output path (e.g. eval/results.csv). Parent dir must exist.
    """
    df.to_csv(path, index=False)
    _logger.info("Results saved to %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """
    Orchestrate the full evaluation:
        1. Run all 20 questions through the pipeline (incremental mode)
        2. Build HuggingFace Dataset
        3. Score with RAGAS metrics
        4. Print summary table
        5. Optionally save CSV
    """
    parser = argparse.ArgumentParser(description="RAGAS evaluation for pubmed_rag.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional path to save results CSV (e.g. eval/results.csv).",
    )
    args = parser.parse_args()

    llm = build_evaluator_llm()
    _logger.info(
        "Starting RAGAS evaluation — %d questions, judge: %s",
        len(EVAL_QUESTIONS),
        os.getenv("RAGAS_EVAL_MODEL", _DEFAULT_EVAL_MODEL),
    )

    rows = []
    for i, item in enumerate(EVAL_QUESTIONS):
        _logger.info("Question %d/%d: %s", i + 1, len(EVAL_QUESTIONS), item["question"][:60])
        rows.append(run_single(item["question"], item["ground_truth"]))

    dataset = build_dataset(rows)

    _logger.info("Scoring with RAGAS — calls Claude internally, may take ~1-2 min...")
    df = score_dataset(dataset, llm)

    print_results(df)

    if args.output is not None:
        save_csv(df, args.output)


if __name__ == "__main__":
    main()
