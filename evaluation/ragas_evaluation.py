"""
evaluation/ragas_evaluation.py
------------------------------
RAGAS evaluation of the stored pipeline outputs (results/rag_outputs.json).

RAGAS (https://docs.ragas.io) is the standard library for scoring RAG
systems. It uses an LLM as a "judge" plus an embedding model, and produces
the same four metrics we implemented by hand in manual_evaluation.py:

    faithfulness        claims in the answer supported by the retrieved context
    answer_relevance    how well the answer addresses the question (RAGAS
                        generates questions from the answer and compares them
                        to the original question with embeddings)
    context_precision   are the relevant retrieved chunks ranked at the top?
                        (judged against the ground-truth "reference")
    context_recall      can every sentence of the ground truth be attributed
                        to the retrieved context?

How the judge LLM is wired
    RAGAS's modern API (`ragas.llms.llm_factory`) accepts any `openai.OpenAI`
    client. Groq's endpoint is OpenAI-compatible, so `src.llm.openai_client
    ("groq")` gives us a FREE judge. Embeddings use the same local
    sentence-transformers model as the pipeline (`ragas.embeddings.
    HuggingFaceEmbeddings`) -- no OpenAI key needed anywhere.

Rate limits
    Each question costs several judge calls (statement extraction, NLI
    verification, question generation, ...). On the free Groq tier that is
    fine for a 12-question set, but calls are made sequentially with a small
    pause between metrics to stay under the per-minute limit. A metric that
    fails (rate limit exhausted, malformed JSON from a small model) is
    recorded as NaN rather than aborting the run; the CSV keeps the error
    text in `ragas_errors` so it can be investigated.

Usage
    python evaluation/ragas_evaluation.py                 # score rag_outputs.json
    python evaluation/ragas_evaluation.py --generate      # run the pipeline first
    python evaluation/ragas_evaluation.py --judge-model llama-3.1-8b-instant --sleep 2

Output: results/ragas_results.csv (+ console table and a side-by-side
comparison with results/evaluation_report.csv when it exists).
"""

import argparse
import csv
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

# Windows consoles default to a legacy code page (cp1252) that cannot print
# characters LLMs like to emit (curly quotes, non-breaking hyphens, ...).
# Switching stdout to UTF-8 avoids UnicodeEncodeError crashes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluation_utils import (  # noqa: E402
    check_dataset_matches_index,
    METRIC_COLUMNS,
    diagnose,
    is_missing,
    load_eval_dataset,
    load_rag_outputs,
    print_summary_table,
    run_pipeline_on_dataset,
    save_rag_outputs,
    summarize,
    write_csv,
)
from src import config  # noqa: E402
from src.llm import MissingAPIKeyError, openai_client, reasoning_kwargs  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)

REPORT_COLUMNS = [
    "id", "category", "question", "ground_truth", "answer",
    *METRIC_COLUMNS,
    "diagnosis", "ragas_errors", "judge_provider", "judge_model", "strategy", "model",
]


def build_ragas_metrics(judge_provider: str, judge_model: str, embedding_model: str) -> dict:
    """Instantiate the four RAGAS metric objects sharing one judge + embedder."""
    # Imported here so the rest of the project never pays RAGAS's import cost.
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    # RAGAS's metric classes call the judge asynchronously, so they need an
    # AsyncOpenAI client. provider="openai" tells RAGAS to speak the OpenAI
    # protocol; the client's base_url decides where requests go (Groq / HF).
    # max_retries is generous because the free plan's 8K tokens/min limit
    # produces HTTP 429s that the SDK waits out using the retry-after header.
    client = openai_client(judge_provider, timeout=120.0, max_retries=8, async_client=True)
    # llm_factory forwards extra kwargs straight to chat.completions.create(),
    # so only SDK-native fields are allowed here: reasoning_effort is one
    # (keeps gpt-oss/qwen "thinking" short), Groq's include_reasoning is not.
    native = {k: v for k, v in reasoning_kwargs(judge_provider, judge_model).items()
              if k == "reasoning_effort"}
    judge = llm_factory(judge_model, provider="openai", client=client, temperature=0.0, **native)
    embeddings = HuggingFaceEmbeddings(model=embedding_model)

    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevance": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }


def score_row(metrics: dict, row: dict, sleep: float = 1.0) -> dict:
    """Run the four metrics on one pipeline output. Failures -> NaN + error note."""
    question, answer = row["question"], row["answer"]
    contexts, reference = row.get("contexts", []), row["ground_truth"]
    out_of_scope = not row.get("expected_sources")
    errors: list[str] = []

    # Each metric needs a different subset of the inputs (mirrors RAGAS's
    # own column requirements: response / retrieved_contexts / reference).
    calls = {
        "faithfulness": lambda: metrics["faithfulness"].score(
            user_input=question, response=answer, retrieved_contexts=contexts),
        "answer_relevance": lambda: metrics["answer_relevance"].score(
            user_input=question, response=answer),
        "context_precision": lambda: metrics["context_precision"].score(
            user_input=question, reference=reference, retrieved_contexts=contexts),
        "context_recall": lambda: metrics["context_recall"].score(
            user_input=question, retrieved_contexts=contexts, reference=reference),
    }

    for name, call in calls.items():
        if out_of_scope and name in ("context_precision", "context_recall"):
            row[name] = None  # undefined when nothing SHOULD be retrieved
            continue
        try:
            result = call()
            value = float(result.value)
            row[name] = None if math.isnan(value) else round(value, 4)
        except Exception as exc:
            row[name] = None
            errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:120]}")
        time.sleep(sleep)  # be polite to the free tier

    row["ragas_errors"] = " | ".join(errors)
    row["diagnosis"] = diagnose(row)
    return row


def evaluate_rows(rows: list[dict], judge_provider: str = config.RAGAS_JUDGE_PROVIDER,
                  judge_model: str = config.RAGAS_JUDGE_MODEL,
                  embedding_model: str = config.EMBEDDING_MODEL, sleep: float = 1.0,
                  progress: Callable[[str, float], None] | None = None) -> list[dict]:
    """Score every row with RAGAS and return the report rows."""
    metrics = build_ragas_metrics(judge_provider, judge_model, embedding_model)
    for index, row in enumerate(rows, start=1):
        if progress:
            progress(f"RAGAS {index}/{len(rows)}: {row['question'][:60]}", index / len(rows))
        score_row(metrics, row, sleep=sleep)
        row["judge_provider"] = judge_provider
        row["judge_model"] = judge_model
    return rows


def compare_with_manual(ragas_rows: list[dict]) -> None:
    """Print RAGAS vs manual metric means side by side (if the manual CSV exists)."""
    if not config.MANUAL_REPORT_FILE.exists():
        return
    with config.MANUAL_REPORT_FILE.open(encoding="utf-8") as fh:
        manual_rows = list(csv.DictReader(fh))
    for row in manual_rows:  # CSV values are strings -> floats/None
        for col in METRIC_COLUMNS:
            row[col] = float(row[col]) if row.get(col) not in (None, "") else None
    manual, ragas = summarize(manual_rows), summarize(ragas_rows)

    print("\nRAGAS vs manual evaluation (mean per metric)")
    print(f"{'metric':<20}{'manual':>10}{'ragas':>10}{'diff':>10}")
    for col in METRIC_COLUMNS:
        m, r = manual[col]["mean"], ragas[col]["mean"]
        diff = "" if m is None or r is None else f"{r - m:+.2f}"
        print(f"{col:<20}{'n/a' if m is None else f'{m:.2f}':>10}"
              f"{'n/a' if r is None else f'{r:.2f}':>10}{diff:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation of the RAG pipeline.")
    parser.add_argument("--generate", action="store_true",
                        help="run the pipeline on the dataset first")
    parser.add_argument("--dataset", type=Path, default=config.EVAL_DATA_FILE,
                        help="question set with ground truth (default: evaluation/evaluation_data.json)")
    parser.add_argument("--force", action="store_true",
                        help="score even if the dataset's expected sources are not in the index")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--strategy", default=config.QUERY_REWRITE_STRATEGY)
    parser.add_argument("--judge-provider", default=config.RAGAS_JUDGE_PROVIDER)
    parser.add_argument("--judge-model", default=config.RAGAS_JUDGE_MODEL)
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="seconds to wait between judge calls (rate limits)")
    args = parser.parse_args()

    if args.generate:
        print(f"Generating answers (k={args.k}, strategy={args.strategy})...")
        pipeline = RAGPipeline()
        dataset = load_eval_dataset(args.dataset)
        missing = check_dataset_matches_index(dataset, pipeline.store.sources())
        if missing and not args.force:
            print(f"The question set {args.dataset.name} expects files that are not in the index: "
                  f"{', '.join(missing)}. Write a matching question set and pass --dataset, or use --force.")
            sys.exit(1)
        rows = run_pipeline_on_dataset(pipeline, dataset, k=args.k,
                                       strategy=args.strategy,
                                       progress=lambda m, f: print(f"  [{f:>4.0%}] {m}"))
        save_rag_outputs(rows)
    else:
        rows = load_rag_outputs()
    print(f"Scoring {len(rows)} outputs with RAGAS "
          f"(judge {args.judge_provider}/{args.judge_model}, embeddings {config.EMBEDDING_MODEL})")

    try:
        rows = evaluate_rows(rows, args.judge_provider, args.judge_model, sleep=args.sleep,
                             progress=lambda m, f: print(f"  [{f:>4.0%}] {m}"))
    except MissingAPIKeyError as exc:
        print(f"\n{exc}")
        sys.exit(1)

    path = write_csv(rows, config.RAGAS_REPORT_FILE, REPORT_COLUMNS)
    print_summary_table(rows, "RAGAS evaluation")
    failed = sum(1 for r in rows if r.get("ragas_errors"))
    if failed:
        print(f"\n{failed} row(s) had metric errors -- see the ragas_errors column.")
    print(f"\nReport written to {path}")
    compare_with_manual(rows)


if __name__ == "__main__":
    main()
