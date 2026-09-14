"""
evaluation/evaluation_utils.py
------------------------------
Shared helpers for the two evaluation scripts (manual + RAGAS) and the
the CLI runs.

The evaluation flow has two separate stages, and this split matters:

    Stage A  run_pipeline_on_dataset()
             Ask every question in evaluation_data.json through the SAME
             RAGPipeline the app uses and store {question, answer,
             retrieved contexts, ground truth, sources} in
             results/rag_outputs.json.

    Stage B  score those stored outputs
             manual_evaluation.py  -> results/evaluation_report.csv
             ragas_evaluation.py   -> results/ragas_results.csv

Because Stage B reads from the JSON file, both evaluators judge EXACTLY the
same answers and contexts, which makes their scores directly comparable --
and you can re-score without paying for new answer generation.

Also here: `diagnose()`, the small rule set that turns four metric values
into a plain-English verdict ("retrieval issue" vs "generation issue"),
which is the point of the whole evaluation stage.
"""

import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Callable

# Make "python evaluation/xxx.py" work from the project root AND from inside
# evaluation/ by putting the project root on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402

# The four RAG quality metrics, in the order they are reported everywhere.
METRIC_COLUMNS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]


# ---------------------------------------------------------------------- #
# Dataset + outputs I/O
# ---------------------------------------------------------------------- #
def load_eval_dataset(path: Path = config.EVAL_DATA_FILE) -> list[dict]:
    """The hand-written questions with ground-truth answers."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_dataset_matches_index(dataset: list[dict], indexed_sources: dict[str, int]) -> list[str]:
    """Return the expected source files that are NOT in the index.

    The shipped evaluation_data.json was written for the seven sample
    documents. If the user replaced data/ with their own files, scoring those
    questions would be meaningless, so the scripts refuse unless --force is
    given or a matching --dataset is supplied.
    """
    expected = {src for item in dataset for src in item.get("expected_sources", [])}
    return sorted(src for src in expected if src not in indexed_sources)


def run_pipeline_on_dataset(
    pipeline: RAGPipeline,
    dataset: list[dict] | None = None,
    k: int = config.TOP_K,
    strategy: str | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> list[dict]:
    """Stage A: generate an answer for every evaluation question.

    Returns one dict per question containing everything both evaluators
    need. `progress(message, fraction)` is an optional progress callback.
    """
    dataset = dataset or load_eval_dataset()
    rows: list[dict] = []
    for index, item in enumerate(dataset, start=1):
        if progress:
            progress(f"Answering {index}/{len(dataset)}: {item['question'][:60]}", index / len(dataset))
        response = pipeline.ask(item["question"], k=k, strategy=strategy)
        row = {
            "id": item["id"],
            "category": item.get("category", ""),
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "expected_sources": item.get("expected_sources", []),
        }
        row.update(response.to_dict())
        rows.append(row)
    return rows


def save_rag_outputs(rows: list[dict], path: Path = config.RAG_OUTPUTS_FILE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_rag_outputs(path: Path = config.RAG_OUTPUTS_FILE) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the pipeline on the dataset first "
            "(python evaluation/manual_evaluation.py --generate)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows: list[dict], path: Path, columns: list[str]) -> Path:
    """Write selected columns of `rows` to CSV (lists are joined with ' | ')."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {}
            for col in columns:
                value = row.get(col)
                if isinstance(value, (list, tuple)):
                    value = " | ".join(str(v) for v in value)
                elif isinstance(value, float):
                    value = "" if math.isnan(value) else round(value, 4)
                flat[col] = value
            writer.writerow(flat)
    return path


# ---------------------------------------------------------------------- #
# Scoring helpers
# ---------------------------------------------------------------------- #
def is_missing(value) -> bool:
    """True for None / NaN -- metrics that could not be computed."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def summarize(rows: list[dict], columns: list[str] = METRIC_COLUMNS) -> dict[str, dict]:
    """NaN-safe mean / min / max / count per metric column."""
    summary = {}
    for col in columns:
        values = [float(r[col]) for r in rows if col in r and not is_missing(r[col])]
        summary[col] = {
            "mean": round(sum(values) / len(values), 4) if values else None,
            "min": round(min(values), 4) if values else None,
            "max": round(max(values), 4) if values else None,
            "count": len(values),
            "missing": len(rows) - len(values),
        }
    return summary


def diagnose(row: dict) -> str:
    """Turn the four metrics into a verdict about WHERE a weak result comes from.

    Reading guide
        context_recall low     -> the needed evidence was never retrieved
                                  (fix: chunking, embeddings, k, rewriting)
        faithfulness low       -> evidence was there but the answer strayed
                                  from it (fix: prompt, temperature, model)
        answer_relevance low   -> the answer does not address the question
                                  (fix: prompt, model)
        context_precision low  -> answer fine but many retrieved chunks were
                                  noise (fix: k, rewriting, re-ranking)
    """
    recall = row.get("context_recall")
    faith = row.get("faithfulness")
    relevance = row.get("answer_relevance")
    precision = row.get("context_precision")

    problems = []
    if not is_missing(recall) and recall < 0.5:
        problems.append("retrieval issue (needed context not retrieved)")
    if not is_missing(faith) and faith < 0.7:
        problems.append("generation issue (answer not supported by context)")
    if not is_missing(relevance) and relevance < 0.5:
        problems.append("generation issue (answer off-topic)")
    if not problems and not is_missing(precision) and precision < 0.5:
        problems.append("retrieval noise (correct but many irrelevant chunks)")
    return "; ".join(problems) if problems else "ok"


def parse_json_reply(text: str) -> dict:
    """Extract the first JSON object from an LLM reply (tolerates code fences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def print_summary_table(rows: list[dict], title: str, columns: list[str] = METRIC_COLUMNS) -> None:
    """Console report: per-question metric table + means + diagnosis."""
    print(f"\n{title}")
    print("=" * len(title))
    header = f"{'id':>3}  {'question':<52} " + " ".join(f"{c[:12]:>12}" for c in columns) + "  diagnosis"
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            cells.append(f"{'n/a':>12}" if is_missing(value) else f"{float(value):>12.2f}")
        print(f"{row['id']:>3}  {row['question'][:52]:<52} " + " ".join(cells) + f"  {row.get('diagnosis', '')}")
    print("-" * len(header))
    summary = summarize(rows, columns)
    means = " ".join(
        f"{'n/a':>12}" if summary[c]['mean'] is None else f"{summary[c]['mean']:>12.2f}" for c in columns
    )
    print(f"{'':>3}  {'MEAN':<52} {means}")
