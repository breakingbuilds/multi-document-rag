"""
evaluation/manual_evaluation.py
-------------------------------
"Manual" evaluation: the four RAG quality metrics implemented by hand, with
simple, inspectable rules -- BEFORE handing the same data to RAGAS.

Why implement them manually first?
    RAGAS is a black box unless you know what each metric measures. Writing
    a plain version of every metric makes the definitions concrete, gives a
    result that needs no paid judge for two of the four metrics, and lets
    you sanity-check RAGAS's numbers against something you understand.

The four metrics (same names and 0-1 scale as RAGAS)

    faithfulness      Is every claim in the ANSWER supported by the retrieved
                      CONTEXT?  (LLM judge: extract claims -> verify each)
                      = supported claims / all claims.  A refusal ("I don't
                      have enough information") has no claims -> 1.0.

    answer_relevance  Does the ANSWER address the QUESTION?  (LLM judge rates
                      1-5, scaled to 0-1). The cosine similarity between the
                      question and answer embeddings is reported alongside as
                      `answer_question_similarity` for a judge-free signal.

    context_precision Of the retrieved chunks, how many are relevant, and are
                      the relevant ones ranked first?  (embedding-based: a
                      chunk is relevant if cosine(chunk, ground truth) >=
                      RELEVANT_SIM_THRESHOLD; then the RAGAS "average
                      precision" formula over ranks). No LLM needed.

    context_recall    Did we retrieve the evidence needed for the ground-truth
                      answer?  (token-based: fraction of the ground truth's
                      content words and numbers that occur in the retrieved
                      contexts). No LLM needed. `expected_source_recall`
                      additionally checks that the expected FILES were hit.

Usage
    python evaluation/manual_evaluation.py --generate     # run pipeline first, then score
    python evaluation/manual_evaluation.py                # score results/rag_outputs.json
    python evaluation/manual_evaluation.py --generate --strategy llm_multi_query --k 6

Output: results/evaluation_report.csv (+ console table).
"""

import argparse
import math
import re
import sys
from pathlib import Path

# Windows consoles default to a legacy code page (cp1252) that cannot print
# characters LLMs like to emit (curly quotes, non-breaking hyphens, ...).
# Switching stdout to UTF-8 avoids UnicodeEncodeError crashes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluation_utils import (  # noqa: E402
    METRIC_COLUMNS,
    check_dataset_matches_index,
    diagnose,
    is_missing,
    load_eval_dataset,
    load_rag_outputs,
    parse_json_reply,
    print_summary_table,
    run_pipeline_on_dataset,
    save_rag_outputs,
    summarize,
    write_csv,
)
from src import config  # noqa: E402
from src.config import STOPWORDS  # noqa: E402
from src.embeddings import EmbeddingModel, cosine_similarity  # noqa: E402
from src.llm import MissingAPIKeyError, get_llm  # noqa: E402
from src.prompt import (  # noqa: E402
    CLAIM_EXTRACTION_SYSTEM,
    CLAIM_EXTRACTION_USER,
    CLAIM_VERIFICATION_SYSTEM,
    CLAIM_VERIFICATION_USER,
    NO_ANSWER_SENTENCE,
    RELEVANCE_RATING_SYSTEM,
    RELEVANCE_RATING_USER,
)
from src.rag_pipeline import RAGPipeline  # noqa: E402

# A retrieved chunk counts as "relevant" when its embedding is at least this
# similar to the ground-truth answer. With all-MiniLM-L6-v2, on-topic
# passages score ~0.5-0.8 against a one-sentence answer, unrelated ~0.1-0.3.
RELEVANT_SIM_THRESHOLD = 0.45

REPORT_COLUMNS = [
    "id", "category", "question", "ground_truth", "answer",
    *METRIC_COLUMNS,
    "answer_question_similarity", "expected_source_recall", "refused",
    "sources", "cited_sources", "strategy", "model", "diagnosis", "notes",
]


# ---------------------------------------------------------------------- #
# Judge-free metrics (embeddings / tokens only)
# ---------------------------------------------------------------------- #
def _content_tokens(text: str) -> set[str]:
    """Lower-cased words and numbers, minus stop-words. Numbers are kept
    because they carry most of the meaning in policy answers ("20", "5")."""
    tokens = re.findall(r"[a-z0-9][a-z0-9'\-]*", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def context_recall_tokens(ground_truth: str, contexts: list[str]) -> float | None:
    """Fraction of ground-truth content tokens present anywhere in the contexts."""
    needed = _content_tokens(ground_truth)
    if not needed:
        return None
    available = _content_tokens(" ".join(contexts))
    return round(len(needed & available) / len(needed), 4)


def context_precision_embeddings(embedder: EmbeddingModel, ground_truth: str,
                                 contexts: list[str]) -> tuple[float | None, list[bool]]:
    """RAGAS-style context precision with an embedding-based relevance judge.

    precision@k = relevant chunks among the first k
    CP = sum_k( precision@k * rel_k ) / number of relevant chunks
    -> 1.0 when every relevant chunk is ranked above every irrelevant one.
    """
    if not contexts:
        return None, []
    gt_vec = embedder.embed_query(ground_truth)
    relevance = [
        cosine_similarity(embedder.embed_query(ctx), gt_vec) >= RELEVANT_SIM_THRESHOLD
        for ctx in contexts
    ]
    if not any(relevance):
        return 0.0, relevance
    score, relevant_so_far = 0.0, 0
    for k, rel in enumerate(relevance, start=1):
        if rel:
            relevant_so_far += 1
            score += relevant_so_far / k
    return round(score / sum(relevance), 4), relevance


def expected_source_recall(expected: list[str], retrieved_sources: list[str]) -> float | None:
    """Fraction of the expected FILES that appear among the retrieved chunks."""
    if not expected:
        return None
    return round(sum(1 for s in expected if s in retrieved_sources) / len(expected), 4)


# ---------------------------------------------------------------------- #
# LLM-judged metrics
# ---------------------------------------------------------------------- #
def faithfulness_llm(llm, question: str, answer: str, contexts: list[str]) -> tuple[float | None, str]:
    """supported claims / total claims, judged by the LLM. Returns (score, note)."""
    if answer.strip().startswith(NO_ANSWER_SENTENCE[:40]):
        return 1.0, "refusal -> no claims to verify"

    extraction = parse_json_reply(llm.generate(
        CLAIM_EXTRACTION_SYSTEM, CLAIM_EXTRACTION_USER.format(question=question, answer=answer),
        temperature=0.0, json_mode=True,
    ))
    claims = [c for c in extraction.get("claims", []) if isinstance(c, str) and c.strip()]
    if not claims:
        return 1.0, "no factual claims extracted"

    context_block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
    claims_block = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
    verification = parse_json_reply(llm.generate(
        CLAIM_VERIFICATION_SYSTEM,
        CLAIM_VERIFICATION_USER.format(context=context_block, claims=claims_block),
        temperature=0.0, json_mode=True,
    ))
    verdicts = verification.get("verdicts", [])
    if not verdicts:
        return None, "judge returned no verdicts"
    supported = sum(1 for v in verdicts if isinstance(v, dict) and bool(v.get("supported")))
    unsupported = [v.get("claim", "?") for v in verdicts if isinstance(v, dict) and not v.get("supported")]
    note = f"{supported}/{len(verdicts)} claims supported"
    if unsupported:
        note += "; unsupported: " + " / ".join(str(u)[:60] for u in unsupported[:2])
    return round(supported / len(verdicts), 4), note


def answer_relevance_llm(llm, question: str, answer: str) -> tuple[float | None, str]:
    """LLM rates 1-5 -> scaled to 0-1."""
    reply = parse_json_reply(llm.generate(
        RELEVANCE_RATING_SYSTEM, RELEVANCE_RATING_USER.format(question=question, answer=answer),
        temperature=0.0, json_mode=True,
    ))
    try:
        score = float(reply.get("score"))
    except (TypeError, ValueError):
        return None, "judge returned no score"
    score = min(max(score, 1.0), 5.0)
    return round((score - 1.0) / 4.0, 4), f"relevance rating {score:.0f}/5: {reply.get('reason', '')[:80]}"


# ---------------------------------------------------------------------- #
# Orchestration
# ---------------------------------------------------------------------- #
def evaluate_rows(rows: list[dict], embedder: EmbeddingModel | None = None,
                  llm=None, progress=None) -> list[dict]:
    """Score every row in place and return them (with diagnosis)."""
    embedder = embedder or EmbeddingModel(config.EMBEDDING_MODEL)
    for index, row in enumerate(rows, start=1):
        if progress:
            progress(f"Scoring {index}/{len(rows)}: {row['question'][:60]}", index / len(rows))
        notes: list[str] = []
        contexts = row.get("contexts", [])
        answer = row.get("answer", "")
        out_of_scope = not row.get("expected_sources")  # e.g. the stock-price question

        row["refused"] = answer.strip().startswith(NO_ANSWER_SENTENCE[:40])

        # --- judge-free metrics --------------------------------------------
        if out_of_scope:
            # There is no "right context" to retrieve, so recall/precision
            # are undefined; what matters is that the model refused.
            row["context_recall"] = None
            row["context_precision"] = None
            notes.append("out-of-scope question: retrieval metrics n/a; refusal expected")
        else:
            row["context_recall"] = context_recall_tokens(row["ground_truth"], contexts)
            row["context_precision"], relevance = context_precision_embeddings(
                embedder, row["ground_truth"], contexts)
            notes.append(f"relevant chunks by rank: {''.join('R' if r else '.' for r in relevance)}")
        row["expected_source_recall"] = expected_source_recall(
            row.get("expected_sources", []), row.get("sources", []))
        row["answer_question_similarity"] = round(cosine_similarity(
            embedder.embed_query(row["question"]), embedder.embed_query(answer or " ")), 4)

        # --- LLM-judged metrics --------------------------------------------
        if llm is None:
            row["faithfulness"] = None
            row["answer_relevance"] = None
            notes.append("no judge LLM: faithfulness / answer_relevance n/a")
        else:
            try:
                row["faithfulness"], note = faithfulness_llm(llm, row["question"], answer, contexts)
                notes.append(note)
                row["answer_relevance"], note = answer_relevance_llm(llm, row["question"], answer)
                notes.append(note)
            except Exception as exc:  # keep going if the judge fails on one row
                row.setdefault("faithfulness", None)
                row.setdefault("answer_relevance", None)
                notes.append(f"judge error: {type(exc).__name__}: {exc}")

        row["diagnosis"] = diagnose(row)
        row["notes"] = "; ".join(notes)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual (hand-written) RAG evaluation.")
    parser.add_argument("--generate", action="store_true",
                        help="run the pipeline on the dataset first (needs an LLM key)")
    parser.add_argument("--dataset", type=Path, default=config.EVAL_DATA_FILE,
                        help="question set with ground truth (default: evaluation/evaluation_data.json)")
    parser.add_argument("--force", action="store_true",
                        help="score even if the dataset's expected sources are not in the index")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--strategy", default=config.QUERY_REWRITE_STRATEGY)
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the LLM-judged metrics (faithfulness, answer relevance)")
    args = parser.parse_args()

    if args.generate:
        print(f"Generating answers for the evaluation set (k={args.k}, strategy={args.strategy})...")
        pipeline = RAGPipeline()
        dataset = load_eval_dataset(args.dataset)
        missing = check_dataset_matches_index(dataset, pipeline.store.sources())
        if missing and not args.force:
            print(f"The question set {args.dataset.name} expects files that are not in the index: "
                  f"{', '.join(missing)}.\nIt was written for the sample documents; with your own files, "
                  f"write a matching question set and pass --dataset, or use --force to score anyway.")
            sys.exit(1)
        rows = run_pipeline_on_dataset(pipeline, dataset, k=args.k, strategy=args.strategy,
                                       progress=lambda m, f: print(f"  [{f:>4.0%}] {m}"))
        save_rag_outputs(rows)
        print(f"Saved pipeline outputs -> {config.RAG_OUTPUTS_FILE}")
    else:
        rows = load_rag_outputs()
        print(f"Loaded {len(rows)} pipeline outputs from {config.RAG_OUTPUTS_FILE}")

    llm = None
    if not args.no_judge:
        try:
            llm = get_llm(config.RAGAS_JUDGE_PROVIDER, config.RAGAS_JUDGE_MODEL)
            print(f"Judge LLM: {llm.provider}/{llm.model}")
        except MissingAPIKeyError as exc:
            print(f"No judge LLM available ({exc}); computing judge-free metrics only.")

    rows = evaluate_rows(rows, llm=llm, progress=lambda m, f: print(f"  [{f:>4.0%}] {m}"))
    path = write_csv(rows, config.MANUAL_REPORT_FILE, REPORT_COLUMNS)
    print_summary_table(rows, "Manual evaluation")
    print(f"\nReport written to {path}")

    summary = summarize(rows)
    weakest = min((c for c in METRIC_COLUMNS if summary[c]["mean"] is not None),
                  key=lambda c: summary[c]["mean"], default=None)
    if weakest:
        print(f"Weakest metric on average: {weakest} ({summary[weakest]['mean']}) -> see diagnosis column.")


if __name__ == "__main__":
    main()
