"""
src/frontend.py
---------------
Helpers shared by the two front-ends -- the command-line app (main.py) and
the web app (app.py) -- so that neither has to import the other and both
behave the same way.

    AnswerEvaluator     scores ONE RAGResponse with the manual metrics (used by
                        the CLI's --evaluate / /eval and by the web app's
                        "evaluate each answer" toggle)
    find_ground_truth   looks a typed question up in evaluation_data.json
    relevance_label     the brief reports answer relevance as a word (High)
    index_settings      what the stored index was built with (the manifest)
    data_changes        files added / removed / modified since ingestion
    resolve_provider    "Hugging Face" / "hf" / "GROQ" -> a PROVIDERS key

Nothing here prints or draws: the CLI renders through src/console.py, the
web app through ui/components.py. Everything else (pipeline, retrieval,
evaluation metrics) lives in src/ and evaluation/ and is imported unchanged.
"""

from __future__ import annotations

import difflib
import json
import re

from evaluation.evaluation_utils import diagnose, load_eval_dataset
from evaluation.manual_evaluation import (
    answer_relevance_llm,
    context_precision_embeddings,
    context_recall_tokens,
    faithfulness_llm,
)
from src import config
from src.document_loader import file_signature
from src.llm import PROVIDERS, get_llm
from src.rag_pipeline import RAGResponse


# ---------------------------------------------------------------------- #
# Ground truth lookup for the live evaluation block
# ---------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    """Lower-case, drop punctuation and extra spaces -- for question matching."""
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def find_ground_truth(question: str, dataset: list[dict]) -> dict | None:
    """Return the evaluation entry whose question matches `question`.

    Exact matches after normalisation win; otherwise a close paraphrase
    (difflib ratio >= 0.9, e.g. a missing question mark or an extra word)
    is accepted. Anything looser would risk scoring an answer against the
    wrong ground truth.
    """
    target = _normalize(question)
    for entry in dataset:
        if _normalize(entry["question"]) == target:
            return entry
    best, best_ratio = None, 0.0
    for entry in dataset:
        ratio = difflib.SequenceMatcher(None, target, _normalize(entry["question"])).ratio()
        if ratio > best_ratio:
            best, best_ratio = entry, ratio
    return best if best_ratio >= 0.9 else None


def relevance_label(score: float | None) -> str:
    """The brief reports answer relevance as a word (High) -- map the 0-1 score."""
    if score is None:
        return "n/a"
    word = "High" if score >= 0.75 else "Medium" if score >= 0.5 else "Low"
    return f"{word} ({score:.2f})"


class AnswerEvaluator:
    """Scores a single RAGResponse with the manual metrics.

    Created once per session (CLI or web): it lazily builds the judge LLM
    (the provider/model from RAGAS_JUDGE_* in .env) and loads the
    evaluation dataset used to look up ground truth. Faithfulness and
    answer relevance are always computed; context precision and recall
    only when the question has a ground truth in the dataset.
    """

    def __init__(self, embedder):
        self.embedder = embedder
        self._judge = None
        self.dataset = load_eval_dataset()

    @property
    def judge(self):
        if self._judge is None:
            self._judge = get_llm(config.RAGAS_JUDGE_PROVIDER, config.RAGAS_JUDGE_MODEL)
        return self._judge

    def evaluate(self, response: RAGResponse) -> dict:
        contexts = response.contexts
        faith, faith_note = faithfulness_llm(self.judge, response.question, response.answer, contexts)
        relevance, _ = answer_relevance_llm(self.judge, response.question, response.answer)

        truth = find_ground_truth(response.question, self.dataset)
        precision = recall = None
        if truth and contexts:
            precision, _ = context_precision_embeddings(self.embedder, truth["ground_truth"], contexts)
            recall = context_recall_tokens(truth["ground_truth"], contexts)

        metrics = {
            "faithfulness": faith,
            "answer_relevance": relevance,
            "context_precision": precision,
            "context_recall": recall,
        }
        return {**metrics, "ground_truth": truth, "faithfulness_note": faith_note,
                "diagnosis": diagnose(metrics)}


# ---------------------------------------------------------------------- #
# Index state (the ingest manifest vs. the data/ folder)
# ---------------------------------------------------------------------- #
def index_settings() -> dict | None:
    """What the stored index was BUILT with, from results/ingest_manifest.json.

    The manifest is the truth for chunk size / overlap: .env may have been
    edited after the last ingestion, in which case the index is stale until
    `python ingest.py` is run again. Returns None if no manifest.
    """
    try:
        return json.loads(config.INGEST_MANIFEST_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def data_changes(manifest: dict) -> dict[str, list[str]]:
    """Compare data/ now with what the manifest recorded at ingest time."""
    before = manifest.get("files_signature") or {}
    now = file_signature(config.DATA_DIR)
    return {
        "added": sorted(f for f in now if f not in before),
        "removed": sorted(f for f in before if f not in now),
        "modified": sorted(f for f in now if f in before and now[f] != before[f]),
    }


# ---------------------------------------------------------------------- #
# Provider names as people type them
# ---------------------------------------------------------------------- #
_PROVIDER_ALIASES = {"hf": "huggingface", "hugging": "huggingface"}


def resolve_provider(text: str) -> str | None:
    """'Hugging Face' / 'hf' / 'GROQ' -> a key of PROVIDERS, or None if it is not a provider."""
    key = "".join(ch for ch in text.lower() if ch.isalnum())
    key = _PROVIDER_ALIASES.get(key, key)
    return key if key in PROVIDERS else None
