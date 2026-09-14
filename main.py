"""
main.py
-------
Command-line application: ask questions, see rewrites, sources and the answer.
Output is framed and coloured by src/console.py (plain ASCII / no colour when
piped, or with NO_COLOR=1 / RAG_ASCII=1).

    python main.py                                # interactive loop
    python main.py --strategy llm_multi_query     # choose a rewriting strategy
    python main.py -q "How many annual leave days are allowed?"   # one-shot
    python main.py --k 8 --provider huggingface --model Qwen/Qwen2.5-7B-Instruct
    python main.py --list-models                  # models available per provider
    python main.py --show-chunks                  # also print the retrieved chunk text
    python main.py --evaluate                     # score every answer (Evaluation: block)

Inside the interactive loop these commands change settings without restarting:

    /strategy <name>    switch rewriting strategy      (/strategies lists them)
    /k <n>              change how many chunks are retrieved
    /provider <name>    switch LLM provider (groq | huggingface | ollama)   (/providers shows readiness)
    /model <id>         switch model (see /models for the list)
    /models             list the models the current provider offers
    /chunks             toggle printing of retrieved chunk text
    /eval               toggle per-answer evaluation
    /help               show the command list again
    /bar                hide / show the one-line command bar under the prompt
    /quit               exit

Per-answer evaluation (--evaluate / /eval) prints the four RAG metrics under
each answer, in the layout of the project brief:

    Evaluation:
    Faithfulness: 1.00
    Context Precision: 0.83
    Answer Relevance: High (0.95)
    Context Recall: 1.00

Faithfulness and answer relevance are judged by the LLM for ANY question.
Context precision and recall need a ground-truth answer, so they are
computed when the question matches one in evaluation/evaluation_data.json
and reported as "n/a" otherwise. The scoring functions are the same ones
evaluation/manual_evaluation.py uses for the batch report.

This CLI, the evaluation scripts and ingest.py all drive the same RAGPipeline
(src/rag_pipeline.py), so a question answered here is answered identically by
the evaluation run.
"""

import argparse
import sys

# Windows consoles default to a legacy code page (cp1252) that cannot print
# characters LLMs like to emit (curly quotes, non-breaking hyphens, ...).
# Switching stdout to UTF-8 avoids UnicodeEncodeError crashes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
from src.console import ui
from src.document_loader import file_signature
from openai import APIError

from src.llm import PROVIDERS, MissingAPIKeyError, get_llm, list_models, provider_status
from src.query_rewriter import STRATEGY_DESCRIPTIONS, RewriteStrategy
from src.rag_pipeline import RAGPipeline, RAGResponse

COMMANDS = [
    ("/strategy <name>", "switch rewriting strategy (shown in the prompt)"),
    ("/strategies", "list the rewriting strategies with a one-line description"),
    ("/k <n>", "change how many chunks are retrieved"),
    ("/provider <name>", "switch LLM provider (groq | huggingface | ollama)"),
    ("/providers", "list the providers and whether each one is ready"),
    ("/model <id>", "switch model on the current provider"),
    ("/models", "list the models the provider offers"),
    ("/chunks", "toggle the retrieved-chunk view"),
    ("/eval", "toggle the per-answer Evaluation block"),
    ("/help", "show this list again"),
    ("/bar", "hide / show the command bar under the prompt"),
    ("/quit", "exit"),
]


def print_response(response: RAGResponse, show_rewrites: bool, show_chunks: bool) -> None:
    """Render a RAGResponse in framed sections (the layout of the project brief,
    dressed up with rules, aligned columns and a little colour)."""
    g = ui.g
    ui.blank()

    # -- Query rewriting -------------------------------------------------
    if show_rewrites and response.rewrite.rewrites:
        ui.section("Query rewriting", response.strategy)
        for note in response.rewrite.notes:
            ui.line(f"{g['bullet']} {ui.dim(note)}", hang=2)
        if response.rewrite.notes:
            ui.line()
        for index, query in enumerate(response.rewrite.all_queries):
            tag = "original" if index == 0 else f"rewrite {index}"
            ui.line(f"{ui.dim(tag.ljust(10))} {query}", hang=11)
        ui.end()
        ui.blank()

    # -- Retrieved sources -----------------------------------------------
    ui.section("Retrieved Sources", f"{g['star']} = cited in the answer")
    if response.sources:
        rows = []
        for index, ref in enumerate(response.sources, start=1):
            mark = ui.warn(g["star"]) if ref.cited else " "
            where = ", ".join(ref.locators) if ref.locators else ""
            nums = ",".join(str(n) for n in ref.citation_numbers)
            rows.append([f"{index}.", mark, ui.bold(ref.source) if ref.cited else ref.source,
                         ui.dim(where), ui.dim(f"passages {nums}")])
        # 3 + 1 + 24 + loc + 14 columns + 4 gaps of 2 must fit inside the frame (width - 6).
        loc_w = max(12, ui.width - 55)
        ui.columns(rows, [3, 1, 24, loc_w, 14], inside=True)
    else:
        ui.line(ui.dim("(none)"))
    ui.end()

    # -- Retrieved chunks (optional) --------------------------------------
    if show_chunks:
        ui.blank()
        ui.section("Retrieved chunks", "similarity · hit count · preview")
        for chunk in response.chunks:
            ui.line(f"{ui.accent(f'[{chunk.rank}]')} {chunk.chunk_id}   "
                    f"{ui.dim('sim')} {chunk.similarity:.3f}   {ui.dim('hits')} {chunk.hit_count}")
            preview = chunk.text[:220].replace("\n", " ")
            ui.line(ui.dim(preview + ("…" if len(chunk.text) > 220 else "")), indent=6)
        ui.end()

    # -- Answer -------------------------------------------------------------
    ui.blank()
    ui.section("Answer")
    for paragraph in response.answer.split("\n"):
        ui.line(ui.warn(paragraph) if response.refused else paragraph)
    ui.end()
    t = response.timings
    ui.print("  " + ui.dim(f"{response.provider}/{response.model}  {g['v']}  rewrite {t['rewrite_s']}s"
                           f"  {g['dot']}  retrieve {t['retrieve_s']}s  {g['dot']}  generate {t['generate_s']}s"))


# ---------------------------------------------------------------------- #
# Per-answer evaluation
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
    """Scores a single RAGResponse with the manual metrics and prints the block.

    Created once per CLI session: it lazily builds the judge LLM (the
    provider/model from RAGAS_JUDGE_* in .env) and loads the evaluation
    dataset used to look up ground truth.
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


def print_evaluation(result: dict) -> None:
    """Render the metrics in the order used by the project brief, with score bars."""
    fmt = lambda v: ("n/a" if v is None else f"{v:.2f}").ljust(12)
    ui.blank()
    ui.section("Evaluation", "0 = worst, 1 = best")
    for label, key in (("Faithfulness", "faithfulness"), ("Context Precision", "context_precision")):
        ui.line(f"{ui.dim(label.ljust(19))} {fmt(result[key])}{ui.bar(result[key])}")
    ui.line(f"{ui.dim('Answer Relevance'.ljust(19))} {relevance_label(result['answer_relevance']).ljust(12)}"
            f"{ui.bar(result['answer_relevance'])}")
    ui.line(f"{ui.dim('Context Recall'.ljust(19))} {fmt(result['context_recall'])}{ui.bar(result['context_recall'])}")
    ui.line()
    if result["ground_truth"] is None:
        ui.line(ui.dim("Precision and recall need a ground-truth answer; this question is not in "
                       "evaluation/evaluation_data.json (run the batch scripts for the full report)."))
    else:
        verdict = result["diagnosis"]
        ui.line(f"{ui.dim('Diagnosis'.ljust(19))} {ui.good(verdict) if verdict == 'ok' else ui.bad(verdict)}")
    ui.end()


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


def print_index_line(count: int, embedding_model: str) -> None:
    """Index: 89 chunks │ chunk size 800 / overlap 150 │ embedding model …"""
    manifest = index_settings()
    v, dot = ui.g["v"], ui.g["dot"]
    if manifest:
        size, overlap = manifest.get("chunk_size"), manifest.get("chunk_overlap")
        ui.print("  " + ui.dim(f"Index: {count} chunks  {v}  chunk size {size} / overlap {overlap}  {v}  "
                               f"embedding model: {embedding_model}"))
        # Warn when .env no longer matches what is stored.
        if (size, overlap) != (config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            ui.status(False, f".env now says chunk size {config.CHUNK_SIZE} / overlap {config.CHUNK_OVERLAP}, "
                             f"but the index was built with {size} / {overlap} {dot} run  python ingest.py  to apply")
        if manifest.get("embedding_model") and manifest["embedding_model"] != embedding_model:
            ui.status(False, f"the manifest was written for {manifest['embedding_model']}; this session uses "
                             f"{embedding_model} (a different collection)")
        # Did the documents change after the index was built?
        if "files_signature" in manifest:
            changes = data_changes(manifest)
            parts = [f"{len(v_)} {k}" for k, v_ in changes.items() if v_]
            if parts:
                names = ", ".join((changes["added"] + changes["modified"] + changes["removed"])[:4])
                ui.status(False, f"data/ changed since the index was built ({', '.join(parts)}: {names}) "
                                 f"{dot} run  python ingest.py  to rebuild")
    else:
        ui.print("  " + ui.dim(f"Index: {count} chunks  {v}  embedding model: {embedding_model}"))


# Accept the ways people actually type a provider name: "Hugging Face", "hf", "HuggingFace".
_PROVIDER_ALIASES = {"hf": "huggingface", "hugging": "huggingface"}


def resolve_provider(text: str) -> str | None:
    """'Hugging Face' / 'hf' / 'GROQ' -> a key of PROVIDERS, or None if it is not a provider."""
    key = "".join(ch for ch in text.lower() if ch.isalnum())
    key = _PROVIDER_ALIASES.get(key, key)
    return key if key in PROVIDERS else None


def print_models(provider: str) -> None:
    """List the models a provider offers (live from its API, static fallback)."""
    status = provider_status()[provider]
    if status["local"]:
        key_note = ui.good(f"running at {status['base_url']}") if status["has_key"] else ui.bad(f"not running: {status['hint']}")
    else:
        key_note = ui.good("key found") if status["has_key"] else ui.bad(f"no key: set {status['env_var']} in .env")
    spec = PROVIDERS[provider]
    models = list_models(provider)
    # Curated models (the provider's recommended handful) as bullets first ...
    curated = [m for m in models if m == spec.default_model or m in spec.fallback_models]
    rest = [m for m in models if m not in curated]
    ui.blank()
    ui.section(status["label"], key_note)
    for model in curated:
        marker = ui.warn("  (default)") if model == status["default_model"] else ""
        ui.line(f"{ui.g['bullet']} {model}{marker}")
    # ... then everything else the provider's live model list reported, in two columns.
    if rest:
        ui.line("")
        ui.line(ui.dim(f"and {len(rest)} more from the live model list (any of them works with /model <id>):"))
        col_w = (ui.width - 8) // 2
        pairs = [rest[i:i + 2] for i in range(0, len(rest), 2)]
        ui.columns([[ui.dim(m) for m in pair] for pair in pairs], [col_w, col_w], inside=True)
    ui.end()


def print_commands() -> None:
    """The command reference: printed at start-up and again by /help."""
    ui.rule("Commands")
    ui.columns([[ui.accent(cmd), desc] for cmd, desc in COMMANDS], [18, ui.width - 22])


def command_bar() -> str:
    """One dim line (two on a narrow terminal) listing the commands, shown
    under the prompt so the reference is always on screen without a
    split-pane UI."""
    names = [cmd.split()[0] for cmd, _ in COMMANDS]
    lines, current = [], ""
    for name in names:
        candidate = f"{current} {name}".strip()
        if current and len(candidate) > ui.width - 4:
            lines.append(current)
            current = name
        else:
            current = candidate
    lines.append(current)
    return "\n".join(ui.dim(f"{ui.g['v']}  {text}") for text in lines)


def print_strategies() -> None:
    """The rewriting strategies with a one-line description each (start-up and /strategies)."""
    ui.rule("Rewriting strategies")
    ui.columns([[ui.bold(s.value), desc] for s, desc in STRATEGY_DESCRIPTIONS.items()], [16, ui.width - 20])


def print_providers() -> None:
    """Providers and their readiness (start-up and /providers)."""
    ui.rule("Providers")
    print_provider_status()


def print_provider_status() -> None:
    """One line per provider so the user knows which ones are ready to use."""
    rows = []
    for key, status in provider_status().items():
        if status["has_key"]:
            state = ui.good(f"{ui.g['ok']} ready") + (ui.dim(f"  {status['base_url']}") if status["local"] else "")
        elif status["local"]:
            state = ui.bad(f"{ui.g['warn']} not running") + ui.dim(f"  {status['hint'].split(' (')[0]}")
        else:
            state = ui.bad(f"{ui.g['warn']} missing {status['env_var']}")
        rows.append([ui.bold(key), status["label"], state])
    ui.columns(rows, [12, 26, ui.width - 42])


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions over the document knowledge base.")
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="chunks to retrieve")
    parser.add_argument("--strategy", default=config.QUERY_REWRITE_STRATEGY,
                        choices=[s.value for s in RewriteStrategy],
                        help="query rewriting strategy")
    parser.add_argument("--provider", default=config.LLM_PROVIDER, choices=list(PROVIDERS))
    parser.add_argument("--model", default=None, help="model id (defaults to provider default)")
    parser.add_argument("--embedding-model", default=config.EMBEDDING_MODEL,
                        help="sentence-transformers model (must match the ingested collection)")
    parser.add_argument("--list-models", action="store_true",
                        help="print the models available for --provider and exit")
    parser.add_argument("--hide-rewrites", action="store_true", help="do not print the query rewrites")
    parser.add_argument("--show-chunks", action="store_true", help="print the retrieved chunk text")
    parser.add_argument("--evaluate", action="store_true",
                        help="score each answer (faithfulness, precision, relevance, recall)")
    args = parser.parse_args()

    if args.list_models:
        print_models(args.provider)
        return

    pipeline = RAGPipeline(embedding_model=args.embedding_model,
                           provider=args.provider, model=args.model)
    strategy = RewriteStrategy.parse(args.strategy)
    k = args.k
    show_rewrites = not args.hide_rewrites
    show_chunks = args.show_chunks
    evaluate = args.evaluate
    show_bar = True   # the one-line command reference under the prompt (/bar toggles it)
    evaluator: AnswerEvaluator | None = None

    if pipeline.store.count() == 0:
        ui.status(False, "The vector store is empty. Put documents in data/ and run  python ingest.py  first.")
        sys.exit(1)

    def ask(question: str) -> None:
        try:
            response = pipeline.ask(question, k=k, strategy=strategy)
        except (MissingAPIKeyError, APIError, RuntimeError) as exc:
            # Missing key, unknown model (404), rate limit exhausted, provider
            # outage: say what happened and keep the interactive session alive.
            ui.blank()
            ui.status(False, f"{pipeline.provider}/{pipeline.model or PROVIDERS[pipeline.provider].default_model}: {exc}")
            if PROVIDERS[pipeline.provider].local and "onnect" in str(exc):
                ui.status(False, f"is the local server running?  {PROVIDERS[pipeline.provider].hint}")
            if args.question:
                sys.exit(1)
            return
        print_response(response, show_rewrites, show_chunks)
        if evaluate:
            nonlocal evaluator
            if evaluator is None:
                evaluator = AnswerEvaluator(pipeline.embedder)
            try:
                print_evaluation(evaluator.evaluate(response))
            except MissingAPIKeyError as exc:
                ui.status(False, f"Evaluation skipped: {exc}")

    if args.question:
        ask(args.question)
        return

    ui.blank()
    manifest = index_settings() or {}
    n_files = len(manifest.get("files", [])) or len(file_signature(config.DATA_DIR))
    ui.banner("Multi-Document RAG", f"{n_files} document(s) in {config.DATA_DIR.name}/")
    ui.blank()
    print_providers()
    ui.blank()
    print_strategies()
    ui.blank()
    print_commands()
    ui.blank()
    print_index_line(pipeline.store.count(), args.embedding_model)
    ui.print("  " + ui.dim("Type a question and press Enter."))

    while True:
        try:
            model_name = pipeline.model or PROVIDERS[pipeline.provider].default_model
            flags = (" " + ui.dim("[eval]") if evaluate else "") + (" " + ui.dim("[chunks]") if show_chunks else "")
            ui.blank()
            ui.print(f"{ui.g['tl']}{ui.g['h']} {ui.accent(f'{pipeline.provider}/{model_name}')} {ui.g['v']} "
                     f"{ui.bold(strategy.value)} {ui.g['v']} k={k}{flags}")
            if show_bar:
                ui.print(command_bar())
            line = input(f"{ui.g['bl']}{ui.g['h']}{ui.g['arrow']} ").strip()
        except (EOFError, KeyboardInterrupt):
            ui.blank()
            ui.status(True, "Bye.")
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "q"):
            break
        if line in ("/help", "/?", "help"):
            ui.blank()
            print_commands()
            continue
        if line == "/bar":
            show_bar = not show_bar
            ui.status(True, f"Command bar {ui.bold('on' if show_bar else 'off')}")
            continue

        # ---- settings commands ------------------------------------------
        if line.startswith("/strategies") or line == "/strategy":
            ui.blank()
            print_strategies()
            ui.status(True, f"Current strategy: {ui.bold(strategy.value)}  {ui.g['dot']}  switch with  /strategy <name>")
            continue
        if line.startswith("/strategy"):
            try:
                strategy = RewriteStrategy.parse(line.split(maxsplit=1)[1])
                ui.status(True, f"Strategy set to {ui.bold(strategy.value)}")
            except (IndexError, ValueError) as exc:
                ui.status(False, f"Usage: /strategy <{'|'.join(s.value for s in RewriteStrategy)}>  ({exc})")
            continue
        if line.startswith("/k"):
            try:
                k = int(line.split()[1])
                ui.status(True, f"k set to {ui.bold(str(k))}")
            except (IndexError, ValueError):
                ui.status(False, "Usage: /k <number>")
            continue
        if line.startswith("/providers") or line == "/provider":
            ui.blank()
            print_providers()
            ui.status(True, f"Current provider: {ui.bold(pipeline.provider)}  {ui.g['dot']}  switch with  /provider <name>")
            continue
        if line.startswith("/provider"):
            try:
                raw = line.split(maxsplit=1)[1].strip()
                name = resolve_provider(raw)
                if name is None:
                    raise ValueError(f"unknown provider '{raw}'")
                # A new provider means a new client; the retriever/store are untouched.
                pipeline.set_llm(provider=name, model=None)
                ui.status(True, f"Provider set to {ui.bold(name)} (model: {PROVIDERS[name].default_model})")
            except (IndexError, ValueError) as exc:
                ui.status(False, f"Usage: /provider <{'|'.join(PROVIDERS)}>  ({exc})")
            continue
        if line.startswith("/models"):
            print_models(pipeline.provider)
            continue
        if line.startswith("/model"):
            try:
                model_id = line.split(maxsplit=1)[1].strip()
            except IndexError:
                ui.status(False, "Usage: /model <model id>   (see /models)")
                continue
            # The most common slip: typing a provider name into /model.
            if resolve_provider(model_id):
                ui.status(False, f"'{model_id}' is a provider, not a model  {ui.g['dot']}  use  /provider {resolve_provider(model_id)}")
                continue
            if model_id not in list_models(pipeline.provider):
                ui.status(False, f"'{model_id}' is not in the model list for {pipeline.provider}  {ui.g['dot']}  see /models")
                continue
            pipeline.set_llm(provider=pipeline.provider, model=model_id)
            ui.status(True, f"Model set to {ui.bold(pipeline.model)}")
            continue
        if line.startswith("/chunks"):
            show_chunks = not show_chunks
            ui.status(True, f"Retrieved-chunk view {ui.bold('on' if show_chunks else 'off')}")
            continue
        if line.startswith("/eval"):
            evaluate = not evaluate
            ui.status(True, f"Per-answer evaluation {ui.bold('on' if evaluate else 'off')}")
            continue

        ask(line)


if __name__ == "__main__":
    main()
