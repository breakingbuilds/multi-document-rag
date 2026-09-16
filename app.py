"""
app.py
------
Web front-end for the Multi-Document RAG system, built with Streamlit.

    streamlit run app.py

It is the SAME pipeline as the command-line app (main.py) behind a browser
page: every question goes through RAGPipeline.ask() and every panel on the
page is a view of the RAGResponse that call returns -- the rewrites, the
retrieved chunks with their scores, the sources with locators, the answer
with [n] citations and the four evaluation metrics. Nothing in src/ or
evaluation/ was changed to add this file; the helpers both front-ends need
live in src/frontend.py and the rendering helpers in ui/components.py.

Tabs
    Ask             chat with the documents; each answer can be unfolded into
                    the rewrites and chunks that produced it and scored live
    Knowledge base  what is indexed, upload documents into data/ or remove
                    them, rebuild the index, browse results/chunks.json
    Evaluation      run Stage A + the manual and RAGAS evaluations from the
                    browser and read the two reports side by side

Heavy objects (embedding model, vector store, judge LLM) are created once
per server process with st.cache_resource and shared by every browser tab;
provider and model switches only rebuild the small LLM client.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import APIError

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import manual_evaluation, ragas_evaluation  # noqa: E402
from evaluation.evaluation_utils import (  # noqa: E402
    METRIC_COLUMNS,
    check_dataset_matches_index,
    load_eval_dataset,
    load_rag_outputs,
    run_pipeline_on_dataset,
    save_rag_outputs,
    write_csv,
)
from ingest import ingest  # noqa: E402
from src import config  # noqa: E402
from src.frontend import AnswerEvaluator, data_changes, index_settings  # noqa: E402
from src.llm import PROVIDERS, MissingAPIKeyError, get_llm, list_models, provider_status  # noqa: E402
from src.query_rewriter import STRATEGY_DESCRIPTIONS, RewriteStrategy  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from ui import components as c  # noqa: E402
from ui.styles import CSS  # noqa: E402

st.set_page_config(page_title="Multi-Document RAG", page_icon="📄", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# ====================================================================== #
# Cached resources -- one per server process, shared by every session
# ====================================================================== #
@st.cache_resource(show_spinner="Loading the embedding model and the vector store…")
def load_pipeline(embedding_model: str) -> RAGPipeline:
    """The pipeline is keyed on the embedding model only: provider / model
    switches go through pipeline.set_llm(), which rebuilds just the client."""
    return RAGPipeline(embedding_model=embedding_model)


@st.cache_resource(show_spinner=False)
def load_evaluator(embedding_model: str) -> AnswerEvaluator:
    return AnswerEvaluator(load_pipeline(embedding_model).embedder)


@st.cache_data(ttl=60, show_spinner=False)
def cached_provider_status() -> dict:
    """Readiness per provider; cached so the Ollama port is not probed on every rerun."""
    return provider_status()


@st.cache_data(ttl=300, show_spinner=False)
def cached_models(provider: str) -> list[str]:
    return list_models(provider)


def reset_index_caches() -> None:
    """After an ingest the store contents changed: drop the cached objects so
    counts, manifest and the evaluator's embedder are rebuilt on next use."""
    load_pipeline.clear()
    load_evaluator.clear()


# ====================================================================== #
# Sidebar -- the same settings the CLI exposes as flags and /commands
# ====================================================================== #
def sidebar() -> dict:
    st.sidebar.title("Multi-Document RAG")
    st.sidebar.caption("Ask questions across PDF, Word, TXT, CSV, JSON, Markdown and HTML.")

    status = cached_provider_status()
    providers = list(PROVIDERS)

    def provider_label(key: str) -> str:
        ready = status[key]["has_key"]
        return f"{'🟢' if ready else '🔴'} {status[key]['label']}"

    provider = st.sidebar.radio(
        "LLM provider", providers, format_func=provider_label,
        index=providers.index(config.LLM_PROVIDER) if config.LLM_PROVIDER in providers else 0,
    )
    if not status[provider]["has_key"]:
        if status[provider]["local"]:
            st.sidebar.warning(f"Not running — {status[provider]['hint']}")
        else:
            st.sidebar.warning(f"No key: set {status[provider]['env_var']} in .env "
                               f"([get one]({status[provider]['signup_url']}))")

    models = cached_models(provider)
    default_model = status[provider]["default_model"]
    model = st.sidebar.selectbox(
        "Model", models, index=models.index(default_model) if default_model in models else 0,
        help="Curated models first, then everything the provider's live list reports.",
    )

    strategies = [s.value for s in RewriteStrategy]
    strategy = st.sidebar.radio(
        "Query rewriting", strategies,
        captions=[STRATEGY_DESCRIPTIONS[s] for s in RewriteStrategy],
        index=strategies.index(config.QUERY_REWRITE_STRATEGY) if config.QUERY_REWRITE_STRATEGY in strategies else 1,
    )
    k = st.sidebar.slider("Chunks retrieved (k)", 1, 12, config.TOP_K,
                          help="How many passages reach the prompt. Each search fetches max(2k, 10) candidates for RRF.")

    st.sidebar.divider()
    show_rewrites = st.sidebar.toggle("Show query rewriting", value=True)
    show_chunks = st.sidebar.toggle("Show retrieved chunks", value=False)
    evaluate = st.sidebar.toggle("Evaluate each answer", value=False,
                                 help="Faithfulness and answer relevance via the judge LLM; precision and recall when the question is in the evaluation set.")

    st.sidebar.divider()
    index_summary()
    return dict(provider=provider, model=model, strategy=strategy, k=k,
                show_rewrites=show_rewrites, show_chunks=show_chunks, evaluate=evaluate)


def index_summary() -> None:
    """The manifest in three lines, plus the same warnings the CLI prints at start-up."""
    manifest = index_settings()
    if not manifest:
        st.sidebar.error("No index yet — go to **Knowledge base** and rebuild it.")
        return
    st.sidebar.markdown(
        f"**Index** · {manifest.get('stored_chunks', manifest.get('chunks', '?'))} chunks from "
        f"{len(manifest.get('files', []))} files  \n"
        f"chunk size {manifest.get('chunk_size')} / overlap {manifest.get('chunk_overlap')}  \n"
        f"<span class='small'>{manifest.get('embedding_model', '')}</span>",
        unsafe_allow_html=True,
    )
    if (manifest.get("chunk_size"), manifest.get("chunk_overlap")) != (config.CHUNK_SIZE, config.CHUNK_OVERLAP):
        st.sidebar.warning(f".env says {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP} but the index was built with "
                           f"{manifest.get('chunk_size')}/{manifest.get('chunk_overlap')} — rebuild to apply.")
    changes = data_changes(manifest)
    if any(changes.values()):
        parts = [f"{len(v)} {kind}" for kind, v in changes.items() if v]
        st.sidebar.warning("data/ changed since the index was built: " + ", ".join(parts) + " — rebuild it.")


# ====================================================================== #
# Tab 1 -- Ask
# ====================================================================== #
def render_turn(turn: dict, settings: dict) -> None:
    """One question/answer pair from the chat history."""
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        if "error" in turn:
            st.error(turn["error"])
            return
        response = turn["response"]
        c.answer_block(response)
        c.timings_line(response)
        st.markdown("**Retrieved sources**")
        c.sources_table(response)
        if settings["show_rewrites"]:
            with st.expander(f"Query rewriting · {response.strategy} · "
                             f"{len(response.rewrite.all_queries)} phrasing(s) searched"):
                c.rewrites_panel(response)
        if settings["show_chunks"]:
            with st.expander(f"Retrieved chunks · {len(response.chunks)}"):
                c.chunks_panel(response)
        if turn.get("evaluation"):
            with st.expander("Evaluation · 0 = worst, 1 = best", expanded=True):
                c.evaluation_block(turn["evaluation"])


def ask_tab(pipeline: RAGPipeline, settings: dict) -> None:
    history = st.session_state.setdefault("history", [])

    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.caption("Every answer is grounded in the retrieved passages and cites them as [n]. "
                   "Open the expanders to see the rewrites and chunks behind it.")
    with top_right:
        if st.button("Clear chat", use_container_width=True, disabled=not history):
            history.clear()
            st.rerun()

    for turn in history:
        render_turn(turn, settings)

    question = st.chat_input("Ask a question about the documents…")
    if not question:
        return

    turn = {"question": question}
    history.append(turn)
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Rewriting → retrieving → generating…"):
                response = pipeline.ask(question, k=settings["k"], strategy=settings["strategy"])
            turn["response"] = response
            if settings["evaluate"]:
                with st.spinner("Scoring the answer…"):
                    turn["evaluation"] = load_evaluator(config.EMBEDDING_MODEL).evaluate(response)
        except MissingAPIKeyError as exc:
            turn["error"] = str(exc)
        except (APIError, RuntimeError) as exc:
            hint = ""
            if PROVIDERS[pipeline.provider].local and "onnect" in str(exc):
                hint = f" — is the local server running? {PROVIDERS[pipeline.provider].hint}"
            turn["error"] = f"{pipeline.provider}/{pipeline.model}: {exc}{hint}"
    st.rerun()   # redraw the whole history through render_turn() so the layout is uniform


# ====================================================================== #
# Tab 2 -- Knowledge base
# ====================================================================== #
def knowledge_tab(pipeline: RAGPipeline) -> None:
    manifest = index_settings()
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("What is indexed")
        if manifest:
            built = manifest.get("ingested_at", "")[:19].replace("T", " ")
            st.caption(f"Built {built} UTC · {manifest.get('documents')} parts → {manifest.get('chunks')} chunks "
                       f"in {manifest.get('seconds')} s · {manifest.get('collection')}")
            c.manifest_table(manifest)
            changes = data_changes(manifest)
            for kind, files in changes.items():
                if files:
                    st.warning(f"{kind.capitalize()} since the index was built: {', '.join(files)} — rebuild the index.")
        else:
            st.info("No index yet. Put documents in data/ (or upload them here) and rebuild.")

        st.subheader("Files in data/")
        files = sorted(p for p in config.DATA_DIR.rglob("*") if p.is_file() and not p.name.startswith("."))
        if files:
            st.dataframe(pd.DataFrame([{
                "file": str(p.relative_to(config.DATA_DIR)).replace("\\", "/"),
                "type": p.suffix.lstrip(".").upper(),
                "size (KB)": round(p.stat().st_size / 1024, 1),
                "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            } for p in files]), hide_index=True, use_container_width=True)
            # removing is the mirror of the uploader: the file leaves data/ here, the index
            # only follows when the user rebuilds (data_changes() flags it until then)
            names = [str(p.relative_to(config.DATA_DIR)).replace("\\", "/") for p in files]
            doomed = st.multiselect("Remove files from data/", names, placeholder="Choose files to delete…")
            if doomed and st.button(f"Delete {len(doomed)} file{'s' if len(doomed) > 1 else ''}", type="secondary"):
                for name in doomed:
                    (config.DATA_DIR / name).unlink(missing_ok=True)
                st.success(f"Deleted: {', '.join(doomed)} — now rebuild the index.")
                time.sleep(0.8)
                st.rerun()
        else:
            st.caption("data/ is empty.")

    with right:
        st.subheader("Add documents")
        uploads = st.file_uploader(
            "Drop files here", type=sorted(e.lstrip(".") for e in config.SUPPORTED_EXTENSIONS), accept_multiple_files=True,
            help="Saved into data/; the index is rebuilt only when you click the button below.",
        )
        overwrite = st.checkbox("Overwrite files with the same name", value=False)
        if uploads and st.button("Save to data/", type="secondary"):
            saved, skipped = [], []
            for up in uploads:
                target = config.DATA_DIR / Path(up.name).name
                if target.exists() and not overwrite:
                    skipped.append(up.name)
                    continue
                target.write_bytes(up.getbuffer())
                saved.append(up.name)
            if saved:
                st.success(f"Saved: {', '.join(saved)} — now rebuild the index.")
            if skipped:
                st.warning(f"Already present, not overwritten: {', '.join(skipped)}")
            st.rerun()

        st.subheader("Rebuild the index")
        col_a, col_b = st.columns(2)
        chunk_size = col_a.number_input("Chunk size (chars)", 200, 2000,
                                        int((manifest or {}).get("chunk_size", config.CHUNK_SIZE)), 50)
        chunk_overlap = col_b.number_input("Overlap (chars)", 0, 500,
                                           int((manifest or {}).get("chunk_overlap", config.CHUNK_OVERLAP)), 25)
        append = st.checkbox("Append to the existing index instead of rebuilding", value=False,
                             help="Rebuild (default) mirrors data/ exactly; append keeps chunks of files that were removed.")
        if st.button("Ingest now", type="primary"):
            with st.status("Ingesting…", expanded=True) as status:
                bar = st.progress(0.0)
                def progress(message: str, fraction: float):
                    status.write(message)
                    bar.progress(min(max(fraction, 0.0), 1.0))
                try:
                    result = ingest(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap),
                                    embedding_model=config.EMBEDDING_MODEL, reset=not append, progress=progress)
                except Exception as exc:  # empty folder, unsupported files, ...
                    status.update(label="Ingestion failed", state="error")
                    st.error(str(exc))
                    return
                status.update(label=f"Done: {result['documents']} parts → {result['chunks']} chunks "
                                    f"in {result['seconds']} s", state="complete")
            reset_index_caches()
            time.sleep(0.5)
            st.rerun()

    st.subheader("Browse the chunks")
    chunks_file = config.CHUNKS_FILE
    if not chunks_file.exists():
        st.caption("results/chunks.json appears after the first ingestion.")
        return
    payload = pd.read_json(chunks_file)
    chunks = pd.DataFrame(list(payload["chunks"]))
    f1, f2 = st.columns([2, 1])
    needle = f1.text_input("Find text in chunks", placeholder="e.g. annual leave")
    sources = f2.multiselect("Source file", sorted(chunks["source"].unique()))
    view = chunks
    if needle:
        view = view[view["text"].str.contains(needle, case=False, regex=False)]
    if sources:
        view = view[view["source"].isin(sources)]
    st.caption(f"{len(view)} of {len(chunks)} chunks · chunk size {payload['chunk_size'][0]} / overlap {payload['chunk_overlap'][0]}")
    st.dataframe(view[["chunk_id", "source", "locator", "characters", "text"]], hide_index=True,
                 use_container_width=True, height=320,
                 column_config={"text": st.column_config.TextColumn(width="large")})


# ====================================================================== #
# Tab 3 -- Evaluation
# ====================================================================== #
def _read_report(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for col in METRIC_COLUMNS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _means(df: pd.DataFrame | None) -> dict | None:
    if df is None:
        return None
    return {m: {"mean": round(float(df[m].mean()), 4) if df[m].notna().any() else None} for m in METRIC_COLUMNS}


def evaluation_tab(pipeline: RAGPipeline, settings: dict) -> None:
    dataset = load_eval_dataset()
    st.caption(f"{len(dataset)} questions with ground truth in evaluation/evaluation_data.json. "
               "Stage A answers them once; both markers score that same file.")

    with st.expander("The question set"):
        st.dataframe(pd.DataFrame([{
            "id": q["id"], "category": q.get("category", ""), "question": q["question"],
            "expected sources": ", ".join(q.get("expected_sources", [])),
        } for q in dataset]), hide_index=True, use_container_width=True)

    missing = check_dataset_matches_index(dataset, pipeline.store.sources())
    if missing:
        st.warning("The question set expects files that are not in the index: " + ", ".join(missing) +
                   ". It was written for the sample documents — with your own files, write a matching "
                   "question set first.")

    stage_a, manual_col, ragas_col = st.columns(3, gap="medium")

    with stage_a:
        st.markdown("**1 · Stage A — generate answers**")
        st.caption(f"k = {settings['k']} · strategy {settings['strategy']} · {settings['provider']}/{settings['model']}")
        outputs = config.RAG_OUTPUTS_FILE
        if outputs.exists():
            st.caption(f"rag_outputs.json · {datetime.fromtimestamp(outputs.stat().st_mtime):%Y-%m-%d %H:%M}")
        if st.button("Generate answers", disabled=bool(missing)):
            with st.status("Answering the question set…", expanded=True) as status:
                bar = st.progress(0.0)
                def progress(message, fraction):
                    status.write(message); bar.progress(fraction)
                try:
                    rows = run_pipeline_on_dataset(pipeline, dataset, k=settings["k"],
                                                   strategy=settings["strategy"], progress=progress)
                    save_rag_outputs(rows)
                    status.update(label=f"Saved {len(rows)} answers to results/rag_outputs.json", state="complete")
                except (MissingAPIKeyError, APIError, RuntimeError) as exc:
                    status.update(label="Failed", state="error"); st.error(str(exc))

    with manual_col:
        st.markdown("**2 · Manual evaluation**")
        no_judge = st.checkbox("Skip the judge LLM (recall + precision only)", value=False)
        st.caption("3 judge calls per question · about a minute")
        if st.button("Run manual evaluation", disabled=not config.RAG_OUTPUTS_FILE.exists()):
            with st.status("Scoring with the hand-written metrics…", expanded=True) as status:
                bar = st.progress(0.0)
                def progress(message, fraction):
                    status.write(message); bar.progress(fraction)
                try:
                    rows = load_rag_outputs()
                    judge = None if no_judge else get_llm(config.RAGAS_JUDGE_PROVIDER, config.RAGAS_JUDGE_MODEL)
                    rows = manual_evaluation.evaluate_rows(rows, embedder=pipeline.embedder, llm=judge, progress=progress)
                    write_csv(rows, config.MANUAL_REPORT_FILE, manual_evaluation.REPORT_COLUMNS)
                    status.update(label="Written to results/evaluation_report.csv", state="complete")
                except (MissingAPIKeyError, APIError, RuntimeError) as exc:
                    status.update(label="Failed", state="error"); st.error(str(exc))

    with ragas_col:
        st.markdown("**3 · RAGAS evaluation**")
        sleep = st.number_input("Pause between calls (s)", 0.0, 10.0, 1.0, 0.5,
                                help="The free plan allows 8K tokens/min; a pause keeps a 12-question run under it.")
        st.caption("~4 judge calls per question · about 12 minutes on the free plan")
        if st.button("Run RAGAS evaluation", disabled=not config.RAG_OUTPUTS_FILE.exists()):
            with st.status("Scoring with RAGAS — leave this tab open…", expanded=True) as status:
                bar = st.progress(0.0)
                def progress(message, fraction):
                    status.write(message); bar.progress(fraction)
                try:
                    rows = load_rag_outputs()
                    rows = ragas_evaluation.evaluate_rows(rows, sleep=float(sleep), progress=progress)
                    write_csv(rows, config.RAGAS_REPORT_FILE, ragas_evaluation.REPORT_COLUMNS)
                    status.update(label="Written to results/ragas_results.csv", state="complete")
                except Exception as exc:  # RAGAS raises its own error types
                    status.update(label="Failed", state="error"); st.error(f"{type(exc).__name__}: {exc}")

    st.divider()
    manual_df, ragas_df = _read_report(config.MANUAL_REPORT_FILE), _read_report(config.RAGAS_REPORT_FILE)
    if manual_df is None and ragas_df is None:
        st.info("No reports yet. Run Stage A, then one or both markers.")
        return

    st.subheader("Results")
    summary = c.summary_frame(_means(manual_df), _means(ragas_df))
    chart_col, table_col = st.columns([3, 2], gap="large")
    with chart_col:
        st.altair_chart(c.results_chart(summary), use_container_width=True)
        st.caption("Mean per metric over the question set · 0 = worst, 1 = best")
    with table_col:
        st.dataframe(summary.style.format("{:.2f}", na_rep="n/a"), use_container_width=True)
        for label, df, path in (("Manual report (CSV)", manual_df, config.MANUAL_REPORT_FILE),
                                ("RAGAS report (CSV)", ragas_df, config.RAGAS_REPORT_FILE)):
            if df is not None:
                st.download_button(label, path.read_bytes(), file_name=path.name, mime="text/csv")

    st.subheader("Per question")
    which = st.radio("Marker", [n for n, d in (("Manual", manual_df), ("RAGAS", ragas_df)) if d is not None],
                     horizontal=True, label_visibility="collapsed")
    df = manual_df if which == "Manual" else ragas_df
    cols = ["id", "category", "question"] + METRIC_COLUMNS + ["diagnosis"]
    st.dataframe(df[[col for col in cols if col in df]].style.format({m: "{:.2f}" for m in METRIC_COLUMNS}, na_rep="n/a"),
                 hide_index=True, use_container_width=True, height=440)
    note_col = "notes" if "notes" in df else "ragas_errors" if "ragas_errors" in df else None
    if note_col:
        with st.expander(f"{note_col} per question"):
            for _, row in df.iterrows():
                if str(row.get(note_col, "")).strip() and str(row.get(note_col)) != "nan":
                    st.markdown(f"**Q{row['id']}** — {row[note_col]}")


# ====================================================================== #
def main() -> None:
    settings = sidebar()
    pipeline = load_pipeline(config.EMBEDDING_MODEL)
    if pipeline.provider != settings["provider"] or (pipeline.model or None) != settings["model"]:
        pipeline.set_llm(provider=settings["provider"], model=settings["model"])

    tab_ask, tab_kb, tab_eval = st.tabs(["Ask", "Knowledge base", "Evaluation"])
    with tab_ask:
        if pipeline.store.count() == 0:
            st.warning("The index is empty — open **Knowledge base**, add documents and rebuild it.")
        ask_tab(pipeline, settings)
    with tab_kb:
        knowledge_tab(pipeline)
    with tab_eval:
        evaluation_tab(pipeline, settings)


main()
