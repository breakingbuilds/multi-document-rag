"""
ui/components.py
----------------
Turns the objects the pipeline already produces into Streamlit widgets.
Every function here is a pure renderer: it receives a RAGResponse, a
manifest dict or report rows and draws them -- no retrieval, no LLM calls.

The CLI renders the same objects through src/console.py; keeping the two
renderers separate means a change to the pipeline output only needs to
be reflected in two small files, and neither front-end imports the other.
"""

from __future__ import annotations

import html
import re

import pandas as pd
import streamlit as st

from loaders.base import describe_location
from src.frontend import relevance_label
from src.rag_pipeline import RAGResponse

_CITATION = re.compile(r"\[(\d+)\]")


# ---------------------------------------------------------------------- #
# Answer
# ---------------------------------------------------------------------- #
def _chip(number: int, response: RAGResponse) -> str:
    """One [n] marker as a chip whose tooltip names the source it points at."""
    if 0 < number <= len(response.chunks):
        chunk = response.chunks[number - 1]
        where = describe_location(chunk.metadata)
        title = f"[{number}] {chunk.source}" + (f" - {where}" if where else "")
    else:
        title = f"[{number}] (no such passage)"
    return f'<span class="cite" title="{html.escape(title)}">{number}</span>'


def answer_block(response: RAGResponse) -> None:
    """The answer text with [n] rendered as chips; refusals get an amber bar."""
    text = html.escape(response.answer)
    text = _CITATION.sub(lambda m: _chip(int(m.group(1)), response), text)
    css = "answer refusal" if response.refused else "answer"
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)


def timings_line(response: RAGResponse) -> None:
    t = response.timings
    st.caption(
        f"{response.provider}/{response.model} · {response.strategy} · k={response.k}  |  "
        f"rewrite {t.get('rewrite_s', 0):.3f}s · retrieve {t.get('retrieve_s', 0):.2f}s · "
        f"generate {t.get('generate_s', 0):.2f}s"
    )


# ---------------------------------------------------------------------- #
# Sources and rewrites
# ---------------------------------------------------------------------- #
def sources_table(response: RAGResponse) -> None:
    """One row per source FILE, star = cited, with locators and passage numbers."""
    rows = [{
        "": "★" if ref.cited else "",
        "source": ref.source,
        "where": ", ".join(ref.locators) if ref.locators else "",
        "passages": ", ".join(str(n) for n in ref.citation_numbers),
        "best score": round(ref.best_score, 3),
    } for ref in response.sources]
    if not rows:
        st.caption("No sources retrieved.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"": st.column_config.TextColumn(width="small"),
                                "passages": st.column_config.TextColumn(width="small"),
                                "best score": st.column_config.NumberColumn(format="%.3f", width="small")})
    st.caption("★ = cited in the answer · passages = the [n] numbers this file's chunks had in the prompt")


def rewrites_panel(response: RAGResponse) -> None:
    """The rewriter's notes plus the numbered phrasings that were searched."""
    rw = response.rewrite
    for note in rw.notes:
        st.markdown(f"- {html.escape(note)}")
    if rw.fallback:
        st.warning("The LLM rewrite failed; the rule-based rewrites were used instead.")
    lines = []
    for index, query in enumerate(rw.all_queries):
        tag = "original" if index == 0 else f"rewrite {index}"
        lines.append(f'<div class="rewrite"><span class="tag">{tag}</span>{html.escape(query)}</div>')
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# ---------------------------------------------------------------------- #
# Retrieved chunks
# ---------------------------------------------------------------------- #
def chunks_panel(response: RAGResponse) -> None:
    """Every retrieved chunk: rank, id, location, scores, which phrasings found it, text."""
    for chunk in response.chunks:
        where = describe_location(chunk.metadata)
        ranks = " · ".join(f"{('original' if q == response.question else q[:28] + ('…' if len(q) > 28 else ''))}: #{r}"
                           for q, r in chunk.per_query_rank.items())
        st.markdown(
            f'<div class="chunk">'
            f'<div class="head">[{chunk.rank}] {html.escape(chunk.source)}'
            f'{(" — " + html.escape(where)) if where else ""}</div>'
            f'<div class="meta">{html.escape(chunk.chunk_id)} · score {chunk.score:.4f} · '
            f'similarity {chunk.similarity:.3f} · found by {chunk.hit_count} phrasing(s)'
            f'{(" · " + html.escape(ranks)) if ranks else ""}</div>'
            f'<pre>{html.escape(chunk.text)}</pre></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------- #
# Evaluation block (same four metrics, same order as the brief)
# ---------------------------------------------------------------------- #
def _metric_row(label: str, value: float | None, shown: str | None = None) -> str:
    if value is None:
        return f'<div class="metric"><span>{label}</span><span class="na">n/a</span><span></span></div>'
    width = max(0.0, min(1.0, value)) * 100
    return (f'<div class="metric"><span>{label}</span><span>{shown or f"{value:.2f}"}</span>'
            f'<span class="bar"><div style="width:{width:.0f}%"></div></span></div>')


def evaluation_block(result: dict) -> None:
    rows = [
        _metric_row("Faithfulness", result["faithfulness"]),
        _metric_row("Context Precision", result["context_precision"]),
        _metric_row("Answer Relevance", result["answer_relevance"], relevance_label(result["answer_relevance"])),
        _metric_row("Context Recall", result["context_recall"]),
    ]
    st.markdown("\n".join(rows), unsafe_allow_html=True)
    if result.get("ground_truth") is None:
        st.caption("Precision and recall need a ground-truth answer; this question is not in "
                   "evaluation/evaluation_data.json. Faithfulness and relevance are scored for any question.")
    else:
        verdict = result["diagnosis"]
        (st.success if verdict == "ok" else st.warning)(f"Diagnosis: {verdict}")
    if result.get("faithfulness_note"):
        st.caption(result["faithfulness_note"])


# ---------------------------------------------------------------------- #
# Knowledge base
# ---------------------------------------------------------------------- #
def manifest_table(manifest: dict) -> None:
    files = manifest.get("files", [])
    if not files:
        st.caption("No files recorded in the manifest.")
        return
    df = pd.DataFrame([{
        "file": f["file"], "type": f.get("file_type", "").upper(), "parts": f.get("documents", 0),
        "characters": f.get("characters", 0), "chunks": f.get("chunks", 0), "status": f.get("status", ""),
    } for f in files])
    st.dataframe(df, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------- #
# Batch evaluation results
# ---------------------------------------------------------------------- #
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer relevance",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
}


def summary_frame(manual: dict | None, ragas: dict | None) -> pd.DataFrame:
    """Mean per metric for the two markers, as a small DataFrame for charts/tables."""
    data = {}
    if manual:
        data["Manual"] = [manual[m]["mean"] for m in METRIC_LABELS]
    if ragas:
        data["RAGAS"] = [ragas[m]["mean"] for m in METRIC_LABELS]
    return pd.DataFrame(data, index=list(METRIC_LABELS.values()))


def results_chart(summary: pd.DataFrame):
    """Grouped (not stacked) bars, metrics in the brief's order, 0-1 axis."""
    import altair as alt

    long = summary.reset_index().melt(id_vars="index", var_name="marker", value_name="score")
    long = long.rename(columns={"index": "metric"})
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("metric:N", sort=list(METRIC_LABELS.values()), title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("marker:N"),
            y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1]), title="mean score"),
            color=alt.Color("marker:N", title=None, scale=alt.Scale(range=["#378ADD", "#FCA311"])),
            tooltip=["metric", "marker", alt.Tooltip("score:Q", format=".2f")],
        )
        .properties(height=280)
    )
