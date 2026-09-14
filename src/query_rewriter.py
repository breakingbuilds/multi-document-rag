"""
src/query_rewriter.py
---------------------
Query rewriting: turning the user's question into one or more search
queries that retrieve better than the raw question would.

Why rewrite at all?
    Users type short, informal questions ("can i wfh fri?"). Documents are
    written formally ("Employees may work remotely up to 3 days per week").
    The embedding of the question and the embedding of the answer passage
    can therefore sit further apart than they should. Rewriting closes that
    gap by producing phrasings that look more like the documents.

Two families of rewriting are implemented, and both can be combined:

  A. MANUAL (rule-based, no API call, fully transparent)
     Ported from the earlier query-rewriting homework and extended:
       1. abbreviation expansion     "wfh" -> "work from home"
       2. keyword-anchored rewrite   "<topic keyword>: <expanded question>"
       3. formal restatement         "What does the <Topic> policy say about: ...?"
       4. keyword-only query         stop-words removed, content words kept
     Each rewrite is then EMBEDDED SEPARATELY and used for retrieval in one
     of two ways (chosen by the strategy):
       - MANUAL_RRF       search once per rewrite, fuse lists with RRF
       - MANUAL_CENTROID  average the rewrite embeddings into ONE vector
                          (original weighted x2) and search once with it

  B. LLM-BASED (one call to the chat model, falls back to MANUAL on error)
       - LLM_MULTI_QUERY  N paraphrases of the question           -> RRF
       - LLM_HYDE         a hypothetical answer passage (HyDE)     -> RRF with original
       - LLM_STEP_BACK    a broader "step-back" question           -> RRF with original

  C. HYBRID = manual rewrites + LLM paraphrases, all fused with RRF.

Rules shared by every strategy (same as the homework):
    * never answer the question during rewriting;
    * keep the original query in the set -- a rewrite can drop a detail;
    * do not invent facts the user did not state.

The output is a `RewriteResult`, which the CLI prints so the user can see
exactly which rewrites were generated and (via the retriever) which chunks
each one found.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from src.config import (
    ABBREVIATIONS,
    LLM_REWRITE_COUNT,
    QUERY_REWRITE_STRATEGY,
    STOPWORDS,
    TOPIC_DISPLAY,
    TOPIC_KEYWORDS,
)
from src.llm import BaseLLM
from src.prompt import (
    HYDE_SYSTEM,
    HYDE_USER,
    MULTI_QUERY_SYSTEM,
    MULTI_QUERY_USER,
    STEP_BACK_SYSTEM,
    STEP_BACK_USER,
)


# ---------------------------------------------------------------------- #
# Strategies
# ---------------------------------------------------------------------- #
class RewriteStrategy(str, Enum):
    NONE = "none"
    MANUAL_RRF = "manual_rrf"
    MANUAL_CENTROID = "manual_centroid"
    LLM_MULTI_QUERY = "llm_multi_query"
    LLM_HYDE = "llm_hyde"
    LLM_STEP_BACK = "llm_step_back"
    HYBRID = "hybrid"

    @classmethod
    def parse(cls, value: "str | RewriteStrategy | None") -> "RewriteStrategy":
        if isinstance(value, cls):
            return value
        value = (value or QUERY_REWRITE_STRATEGY).strip().lower()
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"Unknown rewrite strategy '{value}'. Options: {', '.join(s.value for s in cls)}"
            ) from None

    @property
    def uses_llm(self) -> bool:
        return self in (self.LLM_MULTI_QUERY, self.LLM_HYDE, self.LLM_STEP_BACK, self.HYBRID)

    @property
    def uses_centroid(self) -> bool:
        return self is RewriteStrategy.MANUAL_CENTROID


# One-line explanations printed by the CLI's strategy list.
STRATEGY_DESCRIPTIONS: dict[RewriteStrategy, str] = {
    RewriteStrategy.NONE: "Search with the raw question only (baseline).",
    RewriteStrategy.MANUAL_RRF: "Rule-based rewrites (abbreviations, topic keyword, formal restatement, "
                                "keywords) -> one search per rewrite -> Reciprocal Rank Fusion.",
    RewriteStrategy.MANUAL_CENTROID: "Same rule-based rewrites, but their embeddings are averaged into a "
                                     "single query vector (original weighted x2) -> one search.",
    RewriteStrategy.LLM_MULTI_QUERY: "The LLM writes several paraphrases of the question -> RRF.",
    RewriteStrategy.LLM_HYDE: "The LLM writes a hypothetical answer passage (HyDE); its embedding is "
                              "searched alongside the question -> RRF.",
    RewriteStrategy.LLM_STEP_BACK: "The LLM writes a broader 'step-back' question to pull in background "
                                   "context -> RRF with the original.",
    RewriteStrategy.HYBRID: "Manual rewrites + LLM paraphrases, all fused with RRF (most recall, "
                            "most calls).",
}


@dataclass
class RewriteResult:
    """Everything the pipeline and CLI need to know about a rewriting step."""

    original: str
    rewrites: list[str] = field(default_factory=list)   # WITHOUT the original
    strategy: RewriteStrategy = RewriteStrategy.NONE
    topic: str | None = None                             # detected topic key (manual)
    notes: list[str] = field(default_factory=list)       # human-readable explanation lines
    fallback: bool = False                               # True if the LLM failed -> manual used

    @property
    def all_queries(self) -> list[str]:
        """Original first, then unique rewrites (order preserved)."""
        seen = {self.original.strip().lower()}
        queries = [self.original.strip()]
        for rewrite in self.rewrites:
            key = rewrite.strip().lower()
            if key and key not in seen:
                seen.add(key)
                queries.append(rewrite.strip())
        return queries


# ---------------------------------------------------------------------- #
# A. Manual (rule-based) rewriter
# ---------------------------------------------------------------------- #
class ManualRewriter:
    """Transparent, deterministic rewrites -- no model, no network."""

    def __init__(
        self,
        topic_keywords: dict[str, list[str]] = TOPIC_KEYWORDS,
        topic_display: dict[str, str] = TOPIC_DISPLAY,
        abbreviations: dict[str, str] = ABBREVIATIONS,
        stopwords: set[str] = STOPWORDS,
    ):
        self.topic_keywords = topic_keywords
        self.topic_display = topic_display
        self.abbreviations = abbreviations
        self.stopwords = stopwords

    # -- building blocks --------------------------------------------------
    def expand_abbreviations(self, text: str) -> str:
        """'can i wfh on fri?' -> 'can i work from home on friday?' (meaning unchanged)."""
        result = text
        for abbr, full in self.abbreviations.items():
            result = re.sub(rf"\b{re.escape(abbr)}\b", full, result, flags=re.IGNORECASE)
        return result

    def detect_topics(self, query: str) -> dict[str, int]:
        """{topic: number of keyword hits} for topics mentioned in the query."""
        normalized = self.expand_abbreviations(query).lower()
        scores: dict[str, int] = {}
        for topic, keywords in self.topic_keywords.items():
            hits = sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", normalized))
            if hits:
                scores[topic] = hits
        return scores

    def keyword_only(self, query: str) -> str:
        """Drop stop-words and punctuation: 'how many annual leave days are allowed'
        -> 'annual leave days allowed'. Focuses the embedding on content words."""
        expanded = self.expand_abbreviations(query).lower()
        tokens = re.findall(r"[a-z0-9][a-z0-9\-']*", expanded)
        content = [t for t in tokens if t not in self.stopwords]
        return " ".join(content)

    # -- main entry point --------------------------------------------------
    def rewrite(self, query: str) -> RewriteResult:
        result = RewriteResult(original=query, strategy=RewriteStrategy.MANUAL_RRF)
        expanded = self.expand_abbreviations(query).strip()
        stripped = expanded.rstrip("?.! ")

        # 1. Abbreviation expansion (only worth adding if it changed something).
        if expanded.lower() != query.strip().lower():
            result.rewrites.append(expanded)
            result.notes.append(f"Expanded abbreviations: '{query.strip()}' -> '{expanded}'")

        # 2 + 3. Topic-based rewrites. Unlike the homework version we do not
        # stop to ask for clarification on a tie: vector search copes fine
        # with two anchored rewrites, and RRF rewards whichever one agrees
        # with the original question.
        scores = self.detect_topics(query)
        if scores:
            best = max(scores.values())
            top_topics = [t for t, s in scores.items() if s == best][:2]
            result.topic = top_topics[0]
            for topic in top_topics:
                primary_keyword = self.topic_keywords[topic][0]
                display = self.topic_display.get(topic, topic.replace("_", " ").title())
                result.rewrites.append(f"{primary_keyword}: {expanded}")
                result.rewrites.append(f"What does the {display} policy say about: {stripped}?")
            pretty = ", ".join(self.topic_display.get(t, t) for t in top_topics)
            result.notes.append(f"Detected topic(s): {pretty} (keyword hits: {best})")
        else:
            # No topic keyword at all: still produce a formal restatement so
            # informal phrasing gets normalised.
            result.rewrites.append(f"What do the company documents say about: {stripped}?")
            result.notes.append("No known topic keyword found; used a generic formal restatement.")

        # 4. Keyword-only query.
        keywords = self.keyword_only(query)
        if len(keywords.split()) >= 2 and keywords != expanded.lower():
            result.rewrites.append(keywords)
            result.notes.append(f"Keyword-only query: '{keywords}'")

        return result


# ---------------------------------------------------------------------- #
# B. LLM-based rewriter
# ---------------------------------------------------------------------- #
def _clean_lines(text: str) -> list[str]:
    """Turn a model reply into clean query strings, dropping numbering/bullets."""
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw).strip().strip('"').strip()
        if line and not line.lower().startswith(("here are", "sure", "alternative")):
            lines.append(line)
    return lines


class LLMRewriter:
    """Rewrites that need a chat model. Every method returns strings only."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def multi_query(self, query: str, n: int = LLM_REWRITE_COUNT) -> list[str]:
        reply = self.llm.generate(
            MULTI_QUERY_SYSTEM.format(n=n), MULTI_QUERY_USER.format(question=query),
            temperature=0.7, max_tokens=300,   # a little creativity helps diversity
        )
        return _clean_lines(reply)[:n]

    def hyde(self, query: str) -> str:
        reply = self.llm.generate(
            HYDE_SYSTEM, HYDE_USER.format(question=query), temperature=0.3, max_tokens=250,
        )
        return " ".join(reply.split())  # single paragraph

    def step_back(self, query: str) -> str:
        reply = self.llm.generate(
            STEP_BACK_SYSTEM, STEP_BACK_USER.format(question=query), temperature=0.2, max_tokens=100,
        )
        lines = _clean_lines(reply)
        return lines[0] if lines else query


# ---------------------------------------------------------------------- #
# Facade used by the pipeline
# ---------------------------------------------------------------------- #
class QueryRewriter:
    """Applies a RewriteStrategy and returns a RewriteResult.

    `llm_provider` is a zero-argument callable returning a BaseLLM. It is
    called lazily so that manual strategies never construct an LLM client
    (and therefore never need an API key).
    """

    def __init__(self, llm_provider: Callable[[], BaseLLM] | None = None,
                 manual: ManualRewriter | None = None):
        self.manual = manual or ManualRewriter()
        self._llm_provider = llm_provider

    def _llm_rewriter(self) -> LLMRewriter:
        if self._llm_provider is None:
            raise RuntimeError("An LLM is required for this rewrite strategy but none was configured.")
        return LLMRewriter(self._llm_provider())

    def rewrite(self, query: str, strategy: "str | RewriteStrategy | None" = None) -> RewriteResult:
        strategy = RewriteStrategy.parse(strategy)
        query = query.strip()

        if strategy is RewriteStrategy.NONE:
            return RewriteResult(original=query, strategy=strategy,
                                 notes=["No rewriting: searching with the raw question."])

        if strategy in (RewriteStrategy.MANUAL_RRF, RewriteStrategy.MANUAL_CENTROID):
            result = self.manual.rewrite(query)
            result.strategy = strategy
            if strategy.uses_centroid:
                result.notes.append("Embeddings of the original (x2 weight) and every rewrite are "
                                    "averaged into one query vector.")
            return result

        # ---- LLM strategies (with graceful fallback) ----------------------
        try:
            llm = self._llm_rewriter()
            if strategy is RewriteStrategy.LLM_MULTI_QUERY:
                rewrites = llm.multi_query(query)
                notes = [f"LLM ({llm.llm.model}) generated {len(rewrites)} paraphrase(s)."]
            elif strategy is RewriteStrategy.LLM_HYDE:
                rewrites = [llm.hyde(query)]
                notes = ["LLM wrote a hypothetical answer passage (HyDE); its embedding is "
                         "searched alongside the original question."]
            elif strategy is RewriteStrategy.LLM_STEP_BACK:
                rewrites = [llm.step_back(query)]
                notes = ["LLM wrote a broader step-back question to retrieve background context."]
            else:  # HYBRID
                manual = self.manual.rewrite(query)
                rewrites = manual.rewrites + llm.multi_query(query)
                notes = manual.notes + [f"Plus {len(rewrites) - len(manual.rewrites)} LLM paraphrase(s)."]
                result = RewriteResult(original=query, rewrites=rewrites, strategy=strategy,
                                       topic=manual.topic, notes=notes)
                return result
            return RewriteResult(original=query, rewrites=rewrites, strategy=strategy, notes=notes)

        except Exception as exc:
            # The demo must never crash because a key is missing or a call
            # failed: fall back to the manual rewriter and say so.
            result = self.manual.rewrite(query)
            result.strategy = RewriteStrategy.MANUAL_RRF
            result.fallback = True
            result.notes.insert(0, f"LLM rewriting failed ({type(exc).__name__}: {exc}); "
                                   "fell back to manual rewrites.")
            return result
