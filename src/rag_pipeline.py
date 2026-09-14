"""
src/rag_pipeline.py
-------------------
Connects retrieval and answer generation into the complete RAG workflow:

    question -> (rewrite) -> embed -> similarity search -> top-k chunks
             -> prompt with numbered context -> LLM -> answer + citations

`RAGPipeline.ask()` returns a `RAGResponse` that carries EVERYTHING that
happened -- the rewrites, every retrieved chunk with its scores, the exact
context text sent to the model, the answer, the deduplicated source list and
per-stage timings. The CLI (main.py) and the evaluation scripts all consume
this one object, which keeps them perfectly consistent with each other.

Laziness
    The embedding model and vector store are created on first use and the
    LLM client is created on first `ask()`. Retrieval-only operations
    (`retrieve()`) therefore work without any API key.
"""

import re
import time
from dataclasses import dataclass, field

import numpy as np

from loaders.base import describe_location
from src import config
from src.embeddings import EmbeddingModel, centroid
from src.llm import BaseLLM, get_llm
from src.prompt import NO_ANSWER_SENTENCE, build_rag_messages, format_context
from src.query_rewriter import QueryRewriter, RewriteResult, RewriteStrategy
from src.retriever import RetrievedChunk, Retriever
from src.vector_database import VectorStore, get_vector_store

_CITATION = re.compile(r"\[(\d+)\]")


def normalize_citations(answer: str) -> str:
    """Coerce citation variants into the canonical "[n]" form.

    Models do not always follow the format exactly: gpt-oss likes full-width
    brackets ("【2】", sometimes with a line range "【2†L1-L4】"), others group
    numbers ("[1, 3]") or use ranges ("[2-4]"). Normalising here means the
    CLI's source list, the source grouping and the evaluation all see the
    same markers.
    """
    answer = re.sub(r"【\s*(\d+)(?:\s*†[^】]*)?\s*】", r"[\1]", answer)         # 【2】 / 【2†L1-L4】 -> [2]
    answer = re.sub(r"\[(\d+)\s*[-–]\s*(\d+)\]",                              # [2-4] -> [2][3][4]
                    lambda m: "".join(f"[{i}]" for i in range(int(m.group(1)), int(m.group(2)) + 1)
                                      if int(m.group(2)) - int(m.group(1)) < 20), answer)
    answer = re.sub(r"\[((?:\d+\s*,\s*)+\d+)\]",                              # [1, 3] -> [1][3]
                    lambda m: "".join(f"[{n.strip()}]" for n in m.group(1).split(",")), answer)
    return answer


@dataclass
class SourceRef:
    """One source DOCUMENT (file) that contributed to the answer."""

    source: str
    file_type: str
    locators: list[str] = field(default_factory=list)      # "page 2", "section: Benefits", ...
    chunk_ids: list[str] = field(default_factory=list)
    citation_numbers: list[int] = field(default_factory=list)  # the [n] numbers of its chunks
    best_score: float = 0.0
    cited: bool = False                                     # did the answer actually cite it?


@dataclass
class RAGResponse:
    """Full trace of one question through the pipeline."""

    question: str
    answer: str
    rewrite: RewriteResult
    chunks: list[RetrievedChunk]
    sources: list[SourceRef]
    context_text: str
    timings: dict[str, float]
    provider: str
    model: str
    strategy: str
    k: int
    cited_numbers: list[int] = field(default_factory=list)

    @property
    def contexts(self) -> list[str]:
        """Plain list of retrieved chunk texts -- the format RAGAS expects."""
        return [c.text for c in self.chunks]

    @property
    def refused(self) -> bool:
        """True when the model said it could not answer from the documents."""
        return self.answer.strip().startswith(NO_ANSWER_SENTENCE[:40])

    def to_dict(self) -> dict:
        """JSON-serialisable form used by the evaluation scripts."""
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "chunk_ids": [c.chunk_id for c in self.chunks],
            "sources": [s.source for s in self.sources],
            "cited_sources": [s.source for s in self.sources if s.cited],
            "rewrites": self.rewrite.rewrites,
            "strategy": self.strategy,
            "provider": self.provider,
            "model": self.model,
            "k": self.k,
            "timings": self.timings,
        }


class RAGPipeline:
    """The complete question-answering workflow."""

    def __init__(
        self,
        embedding_model: str = config.EMBEDDING_MODEL,
        vector_db: str = config.VECTOR_DB,
        provider: str | None = None,
        model: str | None = None,
        embedder: EmbeddingModel | None = None,
        store: VectorStore | None = None,
        llm: BaseLLM | None = None,
    ):
        self.embedding_model_name = embedding_model
        self.vector_db = vector_db
        self.provider = provider or config.LLM_PROVIDER
        self.model = model
        # Dependencies can be injected (the evaluation scripts share one
        # embedder across runs) or created lazily below.
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._retriever: Retriever | None = None
        self.rewriter = QueryRewriter(llm_provider=lambda: self.llm)

    # ------------------------------------------------------------------ #
    # Lazy components
    # ------------------------------------------------------------------ #
    @property
    def embedder(self) -> EmbeddingModel:
        if self._embedder is None:
            self._embedder = EmbeddingModel(self.embedding_model_name)
        return self._embedder

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = get_vector_store(self.vector_db, self.embedding_model_name)
        return self._store

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.store, self.embedder)
        return self._retriever

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = get_llm(self.provider, self.model)
        return self._llm

    def set_llm(self, provider: str | None = None, model: str | None = None) -> None:
        """Switch provider and/or model at runtime (used by the CLI's /provider
        and /model commands). The client is rebuilt lazily on the next ask();
        the embedder, store and retriever are untouched."""
        self.provider = provider or self.provider
        self.model = model
        self._llm = None

    # ------------------------------------------------------------------ #
    # Stage 1 + 2: rewrite and retrieve
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        question: str,
        k: int = config.TOP_K,
        strategy: "str | RewriteStrategy | None" = None,
    ) -> tuple[RewriteResult, list[RetrievedChunk], dict[str, float]]:
        """Rewrite the question (per strategy) and fetch the top-k chunks."""
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        rewrite = self.rewriter.rewrite(question, strategy)
        timings["rewrite_s"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        queries = rewrite.all_queries
        if rewrite.strategy is RewriteStrategy.NONE or len(queries) == 1:
            chunks = self.retriever.retrieve(question, k)
        elif rewrite.strategy.uses_centroid:
            # Blend the phrasings into one vector. The original question is
            # weighted twice so the rewrites refine rather than replace it.
            vectors = self.embedder.embed_queries(queries)
            weights = [2.0] + [1.0] * (len(queries) - 1)
            blended = centroid(np.asarray(vectors), weights)
            chunks = self.retriever.retrieve_by_vector(blended, k, label="centroid of rewrites")
        else:
            chunks = self.retriever.retrieve_multi(queries, k)
        timings["retrieve_s"] = round(time.perf_counter() - t0, 3)

        return rewrite, chunks, timings

    # ------------------------------------------------------------------ #
    # Stage 3: generate
    # ------------------------------------------------------------------ #
    def ask(
        self,
        question: str,
        k: int = config.TOP_K,
        strategy: "str | RewriteStrategy | None" = None,
        temperature: float | None = None,
    ) -> RAGResponse:
        """Run the full pipeline and return a RAGResponse."""
        question = question.strip()
        rewrite, chunks, timings = self.retrieve(question, k, strategy)

        messages = build_rag_messages(question, chunks)
        context_text = format_context(chunks)

        t0 = time.perf_counter()
        if chunks:
            answer = normalize_citations(self.llm.chat(messages, temperature=temperature))
        else:
            # Empty vector store (ingest not run yet) -> do not even call the LLM.
            answer = f"{NO_ANSWER_SENTENCE} (No documents have been ingested yet.)"
        timings["generate_s"] = round(time.perf_counter() - t0, 3)
        timings["total_s"] = round(sum(v for k_, v in timings.items() if k_ != "total_s"), 3)

        cited = sorted({int(n) for n in _CITATION.findall(answer) if 0 < int(n) <= len(chunks)})
        sources = self._group_sources(chunks, cited)

        return RAGResponse(
            question=question,
            answer=answer,
            rewrite=rewrite,
            chunks=chunks,
            sources=sources,
            context_text=context_text,
            timings=timings,
            provider=self.llm.provider,
            model=self.llm.model,
            strategy=rewrite.strategy.value,
            k=k,
            cited_numbers=cited,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _group_sources(chunks: list[RetrievedChunk], cited: list[int]) -> list[SourceRef]:
        """Collapse chunk-level results into one entry per source document,
        ordered by the best rank each document achieved."""
        by_source: dict[str, SourceRef] = {}
        for number, chunk in enumerate(chunks, start=1):
            ref = by_source.get(chunk.source)
            if ref is None:
                ref = by_source[chunk.source] = SourceRef(
                    source=chunk.source, file_type=chunk.metadata.get("file_type", ""),
                )
            locator = describe_location(chunk.metadata)
            if locator and locator not in ref.locators:
                ref.locators.append(locator)
            ref.chunk_ids.append(chunk.chunk_id)
            ref.citation_numbers.append(number)
            ref.best_score = max(ref.best_score, chunk.similarity)
            if number in cited:
                ref.cited = True
        # Cited documents first, then by best similarity.
        return sorted(by_source.values(), key=lambda s: (not s.cited, -s.best_score))

    @staticmethod
    def format_sources(response: RAGResponse) -> str:
        """Console rendering used by main.py."""
        lines = []
        for index, ref in enumerate(response.sources, start=1):
            marker = "*" if ref.cited else " "
            where = f" ({', '.join(ref.locators)})" if ref.locators else ""
            nums = ",".join(str(n) for n in ref.citation_numbers)
            lines.append(f"{index}. {marker} {ref.source}{where}  [passages {nums}]")
        return "\n".join(lines) if lines else "(none)"
