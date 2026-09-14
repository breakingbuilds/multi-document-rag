"""
src/retriever.py
----------------
Retrieves the top-k most relevant chunks for a query.

Three ways to retrieve
    retrieve(query, k)            plain single-query similarity search
    retrieve_by_vector(vec, k)    search with a pre-computed vector -- used by
                                  the MANUAL_CENTROID rewriting mode, which
                                  averages several query embeddings first
    retrieve_multi(queries, k)    run one search PER query variation, then
                                  merge the result lists with Reciprocal Rank
                                  Fusion (RRF)

Why Reciprocal Rank Fusion?
    Different phrasings of the same question return overlapping but not
    identical result lists, and their raw similarity scores are not directly
    comparable (a rewrite that repeats the topic keyword inflates scores for
    every chunk on that topic). RRF ignores the raw scores and only uses each
    chunk's RANK in each list:

        rrf_score(chunk) = sum over queries q of  1 / (K + rank_q(chunk))

    with K = 60 (RRF_K in config). A chunk that appears near the top of
    several lists accumulates a high score; a chunk that appears once, far
    down, does not. This is robust, needs no score normalisation, and is what
    most production hybrid-search systems use to combine result lists.

Duplicates are detected by `chunk_id` (stable metadata), NOT by comparing
text, exactly as in the earlier query-rewriting homework: the same chunk
found by three rewrites is ONE result with hit_count = 3 -- and that
agreement is itself a relevance signal, surfaced in the CLI's chunk view.
"""

from dataclasses import dataclass, field

import numpy as np

from src.config import RRF_K, TOP_K
from src.embeddings import EmbeddingModel
from src.vector_database import SearchHit, VectorStore


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever, with explanation of WHY it ranked."""

    hit: SearchHit
    rank: int                         # final 1-based rank handed to the LLM
    score: float                      # fused score (RRF) or cosine similarity
    similarity: float                 # best raw cosine similarity observed
    hit_count: int = 1                # how many query variations found it
    matched_queries: list[str] = field(default_factory=list)
    per_query_rank: dict[str, int] = field(default_factory=dict)

    # Convenience pass-throughs so callers can treat this like a chunk.
    @property
    def chunk_id(self) -> str:
        return self.hit.chunk_id

    @property
    def text(self) -> str:
        return self.hit.text

    @property
    def metadata(self) -> dict:
        return self.hit.metadata

    @property
    def source(self) -> str:
        return self.hit.metadata.get("source", "unknown")


class Retriever:
    """Similarity search on top of a VectorStore + EmbeddingModel pair."""

    def __init__(self, store: VectorStore, embedder: EmbeddingModel, rrf_k: int = RRF_K):
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------ #
    # Single query
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k: int = TOP_K) -> list[RetrievedChunk]:
        """Embed `query` with the SAME model used at ingestion and search."""
        vector = self.embedder.embed_query(query)
        return self.retrieve_by_vector(vector, k, label=query)

    def retrieve_by_vector(self, vector: np.ndarray, k: int = TOP_K,
                           label: str = "<vector>") -> list[RetrievedChunk]:
        """Search with an already-computed query vector."""
        hits = self.store.search(vector, k)
        return [
            RetrievedChunk(
                hit=hit, rank=rank, score=hit.score, similarity=hit.score,
                hit_count=1, matched_queries=[label], per_query_rank={label: rank},
            )
            for rank, hit in enumerate(hits, start=1)
        ]

    # ------------------------------------------------------------------ #
    # Multi query + RRF fusion
    # ------------------------------------------------------------------ #
    def retrieve_multi(self, queries: list[str], k: int = TOP_K,
                       candidate_k: int | None = None) -> list[RetrievedChunk]:
        """Search once per query string and fuse the ranked lists with RRF.

        `candidate_k` controls how deep each individual search goes. Looking
        a bit deeper than k (default 2k, min 10) gives the fusion step a
        richer candidate pool: a chunk ranked 7th by three different rewrites
        deserves to beat one ranked 2nd by a single rewrite.
        """
        if not queries:
            return []
        candidate_k = candidate_k or max(2 * k, 10)

        merged: dict[str, RetrievedChunk] = {}
        for query in queries:
            hits = self.store.search(self.embedder.embed_query(query), candidate_k)
            for rank, hit in enumerate(hits, start=1):
                contribution = 1.0 / (self.rrf_k + rank)
                entry = merged.get(hit.chunk_id)
                if entry is None:
                    merged[hit.chunk_id] = RetrievedChunk(
                        hit=hit, rank=0, score=contribution, similarity=hit.score,
                        hit_count=1, matched_queries=[query], per_query_rank={query: rank},
                    )
                else:
                    entry.score += contribution
                    entry.similarity = max(entry.similarity, hit.score)
                    entry.hit_count += 1
                    entry.matched_queries.append(query)
                    entry.per_query_rank[query] = rank

        # Highest fused score first; ties broken by raw similarity.
        ranked = sorted(merged.values(), key=lambda c: (-c.score, -c.similarity))[:k]
        for position, chunk in enumerate(ranked, start=1):
            chunk.rank = position
            chunk.score = round(chunk.score, 5)
        return ranked
