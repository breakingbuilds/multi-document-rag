"""
src/embeddings.py
-----------------
Converts text (document chunks AND user queries) into dense vectors.

Key idea of RAG retrieval:
    The SAME embedding model must be used for documents at ingestion time
    and for the query at question time. Only then do "close vectors" mean
    "similar meaning". This module is the single place that owns that
    model, so nothing else in the project can accidentally use a different
    one.

Implementation notes
    * Backed by sentence-transformers, which runs locally on CPU -- free, no
      API key, works offline once the model weights are cached (~90 MB for
      all-MiniLM-L6-v2; the first run downloads them from Hugging Face).
    * Vectors are L2-normalised. For unit vectors the dot product equals the
      cosine similarity, which makes the maths in the retriever (and the
      "centroid" query-rewriting mode) simpler and faster.
    * Query embeddings are memoised with an LRU cache: the evaluation scripts
      re-runs scripts on every interaction and the evaluation scripts embed
      the same questions repeatedly, so caching avoids redundant work.
    * `centroid()` averages several vectors and re-normalises the result.
      It implements the MANUAL_CENTROID rewriting mode: embed the original
      question and its rule-based rewrites, average them, and search once
      with the blended vector.
"""

from functools import lru_cache

import numpy as np

from src.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, EMBEDDING_MODELS

# Loaded lazily and cached per model name so that:
#   - importing this module stays instant (no torch initialisation), and
#   - re-using a model name never reloads its weights.
_MODEL_CACHE: dict[str, "SentenceTransformer"] = {}  # noqa: F821 (forward ref)


def _get_model(model_name: str):
    """Return a (cached) SentenceTransformer instance for `model_name`."""
    if model_name not in _MODEL_CACHE:
        # Imported here, not at module top, to keep `import src.config`-style
        # light-weight imports fast (torch takes a couple of seconds to load).
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
    return _MODEL_CACHE[model_name]


class EmbeddingModel:
    """Thin wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        if model_name not in EMBEDDING_MODELS:
            # Unknown names are still allowed (any HF sentence-transformers
            # model works) but we warn so typos are noticed quickly.
            print(f"[embeddings] '{model_name}' is not in EMBEDDING_MODELS registry; "
                  "using it anyway.")
        self.model_name = model_name
        self._model = _get_model(model_name)
        # Ask the model for its real output size rather than trusting config.
        # sentence-transformers 6 renamed the accessor; support both spellings.
        get_dims = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dims = get_dims()

    # ------------------------------------------------------------------ #
    # Documents
    # ------------------------------------------------------------------ #
    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed many texts -> array of shape (len(texts), dims), rows unit-length."""
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit vectors -> dot == cosine
        )
        return vectors.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string -> 1-D unit vector of length dims."""
        return _cached_query_embedding(self.model_name, text.strip())

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Embed several query strings (e.g. original + rewrites) -> (n, dims)."""
        return np.stack([self.embed_query(t) for t in texts]) if texts else np.zeros((0, self.dims))


@lru_cache(maxsize=2048)
def _cached_query_embedding(model_name: str, text: str) -> np.ndarray:
    """Module-level cache so identical queries are embedded once per model."""
    vector = _get_model(model_name).encode(
        text, convert_to_numpy=True, normalize_embeddings=True
    )
    return vector.astype(np.float32)


# ---------------------------------------------------------------------- #
# Vector helpers used by the retriever / query rewriter
# ---------------------------------------------------------------------- #
def centroid(vectors: np.ndarray, weights: list[float] | None = None) -> np.ndarray:
    """Weighted mean of unit vectors, re-normalised to unit length.

    Used by the MANUAL_CENTROID strategy: blending the original query with
    its rewrites produces a vector that sits "between" the phrasings, which
    is more robust to wording than any single one of them.
    """
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("centroid() expects a non-empty 2-D array")
    if weights is None:
        mean = vectors.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        mean = (vectors * w).sum(axis=0) / w.sum()
    norm = np.linalg.norm(mean)
    return (mean / norm).astype(np.float32) if norm > 0 else mean.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (safe for non-unit input)."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0
