"""
src/vector_database.py
----------------------
Stores chunk vectors + metadata and performs similarity search.

Design
    * `VectorStore` is an abstract base class describing the five operations
      the rest of the project needs (add / search / count / sources / reset).
      The retriever and the CLI only ever talk to this interface.
    * `ChromaVectorStore` is the concrete implementation (ChromaDB running
      embedded in-process, persisted to disk under CHROMA_DIR -- no server).
    * `get_vector_store()` is a factory driven by a registry dict, so adding
      another backend later is a matter of writing one subclass and one
      registry line (see the comment above VECTOR_STORES).

Why does the collection name include the embedding model?
    Vectors from different models are NOT comparable (different sizes and
    different spaces). Naming the collection "rag_<model-slug>" guarantees
    that switching the embedding model (--embedding-model) transparently switches to a
    matching collection instead of silently returning garbage.

Similarity metric
    Collections are created with cosine distance. Chroma returns a DISTANCE
    (0 = identical, 2 = opposite); we convert it to a SCORE = 1 - distance
    so higher is always better everywhere else in the code.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import CHROMA_DIR, COLLECTION_PREFIX, EMBEDDING_MODEL, VECTOR_DB
from src.text_splitter import Chunk


@dataclass
class SearchHit:
    """One result from a similarity search."""

    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0  # cosine similarity in [-1, 1]; higher = more similar


def collection_name_for(embedding_model: str) -> str:
    """'sentence-transformers/all-MiniLM-L6-v2' -> 'rag_sentence-transformers_all-minilm-l6-v2'.

    Chroma requires 3-63 chars from [a-zA-Z0-9._-], starting and ending with
    an alphanumeric character.
    """
    slug = re.sub(r"[^a-z0-9._-]+", "_", embedding_model.lower()).strip("._-")
    return f"{COLLECTION_PREFIX}_{slug}"[:63].rstrip("._-")


def _clean_metadata(metadata: dict) -> dict:
    """Chroma only accepts str / int / float / bool values (no None, no lists)."""
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


# ---------------------------------------------------------------------- #
# Abstract interface
# ---------------------------------------------------------------------- #
class VectorStore(ABC):
    """What every vector database backend must provide."""

    name: str = "abstract"

    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Upsert chunks with their vectors (same order, same length)."""

    @abstractmethod
    def search(self, vector: np.ndarray, k: int, where: dict | None = None) -> list[SearchHit]:
        """Return the k most similar chunks to `vector`, best first."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored chunks."""

    @abstractmethod
    def sources(self) -> dict[str, int]:
        """{source file name: number of chunks} for everything stored."""

    @abstractmethod
    def reset(self) -> None:
        """Delete every stored chunk (the default full rebuild in ingest.py)."""


# ---------------------------------------------------------------------- #
# ChromaDB implementation
# ---------------------------------------------------------------------- #
class ChromaVectorStore(VectorStore):
    """Embedded, persistent ChromaDB collection."""

    name = "chroma"
    _ADD_BATCH = 500  # Chroma limits the number of records per add() call

    def __init__(self, embedding_model: str = EMBEDDING_MODEL, persist_dir: Path = CHROMA_DIR):
        import chromadb  # local import keeps module import cheap

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.collection_name = collection_name_for(embedding_model)

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._get_or_create()

    def _get_or_create(self):
        """Create the collection with cosine space (or open the existing one)."""
        return self._client.get_or_create_collection(
            name=self.collection_name,
            # "hnsw:space" selects the distance function for the index.
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model},
        )

    def _call(self, method: str, *args, **kwargs):
        """Invoke a collection method, refreshing a stale handle if needed.

        A collection handle stores the collection's UUID. If another process
        runs `ingest.py` (delete + recreate) while this one -- e.g.
        a long-running CLI session -- is open, the cached handle points at a UUID that
        no longer exists and Chroma raises NotFoundError. Re-opening the
        collection by NAME and retrying once makes the store self-healing.
        """
        from chromadb.errors import NotFoundError

        try:
            return getattr(self._collection, method)(*args, **kwargs)
        except NotFoundError:
            self._collection = self._get_or_create()
            return getattr(self._collection, method)(*args, **kwargs)

    # -- write ----------------------------------------------------------
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for start in range(0, len(chunks), self._ADD_BATCH):
            batch = chunks[start:start + self._ADD_BATCH]
            self._call("upsert",  # upsert: re-ingesting a file overwrites its chunks
                ids=[c.chunk_id for c in batch],
                embeddings=vectors[start:start + self._ADD_BATCH].tolist(),
                documents=[c.text for c in batch],
                metadatas=[_clean_metadata(c.metadata) for c in batch],
            )

    # -- read -----------------------------------------------------------
    def search(self, vector: np.ndarray, k: int, where: dict | None = None) -> list[SearchHit]:
        total = self.count()
        if total == 0:
            return []
        result = self._call("query",
            query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
            n_results=min(k, total),  # Chroma errors if k > stored items
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        # Chroma returns lists-of-lists (one inner list per query vector).
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(SearchHit(
                chunk_id=chunk_id,
                text=text,
                metadata=metadata or {},
                score=round(1.0 - float(distance), 4),  # cosine distance -> similarity
            ))
        return hits

    def count(self) -> int:
        return self._call("count")

    def sources(self) -> dict[str, int]:
        if self.count() == 0:
            return {}
        result = self._call("get", include=["metadatas"])
        counts: dict[str, int] = {}
        for metadata in result["metadatas"]:
            source = (metadata or {}).get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1
        return dict(sorted(counts.items()))

    def get_by_source(self, source: str, limit: int = 1000) -> list[SearchHit]:
        """All chunks of one file, in chunk order."""
        result = self._call(
            "get", where={"source": source}, limit=limit, include=["documents", "metadatas"]
        )
        hits = [
            SearchHit(chunk_id=i, text=t, metadata=m or {})
            for i, t, m in zip(result["ids"], result["documents"], result["metadatas"])
        ]
        return sorted(hits, key=lambda h: (h.metadata.get("global_index", 0)))

    # -- maintenance ----------------------------------------------------
    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._get_or_create()


# ---------------------------------------------------------------------- #
# Factory
# ---------------------------------------------------------------------- #
# To add another backend (FAISS, Qdrant, Pinecone, ...):
#   1. subclass VectorStore above and implement the five methods,
#   2. register it here under the name used in .env (VECTOR_DB=...).
VECTOR_STORES: dict[str, type[VectorStore]] = {
    "chroma": ChromaVectorStore,
}


def get_vector_store(
    backend: str = VECTOR_DB,
    embedding_model: str = EMBEDDING_MODEL,
    **kwargs,
) -> VectorStore:
    """Instantiate the configured vector store for the given embedding model."""
    try:
        store_cls = VECTOR_STORES[backend.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown vector database '{backend}'. Available: {', '.join(VECTOR_STORES)}"
        ) from None
    return store_cls(embedding_model=embedding_model, **kwargs)
