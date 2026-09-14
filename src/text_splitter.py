"""
src/text_splitter.py
--------------------
Splits extracted text into overlapping chunks ready for embedding.

Why chunk at all?
    * Embedding models have a maximum input length, and a single vector for
      a 5-page document would be a blurry average of everything in it.
    * Smaller chunks give sharper vectors, so similarity search finds the
      exact paragraph that answers a question, and the LLM's context window
      is spent only on relevant text.

Why OVERLAP between neighbouring chunks?
    A fact can straddle a chunk boundary ("Employees may carry over a
    maximum of 5 unused annual leave days | into the next calendar year").
    Repeating the last `chunk_overlap` characters of one chunk at the start
    of the next means at least one chunk contains the whole sentence.

Algorithm: "recursive character splitting" (a from-scratch version of the
well-known LangChain approach, no dependency needed):
    1. Try to split on the most natural boundary first: blank lines
       (paragraphs). If the resulting pieces fit in `chunk_size`, merge
       consecutive pieces greedily up to that size.
    2. Any piece that is still too large is split again with the next,
       finer separator (single newline -> sentence end -> space -> character).
    Separators are kept attached to the piece before them so we never lose
    punctuation or newlines.

Output: `Chunk` objects with a stable `chunk_id` such as
    "leave_policy__s2-annual-leave__c0"  (file stem, locator, chunk number)
The id is what the vector store keys on and what the multi-query retriever
uses to detect that two rewrites found the same chunk.
"""

import hashlib
import re
from dataclasses import dataclass, field

from loaders.base import LoadedDocument
from src.config import CHUNK_OVERLAP, CHUNK_SIZE

# Coarse -> fine. "" means "split into single characters" (last resort).
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """One embeddable unit of text with full provenance metadata."""

    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _slugify(value: str, max_len: int = 24) -> str:
    """Make a locator safe for use inside an id: 'Leave at a Glance' -> 'leave-at-a-glance'."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug[:max_len].rstrip("-") or "x"


def _locator_token(metadata: dict) -> str:
    """Short locator fragment for the chunk id (p3 / r12 / i4 / s-section-name)."""
    if "page" in metadata:
        return f"p{metadata['page']}"
    if "row" in metadata:
        return f"r{metadata['row']}"
    if "item_index" in metadata:
        return f"i{metadata['item_index']}"
    if metadata.get("section"):
        # Section names can be long ("Our Offices > Dubai" vs "Our Offices >
        # Karachi" share a prefix), so the slug is truncated AND suffixed with
        # a short hash of the full name to keep ids unique and readable.
        section = str(metadata["section"])
        digest = hashlib.md5(section.encode("utf-8")).hexdigest()[:4]
        return f"s-{_slugify(section)}-{digest}"
    return "s-doc"


class RecursiveCharacterSplitter:
    """Split text into chunks of at most `chunk_size` characters with overlap."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: list[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

    # ------------------------------------------------------------------ #
    # Core algorithm
    # ------------------------------------------------------------------ #
    def _split_on(self, text: str, separator: str) -> list[str]:
        """Split but KEEP the separator attached to the preceding piece."""
        if separator == "":
            return list(text)
        # Positive look-behind: split right after every separator occurrence.
        pieces = re.split(f"(?<={re.escape(separator)})", text)
        return [p for p in pieces if p]

    def _merge(self, pieces: list[str]) -> list[str]:
        """Greedily pack pieces into chunks <= chunk_size, carrying overlap.

        The pieces already contain their separators, so joining is plain
        concatenation. When a chunk is full we emit it, then drop pieces from
        the FRONT of the window until the remaining tail is <= chunk_overlap;
        that tail becomes the start of the next chunk.
        """
        chunks: list[str] = []
        window: list[str] = []
        window_len = 0

        for piece in pieces:
            if window and window_len + len(piece) > self.chunk_size:
                chunks.append("".join(window).strip())
                # Shrink the window to the overlap budget (and make room for
                # the incoming piece so the next chunk is not oversized).
                while window and (
                    window_len > self.chunk_overlap
                    or window_len + len(piece) > self.chunk_size
                ):
                    window_len -= len(window.pop(0))
            window.append(piece)
            window_len += len(piece)

        if window:
            chunks.append("".join(window).strip())
        return [c for c in chunks if c]

    def split_text(self, text: str, separators: list[str] | None = None) -> list[str]:
        """Return a list of chunk strings for `text`."""
        separators = separators if separators is not None else self.separators
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        # Pick the coarsest separator that actually appears in the text.
        separator = separators[-1]
        remaining = []
        for index, candidate in enumerate(separators):
            if candidate == "" or candidate in text:
                separator = candidate
                remaining = separators[index + 1:]
                break

        pieces = self._split_on(text, separator)

        chunks: list[str] = []
        mergeable: list[str] = []  # pieces that already fit inside chunk_size
        for piece in pieces:
            if len(piece) <= self.chunk_size:
                mergeable.append(piece)
                continue
            # This piece alone is too big: flush what we have, then recurse
            # into it with the finer separators.
            if mergeable:
                chunks.extend(self._merge(mergeable))
                mergeable = []
            if remaining:
                chunks.extend(self.split_text(piece, remaining))
            else:  # no finer separator left -> hard cut
                chunks.extend(self._merge(list(piece)))
        if mergeable:
            chunks.extend(self._merge(mergeable))
        return chunks

    # ------------------------------------------------------------------ #
    # Document-level API
    # ------------------------------------------------------------------ #
    def split_documents(self, documents: list[LoadedDocument]) -> list[Chunk]:
        """Chunk every document, attaching provenance metadata to each chunk."""
        chunks: list[Chunk] = []
        used_ids: set[str] = set()   # safety net: ids MUST be unique for the vector store
        global_index = 0

        for doc in documents:
            stem = _slugify(doc.metadata.get("source", "doc").rsplit(".", 1)[0], max_len=40)
            locator = _locator_token(doc.metadata)
            search_from = 0

            for local_index, piece in enumerate(self.split_text(doc.text)):
                # Character offsets record where a chunk sits
                # in its parent document. find() from the last position keeps
                # repeated sentences (overlap) mapped to the right place.
                start = doc.text.find(piece[:40], search_from)
                if start == -1:
                    start = search_from
                end = start + len(piece)
                search_from = max(start + 1, start + len(piece) - self.chunk_overlap)

                metadata = dict(doc.metadata)  # copy: never mutate the source doc
                metadata.update({
                    "chunk_index": local_index,     # position within its document
                    "global_index": global_index,   # position within the whole corpus
                    "char_start": start,
                    "char_end": end,
                    "char_count": len(piece),
                })
                chunk_id = f"{stem}__{locator}__c{local_index}"
                if chunk_id in used_ids:  # two parts with identical locators
                    chunk_id = f"{chunk_id}-{global_index}"
                used_ids.add(chunk_id)
                metadata["chunk_id"] = chunk_id
                chunks.append(Chunk(chunk_id=chunk_id, text=piece, metadata=metadata))
                global_index += 1

        return chunks


def split_documents(
    documents: list[LoadedDocument],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Convenience wrapper used by ingest.py."""
    return RecursiveCharacterSplitter(chunk_size, chunk_overlap).split_documents(documents)
