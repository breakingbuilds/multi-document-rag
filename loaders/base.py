"""
loaders/base.py
---------------
The one data structure every loader returns, plus tiny helpers they share.

Why a common return type?
    The rest of the pipeline (text splitter, embeddings, vector store) should
    not care whether a piece of text came from page 3 of a PDF or row 17 of a
    CSV. Every loader therefore converts its format into a flat list of
    `LoadedDocument` objects: text + a metadata dictionary.

Metadata conventions (used later for citations in the answer):
    source      -> file name, e.g. "leave_policy.txt"        (always present)
    file_type   -> extension without the dot, e.g. "txt"     (always present)
    page        -> 1-based page number                       (PDF only)
    row         -> 1-based data row number                   (CSV only)
    section     -> heading / logical section the text belongs to (DOCX, TXT,
                   MD, HTML) or the JSON path for JSON items
    item_index  -> index of the item inside a JSON list      (JSON only)

Anything else a loader wants to record (e.g. "title" for HTML) is allowed;
the citation formatter simply shows whatever locator keys are present.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LoadedDocument:
    """A unit of extracted text plus where it came from."""

    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")


def base_metadata(path: Path) -> dict:
    """Metadata keys that EVERY document gets, regardless of format."""
    return {
        "source": path.name,
        "file_type": path.suffix.lower().lstrip("."),
    }


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and trim each line while keeping
    paragraph breaks (blank lines) intact. PDF extraction in particular
    produces lots of stray spacing that would otherwise waste chunk budget."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # Collapse 3+ consecutive blank lines into a single blank line.
    collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return collapsed.strip()


def describe_location(metadata: dict) -> str:
    """Turn locator metadata into a short human-readable string for citations.

    Examples: "page 3", "row 9", "item 4 (Payroll)", "section: Leave at a Glance"
    Returns "" when the document has no locator (e.g. a tiny single-part file).
    """
    if "page" in metadata:
        return f"page {metadata['page']}"
    if "row" in metadata:
        return f"row {metadata['row']}"
    if "item_index" in metadata:
        return metadata.get("section") or f"item {metadata['item_index'] + 1}"
    if metadata.get("section"):
        return f"section: {metadata['section']}"
    return ""
