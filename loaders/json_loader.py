"""
loaders/json_loader.py
----------------------
Loads JSON files into `LoadedDocument`s.

JSON can be shaped in countless ways, so this loader follows two rules:

    1. A top-level LIST becomes one document per item. This is the most
       common knowledge-base shape (FAQ entries, product records, tickets).
       Items that look like Q&A pairs (they have "question" and "answer"
       keys) are rendered as "Q: ... A: ..." so the embedding captures both
       halves together.

    2. Anything else (a top-level object, nested objects) is flattened
       recursively into "path.to.key: value" lines and emitted as ONE
       document. That is a safe generic fall-back for config-like JSON.

Metadata: `item_index` (position in the list) and `section` (a short label
such as the item's "category" or "id" when present) so the citation can say
"faq.json, item 4 (Payroll)".
"""

import json
from pathlib import Path

from .base import LoadedDocument, base_metadata


def _flatten(obj, prefix: str = "") -> list[str]:
    """Recursively turn nested dicts/lists into 'a.b.c: value' lines."""
    lines: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            lines.extend(_flatten(value, f"{prefix}{key}."))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            lines.extend(_flatten(value, f"{prefix}{index}."))
    else:
        lines.append(f"{prefix.rstrip('.')}: {obj}")
    return lines


def _item_to_text(item) -> str:
    """Render a single list item as readable text."""
    if isinstance(item, dict):
        # Special case: FAQ-style records read best as an explicit Q/A pair.
        if "question" in item and "answer" in item:
            extras = [f"{k}: {v}" for k, v in item.items() if k not in ("question", "answer")]
            head = " | ".join(extras)
            body = f"Q: {item['question']}\nA: {item['answer']}"
            return f"{head}\n{body}" if head else body
        return "\n".join(_flatten(item))
    return str(item)


def _item_label(item, index: int) -> str:
    """A short human-readable locator for citations."""
    if isinstance(item, dict):
        for key in ("category", "title", "name", "id"):
            if key in item:
                return f"item {index + 1} ({item[key]})"
    return f"item {index + 1}"


def load_json(path: Path) -> list[LoadedDocument]:
    data = json.loads(path.read_text(encoding="utf-8"))
    documents: list[LoadedDocument] = []

    if isinstance(data, list):
        for index, item in enumerate(data):
            text = _item_to_text(item).strip()
            if not text:
                continue
            metadata = base_metadata(path)
            metadata["item_index"] = index
            metadata["section"] = _item_label(item, index)
            documents.append(LoadedDocument(text=text, metadata=metadata))
    else:
        metadata = base_metadata(path)
        metadata["section"] = "root"
        documents.append(LoadedDocument(text="\n".join(_flatten(data)), metadata=metadata))

    return documents
