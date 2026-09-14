"""
loaders/markdown_loader.py
--------------------------
Loads Markdown files, splitting on headings and keeping the heading path.

Markdown headings ("#", "##", "###") give us a ready-made outline. Each
heading starts a new `LoadedDocument`, and the metadata records the full
heading path ("Q3 2026 All-Hands Meeting Notes > People Operations
announcements") so a citation can point at the exact subsection.

Light clean-up is applied so the embedding model is not distracted by
syntax: emphasis markers (**bold**, _italic_), inline code back-ticks, link
targets and task-list check-boxes are stripped, while the words are kept.
"""

import re
from pathlib import Path

from .base import LoadedDocument, base_metadata, normalize_whitespace

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _strip_markdown(text: str) -> str:
    """Remove the most common Markdown syntax while keeping the words."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)          # **bold**
    text = re.sub(r"(?<!\w)[_*](.+?)[_*](?!\w)", r"\1", text)  # _italic_ / *italic*
    text = re.sub(r"`([^`]+)`", r"\1", text)              # `code`
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"^\s*-\s*\[[ xX]\]\s*", "- ", text, flags=re.MULTILINE)  # task boxes
    return text


def load_markdown(path: Path) -> list[LoadedDocument]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    documents: list[LoadedDocument] = []

    # heading_stack[level-1] holds the current heading at that level, so we
    # can rebuild the full path "H1 > H2 > H3" for each section.
    heading_stack: list[str] = []
    buffer: list[str] = []

    def flush():
        if buffer and buffer[0].strip() == (heading_stack[-1] if heading_stack else '').strip() and not any(l.strip() for l in buffer[1:]):
            return  # heading with no body yet: merge it into the next section
        text = normalize_whitespace(_strip_markdown("\n".join(buffer)))
        if text:
            metadata = base_metadata(path)
            metadata["section"] = " > ".join(heading_stack) or "Preamble"
            documents.append(LoadedDocument(text=text, metadata=metadata))
        buffer.clear()

    for line in raw.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            # Truncate the stack to the parent level, then push this heading.
            del heading_stack[level - 1:]
            heading_stack.extend([""] * (level - 1 - len(heading_stack)))
            heading_stack.append(title)
            buffer.append(title)  # keep heading words inside the section text
        else:
            buffer.append(line)

    flush()
    return documents
