"""
loaders/txt_loader.py
---------------------
Loads plain-text files and splits them into logical sections.

Plain text has no formal structure, so we use a light heuristic: a line that
looks like a heading starts a new section. A line is treated as a heading if
it is short and either
    * numbered ("2. ANNUAL LEAVE", "3.1 Carryover"), or
    * written entirely in upper case ("PURPOSE").

Everything between two headings becomes one `LoadedDocument` whose
`section` metadata is the heading text. If the file has no recognisable
headings at all, the whole file becomes a single document -- the text
splitter will still break it into chunks later.
"""

import re
from pathlib import Path

from .base import LoadedDocument, base_metadata, normalize_whitespace

# "1. TITLE", "2.3 Sub title", "10) Title" ...
_NUMBERED_HEADING = re.compile(r"^\s*\d+(\.\d+)*[.)]?\s+\S")


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    if _NUMBERED_HEADING.match(stripped):
        return True
    # All-caps lines with at least one letter, e.g. "PURPOSE" / "LEAVE POLICY"
    letters = [c for c in stripped if c.isalpha()]
    return bool(letters) and stripped.upper() == stripped and len(stripped.split()) <= 8


def load_txt(path: Path) -> list[LoadedDocument]:
    """Return one LoadedDocument per heading-delimited section."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    documents: list[LoadedDocument] = []

    current_heading = "Preamble"
    buffer: list[str] = []

    def flush():
        if buffer and buffer[0].strip() == current_heading.strip() and not any(l.strip() for l in buffer[1:]):
            return  # heading with no body yet: merge it into the next section
        text = normalize_whitespace("\n".join(buffer))
        if text:
            metadata = base_metadata(path)
            metadata["section"] = current_heading
            documents.append(LoadedDocument(text=text, metadata=metadata))
        buffer.clear()

    for line in raw.splitlines():
        if _looks_like_heading(line):
            flush()
            current_heading = line.strip()
            buffer.append(current_heading)  # keep heading text inside the section
        else:
            buffer.append(line)

    flush()
    return documents
