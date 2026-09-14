"""
src/document_loader.py
----------------------
Detects a file's type and dispatches to the matching loader in loaders/.

This is the ONLY place in the project that knows which extension maps to
which loader. Adding support for a new format is a two-step job:

    1. write loaders/my_format_loader.py with load_my_format(path) -> list[LoadedDocument]
    2. add its extension to the LOADERS table below (and to
       SUPPORTED_EXTENSIONS in src/config.py so load_directory() picks it up)

Public API
    load_document(path)   -> list[LoadedDocument]   for one file
    load_directory(dir)   -> (documents, summary)   for every supported file in a folder
"""

from pathlib import Path
from typing import Callable

from loaders import (
    LoadedDocument,
    load_csv,
    load_docx,
    load_html,
    load_json,
    load_markdown,
    load_pdf,
    load_txt,
)
from src.config import DATA_DIR, SUPPORTED_EXTENSIONS

# extension -> loader function. Keys are lower-case and include the dot.
LOADERS: dict[str, Callable[[Path], list[LoadedDocument]]] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
    ".csv": load_csv,
    ".json": load_json,
    ".md": load_markdown,
    ".html": load_html,
    ".htm": load_html,
}

# Sanity check at import time: config and this table must agree, otherwise
# a file could be dropped into data/ that ingestion then silently skips.
assert set(LOADERS) == SUPPORTED_EXTENSIONS, "LOADERS and SUPPORTED_EXTENSIONS are out of sync"


class UnsupportedFileTypeError(ValueError):
    """Raised when load_document() is given a file with an unknown extension."""


def load_document(path: Path | str) -> list[LoadedDocument]:
    """Load ONE file, returning its documents. Raises for unsupported types."""
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedFileTypeError(
            f"{path.name}: unsupported file type '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return loader(path)


# Files that operating systems and editors drop into folders; never documents.
_IGNORED_NAMES = {"desktop.ini", "thumbs.db", ".ds_store", ".gitkeep"}


def iter_files(directory: Path | str = DATA_DIR) -> list[Path]:
    """Every regular file under `directory`, recursively, in a stable order.

    Hidden files (names starting with ".") and OS clutter are skipped, so an
    empty-looking folder really is treated as empty.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.name.lower() in _IGNORED_NAMES or any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        files.append(path)
    return files


def file_signature(directory: Path | str = DATA_DIR) -> dict[str, dict]:
    """{relative path: {size, mtime}} for every file -- cheap change detection.

    Stored in the ingest manifest so main.py can tell the user when the
    documents in data/ changed after the index was built.
    """
    directory = Path(directory)
    return {
        path.relative_to(directory).as_posix(): {"size": path.stat().st_size, "mtime": int(path.stat().st_mtime)}
        for path in iter_files(directory)
    }


def load_directory(directory: Path | str = DATA_DIR) -> tuple[list[LoadedDocument], list[dict]]:
    """Load every supported file under `directory` (sub-folders included).

    Returns
        documents : flat list of LoadedDocument from all files
        summary   : one dict per file -> {file, file_type, documents, characters, status}
                    (used by ingest.py's console table and the manifest)
    The "source" recorded in every document's metadata is the path relative
    to `directory` (just the file name for top-level files), so two files with
    the same name in different sub-folders stay distinguishable in citations
    and chunk ids. Unsupported files are listed with status "skipped" rather
    than raising, so one stray .xlsx does not abort ingestion.
    """
    directory = Path(directory)
    documents: list[LoadedDocument] = []
    summary: list[dict] = []

    for path in iter_files(directory):
        rel = path.relative_to(directory).as_posix()
        entry = {"file": rel, "file_type": path.suffix.lower().lstrip("."),
                 "documents": 0, "characters": 0, "status": "ok"}
        try:
            docs = load_document(path)
            for doc in docs:
                doc.metadata["source"] = rel
            documents.extend(docs)
            entry["documents"] = len(docs)
            entry["characters"] = sum(len(d.text) for d in docs)
        except UnsupportedFileTypeError:
            entry["status"] = "skipped (unsupported type)"
        except Exception as exc:  # a corrupt file should not stop the others
            entry["status"] = f"error: {exc}"
        summary.append(entry)

    return documents, summary
