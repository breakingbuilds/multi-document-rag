"""
loaders package
---------------
One loader per supported file format. Every loader has the same signature:

    load_xxx(path: Path) -> list[LoadedDocument]

and every returned document carries `source` + `file_type` metadata plus a
format-specific locator (page / row / section / item_index) used for
citations. See loaders/base.py for the data structure and conventions.

The mapping from file extension to loader lives in src/document_loader.py
so that this package stays free of any "which loader for which file" logic.
"""

from .base import LoadedDocument
from .csv_loader import load_csv
from .docx_loader import load_docx
from .html_loader import load_html
from .json_loader import load_json
from .markdown_loader import load_markdown
from .pdf_loader import load_pdf
from .txt_loader import load_txt

__all__ = [
    "LoadedDocument",
    "load_csv",
    "load_docx",
    "load_html",
    "load_json",
    "load_markdown",
    "load_pdf",
    "load_txt",
]
