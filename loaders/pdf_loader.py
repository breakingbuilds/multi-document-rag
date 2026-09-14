"""
loaders/pdf_loader.py
---------------------
Extracts text from PDF files, one `LoadedDocument` per page.

Why one document per page (instead of one for the whole file)?
    Page numbers are the most natural citation unit for PDFs ("see page 3 of
    the policy"). Keeping pages separate means every chunk produced later
    inherits its page number automatically, so the final answer can show
    "company_policy.pdf, page 3".

Library: pypdf (pure Python, no system dependencies). It works well for
text-based PDFs (scanned images need OCR, which is not included).
Scanned/image-only PDFs would need OCR, which is out of scope here.
"""

from pathlib import Path

from pypdf import PdfReader

from .base import LoadedDocument, base_metadata, normalize_whitespace


def load_pdf(path: Path) -> list[LoadedDocument]:
    """Return one LoadedDocument per non-empty page of the PDF."""
    reader = PdfReader(str(path))
    documents: list[LoadedDocument] = []

    for page_number, page in enumerate(reader.pages, start=1):
        # extract_text() can return None for pages with no text layer.
        raw_text = page.extract_text() or ""
        text = normalize_whitespace(raw_text)
        if not text:
            continue  # skip blank pages: nothing to embed

        metadata = base_metadata(path)
        metadata["page"] = page_number
        metadata["total_pages"] = len(reader.pages)
        documents.append(LoadedDocument(text=text, metadata=metadata))

    return documents
