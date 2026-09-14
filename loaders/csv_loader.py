"""
loaders/csv_loader.py
---------------------
Loads CSV files, one `LoadedDocument` per data row plus one summary document.

Why one document per row?
    Tabular data is fundamentally different from prose. A single row such as
    "AD-009, Fatima Noor, Data Engineering, Senior Data Engineer, Karachi ..."
    is a self-contained fact, so it makes a natural retrieval unit. We render
    each row as "column: value; column: value" so the embedding model sees
    the column names as context ("manager: Hira Siddiqui" is far more
    informative than a bare "Hira Siddiqui").

Why an extra summary document?
    Questions such as "How many employees are in Data Engineering?" cannot
    be answered from any single row. The summary document contains the
    column list, the total row count and value counts for low-cardinality
    text columns (e.g. department, location, status), which gives the LLM
    enough evidence to answer aggregate questions honestly.
"""

from pathlib import Path

import pandas as pd

from .base import LoadedDocument, base_metadata

# Only columns with at most this many distinct values get value counts in
# the summary -- counting 25 distinct employee names is useless noise.
_MAX_CATEGORIES_FOR_COUNTS = 12


def _row_to_text(row: pd.Series) -> str:
    parts = []
    for column, value in row.items():
        if pd.isna(value) or str(value).strip() == "":
            continue  # skip empty cells (e.g. the CEO has no manager)
        parts.append(f"{column}: {value}")
    return "; ".join(parts)


def _summary_text(path: Path, df: pd.DataFrame) -> str:
    lines = [
        f"Summary of {path.name}",
        f"Columns: {', '.join(df.columns)}",
        f"Total rows (records): {len(df)}",
    ]
    for column in df.columns:
        series = df[column].dropna().astype(str)
        series = series[series.str.strip() != ""]
        if series.empty or series.nunique() > _MAX_CATEGORIES_FOR_COUNTS:
            continue
        counts = series.value_counts()
        rendered = ", ".join(f"{value}: {count}" for value, count in counts.items())
        lines.append(f"Count by {column}: {rendered}")
    return "\n".join(lines)


def load_csv(path: Path) -> list[LoadedDocument]:
    """Return one document per row followed by one aggregate summary document."""
    # dtype=str keeps IDs like "AD-001" and dates exactly as written.
    df = pd.read_csv(path, dtype=str)
    documents: list[LoadedDocument] = []

    for index, row in df.iterrows():
        text = _row_to_text(row)
        if not text:
            continue
        metadata = base_metadata(path)
        metadata["row"] = int(index) + 1  # 1-based, matches spreadsheet numbering
        documents.append(LoadedDocument(text=text, metadata=metadata))

    summary_metadata = base_metadata(path)
    summary_metadata["section"] = "summary"
    documents.append(LoadedDocument(text=_summary_text(path, df), metadata=summary_metadata))

    return documents
