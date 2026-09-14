"""
ingest.py
---------
Ingestion pipeline:  Documents -> Load & extract -> Chunk -> Embed -> Vector DB

Put your documents (PDF, DOCX, TXT, CSV, JSON, Markdown, HTML -- sub-folders
allowed) in data/ and run this once, and again whenever the files change:

    python ingest.py            # REBUILD the index from everything in data/
    python ingest.py --append   # add/refresh files, keep chunks of files no longer present
    python ingest.py --chunk-size 500 --chunk-overlap 100
    python ingest.py --embedding-model BAAI/bge-small-en-v1.5
    python ingest.py --data-dir "D:/other/folder"

A full rebuild is the default because it is the only way to make the index
match the folder exactly: chunks are upserted by id, so a file you DELETED
would otherwise linger in the collection.

What happens, step by step
    1. src.document_loader.load_directory() reads every supported file in
       data/ through the format-specific loaders (PDF, DOCX, TXT, CSV, JSON,
       Markdown, HTML) and returns LoadedDocument objects with metadata.
    2. src.text_splitter splits each document into overlapping chunks and
       assigns every chunk a stable chunk_id.
    3. src.embeddings turns each chunk's text into a vector.
    4. src.vector_database stores vector + text + metadata in ChromaDB.
    5. Two inspection files are written to results/:
         - chunks.json          every chunk exactly as it was embedded (id,
                                source, locator, text) -- open it to see how
                                the splitter cut your documents;
         - ingest_manifest.json per-file counts, model, timings.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# Windows consoles default to a legacy code page (cp1252) that cannot print
# characters LLMs like to emit (curly quotes, non-breaking hyphens, ...).
# Switching stdout to UTF-8 avoids UnicodeEncodeError crashes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tqdm import tqdm

from src.console import ui
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_FILE,
    DATA_DIR,
    EMBEDDING_MODEL,
    INGEST_MANIFEST_FILE,
    VECTOR_DB,
)
from src.document_loader import file_signature, iter_files, load_directory
from src.embeddings import EmbeddingModel
from src.text_splitter import split_documents
from src.vector_database import get_vector_store


def _locator(metadata: dict) -> str:
    """Human-readable position inside the source file: 'page 2', 'row 7',
    'item 3', 'section: Benefits' -- whichever the loader recorded."""
    if "page" in metadata:
        return f"page {metadata['page']}"
    if "row" in metadata:
        return f"row {metadata['row']}"
    if "item_index" in metadata:
        return f"item {metadata['item_index']}"
    if metadata.get("section"):
        return f"section: {metadata['section']}"
    return ""


def write_chunks_file(chunks, chunk_size: int, chunk_overlap: int, path: Path = CHUNKS_FILE) -> Path:
    """Dump every chunk to results/chunks.json so the split can be inspected.

    The file is purely for humans (and for debugging retrieval): the vector
    store holds the same text and metadata, but there it is hidden behind
    the ChromaDB API. Chunks are written in ingestion order, which is also
    document order, so neighbouring entries show the overlap between them.
    """
    records = []
    for order, chunk in enumerate(chunks):
        meta = chunk.metadata
        records.append({
            "order": order,
            "chunk_id": chunk.chunk_id,
            "source": meta.get("source"),
            "file_type": meta.get("file_type"),
            "locator": _locator(meta),
            "chunk_index": meta.get("chunk_index"),
            "characters": len(chunk.text),
            "text": chunk.text,
            "metadata": meta,
        })
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "total_chunks": len(records),
        "chunks": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def ingest(
    data_dir: Path = DATA_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedding_model: str = EMBEDDING_MODEL,
    vector_db: str = VECTOR_DB,
    reset: bool = True,
    progress: Callable[[str, float], None] | None = None,
) -> dict:
    """Run the full ingestion pipeline and return a manifest dict.

    `progress(message, fraction)` is an optional callback for callers that
    want to drive their own progress display; the CLI just prints.
    """
    def report(message: str, fraction: float):
        if progress:
            progress(message, fraction)
        else:
            print(f"  {ui.dim(f'[{fraction:>4.0%}]')} {message}")

    started = time.perf_counter()

    # ---- 1. Load ------------------------------------------------------
    report("Loading documents...", 0.05)
    if not iter_files(data_dir):
        raise RuntimeError(
            f"{data_dir} is empty. Put your documents there (PDF, DOCX, TXT, CSV, JSON, "
            f"Markdown, HTML -- sub-folders are fine) and run ingest.py again."
        )
    documents, file_summary = load_directory(data_dir)
    if not documents:
        raise RuntimeError(
            f"{data_dir} contains {len(iter_files(data_dir))} file(s) but none in a supported format "
            f"(pdf, docx, txt, csv, json, md, html)."
        )

    # ---- 2. Chunk -----------------------------------------------------
    report(f"Chunking {len(documents)} document parts "
           f"(chunk_size={chunk_size}, overlap={chunk_overlap})...", 0.20)
    chunks = split_documents(documents, chunk_size, chunk_overlap)

    # Per-file chunk counts for the manifest / console table.
    chunks_per_file: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        chunks_per_file[source] = chunks_per_file.get(source, 0) + 1
    for entry in file_summary:
        entry["chunks"] = chunks_per_file.get(entry["file"], 0)

    # Write the chunks out BEFORE embedding: even if the embedding step fails
    # (no model download, out of memory) the split can still be inspected.
    chunks_path = write_chunks_file(chunks, chunk_size, chunk_overlap)
    report(f"Wrote {len(chunks)} chunks to {chunks_path.name} for inspection", 0.25)

    # ---- 3. Embed -----------------------------------------------------
    report(f"Loading embedding model {embedding_model}...", 0.30)
    embedder = EmbeddingModel(embedding_model)
    report(f"Embedding {len(chunks)} chunks ({embedder.dims}-dim vectors)...", 0.40)
    vectors = embedder.embed_documents(
        [c.text for c in chunks], show_progress=progress is None
    )

    # ---- 4. Store -----------------------------------------------------
    report(f"Writing to {vector_db} vector store...", 0.80)
    store = get_vector_store(vector_db, embedding_model)
    if reset:
        store.reset()
    store.add(chunks, vectors)

    # ---- 5. Manifest --------------------------------------------------
    manifest = {
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "vector_db": vector_db,
        "collection": getattr(store, "collection_name", None),
        "embedding_model": embedding_model,
        "embedding_dims": embedder.dims,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "documents": len(documents),
        "chunks": len(chunks),
        "stored_chunks": store.count(),
        "files": file_summary,
        "files_signature": file_signature(data_dir),   # for change detection in main.py
        "rebuild": reset,
        "chunks_file": str(chunks_path),
        "seconds": round(time.perf_counter() - started, 2),
    }
    INGEST_MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    INGEST_MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report("Done.", 1.0)
    return manifest


def _print_table(manifest: dict) -> None:
    """Pretty console summary of what was ingested."""
    ui.blank()
    mode = "full rebuild" if manifest.get("rebuild", True) else "append"
    ui.section("Ingestion summary", f"{manifest['documents']} parts {ui.g['arrow']} {manifest['chunks']} chunks  {ui.g['dot']}  {mode}")
    rows = [["File", "Type", "Parts", "Chars", "Chunks", "Status"]]
    for f in manifest["files"]:
        rows.append([f["file"], f["file_type"], str(f["documents"]), str(f["characters"]),
                     str(f.get("chunks", 0)), ui.good(f["status"]) if f["status"] == "ok" else ui.bad(f["status"])])
    ui.columns(rows, [26, 5, 5, 6, 6, 20], inside=True, header=True)
    ui.line()
    ui.kv("Embedding model", f"{manifest['embedding_model']}  ({manifest['embedding_dims']}-dim)", inside=True)
    ui.kv("Vector store", f"{manifest['vector_db']}  {ui.g['dot']}  collection {manifest['collection']}", inside=True)
    ui.kv("Stored chunks", str(manifest["stored_chunks"]), inside=True)
    ui.kv("Chunks file", f"{manifest['chunks_file']}", inside=True)
    ui.kv("Time", f"{manifest['seconds']} s", inside=True)
    ui.end()
    ui.status(True, "Done. Next step:  python main.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector database.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--vector-db", default=VECTOR_DB)
    parser.add_argument("--append", action="store_true",
                        help="keep the existing collection and add/refresh files (default: full rebuild)")
    parser.add_argument("--reset", action="store_true", help=argparse.SUPPRESS)  # kept for old habits
    args = parser.parse_args()

    try:
        manifest = ingest(
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model,
            vector_db=args.vector_db,
            reset=not args.append,
        )
    except Exception as exc:
        ui.status(False, f"Ingestion failed: {exc}")
        sys.exit(1)

    _print_table(manifest)


if __name__ == "__main__":
    main()
