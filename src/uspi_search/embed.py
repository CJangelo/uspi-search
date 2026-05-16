"""
embed.py — Chunk section text, embed via sentence-transformers, store in ChromaDB.

Usage:
    uv run embed --db data/labels.db --chroma-dir data/chroma
    uv run embed --model BAAI/bge-small-en-v1.5   # swap model; re-embeds stale labels
"""

import json
import logging
import sqlite3
from pathlib import Path

import click
import chromadb
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_COLLECTION = "label_sections"


# --------------------------------------------------------------------------- #
# Chunking                                                                      #
# --------------------------------------------------------------------------- #

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split *text* into word-count chunks with *overlap* words of carry-over context."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks


# --------------------------------------------------------------------------- #
# DB helpers                                                                    #
# --------------------------------------------------------------------------- #

def _labels_pending(conn: sqlite3.Connection, model_version: str) -> list[str]:
    """Return label_ids whose embeddings are absent or were made with a different model."""
    rows = conn.execute(
        """
        SELECT label_id FROM sync_state
        WHERE embedding_model_version IS NULL
           OR embedding_model_version != ?
        """,
        (model_version,),
    ).fetchall()
    return [r[0] for r in rows]


def _fetch_sections(conn: sqlite3.Connection, label_id: str) -> list[tuple[str, str]]:
    return conn.execute(
        "SELECT section_name, text FROM sections WHERE label_id = ?", (label_id,)
    ).fetchall()


def _fetch_metadata(conn: sqlite3.Connection, label_id: str) -> dict:
    row = conn.execute(
        "SELECT metadata FROM labels WHERE label_id = ?", (label_id,)
    ).fetchone()
    return json.loads(row[0]) if row and row[0] else {}


# --------------------------------------------------------------------------- #
# Core embed logic                                                               #
# --------------------------------------------------------------------------- #

def _embed_one(
    conn: sqlite3.Connection,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    model_version: str,
    label_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """Embed a single label. Returns number of chunks upserted (0 = skipped)."""
    sections = _fetch_sections(conn, label_id)
    if not sections:
        log.warning("No sections for %s — skipping", label_id)
        return 0

    meta = _fetch_metadata(conn, label_id)
    base_meta = {
        "label_id": label_id,
        "brand_name": meta.get("brand_name") or "",
        "generic_name": meta.get("generic_name") or "",
        "application_number": meta.get("application_number") or "",
    }

    ids: list[str] = []
    docs: list[str] = []
    metadatas: list[dict] = []

    for section_name, text in sections:
        for i, chunk in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
            ids.append(f"{label_id}__{section_name}__{i}")
            docs.append(chunk)
            metadatas.append({**base_meta, "section_name": section_name, "chunk_index": i})

    embeddings = model.encode(docs, batch_size=32, show_progress_bar=False).tolist()
    collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

    with conn:
        conn.execute(
            "UPDATE sync_state SET embedding_model_version = ? WHERE label_id = ?",
            (model_version, label_id),
        )

    return len(ids)


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

@click.command()
@click.option("--db", default="data/labels.db", show_default=True, type=click.Path())
@click.option(
    "--chroma-dir", default="data/chroma", show_default=True, type=click.Path(),
    help="ChromaDB persist directory.",
)
@click.option(
    "--model", default=_DEFAULT_MODEL, show_default=True,
    help="sentence-transformers model name. Changing this re-embeds all labels.",
)
@click.option(
    "--collection", default=_DEFAULT_COLLECTION, show_default=True,
    help="ChromaDB collection name.",
)
@click.option(
    "--chunk-size", default=400, show_default=True, type=int,
    help="Target words per chunk.",
)
@click.option(
    "--chunk-overlap", default=50, show_default=True, type=int,
    help="Words of overlap between consecutive chunks.",
)
@click.option("--verbose", "-v", is_flag=True)
def main(
    db: str,
    chroma_dir: str,
    model: str,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
    verbose: bool,
) -> None:
    """Embed label sections into ChromaDB."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    db_path = Path(db)
    if not db_path.exists():
        raise click.ClickException(f"Database not found: {db_path} — run `parse` first.")

    log.info("Loading model: %s", model)
    st_model = SentenceTransformer(model)

    chroma_path = Path(chroma_dir)
    chroma_path.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    chroma_collection = chroma_client.get_or_create_collection(
        name=collection,
        metadata={"hnsw:space": "cosine"},
    )

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        pending = _labels_pending(conn, model)
        log.info("%d label(s) pending embedding", len(pending))

        labels_done = total_chunks = 0
        for label_id in pending:
            n = _embed_one(
                conn, chroma_collection, st_model, model,
                label_id, chunk_size, chunk_overlap,
            )
            if n:
                labels_done += 1
                total_chunks += n
                log.info("[%d/%d] %s — %d chunk(s)", labels_done, len(pending), label_id, n)

        click.echo(
            f"Done. {labels_done} label(s) embedded, {total_chunks} chunk(s) → {chroma_dir}"
        )
    finally:
        conn.close()
