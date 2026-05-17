"""
parse.py — Read raw label JSON, extract sections, write to SQLite + FTS5.

Usage:
    uv run parse --in-file data/raw/deduped_atopic_dermatitis_*.json --db data/labels.db
    uv run parse --in-dir data/raw --db data/labels.db
"""

import importlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import click

from uspi_search.models import LabelRecord

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Schema                                                                        #
# --------------------------------------------------------------------------- #

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS labels (
        label_id  TEXT PRIMARY KEY,
        source    TEXT NOT NULL,
        metadata  TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sections (
        id           INTEGER PRIMARY KEY,
        label_id     TEXT NOT NULL REFERENCES labels(label_id) ON DELETE CASCADE,
        section_name TEXT NOT NULL,
        text         TEXT NOT NULL
    )
    """,
    # FTS5 content table — index is maintained by triggers below; text lives in sections.
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
        label_id     UNINDEXED,
        section_name UNINDEXED,
        text,
        content='sections',
        content_rowid='id',
        tokenize='porter unicode61'
    )
    """,
    # Triggers keep FTS in sync with sections automatically.
    """
    CREATE TRIGGER IF NOT EXISTS sections_ai AFTER INSERT ON sections BEGIN
        INSERT INTO sections_fts(rowid, label_id, section_name, text)
        VALUES (new.id, new.label_id, new.section_name, new.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS sections_ad AFTER DELETE ON sections BEGIN
        INSERT INTO sections_fts(sections_fts, rowid, label_id, section_name, text)
        VALUES ('delete', old.id, old.label_id, old.section_name, old.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS sections_au AFTER UPDATE ON sections BEGIN
        INSERT INTO sections_fts(sections_fts, rowid, label_id, section_name, text)
        VALUES ('delete', old.id, old.label_id, old.section_name, old.text);
        INSERT INTO sections_fts(rowid, label_id, section_name, text)
        VALUES (new.id, new.label_id, new.section_name, new.text);
    END
    """,
    """
    CREATE TABLE IF NOT EXISTS sync_state (
        label_id                TEXT PRIMARY KEY,
        spl_version             INTEGER,
        last_pull_timestamp     TEXT,
        embedding_model_version TEXT
    )
    """,
]


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    for stmt in _DDL:
        conn.execute(stmt)
    conn.commit()


# --------------------------------------------------------------------------- #
# Upsert logic                                                                  #
# --------------------------------------------------------------------------- #

def _stored_spl_version(conn: sqlite3.Connection, label_id: str) -> int | None:
    row = conn.execute(
        "SELECT spl_version FROM sync_state WHERE label_id = ?", (label_id,)
    ).fetchone()
    return row[0] if row else None


def upsert_label(conn: sqlite3.Connection, record: LabelRecord, now: str) -> bool:
    """Write *record* to the DB. Returns True if written, False if skipped."""
    spl_version = int(record.metadata.get("spl_version") or 0)
    stored = _stored_spl_version(conn, record.label_id)

    if stored is not None and stored >= spl_version:
        log.debug(
            "Skip %s — stored v%d >= incoming v%d", record.label_id, stored, spl_version
        )
        return False

    with conn:
        # ON DELETE CASCADE removes old sections, which fires sections_ad trigger
        # and cleans up FTS automatically.
        conn.execute("DELETE FROM labels WHERE label_id = ?", (record.label_id,))
        conn.execute(
            "INSERT INTO labels(label_id, source, metadata) VALUES (?, ?, ?)",
            (record.label_id, record.source, json.dumps(record.metadata)),
        )
        for section_name, text in record.sections:
            conn.execute(
                "INSERT INTO sections(label_id, section_name, text) VALUES (?, ?, ?)",
                (record.label_id, section_name, text),
            )
        conn.execute(
            """
            INSERT INTO sync_state(label_id, spl_version, last_pull_timestamp)
            VALUES (?, ?, ?)
            ON CONFLICT(label_id) DO UPDATE SET
                spl_version          = excluded.spl_version,
                last_pull_timestamp  = excluded.last_pull_timestamp
            """,
            (record.label_id, spl_version, now),
        )

    return True


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

@click.command()
@click.option(
    "--in-file", "in_files",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Deduped JSON file(s) produced by ingest.py. Repeatable.",
)
@click.option(
    "--in-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Directory to scan for deduped_*.json files (alternative to --in-file).",
)
@click.option("--db", default="data/labels.db", show_default=True, type=click.Path())
@click.option(
    "--parser",
    default="openfda_json",
    show_default=True,
    help="Parser module under uspi_search.parsers.*",
)
@click.option("--verbose", "-v", is_flag=True)
def main(
    in_files: tuple[str, ...],
    in_dir: str | None,
    db: str,
    parser: str,
    verbose: bool,
) -> None:
    """Parse label JSON into SQLite + FTS5."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    files: list[Path] = [Path(f) for f in in_files]
    if in_dir:
        files.extend(sorted(Path(in_dir).glob("deduped_*.json")))
    if not files:
        raise click.UsageError("Provide at least one --in-file or --in-dir.")

    parser_mod = importlib.import_module(f"uspi_search.parsers.{parser}")

    db_path = Path(db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(db_path)
    try:
        init_db(conn)
        inserted = skipped = 0

        for path in files:
            log.info("Parsing %s", path.name)
            for record in parser_mod.parse(path):
                if upsert_label(conn, record, now):
                    inserted += 1
                    log.debug("Wrote  %s (%d sections)", record.label_id, len(record.sections))
                else:
                    skipped += 1

        click.echo(f"Done. {inserted} written, {skipped} skipped -> {db_path}")
    finally:
        conn.close()
