"""
query.py — Section-aware hybrid search over FTS5 (keyword) + ChromaDB (semantic).
Results are merged with Reciprocal Rank Fusion (RRF). No generation step.

Usage:
    uv run query "dupilumab mechanism of action"
    uv run query "IL-13 signaling" --section indications_and_usage --top-k 5
    uv run query "hepatotoxicity" --section warnings_and_cautions --json
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import click
import chromadb
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_COLLECTION = "label_sections"
_RRF_K = 60  # standard constant; higher K = more conservative blending


# --------------------------------------------------------------------------- #
# FTS5 keyword search                                                           #
# --------------------------------------------------------------------------- #

def search_fts(
    conn: sqlite3.Connection,
    query_text: str,
    sections: tuple[str, ...],
    limit: int,
    snippet_tokens: int = 40,
) -> list[tuple[str, str, str]]:
    """Return [(label_id, section_name, snippet)] ranked by FTS5 BM25 (best first).

    snippet() returns the portion of the section text that contains the match,
    with matched terms wrapped in [ ]. Column index 2 = the text column.
    """
    where_clauses = ["sections_fts MATCH ?"]

    # snippet(? ) appears in SELECT (before WHERE), so it must be first in params.
    params: list[Any] = [snippet_tokens, query_text]

    if sections:
        placeholders = ",".join("?" * len(sections))
        where_clauses.append(f"s.section_name IN ({placeholders})")
        params.extend(sections)

    params.append(limit)
    sql = f"""
        SELECT s.label_id, s.section_name,
               snippet(sections_fts, 2, '[', ']', '...', ?)
        FROM sections_fts fts
        JOIN sections s ON s.id = fts.rowid
        WHERE {' AND '.join(where_clauses)}
        ORDER BY fts.rank
        LIMIT ?
    """
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        log.warning("FTS5 error (check query syntax): %s", exc)
        rows = []

    return [(r[0], r[1], r[2]) for r in rows]


# --------------------------------------------------------------------------- #
# Semantic search                                                                #
# --------------------------------------------------------------------------- #

def search_semantic(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    query_text: str,
    sections: tuple[str, ...],
    limit: int,
) -> list[tuple[str, str]]:
    """Return [(label_id, section_name)] ranked by cosine similarity (best first).

    ChromaDB returns chunk-level results; we aggregate to section level by
    keeping the best (lowest) distance per (label_id, section_name) pair.
    """
    n = min(limit * 5, collection.count())
    if n == 0:
        return []

    embedding = model.encode(query_text).tolist()
    where = {"section_name": {"$in": list(sections)}} if sections else None

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where,
            include=["metadatas", "distances"],
        )
    except Exception as exc:
        log.warning("ChromaDB query error: %s", exc)
        return []

    best: dict[tuple[str, str], float] = {}
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        key = (meta["label_id"], meta["section_name"])
        if key not in best or dist < best[key]:
            best[key] = dist

    return [k for k, _ in sorted(best.items(), key=lambda x: x[1])][:limit]


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion                                                        #
# --------------------------------------------------------------------------- #

def _rrf(rank: int) -> float:
    return 1.0 / (_RRF_K + rank + 1)


def rrf_merge(
    fts_hits: list[tuple[str, str]],
    sem_hits: list[tuple[str, str]],
    fts_weight: float,
    sem_weight: float,
    top_k: int,
) -> list[dict]:
    scores: dict[tuple[str, str], float] = {}
    sources: dict[tuple[str, str], set[str]] = {}

    for rank, key in enumerate(fts_hits):
        scores[key] = scores.get(key, 0.0) + fts_weight * _rrf(rank)
        sources.setdefault(key, set()).add("fts")

    for rank, key in enumerate(sem_hits):
        scores[key] = scores.get(key, 0.0) + sem_weight * _rrf(rank)
        sources.setdefault(key, set()).add("semantic")

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "label_id": label_id,
            "section_name": section_name,
            "score": round(score, 6),
            "match_source": "+".join(sorted(sources[(label_id, section_name)])),
        }
        for (label_id, section_name), score in ranked
    ]


# --------------------------------------------------------------------------- #
# Enrich with text + metadata from SQLite                                       #
# --------------------------------------------------------------------------- #

def enrich(conn: sqlite3.Connection, results: list[dict]) -> list[dict]:
    enriched = []
    for hit in results:
        row = conn.execute(
            """
            SELECT s.text, l.metadata
            FROM sections s
            JOIN labels l ON l.label_id = s.label_id
            WHERE s.label_id = ? AND s.section_name = ?
            """,
            (hit["label_id"], hit["section_name"]),
        ).fetchone()
        if not row:
            continue
        meta = json.loads(row[1]) if row[1] else {}
        enriched.append({
            **hit,
            "text": row[0],
            "brand_name": meta.get("brand_name") or "",
            "generic_name": meta.get("generic_name") or "",
            "application_number": meta.get("application_number") or "",
        })
    return enriched


# --------------------------------------------------------------------------- #
# Group section-level hits by drug                                              #
# --------------------------------------------------------------------------- #

def _group_by_drug(results: list[dict]) -> list[dict]:
    """Collapse per-section results into one entry per drug.

    Drug order follows RRF ranking (first occurrence = best-scoring section).
    """
    drugs: dict[str, dict] = {}
    for hit in results:
        lid = hit["label_id"]
        if lid not in drugs:
            drugs[lid] = {
                "label_id": lid,
                "brand_name": hit["brand_name"],
                "generic_name": hit["generic_name"],
                "application_number": hit["application_number"],
                "best_score": hit["score"],
                "sections": [],
            }
        drugs[lid]["sections"].append({
            "section_name": hit["section_name"],
            "score": hit["score"],
            "match_source": hit["match_source"],
            "fts_snippet": hit.get("fts_snippet"),
            "text": hit.get("text", ""),
        })
    return list(drugs.values())


# --------------------------------------------------------------------------- #
# Display                                                                        #
# --------------------------------------------------------------------------- #

def _print_results(drugs: list[dict], query_text: str) -> None:
    click.echo(f'\nQuery: "{query_text}"')
    click.echo("-" * 72)
    if not drugs:
        click.echo("No results.")
        return
    for i, drug in enumerate(drugs, 1):
        label = drug.get("brand_name") or drug.get("generic_name") or drug["label_id"]
        app_no = f"({drug['application_number']})" if drug.get("application_number") else ""
        click.echo(f"#{i}  {label} {app_no}  score={drug['best_score']:.4f}")
        for sec in drug["sections"]:
            excerpt = sec.get("fts_snippet") or sec.get("text", "")[:300]
            excerpt = excerpt.replace("\n", " ").encode("ascii", errors="replace").decode("ascii")
            click.echo(f"    [{sec['section_name']}][{sec['match_source']}] {excerpt}")
        click.echo("")


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

@click.command()
@click.argument("query_text")
@click.option("--db", default="data/labels.db", show_default=True, type=click.Path())
@click.option("--chroma-dir", default="data/chroma", show_default=True, type=click.Path())
@click.option(
    "--model", default=_DEFAULT_MODEL, show_default=True,
    help="sentence-transformers model (must match what embed.py used).",
)
@click.option("--collection", default=_DEFAULT_COLLECTION, show_default=True)
@click.option(
    "--section", "sections", multiple=True,
    help="Restrict to specific section name(s). Repeatable.",
)
@click.option("--top-k", default=10, show_default=True, type=int)
@click.option(
    "--fts-weight", default=1.0, show_default=True, type=float,
    help="RRF weight applied to keyword results.",
)
@click.option(
    "--sem-weight", default=1.0, show_default=True, type=float,
    help="RRF weight applied to semantic results.",
)
@click.option("--snippet-tokens", default=40, show_default=True, type=int,
              help="Approximate tokens in each FTS5 snippet.")
@click.option("--json", "output_json", is_flag=True, help="Emit results as JSON.")
@click.option("--verbose", "-v", is_flag=True)
def main(
    query_text: str,
    db: str,
    chroma_dir: str,
    model: str,
    collection: str,
    sections: tuple[str, ...],
    top_k: int,
    fts_weight: float,
    sem_weight: float,
    snippet_tokens: int,
    output_json: bool,
    verbose: bool,
) -> None:
    """Hybrid keyword + semantic search over FDA label sections."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    db_path = Path(db)
    if not db_path.exists():
        raise click.ClickException(f"Database not found: {db_path} — run `parse` first.")

    chroma_client = chromadb.PersistentClient(path=str(Path(chroma_dir)))
    try:
        chroma_col = chroma_client.get_collection(name=collection)
        do_semantic = chroma_col.count() > 0
    except Exception:
        chroma_col = None
        do_semantic = False
        log.warning("Collection '%s' not found — semantic search disabled.", collection)

    # Only load the model if the collection exists and has data
    st_model = SentenceTransformer(model) if do_semantic else None

    candidate_limit = top_k * 5

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        fts_hits_raw = search_fts(conn, query_text, sections, candidate_limit, snippet_tokens)
        fts_snippets = {(lid, sec): snip for lid, sec, snip in fts_hits_raw}
        fts_hits = [(lid, sec) for lid, sec, _ in fts_hits_raw]

        sem_hits = (
            search_semantic(chroma_col, st_model, query_text, sections, candidate_limit)
            if do_semantic
            else []
        )
        merged = rrf_merge(fts_hits, sem_hits, fts_weight, sem_weight, top_k)
        for hit in merged:
            hit["fts_snippet"] = fts_snippets.get((hit["label_id"], hit["section_name"]))
        results = enrich(conn, merged)
        drugs = _group_by_drug(results)
    finally:
        conn.close()

    if output_json:
        click.echo(json.dumps(drugs, indent=2))
    else:
        _print_results(drugs, query_text)
