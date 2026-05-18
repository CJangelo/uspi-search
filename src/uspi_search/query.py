"""
query.py — Two search modes over FDA drug labels.

Usage:
    uv run query keyword "vIGA"                          # exact match, all drugs
    uv run query keyword "vIGA" --section clinical_studies
    uv run query search "pediatric itch"                 # semantic/concept search
    uv run query search "IL-13 signaling" --top-k 5
    uv run query search "hepatotoxicity" --fts-weight 1  # hybrid
    uv run query --help                                  # show subcommands
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
_UNLIMITED = 10_000  # practical cap when top_k == 0 (all results)


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


def count_fts_drugs(
    conn: sqlite3.Connection,
    query_text: str,
    sections: tuple[str, ...],
) -> int | None:
    """Return total unique drugs matching the FTS query (for the info line)."""
    where_clauses = ["sections_fts MATCH ?"]
    params: list[Any] = [query_text]
    if sections:
        placeholders = ",".join("?" * len(sections))
        where_clauses.append(f"s.section_name IN ({placeholders})")
        params.extend(sections)
    sql = f"""
        SELECT COUNT(DISTINCT s.label_id)
        FROM sections_fts fts
        JOIN sections s ON s.id = fts.rowid
        WHERE {' AND '.join(where_clauses)}
    """
    try:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return None


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

def _group_by_drug(results: list[dict], limit: int) -> list[dict]:
    """Collapse per-section results into one entry per drug.

    Drug order follows RRF ranking (first occurrence = best-scoring section).
    limit=0 means return all.
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
    result_list = list(drugs.values())
    return result_list if limit == 0 else result_list[:limit]


# --------------------------------------------------------------------------- #
# Display                                                                        #
# --------------------------------------------------------------------------- #

def _print_results(drugs: list[dict], query_text: str, fts_total: int | None, top_k: int) -> None:
    click.echo(f'\nQuery: "{query_text}"')
    if fts_total is not None:
        if top_k > 0 and len(drugs) < fts_total:
            click.echo(
                f"Showing {len(drugs)} of {fts_total} matching drugs. "
                f"Use --top-k 0 to see all {fts_total}."
            )
        else:
            click.echo(f"{fts_total} matching drug(s).")
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

@click.group()
@click.option("--db", default="data/labels.db", show_default=True, type=click.Path())
@click.option("--chroma-dir", default="data/chroma", show_default=True, type=click.Path())
@click.option(
    "--model", default=_DEFAULT_MODEL, show_default=True,
    help="sentence-transformers model (must match what embed used).",
)
@click.option("--collection", default=_DEFAULT_COLLECTION, show_default=True)
@click.option("--verbose", "-v", is_flag=True)
@click.pass_context
def main(ctx: click.Context, db: str, chroma_dir: str, model: str, collection: str, verbose: bool) -> None:
    """Search FDA drug labels: 'keyword' for exact terms, 'search' for concepts."""
    ctx.ensure_object(dict)
    ctx.obj.update(db=db, chroma_dir=chroma_dir, model=model, collection=collection)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )


@main.command("keyword")
@click.argument("query_text")
@click.option(
    "--section", "sections", multiple=True,
    help="Restrict to specific section name(s). Repeatable.",
)
@click.option("--top-k", default=0, show_default=True, type=int,
              help="Drugs to return (0 = all matching).")
@click.option("--snippet-tokens", default=40, show_default=True, type=int,
              help="Approximate tokens in each FTS5 snippet.")
@click.option("--json", "output_json", is_flag=True, help="Emit results as JSON.")
@click.pass_context
def keyword_cmd(
    ctx: click.Context,
    query_text: str,
    sections: tuple[str, ...],
    top_k: int,
    snippet_tokens: int,
    output_json: bool,
) -> None:
    """Exact keyword search (FTS5). Returns every drug label containing the term."""
    o = ctx.obj
    db_path = Path(o["db"])
    if not db_path.exists():
        raise click.ClickException(f"Database not found: {db_path} — run 'parse' first.")

    candidate_limit = _UNLIMITED if top_k == 0 else top_k * 10
    section_limit   = _UNLIMITED if top_k == 0 else top_k * 5

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        fts_total = count_fts_drugs(conn, query_text, sections)
        fts_hits_raw = search_fts(conn, query_text, sections, candidate_limit, snippet_tokens)
        fts_snippets = {(lid, sec): snip for lid, sec, snip in fts_hits_raw}
        fts_hits = [(lid, sec) for lid, sec, _ in fts_hits_raw]
        merged = rrf_merge(fts_hits, [], 1.0, 0.0, section_limit)
        for hit in merged:
            hit["fts_snippet"] = fts_snippets.get((hit["label_id"], hit["section_name"]))
        results = enrich(conn, merged)
        drugs = _group_by_drug(results, top_k)
    finally:
        conn.close()

    if output_json:
        click.echo(json.dumps({"total_fts_drugs": fts_total, "drugs": drugs}, indent=2))
    else:
        _print_results(drugs, query_text, fts_total, top_k)


@main.command("search")
@click.argument("query_text")
@click.option(
    "--section", "sections", multiple=True,
    help="Restrict to specific section name(s). Repeatable.",
)
@click.option("--top-k", default=10, show_default=True, type=int,
              help="Drugs to return.")
@click.option(
    "--fts-weight", default=0.0, show_default=True, type=float,
    help="Set >0 to add keyword results (hybrid mode).",
)
@click.option("--snippet-tokens", default=40, show_default=True, type=int,
              help="Approximate tokens in each FTS5 snippet (hybrid mode only).")
@click.option("--json", "output_json", is_flag=True, help="Emit results as JSON.")
@click.pass_context
def search_cmd(
    ctx: click.Context,
    query_text: str,
    sections: tuple[str, ...],
    top_k: int,
    fts_weight: float,
    snippet_tokens: int,
    output_json: bool,
) -> None:
    """Concept search (semantic). Finds labels relevant to the idea, not exact words."""
    o = ctx.obj
    db_path = Path(o["db"])
    if not db_path.exists():
        raise click.ClickException(f"Database not found: {db_path} — run 'parse' first.")

    chroma_client = chromadb.PersistentClient(path=str(Path(o["chroma_dir"])))
    try:
        chroma_col = chroma_client.get_collection(name=o["collection"])
        if chroma_col.count() == 0:
            raise click.ClickException("ChromaDB collection is empty -- run 'embed' first.")
    except click.ClickException:
        raise
    except Exception:
        raise click.ClickException("ChromaDB collection not found -- run 'embed' first.")

    st_model = SentenceTransformer(o["model"])
    candidate_limit = top_k * 10
    section_limit   = top_k * 5

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        fts_snippets: dict = {}
        fts_hits: list = []
        if fts_weight > 0:
            fts_hits_raw = search_fts(conn, query_text, sections, candidate_limit, snippet_tokens)
            fts_snippets = {(lid, sec): snip for lid, sec, snip in fts_hits_raw}
            fts_hits = [(lid, sec) for lid, sec, _ in fts_hits_raw]

        sem_hits = search_semantic(chroma_col, st_model, query_text, sections, candidate_limit)
        merged = rrf_merge(fts_hits, sem_hits, fts_weight, 1.0, section_limit)
        for hit in merged:
            hit["fts_snippet"] = fts_snippets.get((hit["label_id"], hit["section_name"]))
        results = enrich(conn, merged)
        drugs = _group_by_drug(results, top_k)
    finally:
        conn.close()

    if output_json:
        click.echo(json.dumps({"drugs": drugs}, indent=2))
    else:
        _print_results(drugs, query_text, None, top_k)
