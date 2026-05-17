"""
pipeline.py — Orchestrates ingest -> parse -> embed for weekly incremental sync.

Each stage is invoked as a programmatic Click call (standalone_mode=False),
so all validation, defaults, and error handling from the individual CLIs apply.
The incremental logic lives in each stage: parse skips labels whose spl_version
hasn't changed; embed skips labels whose embedding_model_version matches.

Usage:
    uv run pipeline --indication "atopic dermatitis"
    uv run pipeline --indication "psoriasis" --skip-embed
    uv run pipeline --skip-ingest --db data/labels.db   # re-parse + re-embed existing raw
"""

import logging
import time
from pathlib import Path

import click

from uspi_search import embed as embed_mod
from uspi_search import ingest as ingest_mod
from uspi_search import parse as parse_mod

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Stage runner                                                                  #
# --------------------------------------------------------------------------- #

def _run_stage(label: str, cmd: click.BaseCommand, args: list[str]) -> None:
    log.info("── %s ──", label.upper())
    t0 = time.monotonic()
    try:
        cmd(args, standalone_mode=False)
    except click.ClickException as exc:
        raise click.ClickException(f"{label} failed: {exc.format_message()}") from exc
    except Exception as exc:
        raise click.ClickException(f"{label} failed: {exc}") from exc
    log.info("── %s done (%.1fs) ──", label.upper(), time.monotonic() - t0)


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

@click.command()
@click.option(
    "--indication", default="atopic dermatitis", show_default=True,
    help="Indication keyword (passed to ingest).",
)
@click.option(
    "--out-dir", default="data/raw", show_default=True, type=click.Path(),
    help="Raw JSON directory (ingest output / parse input).",
)
@click.option("--db", default="data/labels.db", show_default=True, type=click.Path())
@click.option("--chroma-dir", default="data/chroma", show_default=True, type=click.Path())
@click.option(
    "--embed-model", default="all-MiniLM-L6-v2", show_default=True,
    help="sentence-transformers model (passed to embed).",
)
@click.option(
    "--api-key", envvar="OPENFDA_API_KEY", default=None,
    help="openFDA API key (or set $OPENFDA_API_KEY).",
)
@click.option("--skip-ingest", is_flag=True, help="Skip ingest; use existing raw files.")
@click.option("--skip-embed", is_flag=True, help="Stop after parse; skip embed stage.")
@click.option("--verbose", "-v", is_flag=True)
def main(
    indication: str,
    out_dir: str,
    db: str,
    chroma_dir: str,
    embed_model: str,
    api_key: str | None,
    skip_ingest: bool,
    skip_embed: bool,
    verbose: bool,
) -> None:
    """Run the full ingest -> parse -> embed pipeline."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    t_start = time.monotonic()
    stages = []
    if not skip_ingest:
        stages.append("ingest")
    stages.append("parse")
    if not skip_embed:
        stages.append("embed")

    click.echo(f'Pipeline [{" -> ".join(stages)}]  indication="{indication}"')

    # ── Stage 1: ingest ────────────────────────────────────────────────────
    if not skip_ingest:
        ingest_args = [
            "--indication", indication,
            "--out-dir", out_dir,
        ]
        if api_key:
            ingest_args += ["--api-key", api_key]
        if verbose:
            ingest_args.append("--verbose")
        _run_stage("ingest", ingest_mod.main, ingest_args)

    # Guard: parse needs at least one deduped file to read
    raw_files = sorted(Path(out_dir).glob("deduped_*.json")) if Path(out_dir).exists() else []
    if not raw_files:
        raise click.ClickException(
            f"No deduped_*.json files in {out_dir!r}. "
            "Run without --skip-ingest first, or point --out-dir at an existing raw directory."
        )

    # ── Stage 2: parse ─────────────────────────────────────────────────────
    parse_args = ["--in-dir", out_dir, "--db", db]
    if verbose:
        parse_args.append("--verbose")
    _run_stage("parse", parse_mod.main, parse_args)

    # ── Stage 3: embed ─────────────────────────────────────────────────────
    if not skip_embed:
        embed_args = [
            "--db", db,
            "--chroma-dir", chroma_dir,
            "--model", embed_model,
        ]
        if verbose:
            embed_args.append("--verbose")
        _run_stage("embed", embed_mod.main, embed_args)

    click.echo(f"Pipeline complete in {time.monotonic() - t_start:.1f}s")
