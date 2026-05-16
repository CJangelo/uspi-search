"""
ingest.py — Query openFDA /drug/label.json, paginate, and save raw JSON locally.

API-side filters:
    openfda.product_type:"HUMAN PRESCRIPTION DRUG"
    openfda.application_number:(NDA* OR BLA*)
    indications_and_usage:{indication}

Client-side filters (applied after fetch):
    openfda.marketing_status contains "prescription" (list field, nested under openfda)
    Dedup by openfda.spl_set_id, keeping record with highest spl_version

Output files:
    {out_dir}/batch_{skip:06d}.json      — raw API response per page
    {out_dir}/deduped_{slug}_{ts}.json   — final deduplicated records
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import click
import requests

OPENFDA_BASE = "https://api.fda.gov/drug/label.json"
# API hard cap: skip + limit <= 25,000; with limit=1000 the max useful skip is 24,000
_MAX_SKIP = 24_000
_RETRY_ATTEMPTS = 3

log = logging.getLogger(__name__)


def build_search_query(indication: str) -> str:
    parts = [
        'openfda.product_type:"HUMAN PRESCRIPTION DRUG"',
        "(openfda.application_number:NDA* OR openfda.application_number:BLA*)",
        f'indications_and_usage:"{indication}"',
    ]
    return " AND ".join(parts)


def _fetch_page(
    session: requests.Session,
    query: str,
    skip: int,
    limit: int,
    api_key: str | None,
) -> dict:
    params: dict = {"search": query, "limit": limit, "skip": skip}
    if api_key:
        params["api_key"] = api_key

    for attempt in range(_RETRY_ATTEMPTS):
        resp = session.get(OPENFDA_BASE, params=params, timeout=30)

        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
            log.warning(
                "Rate limited; waiting %ds (attempt %d/%d)", wait, attempt + 1, _RETRY_ATTEMPTS
            )
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp.json()

    # exhausted retries — final raise
    resp.raise_for_status()


def paginate(
    session: requests.Session,
    query: str,
    limit: int,
    api_key: str | None,
) -> Iterator[list[dict]]:
    skip = 0
    total: int | None = None

    while True:
        if skip > _MAX_SKIP:
            log.warning(
                "openFDA skip cap reached (%d); stopping early — results may be truncated.",
                _MAX_SKIP,
            )
            break

        data = _fetch_page(session, query, skip, limit, api_key)

        if total is None:
            total = data["meta"]["results"]["total"]
            log.info("API reports %d total matching records", total)
            if total > _MAX_SKIP + limit:
                log.warning(
                    "Result set (%d) exceeds openFDA pagination cap (%d); "
                    "only the first %d records will be retrieved.",
                    total,
                    _MAX_SKIP + limit,
                    _MAX_SKIP + limit,
                )

        results: list[dict] = data.get("results", [])
        if not results:
            break

        yield results

        skip += len(results)
        if skip >= total:
            break


def _spl_version_int(record: dict) -> int:
    try:
        return int(record.get("openfda", {}).get("spl_version", ["0"])[0])
    except (ValueError, IndexError, TypeError):
        return 0


def dedup_by_spl_set_id(records: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    no_set_id = 0

    for rec in records:
        set_ids: list = rec.get("openfda", {}).get("spl_set_id", [])
        if not set_ids:
            no_set_id += 1
            continue
        set_id = set_ids[0]
        if set_id not in best or _spl_version_int(rec) > _spl_version_int(best[set_id]):
            best[set_id] = rec

    if no_set_id:
        log.warning("Dropped %d records with no spl_set_id during dedup.", no_set_id)

    return list(best.values())


def _is_currently_marketed(record: dict) -> bool:
    statuses = record.get("openfda", {}).get("marketing_status", [])
    return any("prescription" in s.lower() for s in statuses)


def _indication_slug(indication: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in indication.lower())[:40]


@click.command()
@click.option(
    "--indication",
    default="atopic dermatitis",
    show_default=True,
    help="Indication keyword searched in indications_and_usage.",
)
@click.option(
    "--out-dir",
    default="data/raw",
    show_default=True,
    type=click.Path(),
    help="Directory for raw JSON output.",
)
@click.option(
    "--limit",
    default=1000,
    show_default=True,
    type=click.IntRange(1, 1000),
    help="Records per API call (1–1000).",
)
@click.option(
    "--api-key",
    envvar="OPENFDA_API_KEY",
    default=None,
    help="openFDA API key (or set $OPENFDA_API_KEY). Unauthenticated limit: 40 req/min.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def main(
    indication: str,
    out_dir: str,
    limit: int,
    api_key: str | None,
    verbose: bool,
) -> None:
    """Ingest FDA drug labels from openFDA for a given indication."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    query = build_search_query(indication)
    log.info("Indication: %r", indication)
    log.debug("Search query: %s", query)

    all_records: list[dict] = []
    session = requests.Session()

    for batch in paginate(session, query, limit, api_key):
        skip_offset = len(all_records)
        batch_file = out_path / f"batch_{skip_offset:06d}.json"
        batch_file.write_text(json.dumps(batch, indent=2), encoding="utf-8")
        log.info("Fetched %d records → %s", len(batch), batch_file.name)
        all_records.extend(batch)

    log.info("Total fetched: %d", len(all_records))

    marketed = [r for r in all_records if _is_currently_marketed(r)]
    log.info("After marketing_status filter: %d (dropped %d)", len(marketed), len(all_records) - len(marketed))

    deduped = dedup_by_spl_set_id(marketed)
    log.info("After spl_set_id dedup: %d (dropped %d)", len(deduped), len(marketed) - len(deduped))

    slug = _indication_slug(indication)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = out_path / f"deduped_{slug}_{ts}.json"
    out_file.write_text(json.dumps(deduped, indent=2), encoding="utf-8")

    click.echo(f"Done. {len(deduped)} labels → {out_file}")
