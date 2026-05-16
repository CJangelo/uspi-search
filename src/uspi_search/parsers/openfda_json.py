"""Parser for openFDA /drug/label.json output (deduped JSON arrays)."""

import html
import json
import re
from pathlib import Path
from typing import Iterator

from uspi_search.models import LabelRecord

# Top-level keys that are metadata, not searchable label text
_SKIP_KEYS = frozenset({
    "id", "set_id", "version", "effective_time",
    "openfda", "spl_product_data_elements",
})


def _clean(raw: str) -> str:
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_sections(record: dict) -> list[tuple[str, str]]:
    """Return [(section_name, cleaned_text)] for every text-bearing field."""
    out = []
    for key, value in record.items():
        if key in _SKIP_KEYS:
            continue
        if isinstance(value, list) and value and isinstance(value[0], str):
            text = _clean(" ".join(value))
            if text:
                out.append((key, text))
    return out


def _build_metadata(record: dict) -> dict:
    openfda = record.get("openfda", {})

    def first(key: str) -> str | None:
        vals = openfda.get(key, [])
        return vals[0] if vals else None

    return {
        "application_number": first("application_number"),
        "brand_name": first("brand_name"),
        "generic_name": first("generic_name"),
        "manufacturer_name": first("manufacturer_name"),
        "route": first("route"),
        "substance_name": first("substance_name"),
        "spl_version": first("spl_version"),
        "product_type": first("product_type"),
    }


def parse(path: Path) -> Iterator[LabelRecord]:
    """Yield one LabelRecord per drug label found in *path*."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    # Accept both a bare list (our deduped output) and a full API envelope
    records: list[dict] = data if isinstance(data, list) else data.get("results", [])

    for rec in records:
        set_ids: list = rec.get("openfda", {}).get("spl_set_id", [])
        if not set_ids:
            continue

        yield LabelRecord(
            label_id=set_ids[0],
            source="openfda_json",
            metadata=_build_metadata(rec),
            sections=_extract_sections(rec),
        )
