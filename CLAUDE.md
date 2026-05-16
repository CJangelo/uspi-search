# uspi-search — Claude Code context

## What this project is
Hybrid keyword + semantic retrieval system for FDA drug labels via the openFDA
`/drug/label.json` API. **Not RAG** — no generation step. Returns merged ranked
results from FTS5 (keyword) and ChromaDB (semantic) searches.

## Pipeline overview

| Script | Role |
|---|---|
| `ingest.py` | Query openFDA, paginate, save raw JSON to `data/raw/` |
| `parse.py` | Read raw JSON, extract label sections, write to SQLite + FTS5 |
| `embed.py` | Chunk section text, embed via sentence-transformers, store in ChromaDB |
| `query.py` | Section-aware search across FTS5 + ChromaDB, return merged results |
| `pipeline.py` | Orchestrate ingest → parse → embed for weekly incremental sync |

## openFDA ingest filters

Applied at query time (API-side):
- `openfda.product_type:"HUMAN PRESCRIPTION DRUG"`
- `openfda.application_number:(NDA* BLA*)` — innovator drugs only (NDA = small molecule, BLA = biologic)
- `indications_and_usage:"{indication}"` — parameterised, default `"atopic dermatitis"`

Applied post-download (client-side):
- Marketing status: keep records where top-level `marketing_status` includes "Prescription" (field is inconsistently populated in the API, so filter after fetch)
- Dedup: group by `openfda.spl_set_id`, keep record with highest `openfda.spl_version`

openFDA pagination: `limit=1000` (API max), `skip` increments by 1000. Hard cap: `skip + limit ≤ 25,000`.

## Database schema (SQLite)

Schema is **source-agnostic** — not openFDA-shape-specific. A DailyMed XML parser
produces the same row shape.

```sql
CREATE TABLE labels (
    label_id     TEXT PRIMARY KEY,   -- spl_set_id or equivalent
    source       TEXT NOT NULL,      -- 'openfda_json' | 'dailymed_xml'
    metadata     TEXT                -- JSON blob: application_number, brand_name, etc.
);

CREATE TABLE sections (
    id           INTEGER PRIMARY KEY,
    label_id     TEXT NOT NULL REFERENCES labels(label_id),
    section_name TEXT NOT NULL,      -- e.g. 'indications_and_usage'
    text         TEXT NOT NULL
);

CREATE VIRTUAL TABLE sections_fts USING fts5(
    label_id UNINDEXED,
    section_name UNINDEXED,
    text,
    content='sections',
    content_rowid='id'
);

CREATE TABLE sync_state (
    label_id              TEXT PRIMARY KEY,
    spl_version           INTEGER,
    last_pull_timestamp   TEXT,      -- ISO-8601
    embedding_model_version TEXT
);
```

## Pluggable parser contract

Any parser module must expose:

```python
def parse(raw_path: Path) -> Iterator[LabelRecord]:
    ...

@dataclass
class LabelRecord:
    label_id: str
    source: str                     # 'openfda_json' | 'dailymed_xml'
    metadata: dict
    sections: list[tuple[str, str]] # [(section_name, text), ...]
```

`parse.py` calls `parser.parse(path)` — the parser module is injected, not hardcoded.

## Dependency stages

- **Now (ingest + parse):** `requests`, `click`
- **Stage 3 (embed):** `sentence-transformers`, `chromadb`
- Do not add stage-3 deps until embed.py is being implemented.

## Data directories

- `data/raw/` — raw openFDA JSON responses (gitignored)
- `data/chroma/` — ChromaDB persistent store (gitignored)
- SQLite DB path: `data/labels.db` (gitignored)

## Environment variables

| Variable | Purpose |
|---|---|
| `OPENFDA_API_KEY` | Optional; unauthenticated rate limit is 40 req/min |
