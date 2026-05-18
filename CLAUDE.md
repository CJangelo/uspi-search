# uspi-search — Claude Code context

## What this project is

Keyword and semantic retrieval system for FDA drug labels via the openFDA
`/drug/label.json` API. **Not RAG** — no generation step. Two distinct search
modes: exact keyword (FTS5) and concept search (ChromaDB semantic, optional hybrid).

## Implementation status

All five pipeline stages are fully implemented and committed.

## File structure

```
src/uspi_search/
    ingest.py          # Stage 1: fetch from openFDA, save raw JSON
    parse.py           # Stage 2: extract sections, write SQLite + FTS5
    embed.py           # Stage 3: chunk + embed via sentence-transformers -> ChromaDB
    query.py           # Two subcommands: 'keyword' (FTS5) and 'search' (semantic/hybrid)
    pipeline.py        # Orchestrator: ingest -> parse -> embed
    models.py          # LabelRecord dataclass (shared parser contract)
    parsers/
        __init__.py
        openfda_json.py # openFDA JSON parser (strips HTML, extracts all text sections)
data/
    raw/               # gitignored — raw openFDA JSON batches + deduped files
    chroma/            # gitignored — ChromaDB persistent store
    labels.db          # gitignored — SQLite database
```

## Quick-start usage

```bash
# Full pipeline (downloads model on first run, ~400MB)
uv run pipeline --indication "atopic dermatitis"

# Partial runs
uv run pipeline --skip-ingest          # re-parse + re-embed existing raw files
uv run pipeline --skip-embed           # ingest + parse only (no model download)

# Individual stages
uv run ingest --indication "atopic dermatitis" --out-dir data/raw
uv run parse  --in-dir data/raw --db data/labels.db
uv run embed  --db data/labels.db --chroma-dir data/chroma

# Keyword search — exact term, returns all matching drugs (FTS5 only)
uv run query keyword "vIGA"
uv run query keyword "dupilumab" --section indications_and_usage
uv run query keyword "hepatotoxicity" --json

# Concept search — finds labels relevant to an idea (semantic)
uv run query search "pediatric itch"
uv run query search "IL-13 inhibitor" --top-k 5
uv run query search "mechanism of action JAK inhibitor" --fts-weight 1.0  # hybrid
```

## openFDA ingest filters

Applied at query time (API-side):
- `openfda.product_type:"HUMAN PRESCRIPTION DRUG"`
- `(openfda.application_number:NDA* OR openfda.application_number:BLA*)` — innovator drugs only
- `indications_and_usage:"{indication}"` — parameterised, default `"atopic dermatitis"`

Applied post-download (client-side, in `ingest.py`):
- Marketing status: keep records where `openfda.marketing_status` list contains
  `"Prescription"` (field is inconsistently populated, so filter after fetch)
- Dedup: group by `openfda.spl_set_id`, keep record with highest `openfda.spl_version`
  (cast to int; missing treated as 0)

openFDA pagination: `limit=1000` (API max), `skip` increments by 1000.
Hard cap: `skip + limit <= 25,000`. Warning logged if hit.

## Database schema (SQLite)

Schema is **source-agnostic** — not openFDA-shape-specific. A DailyMed XML parser
produces the same row shape.

```sql
CREATE TABLE labels (
    label_id  TEXT PRIMARY KEY,  -- spl_set_id or equivalent
    source    TEXT NOT NULL,     -- 'openfda_json' | 'dailymed_xml'
    metadata  TEXT               -- JSON blob: application_number, brand_name, etc.
);

CREATE TABLE sections (
    id           INTEGER PRIMARY KEY,
    label_id     TEXT NOT NULL REFERENCES labels(label_id) ON DELETE CASCADE,
    section_name TEXT NOT NULL,
    text         TEXT NOT NULL
);

-- FTS5 content table backed by sections; triggers keep index in sync.
CREATE VIRTUAL TABLE sections_fts USING fts5(
    label_id     UNINDEXED,
    section_name UNINDEXED,
    text,
    content='sections',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Triggers (created by init_db in parse.py):
--   sections_ai  AFTER INSERT  -> populates FTS index
--   sections_ad  AFTER DELETE  -> removes from FTS index
--   sections_au  AFTER UPDATE  -> updates FTS index
-- Deleting a label cascades to sections, which fires sections_ad automatically.

CREATE TABLE sync_state (
    label_id                TEXT PRIMARY KEY,
    spl_version             INTEGER,
    last_pull_timestamp     TEXT,   -- ISO-8601
    embedding_model_version TEXT    -- e.g. 'all-MiniLM-L6-v2'; NULL = not yet embedded
);
```

WAL mode and foreign keys are enabled by `init_db` at runtime.

## Pluggable parser contract

`LabelRecord` is defined in `src/uspi_search/models.py`:

```python
@dataclass
class LabelRecord:
    label_id: str
    source: str                      # 'openfda_json' | 'dailymed_xml'
    metadata: dict
    sections: list[tuple[str, str]]  # [(section_name, text), ...]
```

Any parser module under `src/uspi_search/parsers/` must expose:

```python
def parse(path: Path) -> Iterator[LabelRecord]: ...
```

`parse.py` dynamically imports the parser via `--parser` CLI flag (default `openfda_json`).
Adding DailyMed XML support = add `parsers/dailymed_xml.py` implementing the same interface.

### Current parser: `openfda_json.py`

- Reads a bare JSON array (deduped output from ingest) or a full API envelope
- Extracts `spl_set_id` as `label_id`; skips records with no `spl_set_id`
- Extracts all keys holding `list[str]` values (except metadata keys) as sections
- Cleans text: strips HTML/XML tags, decodes HTML entities, normalises whitespace

## Embed stage details

- **Chunking**: word-count based, default 400 words / 50-word overlap (`chunk_text` in `embed.py`)
- **Model**: `all-MiniLM-L6-v2` (default); swap with `--model` to re-embed everything
- **ChromaDB**: `PersistentClient`, one collection (`label_sections`), cosine distance
- **Chunk ID**: `{label_id}__{section_name}__{chunk_index}` — upsert is idempotent
- **Metadata per chunk**: `label_id`, `section_name`, `chunk_index`, `brand_name`,
  `generic_name`, `application_number` — all filterable at query time via ChromaDB `where`
- **Incremental**: skips labels where `sync_state.embedding_model_version` matches current model

## Query stage details

`query.py` is a Click group with two subcommands. Results are always grouped
by drug (one entry per label, sections listed underneath).

### `keyword` subcommand — exact term search (FTS5 only)
- Runs `sections_fts MATCH ?` (BM25), no ChromaDB, no model load
- Returns **all** matching drugs by default (`--top-k 0`); pass `--top-k N` to limit
- Always shows total match count; if truncated: "Showing X of Y. Use --top-k 0 to see all Y."
- Each matching section shows an FTS5 snippet with the term bracketed: `[vIGA]`
- Snippet length configurable via `--snippet-tokens` (default 40 tokens)
- Section filter: `--section clinical_studies` (repeatable)

### `search` subcommand — concept search (semantic, optional hybrid)
- Encodes query with `SentenceTransformer`, queries ChromaDB by cosine distance
- Aggregates chunk-level hits to section level (best distance per `(label_id, section_name)`)
- Returns top-K drugs ranked by relevance (`--top-k`, default 10); no total count shown
- Add `--fts-weight 1.0` to enable hybrid mode: FTS5 + semantic merged via RRF (k=60)
- In hybrid mode, matching sections show FTS snippets; semantic-only sections show first 300 chars
- Non-ASCII characters in display output are replaced with `?` (Windows cp1252 safety)

### Shared options (go before the subcommand)
`--db`, `--chroma-dir`, `--model`, `--collection`, `--verbose`

### Output
- Human-readable by default; `--json` emits a JSON object with a `drugs` list
- `keyword --json` also includes `total_fts_drugs` at the top level

## Dependencies

All deps are installed. `uv sync` restores the exact locked environment.

| Package | Version | Stage |
|---|---|---|
| `requests` | >=2.32 | ingest |
| `click` | >=8.1 | all CLIs |
| `sentence-transformers` | >=3.0 | embed, query |
| `chromadb` | >=0.6 | embed, query |

## Data directories

- `data/raw/` — gitignored; batch JSON files (`batch_000000.json`) and deduped output
  (`deduped_{indication}_{timestamp}.json`) written by ingest
- `data/chroma/` — gitignored; ChromaDB persistent store written by embed
- `data/labels.db` — gitignored; SQLite written by parse

## Environment variables

| Variable | Purpose |
|---|---|
| `OPENFDA_API_KEY` | Optional; unauthenticated rate limit is 40 req/min |

## Windows notes

- Avoid Unicode characters (arrows, em-dashes, etc.) in strings that Click renders
  as help text — the Windows console defaults to cp1252, which rejects many Unicode
  code points. Use ASCII equivalents (`->` not `->`, `-` not `--`).
- Shell glob expansion does not work in PowerShell for `--in-file data/raw/*.json`;
  use `--in-dir data/raw` instead (parse and pipeline both support it).
