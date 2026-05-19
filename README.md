# uspi-search

Retrieval system for FDA drug labels via openFDA. Two distinct search modes:

- **keyword** — exact term search (FTS5 BM25). Returns every label containing the term with a snippet showing where it appears.
- **search** — concept search (semantic via ChromaDB). Finds labels relevant to an idea, not just exact words. Optional hybrid mode adds keyword scoring.

Not RAG — no generation step, pure retrieval.

## Setup

```bash
uv sync
```

## Full pipeline

```bash
# Ingest, parse, and embed in one step (downloads ~400MB model on first run)
uv run pipeline --indication "atopic dermatitis"

# Skip ingest (re-parse + re-embed existing raw files)
uv run pipeline --skip-ingest

# Skip embed (ingest + parse only, no model download)
uv run pipeline --skip-embed
```

## Individual stages

```bash
# Stage 1 — fetch labels from openFDA
uv run ingest --indication "atopic dermatitis" --out-dir data/raw

# Stage 2 — parse raw JSON into SQLite + FTS5
uv run parse --in-dir data/raw --db data/labels.db

# Stage 3 — embed sections into ChromaDB
uv run embed --db data/labels.db --chroma-dir data/chroma
```

## Searching

### Keyword search — exact term

Returns every drug label containing the term. Results are grouped by drug with
FTS5 snippets showing the match in context (`[term]` bracketed).

```bash
uv run query keyword "vIGA"
uv run query keyword "dupilumab" --section indications_and_usage
uv run query keyword "vIGA" --top-k 5        # limit to 5 drugs (default: all)
uv run query keyword "vIGA" --json
```

### Concept search — semantic

Finds labels relevant to the idea, ranked by semantic similarity. Requires
`embed` to have been run first.

```bash
uv run query search "pediatric itch"
uv run query search "IL-13 signaling pathway" --top-k 5
uv run query search "hepatotoxicity risk" --fts-weight 1.0   # hybrid mode
uv run query search "pediatric itch" --json
```

### Shared options

Placed before the subcommand:

```bash
uv run query --db data/labels.db --verbose keyword "vIGA"
```

| Option | Default | Description |
|---|---|---|
| `--db` | `data/labels.db` | SQLite database path |
| `--chroma-dir` | `data/chroma` | ChromaDB directory |
| `--model` | `all-MiniLM-L6-v2` | Embedding model (must match what `embed` used) |
| `--verbose` / `-v` | off | Debug logging |

## Environment

| Variable | Purpose |
|---|---|
| `OPENFDA_API_KEY` | Optional; unauthenticated rate limit is 40 req/min |

Set it for the session:
```bash
$env:OPENFDA_API_KEY = "your_key_here"   # PowerShell
export OPENFDA_API_KEY="your_key_here"   # bash
```
