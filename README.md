# uspi-search

Hybrid keyword + semantic retrieval system for FDA drug labels via openFDA.

## Setup

```bash
uv sync
```

## Usage

```bash
# Ingest labels for an indication
uv run ingest --indication "atopic dermatitis"

# Parse raw JSON into SQLite
uv run parse

# Embed sections into ChromaDB
uv run embed

# Query
uv run query "dupilumab mechanism of action"

# Or run the full pipeline
uv run pipeline --indication "atopic dermatitis"
```

## Environment

Copy `.env.example` to `.env` and set `OPENFDA_API_KEY` (optional; unauthenticated rate limit is 40 req/min).
