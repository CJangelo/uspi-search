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
uv run query keyword "vIGA" --top-k 5           # limit to 5 drugs (default: all)
uv run query keyword "vIGA" --snippet-tokens 80  # longer context around each match (default: 40)
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

## Examples

### End-to-end: build the database then search

```bash
# 1. Download labels for atopic dermatitis and build the full index
uv run pipeline --indication "atopic dermatitis"
# Pipeline [ingest -> parse -> embed]  indication="atopic dermatitis"
# Done. 23 labels -> data/labels.db
# Done. 23 label(s) embedded, 1,847 chunk(s) -> data/chroma
```

```bash
# 2. Find every label that mentions a specific clinical endpoint
uv run query keyword "vIGA"

# Query: "vIGA"
# 3 matching drug(s).
# ------------------------------------------------------------------------
# #1  ZORYVE (NDA215985)
#     [clinical_studies_table] ...[vIGA]-AD Success [vIGA]-AD success was
#     defined as a [vIGA]-AD score of "Clear" (0) or "Almost Clear" (1), plus a
#     2-grade [vIGA]-AD score improvement from baseline at Week 4...
#     [clinical_studies] ...[vIGA]-AD Success [vIGA]-AD success was defined...
#
# #2  VTAMA (NDA215272)
#     [clinical_studies] ...the 5-point validated Investigator's Global
#     Assessment ([vIGA]-AD). The majority of subjects had "Moderate" disease...
#     [clinical_studies_table] ...[vIGA]-AD Treatment Success was defined
#     as a [vIGA]-AD score of "Clear" or "Almost Clear" and at least a 2-grade...
#
# #3  Rinvoq (NDA211675)
#     [clinical_studies_table] ...Responder was defined as a patient with
#     [vIGA]-AD 0 or 1 ("clear" or "almost clear") with a reduction of >= 2 points...
```

Snippets default to ~40 tokens of context around the match. Use `--snippet-tokens` to get more:

```bash
uv run query keyword "vIGA" --snippet-tokens 80
```

```bash
# 3. Search for a concept — no exact term needed
uv run query search "pediatric itch"

# Query: "pediatric itch"
# ------------------------------------------------------------------------
# #1  Derma-Smoothe/FS (NDA019452)  score=0.0164
#     [pediatric_use][semantic] ...HPA axis suppression, Cushing's syndrome, and
#     intracranial hypertension have been reported in children receiving topical
#     corticosteroids...
#     [indications_and_usage][semantic] ...topical treatment of moderate to severe
#     atopic dermatitis in pediatric patients 3 months and older...
#
# #2  Doxepin Hydrochloride (NDA020126)  score=0.0161
#     [pediatric_use][semantic] ...use of Doxepin Hydrochloride Cream, 5% in
#     pediatric patients is not recommended...
# ...
```

### Narrow by section

```bash
# Only search the indications_and_usage section
uv run query keyword "dupilumab" --section indications_and_usage

# Stack multiple sections
uv run query keyword "EASI-75" --section clinical_studies --section clinical_studies_table
```

### Get all matching drugs (no limit)

By default `keyword` returns all matches. If you've set `--top-k`, pass `0` to remove the limit:

```bash
uv run query keyword "JAK inhibitor" --top-k 0
# "Showing 8 of 8 matching drugs." — no truncation message means all results are shown
```

### Machine-readable output

```bash
uv run query keyword "vIGA" --json | python -m json.tool
# {
#   "total_fts_drugs": 3,
#   "drugs": [
#     {
#       "label_id": "ec1bb0d1-...",
#       "brand_name": "ZORYVE",
#       "application_number": "NDA215985",
#       "best_score": 0.016393,
#       "sections": [
#         {
#           "section_name": "clinical_studies_table",
#           "fts_snippet": "...[vIGA]-AD Success...",
#           "text": "..."
#         }
#       ]
#     }
#   ]
# }
```

## Environment

| Variable | Purpose |
|---|---|
| `OPENFDA_API_KEY` | Optional; unauthenticated rate limit is 40 req/min |

Set it for the session:
```bash
$env:OPENFDA_API_KEY = "your_key_here"   # PowerShell
export OPENFDA_API_KEY="your_key_here"   # bash
```
