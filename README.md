# uspi-search

Retrieval system for FDA drug labels via openFDA. Two distinct search modes:

- **keyword** — exact term search (FTS5 BM25). Returns every label containing the term with a snippet showing where it appears.
- **search** — concept search (semantic via ChromaDB). Finds labels relevant to an idea, not just exact words. Uses [PubMedBERT](https://huggingface.co/NeuML/pubmedbert-base-embeddings) embeddings by default — a biomedical model that understands clinical vocabulary (pruritus = itch, IL-31 pathway, pediatric populations, etc.). Optional hybrid mode adds keyword scoring.

Not RAG — no generation step, pure retrieval.

## Setup

```bash
uv sync
```

## Building the index

The index is built in three stages that must run in order:

1. **Ingest** — fetches drug label JSON from openFDA and saves it locally
2. **Parse** — reads the raw JSON and writes label text into SQLite with a full-text search index
3. **Embed** — chunks each label section, embeds it with PubMedBERT, and stores the vectors in ChromaDB

Once all three stages have run, the `keyword` and `search` commands are available.

### Full pipeline (all three stages at once)

The `pipeline` command runs ingest → parse → embed in sequence. Use this for a
fresh build or when you want to pull updated labels from openFDA.

On the first run it downloads the embedding model (~440MB):

```bash
uv run pipeline --indication "atopic dermatitis"
```

If you already have raw label files downloaded (in `data/raw/`) and only want
to re-parse and re-embed them — without hitting the openFDA API again — use
`--skip-ingest`:

```bash
uv run pipeline --skip-ingest
```

If you only need the SQLite database for keyword search and want to skip the
model download entirely, use `--skip-embed`:

```bash
uv run pipeline --skip-embed
```

### Individual stages

Run stages individually when you need more control — for example, to use a
different indication for ingest, a different database path, or to try a
different embedding model without re-downloading labels.

**Stage 1 — Ingest**

Fetches labels matching an indication from openFDA and saves raw JSON to disk.
The `--indication` text is searched inside the `indications_and_usage` field.

```bash
uv run ingest --indication "atopic dermatitis" --out-dir data/raw
```

Output: `data/raw/deduped_atopic_dermatitis_<timestamp>.json`

**Stage 2 — Parse**

Reads the raw JSON, extracts all text sections, and writes them into SQLite
with an FTS5 full-text search index. Subsequent runs are incremental — labels
already in the database are skipped unless their version has changed.

```bash
uv run parse --in-dir data/raw --db data/labels.db
```

Output: `data/labels.db`

**Stage 3 — Embed**

Chunks each section, embeds with the specified model, and upserts vectors into
ChromaDB. Each model gets its own subdirectory under `--chroma-dir`, so
switching models does not overwrite existing embeddings.

```bash
uv run embed --db data/labels.db --chroma-dir data/chroma
```

To try a different model, pass `--model`. The same flag must be passed to
`query` at search time:

```bash
uv run embed --db data/labels.db --chroma-dir data/chroma --model BAAI/bge-base-en-v1.5
uv run query --model BAAI/bge-base-en-v1.5 search "pediatric itch"
```

Output: `data/chroma/pubmedbert-base-embeddings/` (or the slug of whatever model you used)

## Searching

### Shared options

Placed before the subcommand:

```bash
uv run query --db data/labels.db --verbose keyword "vIGA"
```

| Option | Default | Description |
|---|---|---|
| `--db` | `data/labels.db` | SQLite database path |
| `--chroma-dir` | `data/chroma` | ChromaDB directory |
| `--model` | `NeuML/pubmedbert-base-embeddings` | Embedding model (must match what `embed` used) |
| `--verbose` / `-v` | off | Debug logging |

### Keyword search — exact term

Returns every drug label containing the term. Results are grouped by drug with
FTS5 snippets showing the match in context (`[term]` bracketed).

```bash
uv run query keyword "vIGA"
uv run query keyword "dupilumab" --section indications_and_usage
uv run query keyword "EASI-75" --section clinical_studies --section clinical_studies_table
```

| Option | Default | Description |
|---|---|---|
| `--section` | (all sections) | Restrict to a section name. Repeatable. |
| `--top-k` | 0 (all) | Max drugs to return. 0 = return all matching drugs. |
| `--snippet-tokens` | 100 | Approximate tokens of context around each match. |
| `--json` | off | Emit results as JSON. |

### Concept search — semantic

Finds labels relevant to the idea, ranked by semantic similarity. Requires
`embed` to have been run first.

```bash
uv run query search "pediatric itch"
uv run query search "IL-13 signaling pathway" --top-k 5
uv run query search "hepatotoxicity risk" --fts-weight 1.0   # hybrid: semantic + keyword
```

| Option | Default | Description |
|---|---|---|
| `--section` | (all sections) | Restrict to a section name. Repeatable. |
| `--top-k` | 10 | Drugs to return. |
| `--fts-weight` | 0.0 | Set >0 to blend keyword results (hybrid mode). |
| `--snippet-tokens` | 100 | Context tokens per FTS snippet (hybrid mode only). |
| `--json` | off | Emit results as JSON. |

## Examples

### End-to-end: build the database then search

```bash
# 1. Download labels for atopic dermatitis and build the full index
uv run pipeline --indication "atopic dermatitis"
# Pipeline [ingest -> parse -> embed]  indication="atopic dermatitis"
# Done. 70 labels -> data/labels.db
# Done. 70 label(s) embedded, 2,703 chunk(s) -> data/chroma/pubmedbert-base-embeddings
```

```bash
# 2. Find every label that mentions a specific clinical endpoint
uv run query keyword "vIGA"

# Query: "vIGA"
# 3 matching drug(s).
# ------------------------------------------------------------------------
# #1  ZORYVE (NDA215985)
#     [clinical_studies_table] ...N=433 N=221 N=451 N=232 [vIGA]-AD success
#     32.0% 15.2% 28.9% 12.0% Difference from vehicle (95% CI) ...
#     17.4% (11.09%, 23.75%) 16.5% (10.61%, 22.42%) Figure 2: [vIGA]-AD
#     Success [vIGA]-AD success was defined as a [vIGA]-AD score of "Clear"
#     (0) or "Almost Clear" (1), plus a 2-grade [vIGA]-AD score improvement...
#     [clinical_studies] ...(same excerpt)
#
# #2  VTAMA (NDA215272)
#     [clinical_studies] ...Number of subjects randomized 270 137 271 135
#     [vIGA]-AD Treatment Success ... treatment success was defined as a
#     [vIGA]-AD score of "Clear" or "Almost Clear" and at least a 2-grade
#     improvement from baseline. Difference from Vehicle (95% CI)...
#
# #3  Rinvoq (NDA211675)
#     [clinical_studies_table] ...Responder was defined as a patient with
#     [vIGA]-AD 0 or 1 ("clear" or "almost clear") with a reduction of
#     >= 2 points on a 0-4 ordinal scale...
```

```bash
# 3. Search for a concept — no exact term needed
uv run query search "pediatric itch"

# Query: "pediatric itch"
# ------------------------------------------------------------------------
# #1  NEMLUVIO (BLA761390)  score=0.0164
#     [pediatric_use][semantic] ...The safety and effectiveness of NEMLUVIO for
#     the treatment of moderate-to-severe atopic dermatitis in combination with
#     topical corticosteroids and/or calcineurin inhibitors have been established
#     in pediatric patients 12 years of age and older...
#     [indications_and_usage][semantic] NEMLUVIO is an interleukin-31 receptor
#     antagonist indicated for the treatment of adults and pediatric patients 12
#     years of age and older with moderate-to-severe atopic dermatitis...
#
# #2  Tacrolimus (NDA050777)  score=0.0161
#     [pediatric_use][semantic] ...Tacrolimus Ointment 0.03% is recommended for
#     use as second-line therapy for moderate to severe atopic dermatitis in
#     non-immunocompromised children aged 2 to 15 years...
#
# #3  Dupixent (BLA761055)  score=0.0147
#     [pediatric_use][semantic] ...The safety and effectiveness of DUPIXENT have
#     been established in pediatric patients 6 months of age and older with
#     moderate-to-severe AD...
# ...
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
