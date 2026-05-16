"""
parse.py — Read raw openFDA JSON, extract label sections, write to SQLite + FTS5.

Pluggable parser contract: any parser module must expose
    parse(raw_path: Path) -> Iterator[LabelRecord]
"""
