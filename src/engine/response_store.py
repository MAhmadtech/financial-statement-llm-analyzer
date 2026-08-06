"""Phase 5: persist every experiment with complete metadata.

Each run is stored as outputs/responses/<statement_id>_<strategy>_<ts>.json
holding the full record from the plan (statement_id, strategy,
prompt_version, model, extracted_text, prompt, response, structured_json,
latency, timestamp, input_tokens, output_tokens), and a summary row is
appended to outputs/experiments/experiments_index.csv for fast loading.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from config import settings

INDEX_COLUMNS = [
    "response_file", "statement_id", "doc_name", "strategy", "prompt_version",
    "model", "latency", "timestamp", "input_tokens", "output_tokens",
    "json_parsed", "error",
]


def save_response(record: dict) -> Path:
    """Write the full experiment record and index it. Returns the JSON path."""
    settings.RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    settings.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{record['statement_id']}_{record['strategy']}_{ts}.json"
    path = settings.RESPONSES_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    index_path = settings.EXPERIMENTS_DIR / "experiments_index.csv"
    new_file = not index_path.exists()
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(INDEX_COLUMNS)
        writer.writerow(
            [
                filename,
                record.get("statement_id"),
                record.get("doc_name"),
                record.get("strategy"),
                record.get("prompt_version"),
                record.get("model"),
                record.get("latency"),
                record.get("timestamp"),
                record.get("input_tokens"),
                record.get("output_tokens"),
                record.get("structured_json") is not None,
                record.get("error") or "",
            ]
        )
    return path


def load_responses(
    statement_id: str | None = None, strategy: str | None = None
) -> list[dict]:
    """Load stored experiment records, optionally filtered."""
    records = []
    if not settings.RESPONSES_DIR.exists():
        return records
    for path in sorted(settings.RESPONSES_DIR.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Interrupted write (crash/kill mid-save); skip, don't poison runs
            print(f"WARNING: skipping unreadable response file {path.name}")
            continue
        if statement_id and record.get("statement_id") != statement_id:
            continue
        if strategy and record.get("strategy") != strategy:
            continue
        record["response_file"] = path.name
        records.append(record)
    return records


def latest_response(statement_id: str, strategy: str) -> dict | None:
    """Most recent stored record for a statement/strategy pair."""
    records = load_responses(statement_id, strategy)
    return records[-1] if records else None
