"""Phase 9: batch experiments - all statements x all prompt strategies.

Resumable: a (statement, strategy) pair is skipped when a successful stored
response already exists, so interrupted runs can simply be restarted.

Run:  python scripts/run_experiments.py
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from src.engine.analyzer import analyze_statement
from src.engine.response_store import load_responses
from src.evaluation.evaluator import evaluate_all
from src.extraction.pdf_extractor import load_processed_text
from src.llm.base import get_client
from src.utils.logging_setup import get_logger

logger = get_logger("batch")


def completed_pairs() -> set[tuple[str, str]]:
    return {
        (r["statement_id"], r["strategy"])
        for r in load_responses()
        if not r.get("error")
    }


def main() -> None:
    manifest = settings.EVAL_DATASET_DIR / "manifest.csv"
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    done = completed_pairs()
    todo = [
        (row, strategy)
        for row in rows
        for strategy in settings.PROMPT_STRATEGIES
        if (row["statement_id"], strategy) not in done
    ]
    total = len(rows) * len(settings.PROMPT_STRATEGIES)
    print(f"{total} experiments total; {total - len(todo)} already done; "
          f"{len(todo)} to run")

    client = get_client()
    failures = 0
    for i, (row, strategy) in enumerate(todo, start=1):
        sid, doc = row["statement_id"], row["doc_name"]
        try:
            text = load_processed_text(doc)
        except FileNotFoundError as exc:
            print(f"[{i}/{len(todo)}] {sid}/{strategy}: SKIP - {exc}")
            continue

        record = analyze_statement(text, strategy, sid, doc, client)
        if record["error"]:
            failures += 1
            print(f"[{i}/{len(todo)}] {sid}/{strategy}: FAILED - "
                  f"{record['error'][:100]}")
        else:
            print(f"[{i}/{len(todo)}] {sid}/{strategy}: ok "
                  f"{record['latency']}s, json={record['structured_json'] is not None}")
        time.sleep(settings.BATCH_SLEEP_SECONDS)

    # Evaluate every successful stored record and write the flat metrics CSV
    records = [r for r in load_responses() if not r.get("error")]
    # Keep only the latest record per (statement, strategy)
    latest: dict[tuple[str, str], dict] = {}
    for r in records:
        latest[(r["statement_id"], r["strategy"])] = r
    results = evaluate_all(
        list(latest.values()),
        out_csv=settings.EXPERIMENTS_DIR / "evaluations.csv",
    )
    print(f"\nEvaluated {len(results)} experiments -> "
          f"{settings.EXPERIMENTS_DIR / 'evaluations.csv'}")
    if failures:
        print(f"WARNING: {failures} experiments failed; rerun this script to retry.")


if __name__ == "__main__":
    main()
