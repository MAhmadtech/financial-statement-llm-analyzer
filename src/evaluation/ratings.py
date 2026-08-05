"""User ratings (Phase 6 metric 5): local CSV store.

The Streamlit app appends one row per 1-5 rating; the evaluator reads means
per (statement_id, strategy).
"""

import csv
from datetime import datetime

from config import settings

COLUMNS = ["timestamp", "statement_id", "doc_name", "strategy", "rating", "comment"]


def append_rating(
    statement_id: str,
    strategy: str,
    rating: int,
    doc_name: str = "",
    comment: str = "",
) -> None:
    if not 1 <= int(rating) <= 5:
        raise ValueError("rating must be between 1 and 5")

    settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not settings.RATINGS_CSV.exists()
    with open(settings.RATINGS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(COLUMNS)
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                statement_id,
                doc_name,
                strategy,
                int(rating),
                comment.replace("\n", " ").strip(),
            ]
        )
