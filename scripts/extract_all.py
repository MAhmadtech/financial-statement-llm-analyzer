"""Phase 2: extract and preprocess all evaluation statements.

Reads data/eval_dataset/manifest.csv and processes each PDF in
data/raw_statements into data/processed + data/clean_text_backup.

Run:  python scripts/extract_all.py
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from src.extraction.pdf_extractor import process_pdf


def main() -> None:
    manifest = settings.EVAL_DATASET_DIR / "manifest.csv"
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows, start=1):
        doc_name = row["doc_name"]
        pdf_path = settings.RAW_STATEMENTS_DIR / f"{doc_name}.pdf"
        out_json = settings.PROCESSED_DIR / f"{doc_name}.json"
        if out_json.exists():
            print(f"[{i}/{len(rows)}] {doc_name}: already processed, skipping")
            continue

        start = time.time()
        record = process_pdf(pdf_path, doc_name)
        print(
            f"[{i}/{len(rows)}] {doc_name}: {record['extractor_used']}, "
            f"{record['num_pages']} pages, {record['num_tables']} tables, "
            f"{record['char_count']:,} chars, "
            f"{record['removed_header_footer_lines']} header/footer lines removed "
            f"({time.time() - start:.1f}s)"
        )


if __name__ == "__main__":
    main()
