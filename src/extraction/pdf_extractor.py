"""Phase 2: PDF extraction with pdfplumber (primary) and PyMuPDF (fallback).

Responsibilities (per plan):
  - Extract text and tables
  - Preserve financial values, negative numbers, currency formatting
  - Remove headers/footers, normalize whitespace (see preprocessing module)
"""

import datetime
import json
from pathlib import Path

from config import settings
from src.preprocessing.text_cleaner import clean_pages

# Heuristic: below this many characters per page on average, the primary
# extractor probably failed (scanned/odd encoding) and we try the fallback.
MIN_AVG_CHARS_PER_PAGE = 80


def extract_with_pdfplumber(pdf_path: Path) -> dict:
    import pdfplumber

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            pages.append(
                {"page_num": page.page_number, "text": text, "tables": tables}
            )
    return {"extractor": "pdfplumber", "pages": pages}


def extract_with_pymupdf(pdf_path: Path) -> dict:
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            tables = []
            try:
                tables = [t.extract() for t in page.find_tables().tables]
            except Exception:
                pass  # table detection is best-effort in the fallback path
            pages.append(
                {"page_num": page.number + 1, "text": text, "tables": tables}
            )
    return {"extractor": "pymupdf", "pages": pages}


def _text_quality_ok(result: dict) -> bool:
    pages = result["pages"]
    if not pages:
        return False
    total_chars = sum(len(p["text"]) for p in pages)
    return total_chars / len(pages) >= MIN_AVG_CHARS_PER_PAGE


def extract_document(pdf_path: Path) -> dict:
    """Extract a PDF, falling back to PyMuPDF when pdfplumber under-delivers."""
    pdf_path = Path(pdf_path)
    try:
        result = extract_with_pdfplumber(pdf_path)
        if _text_quality_ok(result):
            return result
        fallback_reason = "low text yield from pdfplumber"
    except Exception as exc:
        fallback_reason = f"pdfplumber error: {exc}"

    result = extract_with_pymupdf(pdf_path)
    result["fallback_reason"] = fallback_reason
    return result


def process_pdf(pdf_path: Path, doc_name: str | None = None) -> dict:
    """Extract + clean a PDF and persist processed JSON/text artifacts.

    Writes:
      data/processed/<doc_name>.json  (full record incl. tables)
      data/processed/<doc_name>.txt   (clean text only)
      data/clean_text_backup/<doc_name>.txt
    """
    pdf_path = Path(pdf_path)
    doc_name = doc_name or pdf_path.stem

    raw = extract_document(pdf_path)
    clean_text, removed_lines = clean_pages([p["text"] for p in raw["pages"]])

    num_tables = sum(len(p["tables"]) for p in raw["pages"])
    record = {
        "doc_name": doc_name,
        "pdf_path": str(pdf_path),
        "extractor_used": raw["extractor"],
        "fallback_reason": raw.get("fallback_reason"),
        "num_pages": len(raw["pages"]),
        "num_tables": num_tables,
        "char_count": len(clean_text),
        "removed_header_footer_lines": removed_lines,
        "extracted_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "text": clean_text,
        "tables": [
            {"page_num": p["page_num"], "tables": p["tables"]}
            for p in raw["pages"]
            if p["tables"]
        ],
    }

    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    settings.CLEAN_TEXT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    json_path = settings.PROCESSED_DIR / f"{doc_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)

    for txt_path in (
        settings.PROCESSED_DIR / f"{doc_name}.txt",
        settings.CLEAN_TEXT_BACKUP_DIR / f"{doc_name}.txt",
    ):
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(clean_text)

    return record


def load_processed_text(doc_name: str) -> str:
    """Return clean text for a processed document, using the backup if needed."""
    for candidate in (
        settings.PROCESSED_DIR / f"{doc_name}.txt",
        settings.CLEAN_TEXT_BACKUP_DIR / f"{doc_name}.txt",
    ):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"No processed text for {doc_name}; run scripts/extract_all.py first"
    )
