"""Phase 2: text preprocessing for extracted PDF pages.

Rules (per plan):
  - Remove repeated headers/footers (lines recurring at page tops/bottoms)
  - Normalize whitespace
  - Preserve negative numbers, e.g. (1,234) stays (1,234)
  - Preserve currency formatting, e.g. $1,577 stays $1,577
"""

import re
from collections import Counter

# Lines at the top/bottom of a page are header/footer candidates
EDGE_LINES = 2
# A candidate is removed if it recurs on at least this share of pages
RECURRENCE_THRESHOLD = 0.5
MIN_RECURRENCE_PAGES = 3

_PAGE_NUMBER_RE = re.compile(r"^\s*(page\s+)?\d{1,4}(\s+of\s+\d{1,4})?\s*$", re.I)


def _normalize_line(line: str) -> str:
    """Collapse runs of spaces/tabs; keeps every character otherwise."""
    return re.sub(r"[ \t]+", " ", line).strip()


def _edge_lines(page_lines: list[str]) -> list[str]:
    edges = page_lines[:EDGE_LINES] + page_lines[-EDGE_LINES:]
    return [e for e in edges if e]


def _recurring_edges(pages_lines: list[list[str]]) -> set[str]:
    counter: Counter[str] = Counter()
    for page_lines in pages_lines:
        for line in set(_edge_lines(page_lines)):
            # Ignore digits when fingerprinting so "Page 12" == "Page 13"
            fingerprint = re.sub(r"\d+", "#", line)
            counter[fingerprint] += 1

    total_pages = max(len(pages_lines), 1)
    return {
        fp
        for fp, count in counter.items()
        if count >= MIN_RECURRENCE_PAGES and count / total_pages >= RECURRENCE_THRESHOLD
    }


def clean_pages(page_texts: list[str]) -> tuple[str, int]:
    """Clean a list of per-page texts into one document string.

    Returns (clean_text, number_of_header_footer_lines_removed).
    """
    pages_lines = [
        [_normalize_line(l) for l in text.splitlines()] for text in page_texts
    ]
    recurring = _recurring_edges(pages_lines)

    removed = 0
    out_pages = []
    for page_lines in pages_lines:
        kept = []
        for i, line in enumerate(page_lines):
            if not line:
                continue
            is_edge = i < EDGE_LINES or i >= len(page_lines) - EDGE_LINES
            if is_edge:
                fingerprint = re.sub(r"\d+", "#", line)
                if fingerprint in recurring or _PAGE_NUMBER_RE.match(line):
                    removed += 1
                    continue
            kept.append(line)
        if kept:
            out_pages.append("\n".join(kept))

    return "\n\n".join(out_pages), removed
