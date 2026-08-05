"""Numeric matching utilities for evaluation (Phase 6).

Financial figures appear in many surface forms ("$1,577", "1577.00",
"1.577 billion", "(423)"). We normalize both sides to floats and compare
with a small relative tolerance across common unit scales, so "$1.577B"
in a response matches "1,577" from a statement reported in millions.
"""

import re

# Matches 1,577 / 1577.00 / (423) / $1.5 / 12.3% etc.
_NUMBER_RE = re.compile(
    r"\(?\$?\s*(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*%?\)?"
)

# A trailing magnitude word directly after the number
_MAGNITUDE_RE = re.compile(
    r"\(?\$?\s*(-?[\d,]+(?:\.\d+)?)\s*(billion|bn|million|mm|mn|thousand|k)\b",
    re.I,
)

_MAGNITUDES = {
    "billion": 1e9, "bn": 1e9,
    "million": 1e6, "mm": 1e6, "mn": 1e6,
    "thousand": 1e3, "k": 1e3,
}

RELATIVE_TOLERANCE = 0.01
# Statements report in thousands/millions/billions inconsistently with
# responses, so a value counts as present if it matches at any common scale.
SCALES = (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)


def _to_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def extract_numbers(text: str) -> set[float]:
    """All numeric values found in text, magnitude words applied."""
    if not text:
        return set()
    values: set[float] = set()

    for match in _MAGNITUDE_RE.finditer(text):
        value = _to_float(match.group(1))
        if value is not None:
            values.add(value * _MAGNITUDES[match.group(2).lower()])

    for match in _NUMBER_RE.finditer(text):
        value = _to_float(match.group(1))
        if value is not None:
            values.add(value)
            values.add(abs(value))
    return values


def value_present(value: float, pool: set[float]) -> bool:
    """True if `value` appears in `pool` at any scale within tolerance."""
    if value == 0:
        return 0 in pool
    for scale in SCALES:
        target = value * scale
        for candidate in pool:
            if candidate == 0:
                continue
            if abs(candidate - target) / max(abs(target), 1e-12) <= RELATIVE_TOLERANCE:
                return True
            if abs(abs(candidate) - abs(target)) / max(abs(target), 1e-12) \
                    <= RELATIVE_TOLERANCE:
                return True
    return False


_STOPWORDS = {
    "the", "and", "for", "was", "were", "that", "this", "with", "from",
    "have", "has", "had", "are", "is", "of", "in", "to", "a", "an", "by",
    "as", "on", "its", "it", "be", "or", "at", "which", "their", "than",
    "into", "during", "fiscal", "year", "company", "companys", "what",
    "does", "did", "how", "when", "give", "answer", "question", "based",
    "using", "according", "million", "millions", "billion", "billions",
    "usd", "amount", "value", "please", "state", "statement",
}


def significant_terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z&'-]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def keyword_overlap(expected: str, response: str) -> float:
    """Share of the expected answer's significant terms found in response."""
    expected_terms = significant_terms(expected)
    if not expected_terms:
        return 0.0
    response_terms = significant_terms(response)
    return len(expected_terms & response_terms) / len(expected_terms)
