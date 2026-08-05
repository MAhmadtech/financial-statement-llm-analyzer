"""Extract the structured JSON block from an LLM response (Phase 5)."""

import json
import re

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def extract_json_block(text: str) -> dict | None:
    """Return the structured JSON dict from a response, or None.

    Tries, in order: fenced ```json block, then the largest brace-balanced
    candidate found in the text.
    """
    if not text:
        return None

    for match in _FENCED_JSON_RE.finditer(text):
        parsed = _try_parse(match.group(1))
        if parsed is not None:
            return parsed

    # Fall back to brace matching (model sometimes omits the fence)
    for candidate in _brace_candidates(text):
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed
    return None


def _try_parse(candidate: str) -> dict | None:
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _brace_candidates(text: str) -> list[str]:
    """Balanced {...} spans, longest first."""
    spans = []
    stack = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:  # top-level span closed
                spans.append(text[start : i + 1])
    return sorted(spans, key=len, reverse=True)
