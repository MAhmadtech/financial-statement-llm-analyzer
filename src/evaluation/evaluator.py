"""Phase 6: evaluation framework - the core research contribution.

Metrics per stored experiment record vs its ground truth file:

1. fact_accuracy        matched expected facts+ratios / total expected
2. completeness         covered checklist topics / expected topics
3. hallucination_rate   ungrounded numeric claims / numeric claims checked
4. response_time        latency captured by the client
5. user_rating          mean 1-5 rating from the Streamlit CSV (if any)
6. precision/recall/F1  optional advanced metrics over numeric fact claims

Matching rules (documented for the methodology chapter):
  - An expected fact/ratio counts as matched when the numbers in its answer
    appear in the response (1% tolerance, unit-scale aware), or - for
    qualitative answers - when >= 40% of its significant terms appear.
  - Hallucination is measured over the structured JSON's factual fields
    (Revenue, Expenses, Net Profit): a numeric claim is ungrounded if absent
    from the source statement text. Derived ratio fields are excluded
    because correct ratios rarely appear verbatim in filings.
"""

import csv
import json
from pathlib import Path

from config import settings
from config.constants import EXPECTED_TOPICS
from src.evaluation.numeric_match import (
    extract_numbers,
    keyword_overlap,
    value_present,
)

QUALITATIVE_OVERLAP_THRESHOLD = 0.4

# Topic -> keyword families for the completeness checklist
TOPIC_KEYWORDS = {
    "revenue": ["revenue", "sales", "top line", "top-line"],
    "expenses": ["expense", "cost", "cogs", "sg&a", "opex"],
    "profit": ["profit", "net income", "earnings", "margin", "bottom line"],
    "liquidity": ["liquidity", "current ratio", "quick ratio",
                  "working capital", "cash position"],
    "solvency": ["solvency", "debt", "leverage", "debt-to-equity",
                 "interest coverage", "gearing"],
    "risks": ["risk", "concern", "headwind", "uncertainty"],
    "recommendations": ["recommend", "suggest", "advis", "should"],
}

JSON_FACT_FIELDS = ["Revenue", "Expenses", "Net Profit"]


def load_ground_truth(statement_id: str) -> dict:
    path = settings.EVAL_DATASET_DIR / f"{statement_id}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _expected_item_matched(item: dict, response_text: str,
                           response_numbers: set[float]) -> bool:
    answer = item["answer"]
    expected_numbers = extract_numbers(answer)
    if expected_numbers:
        matched = sum(
            1 for v in expected_numbers if value_present(v, response_numbers)
        )
        return matched / len(expected_numbers) >= 0.5
    return keyword_overlap(answer, response_text) >= QUALITATIVE_OVERLAP_THRESHOLD


def fact_accuracy(record: dict, ground_truth: dict) -> dict:
    response_text = record.get("response") or ""
    if record.get("structured_json"):
        response_text += "\n" + json.dumps(record["structured_json"])
    response_numbers = extract_numbers(response_text)

    expected = ground_truth["expected_facts"] + ground_truth["expected_ratios"]
    matched, missed = [], []
    for item in expected:
        if _expected_item_matched(item, response_text, response_numbers):
            matched.append(item["financebench_id"])
        else:
            missed.append(item["financebench_id"])

    total = len(expected)
    return {
        "fact_accuracy": len(matched) / total if total else None,
        "facts_matched": matched,
        "facts_missed": missed,
        "facts_total": total,
    }


def completeness(record: dict) -> dict:
    text = (record.get("response") or "").lower()
    covered = [
        topic
        for topic, keywords in TOPIC_KEYWORDS.items()
        if any(k in text for k in keywords)
    ]
    return {
        "completeness": len(covered) / len(EXPECTED_TOPICS),
        "topics_covered": covered,
        "topics_missing": [t for t in EXPECTED_TOPICS if t not in covered],
    }


def hallucination(record: dict) -> dict:
    structured = record.get("structured_json") or {}
    source_numbers = extract_numbers(record.get("extracted_text") or "")

    claims_checked = 0
    ungrounded = []
    for field in JSON_FACT_FIELDS:
        raw = structured.get(field)
        if raw is None:
            continue
        for value in extract_numbers(str(raw)):
            # Skip tiny values (years, percentages) that inflate the metric
            if abs(value) < 100:
                continue
            claims_checked += 1
            if not value_present(value, source_numbers):
                ungrounded.append({"field": field, "value": value})

    return {
        "hallucination_rate": (
            len(ungrounded) / claims_checked if claims_checked else None
        ),
        "numeric_claims_checked": claims_checked,
        "ungrounded_claims": ungrounded,
    }


def advanced_metrics(record: dict, ground_truth: dict, halluc: dict,
                     accuracy: dict) -> dict:
    """Optional precision/recall/F1 over numeric fact claims.

    precision: grounded JSON numeric claims / all JSON numeric claims
    recall:    fact_accuracy (expected items recovered)
    """
    checked = halluc["numeric_claims_checked"]
    precision = (
        (checked - len(halluc["ungrounded_claims"])) / checked if checked else None
    )
    recall = accuracy["fact_accuracy"]
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def user_ratings() -> dict[tuple[str, str], float]:
    """Mean rating per (statement_id, strategy) from the ratings CSV."""
    path = settings.RATINGS_CSV
    if not Path(path).exists():
        return {}
    sums: dict[tuple[str, str], list[float]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rating = float(row["rating"])
            except (KeyError, ValueError):
                continue
            key = (row.get("statement_id", ""), row.get("strategy", ""))
            sums.setdefault(key, []).append(rating)
    return {k: sum(v) / len(v) for k, v in sums.items()}


def evaluate_record(record: dict, ground_truth: dict | None = None) -> dict:
    """Evaluate one stored experiment record. Returns metrics + details."""
    if ground_truth is None:
        ground_truth = load_ground_truth(record["statement_id"])

    accuracy = fact_accuracy(record, ground_truth)
    complete = completeness(record)
    halluc = hallucination(record)
    advanced = advanced_metrics(record, ground_truth, halluc, accuracy)
    ratings = user_ratings()

    return {
        "statement_id": record["statement_id"],
        "doc_name": record.get("doc_name", ""),
        "strategy": record["strategy"],
        "prompt_version": record.get("prompt_version"),
        "model": record.get("model"),
        "response_file": record.get("response_file", ""),
        "error": record.get("error"),
        **accuracy,
        **complete,
        **halluc,
        "response_time": record.get("latency"),
        "user_rating": ratings.get(
            (record["statement_id"], record["strategy"])
        ),
        **advanced,
    }


METRIC_COLUMNS = [
    "statement_id", "doc_name", "strategy", "prompt_version", "model",
    "response_file", "error", "fact_accuracy", "facts_total", "completeness",
    "hallucination_rate", "numeric_claims_checked", "response_time",
    "user_rating", "precision", "recall", "f1",
]


def evaluate_all(records: list[dict], out_csv: Path | None = None) -> list[dict]:
    """Evaluate many records; optionally write a flat CSV for analysis."""
    results = [evaluate_record(r) for r in records]

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    return results
