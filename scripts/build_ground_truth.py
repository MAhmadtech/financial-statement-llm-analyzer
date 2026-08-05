"""Phase 1: Build ground truth files from the FinanceBench dataset.

Selects the NUM_EVAL_STATEMENTS documents with the richest question coverage
from financebench_open_source.jsonl and produces, for each:

  data/eval_dataset/statement_XX.json   with pdf_path, expected_summary,
                                        expected_facts, expected_ratios,
                                        expected_risks, expected_recommendations
  data/raw_statements/<doc_name>.pdf    copy of the source PDF
  data/expected_outputs/*.csv           CSV versions of facts/ratios (fallback)
  data/eval_dataset/manifest.csv        statement_id -> document mapping

Derivation rules (documented for the methodology chapter):
  - expected_facts: QA pairs whose question asks for a reported figure.
  - expected_ratios: QA pairs whose question involves a ratio/margin/rate.
  - expected_risks: QA pairs whose question or answer flags a risk/concern.
  - expected_recommendations: FinanceBench contains no analyst
    recommendations, so these are coverage expectations derived from each
    document's ratio/risk topics (used by the completeness metric, not the
    accuracy metric).
  - expected_summary: deterministic template from document metadata plus the
    document's key reported facts.

Run:  python scripts/build_ground_truth.py
"""

import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings

RATIO_KEYWORDS = [
    "ratio", "margin", "roe", "roa", "roic", "turnover", "coverage",
    "cagr", "growth rate", "tax rate", "yield", "days payable",
    "days of inventory", "dpo", "dso", "interest coverage", "quick ratio",
    "current ratio", "debt-to", "payout", "conversion cycle",
    "capital intensity", "retention", "asset turnover", "leverage",
]

RISK_KEYWORDS = [
    "risk", "headwind", "concern", "adverse", "litigation", "lawsuit",
    "impair", "restructur", "covenant", "default", "going concern",
    "volatil", "uncertain", "downturn", "decline", "liquidity profile",
    "solvency", "solvent", "distress", "warning",
]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def first_evidence(record: dict, max_chars: int = 400) -> str:
    evidence = record.get("evidence") or []
    if evidence and isinstance(evidence, list):
        text = evidence[0].get("evidence_text", "")
        return " ".join(text.split())[:max_chars]
    return ""


def classify(record: dict) -> str:
    """Primary bucket for a QA pair: 'ratio' or 'fact'."""
    if contains_any(record["question"], RATIO_KEYWORDS):
        return "ratio"
    return "fact"


def is_risk_related(record: dict) -> bool:
    return contains_any(record["question"], RISK_KEYWORDS) or contains_any(
        str(record["answer"]), RISK_KEYWORDS
    )


def qa_entry(record: dict) -> dict:
    return {
        "financebench_id": record["financebench_id"],
        "question_type": record["question_type"],
        "question": record["question"],
        "answer": str(record["answer"]),
        "justification": record.get("justification") or "",
        "evidence_excerpt": first_evidence(record),
    }


def build_summary(doc_info: dict, facts: list[dict]) -> str:
    company = doc_info.get("company", "the company")
    period = doc_info.get("doc_period", "")
    doc_type = str(doc_info.get("doc_type", "filing")).upper()
    sector = doc_info.get("gics_sector", "")
    key_points = "; ".join(
        f"{f['question'].split('?')[0].strip()}? -> {f['answer']}"
        for f in facts[:5]
    )
    summary = (
        f"{company} {period} {doc_type} filing"
        f"{f' ({sector} sector)' if sector else ''}. "
        f"An accurate summary must be consistent with these reported figures: "
        f"{key_points}"
    )
    return summary


def build_recommend_expectations(ratios: list[dict], risks: list[dict]) -> list[dict]:
    topics = set()
    for entry in ratios + risks:
        lowered = entry["question"].lower()
        for kw in RATIO_KEYWORDS + RISK_KEYWORDS:
            if kw in lowered:
                topics.add(kw)
    expectations = [
        {
            "expectation": (
                "Analysis must provide at least two actionable recommendations "
                "grounded in figures reported in the statement."
            )
        }
    ]
    for topic in sorted(topics)[:5]:
        expectations.append(
            {"expectation": f"Recommendations should address the company's {topic}."}
        )
    return expectations


def main() -> None:
    qa_records = load_jsonl(settings.FINANCEBENCH_QA_FILE)
    doc_records = load_jsonl(settings.FINANCEBENCH_DOCINFO_FILE)
    doc_info_by_name = {d["doc_name"]: d for d in doc_records}

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for record in qa_records:
        by_doc[record["doc_name"]].append(record)

    # Rank documents: most QA coverage first, name as deterministic tie-break.
    ranked = sorted(by_doc.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    selected = ranked[: settings.NUM_EVAL_STATEMENTS]

    settings.EVAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    settings.RAW_STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.EXPECTED_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for idx, (doc_name, records) in enumerate(selected, start=1):
        statement_id = f"statement_{idx:02d}"
        doc_info = doc_info_by_name.get(doc_name, {})

        facts, ratios, risks = [], [], []
        for record in records:
            entry = qa_entry(record)
            if classify(record) == "ratio":
                ratios.append(entry)
            else:
                facts.append(entry)
            if is_risk_related(record):
                risks.append(entry)

        source_pdf = settings.FINANCEBENCH_PDF_DIR / f"{doc_name}.pdf"
        target_pdf = settings.RAW_STATEMENTS_DIR / f"{doc_name}.pdf"
        if source_pdf.exists() and not target_pdf.exists():
            shutil.copy2(source_pdf, target_pdf)

        ground_truth = {
            "statement_id": statement_id,
            "doc_name": doc_name,
            "company": doc_info.get("company", ""),
            "doc_type": doc_info.get("doc_type", ""),
            "doc_period": doc_info.get("doc_period", ""),
            "gics_sector": doc_info.get("gics_sector", ""),
            "pdf_path": str(target_pdf.relative_to(settings.PROJECT_ROOT)),
            "expected_summary": build_summary(doc_info, facts),
            "expected_facts": facts,
            "expected_ratios": ratios,
            "expected_risks": risks,
            "expected_recommendations": build_recommend_expectations(ratios, risks),
        }

        out_path = settings.EVAL_DATASET_DIR / f"{statement_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2, ensure_ascii=False)

        # CSV fallback versions of the expected values
        csv_path = settings.EXPECTED_OUTPUTS_DIR / f"{statement_id}_expected.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "financebench_id", "question", "answer"])
            for category, entries in (
                ("fact", facts), ("ratio", ratios), ("risk", risks)
            ):
                for e in entries:
                    writer.writerow(
                        [category, e["financebench_id"], e["question"], e["answer"]]
                    )

        manifest_rows.append(
            [statement_id, doc_name, doc_info.get("company", ""),
             doc_info.get("doc_period", ""), len(facts), len(ratios), len(risks),
             source_pdf.exists()]
        )
        print(f"{statement_id}: {doc_name} "
              f"(facts={len(facts)}, ratios={len(ratios)}, risks={len(risks)})")

    manifest_path = settings.EVAL_DATASET_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["statement_id", "doc_name", "company", "doc_period",
             "num_facts", "num_ratios", "num_risks", "pdf_found"]
        )
        writer.writerows(manifest_rows)

    print(f"\nWrote {len(selected)} ground truth files to {settings.EVAL_DATASET_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
