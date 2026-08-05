"""Project-wide constants: checklist topics, JSON schema keys, labels.

These describe *what* the analyses must contain; generation parameters live
in settings.py.
"""

# Sections a complete analysis must discuss (completeness checklist, Phase 6)
EXPECTED_TOPICS = [
    "revenue",
    "expenses",
    "profit",
    "liquidity",
    "solvency",
    "risks",
    "recommendations",
]

# Keys the structured JSON output must contain (Phase 5)
STRUCTURED_JSON_KEYS = [
    "Revenue",
    "Expenses",
    "Net Profit",
    "Current Ratio",
    "Debt to Equity",
    "ROE",
    "Risks",
    "Recommendations",
]

# Sections forced by the structured prompt strategy (Phase 3)
STRUCTURED_SECTIONS = [
    "Executive Summary",
    "Revenue",
    "Expenses",
    "Profitability",
    "Liquidity",
    "Solvency",
    "Financial Ratios",
    "Risks",
    "Recommendations",
]

# Display names for the four prompt strategies
STRATEGY_LABELS = {
    "basic": "Basic Prompt",
    "role_based": "Role-Based Prompt",
    "structured": "Structured Prompt",
    "few_shot": "Few-Shot Prompt",
}

# Ground truth file fields (Phase 1)
GROUND_TRUTH_FIELDS = [
    "pdf_path",
    "expected_summary",
    "expected_facts",
    "expected_ratios",
    "expected_risks",
    "expected_recommendations",
]
