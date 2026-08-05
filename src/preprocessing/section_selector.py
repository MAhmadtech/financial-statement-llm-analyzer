"""Select the most financially relevant text when a filing exceeds the
prompt budget (Phase 2/3 support).

Long 10-Ks put the consolidated statements deep in the document, so cutting
the *first* N characters would feed the model business prose and drop the
numbers. Instead we split the clean text into blocks, score each block by
financial-content density, and keep the highest-value blocks (in original
order) within the character budget.
"""

import re

BLOCK_CHARS = 3000

_FINANCIAL_KEYWORDS = [
    "revenue", "net income", "net earnings", "net sales", "total assets",
    "total liabilities", "stockholders' equity", "shareholders' equity",
    "cash flow", "operating income", "operating activities", "gross profit",
    "cost of sales", "cost of goods", "balance sheet", "income statement",
    "consolidated statements", "consolidated balance", "earnings per share",
    "long-term debt", "current assets", "current liabilities", "cash and cash",
    "operating expenses", "interest expense", "income tax", "dividends",
    "capital expenditures", "depreciation", "amortization", "inventories",
    "accounts receivable", "accounts payable", "retained earnings",
]

_DIGIT_RE = re.compile(r"\d")


def _score_block(block: str) -> float:
    lowered = block.lower()
    keyword_hits = sum(lowered.count(k) for k in _FINANCIAL_KEYWORDS)
    digit_density = len(_DIGIT_RE.findall(block)) / max(len(block), 1)
    # Keyword presence dominates; digit density separates statement tables
    # from prose that merely mentions "revenue".
    return keyword_hits + 40 * digit_density


def select_relevant_text(text: str, budget_chars: int) -> tuple[str, bool]:
    """Return (text_within_budget, was_reduced).

    Text under budget passes through untouched. Otherwise the top-scoring
    blocks are kept in document order, joined with an ellipsis marker so the
    model knows material was skipped.
    """
    if len(text) <= budget_chars:
        return text, False

    blocks = [
        (i, text[start : start + BLOCK_CHARS])
        for i, start in enumerate(range(0, len(text), BLOCK_CHARS))
    ]
    scored = sorted(blocks, key=lambda b: _score_block(b[1]), reverse=True)

    marker = "\n\n[... less relevant filing sections omitted ...]\n\n"
    # Reserve room for join markers (worst case: every selected block disjoint)
    max_blocks = max(1, (budget_chars) // (BLOCK_CHARS + len(marker)))
    selected = sorted(scored[:max_blocks], key=lambda b: b[0])

    pieces = []
    previous_index = None
    for index, block in selected:
        if previous_index is not None and index != previous_index + 1:
            pieces.append(marker)
        pieces.append(block)
        previous_index = index
    return "".join(pieces), True
