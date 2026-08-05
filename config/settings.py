"""Central configuration for the Financial LLM Prompting project.

All tunable values live here (loaded from .env where present) so that no
module ever hardcodes model names, generation parameters, or paths.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_STATEMENTS_DIR = DATA_DIR / "raw_statements"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DATASET_DIR = DATA_DIR / "eval_dataset"
EXPECTED_OUTPUTS_DIR = DATA_DIR / "expected_outputs"
CLEAN_TEXT_BACKUP_DIR = DATA_DIR / "clean_text_backup"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESPONSES_DIR = OUTPUTS_DIR / "responses"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
REPORTS_DIR = OUTPUTS_DIR / "reports"
CHARTS_DIR = OUTPUTS_DIR / "charts"
LOGS_DIR = OUTPUTS_DIR / "logs"

CONFIG_DIR = PROJECT_ROOT / "config"
PROMPTS_FILE = CONFIG_DIR / "prompts.yaml"

# Source dataset (FinanceBench clone)
FINANCEBENCH_DIR = PROJECT_ROOT / "financebench"
FINANCEBENCH_PDF_DIR = FINANCEBENCH_DIR / "pdfs"
FINANCEBENCH_QA_FILE = FINANCEBENCH_DIR / "data" / "financebench_open_source.jsonl"
FINANCEBENCH_DOCINFO_FILE = (
    FINANCEBENCH_DIR / "data" / "financebench_document_information.jsonl"
)

# User ratings collected from the Streamlit app (local CSV store)
RATINGS_CSV = OUTPUTS_DIR / "ratings.csv"

# ---------------------------------------------------------------------------
# Environment / secrets
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# LLM configuration (env-overridable, never hardcoded elsewhere)
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))
TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
RETRY_COUNT = int(os.getenv("LLM_RETRY_COUNT", "3"))

# Statement text passed into a prompt is truncated to this many characters to
# keep latency and cost bounded on 100+ page 10-K filings.
MAX_INPUT_CHARS = int(os.getenv("LLM_MAX_INPUT_CHARS", "60000"))

# Backend switch. "gemini" is the only live backend; "gpt41mini" and "llama"
# are adapter stubs kept for future multi-model comparison (see src/llm/).
LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
NUM_EVAL_STATEMENTS = int(os.getenv("NUM_EVAL_STATEMENTS", "20"))
PROMPT_STRATEGIES = ["basic", "role_based", "structured", "few_shot"]

# Pause between batch API calls to respect free-tier rate limits
BATCH_SLEEP_SECONDS = float(os.getenv("BATCH_SLEEP_SECONDS", "10"))


def validate() -> list[str]:
    """Return a list of configuration problems (empty list = all good)."""
    problems = []
    if not GEMINI_API_KEY:
        problems.append("GEMINI_API_KEY is missing - copy .env.example to .env")
    if LLM_BACKEND not in ("gemini", "gpt41mini", "llama"):
        problems.append(f"Unknown LLM_BACKEND: {LLM_BACKEND}")
    if not FINANCEBENCH_QA_FILE.exists():
        problems.append(f"FinanceBench QA file not found: {FINANCEBENCH_QA_FILE}")
    return problems
