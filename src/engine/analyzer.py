"""Phase 4: analysis pipeline.

Financial statement -> prompt builder -> LLM backend -> response parsing ->
stored experiment record (Phase 5).
"""

from datetime import datetime

from config import settings
from src.engine.response_store import save_response
from src.llm.base import LLMClient, get_client
from src.prompts.prompt_builder import build_prompt
from src.utils.json_parser import extract_json_block
from src.utils.logging_setup import get_logger

logger = get_logger("analyzer")


def analyze_statement(
    statement_text: str,
    strategy: str,
    statement_id: str = "adhoc",
    doc_name: str = "",
    client: LLMClient | None = None,
    save: bool = True,
) -> dict:
    """Run one experiment: analyze one statement with one prompt strategy.

    Returns the full experiment record (and stores it when save=True).
    A failed LLM call still produces a stored record with the error field
    set, so batch runs are never silently lossy.
    """
    client = client or get_client()
    built = build_prompt(strategy, statement_text)

    record = {
        "statement_id": statement_id,
        "doc_name": doc_name,
        "strategy": strategy,
        "prompt_version": built["prompt_version"],
        "model": settings.GEMINI_MODEL if client.name == "gemini" else client.name,
        "backend": client.name,
        "extracted_text": statement_text,
        "statement_truncated": built["statement_truncated"],
        "prompt": built["prompt"],
        "response": None,
        "structured_json": None,
        "latency": None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_tokens": None,
        "output_tokens": None,
        "attempts": None,
        "error": None,
    }

    try:
        result = client.generate(built["prompt"])
        record.update(
            response=result.text,
            structured_json=extract_json_block(result.text),
            latency=round(result.latency_seconds, 3),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            attempts=result.attempts,
        )
        if record["structured_json"] is None:
            logger.warning(
                "%s/%s: no structured JSON block found in response",
                statement_id, strategy,
            )
    except Exception as exc:
        record["error"] = str(exc)
        logger.error("%s/%s failed: %s", statement_id, strategy, exc)

    if save:
        save_response(record)
    return record


def run_all_strategies(
    statement_text: str,
    statement_id: str = "adhoc",
    doc_name: str = "",
    save: bool = True,
) -> dict[str, dict]:
    """Run every configured prompt strategy against one statement."""
    client = get_client()
    results = {}
    for strategy in settings.PROMPT_STRATEGIES:
        logger.info("running %s / %s", statement_id, strategy)
        results[strategy] = analyze_statement(
            statement_text, strategy, statement_id, doc_name, client, save
        )
    return results
