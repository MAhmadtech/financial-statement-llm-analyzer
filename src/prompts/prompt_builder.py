"""Phase 3: prompt construction from versioned strategies in prompts.yaml.

Usage:
    from src.prompts.prompt_builder import build_prompt
    built = build_prompt("structured", statement_text)
    built["prompt"], built["strategy"], built["prompt_version"]
"""

from functools import lru_cache

import yaml

from config import settings
from src.preprocessing.section_selector import select_relevant_text


@lru_cache(maxsize=1)
def load_prompt_config() -> dict:
    with open(settings.PROMPTS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def available_strategies() -> list[str]:
    return list(load_prompt_config()["strategies"].keys())


def get_active_version(strategy: str) -> str:
    return load_prompt_config()["strategies"][strategy]["active_version"]


def _fit_to_budget(text: str) -> tuple[str, bool]:
    """Keep the most financially relevant sections within MAX_INPUT_CHARS."""
    return select_relevant_text(text, settings.MAX_INPUT_CHARS)


def build_prompt(
    strategy: str, statement_text: str, version: str | None = None
) -> dict:
    """Render the full prompt for a strategy.

    Returns a dict with the prompt plus the metadata Phase 5 stores:
    strategy, prompt_version, truncated flag.
    """
    config = load_prompt_config()
    strategies = config["strategies"]
    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Available: {list(strategies)}"
        )

    version = version or strategies[strategy]["active_version"]
    versions = strategies[strategy]["versions"]
    if version not in versions:
        raise ValueError(
            f"Unknown version '{version}' for strategy '{strategy}'. "
            f"Available: {list(versions)}"
        )

    text, reduced = _fit_to_budget(statement_text)
    prompt = versions[version]["template"].format(statement_text=text)
    prompt = prompt.rstrip() + "\n\n" + config["json_output_instruction"].rstrip()

    return {
        "strategy": strategy,
        "prompt_version": f"{strategy}_{version}",
        "prompt": prompt,
        "statement_truncated": reduced,
    }
