"""Phase 4: Gemini API client.

Responsibilities: API connection, retry logic, timeout handling, token
logging, error handling. All generation parameters come from config/settings.
"""

import time

from google import genai
from google.genai import types

from config import settings
from src.llm.base import LLMClient, LLMResult
from src.utils.logging_setup import get_logger

logger = get_logger("gemini_client")

# Transient failures worth retrying (rate limit, server errors, timeouts)
_RETRYABLE_CODES = {429, 500, 502, 503, 504}


class GeminiClient(LLMClient):
    name = "gemini"

    def __init__(self) -> None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY missing - copy .env.example to .env and set it"
            )
        self._client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options=types.HttpOptions(
                timeout=settings.TIMEOUT_SECONDS * 1000  # milliseconds
            ),
        )
        self._config = types.GenerateContentConfig(
            temperature=settings.TEMPERATURE,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
        )

    def generate(self, prompt: str) -> LLMResult:
        last_error: Exception | None = None
        start = time.time()

        for attempt in range(1, settings.RETRY_COUNT + 1):
            try:
                response = self._client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt,
                    config=self._config,
                )
                latency = time.time() - start
                usage = response.usage_metadata
                input_tokens = getattr(usage, "prompt_token_count", None)
                output_tokens = getattr(usage, "candidates_token_count", None)
                logger.info(
                    "model=%s attempt=%d latency=%.2fs input_tokens=%s "
                    "output_tokens=%s",
                    settings.GEMINI_MODEL, attempt, latency, input_tokens,
                    output_tokens,
                )
                return LLMResult(
                    text=response.text or "",
                    model=settings.GEMINI_MODEL,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=latency,
                    attempts=attempt,
                )
            except Exception as exc:
                last_error = exc
                if not self._is_retryable(exc) or attempt == settings.RETRY_COUNT:
                    logger.error("attempt %d failed permanently: %s", attempt, exc)
                    raise
                wait = 2**attempt  # 2s, 4s, 8s...
                logger.warning(
                    "attempt %d failed (%s), retrying in %ds", attempt, exc, wait
                )
                time.sleep(wait)

        raise RuntimeError(f"Gemini call failed: {last_error}")

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code in _RETRYABLE_CODES:
            return True
        text = str(exc).lower()
        return any(
            marker in text
            for marker in ("timeout", "timed out", "unavailable", "overloaded",
                           "resource_exhausted", "429", "503")
        )
