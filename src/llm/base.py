"""Adapter interface for LLM backends (Phase 4).

The main pipeline only talks to LLMClient, so additional backends
(GPT-4.1 Mini, Llama 3.1 8B) can be enabled via the LLM_BACKEND config flag
without changing pipeline code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float
    attempts: int


class LLMClient(ABC):
    name: str = "abstract"

    @abstractmethod
    def generate(self, prompt: str) -> LLMResult:
        """Send a prompt, return the model response with usage metadata."""


def get_client() -> LLMClient:
    """Instantiate the backend selected by settings.LLM_BACKEND."""
    from config import settings

    if settings.LLM_BACKEND == "gemini":
        from src.llm.gemini_client import GeminiClient

        return GeminiClient()
    if settings.LLM_BACKEND == "gpt41mini":
        from src.llm.adapters import Gpt41MiniClient

        return Gpt41MiniClient()
    if settings.LLM_BACKEND == "llama":
        from src.llm.adapters import LlamaClient

        return LlamaClient()
    raise ValueError(f"Unknown LLM_BACKEND: {settings.LLM_BACKEND}")
