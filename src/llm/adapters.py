"""Adapter stubs for optional backends (Phase 4, plan section 'Optional').

Per project scope these are interface-only: selecting them via LLM_BACKEND
raises a clear error instead of calling a second live API. Implementing one
means subclassing LLMClient.generate() - no pipeline changes needed.
"""

from src.llm.base import LLMClient, LLMResult


class Gpt41MiniClient(LLMClient):
    name = "gpt41mini"

    def generate(self, prompt: str) -> LLMResult:
        raise NotImplementedError(
            "GPT-4.1 Mini adapter is a stub (out of scope). Set "
            "LLM_BACKEND=gemini, or implement this adapter with the OpenAI "
            "SDK: generate() must return an LLMResult."
        )


class LlamaClient(LLMClient):
    name = "llama"

    def generate(self, prompt: str) -> LLMResult:
        raise NotImplementedError(
            "Llama 3.1 8B adapter is a stub (out of scope). Set "
            "LLM_BACKEND=gemini, or implement this adapter against a local "
            "inference server: generate() must return an LLMResult."
        )
