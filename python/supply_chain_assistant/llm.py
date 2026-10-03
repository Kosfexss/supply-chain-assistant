"""Provider-neutral interface for an optional language-model adapter."""

from typing import Protocol


class LLMClient(Protocol):
    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        """Generate text from trusted instructions and supplied business context."""
        ...