"""Utilities for recording Gemini RAG-call metrics."""

from dataclasses import dataclass, field
from datetime import datetime


GEMINI_FLASH_INPUT_COST_PER_MILLION = 0.15
GEMINI_FLASH_OUTPUT_COST_PER_MILLION = 0.60


@dataclass
class LLMCallRecord:
    """Stores answer-generation metadata for one RAG request."""

    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)


def calculate_cost(model: str, usage_metadata) -> float:
    """Estimate Gemini request cost from response usage metadata."""
    if "gemini-2.5-flash" not in model or usage_metadata is None:
        return 0.0

    prompt_tokens = usage_metadata.prompt_token_count or 0
    completion_tokens = usage_metadata.candidates_token_count or 0

    input_cost = (
        prompt_tokens * GEMINI_FLASH_INPUT_COST_PER_MILLION / 1_000_000
    )

    output_cost = (
        completion_tokens * GEMINI_FLASH_OUTPUT_COST_PER_MILLION / 1_000_000
    )

    return input_cost + output_cost