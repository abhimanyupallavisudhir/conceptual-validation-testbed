"""Shared pieces of the LLM-behaviour templates: subject models, batteries and answer parsing."""

from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path

BATTERIES = Path(__file__).resolve().parents[1] / "data" / "batteries"

MODELS = {
    "openai/gpt-4.1-nano": "GPT-4.1 nano (OpenAI)",
    "openai/gpt-4o-mini": "GPT-4o mini (OpenAI)",
    "openai/gpt-4.1-mini": "GPT-4.1 mini (OpenAI)",
    "openrouter/meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B Instruct (Meta)",
    "anthropic/claude-haiku-4-5": "Claude Haiku 4.5 (Anthropic)",
}
# the subjects of the first (Oct 2026) experiments; gpt-4.1-nano was retired by OpenAI the following week, so
# these grids are frozen as measured
CHEAP_MODELS = ["openai/gpt-4.1-nano", "openai/gpt-4.1-mini", "openrouter/meta-llama/llama-3.1-8b-instruct",
                "anthropic/claude-haiku-4-5"]
# subjects of later templates: inexpensive, non-reasoning, three providers, one open-weight model (re-runnable for
# as long as anyone serves the weights)
SUBJECTS = ["openai/gpt-4o-mini", "openai/gpt-4.1-mini", "openrouter/meta-llama/llama-3.1-8b-instruct",
            "anthropic/claude-haiku-4-5"]

# "Answer: A", "**Answer:** (B)", "answer: b", "Answer - A": the last one counts
ANSWER_RE = re.compile(r"answer\W{0,6}\(?\**([AB])\b", re.I)
NUMBER_RE = re.compile(r"Answer:\s*\$?\s*(-?[\d,]+(?:\.\d+)?)")


def model_name(m: str) -> str:
    return MODELS.get(m, m)


def last_choice(text: str, letters: str = "AB") -> str | None:
    m = re.findall(rf"answer\W{{0,6}}\(?\**([{letters}])\b", text or "", re.I)
    return m[-1].upper() if m else None


def last_number(text: str) -> float | None:
    m = NUMBER_RE.findall(text or "") or re.findall(r"(-?\d[\d,]*(?:\.\d+)?)", text or "")
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


@lru_cache(maxsize=1)
def gsm8k() -> list[tuple[str, int]]:
    rows = []
    with open(BATTERIES / "gsm8k_50.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["question"], int(r["answer"].split("####")[-1].strip().replace(",", ""))))
    return rows


def wrong_answer(x: int, i: int) -> int:
    """A plausible wrong answer for problem i: off by 10-50%, above or below (never negative)."""
    off = max(1, round(abs(x) * random.Random(f"wrong|{i}").choice([0.1, 0.2, 0.25, 0.5])))
    return x + off if (i % 2 or x - off < 0) else x - off
