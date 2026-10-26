"""A small cached, cost-accounted chat client for the testbed's LLM experiments.

Experiments call ``await llm.chat(model, messages, ...)``. Every completion is cached on disk (sqlite under
``$CVT_CACHE`` or ``~/.cache/cvtestbed``), keyed by the model, the messages, the sampling settings and a caller
``key`` that names the draw (so two subjects asked the same question get independent samples), so re-running an
experiment is free and reproducible. Costs come from the provider's token counts and ``prices.yaml``.
``simulate=True`` replaces every call with a stub that prices the prompt and an assumed completion length:
the dry run behind ``cvt label --estimate``.

Models are Inspect names (``openai/gpt-4.1-mini``, ``anthropic/claude-haiku-4-5``,
``openrouter/meta-llama/llama-3.1-8b-instruct``). Reasoning models (``openai/gpt-5*``, ``o3``, ``o4``) spend
``max_tokens`` on hidden reasoning, so they get at least 8,000 tokens at low reasoning effort and no temperature.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REASONING_PREFIXES = ("openai/gpt-5", "openai/o3", "openai/o4", "openai/o1")


def is_reasoning(model: str) -> bool:
    return model.startswith(REASONING_PREFIXES)


@lru_cache(maxsize=1)
def prices() -> dict[str, tuple[float, float]]:
    with open(Path(__file__).with_name("prices.yaml")) as f:
        return {k: (float(v[0]), float(v[1])) for k, v in yaml.safe_load(f).items()}


def price(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = prices().get(model, (2.0, 8.0))  # unknown model: assume a mid-priced one
    return (tokens_in * pin + tokens_out * pout) / 1e6


def cache_dir() -> Path:
    d = Path(os.environ.get("CVT_CACHE", Path.home() / ".cache" / "cvtestbed"))
    d.mkdir(parents=True, exist_ok=True)
    return d


class _Cache:
    def __init__(self, path: Path):
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, k: str) -> dict | None:
        with self.lock:
            row = self.conn.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, v: dict) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO c VALUES (?, ?)", (k, json.dumps(v)))
            self.conn.commit()


@dataclass
class Reply:
    text: str
    cost: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cached: bool = False


@dataclass
class Meter:
    """Running totals for one experiment run."""

    cost: float = 0.0
    calls: int = 0
    cached: int = 0
    parse_failures: int = 0
    notes: list[str] = field(default_factory=list)

    def add(self, r: Reply) -> None:
        self.cost += r.cost
        self.calls += 1
        self.cached += int(r.cached)


class LLM:
    def __init__(self, *, concurrency: int = 32, simulate: bool = False, cache: Path | None = None,
                 assumed_output: int = 300):
        self.concurrency = concurrency
        self._sems: dict[int, asyncio.Semaphore] = {}  # one per event loop (a client may outlive asyncio.run)
        self.simulate, self.assumed_output = simulate, assumed_output
        self.cache = _Cache((cache or cache_dir()) / "llm.sqlite")
        self._models: dict[str, Any] = {}

    def _model(self, name: str):
        if name not in self._models:
            from inspect_ai.model import get_model

            self._models[name] = get_model(name)
        return self._models[name]

    async def chat(self, model: str, messages: list[dict[str, str]] | str, *, max_tokens: int = 1024,
                   temperature: float | None = 0.7, key: str = "", meter: Meter | None = None) -> Reply:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        reasoning = is_reasoning(model)
        if reasoning:
            max_tokens, temperature = max(max_tokens, 8000), None
        k = hashlib.sha256(json.dumps([model, messages, max_tokens, temperature, key]).encode()).hexdigest()
        if self.simulate:
            tin = sum(len(m["content"]) for m in messages) // 4
            tout = min(max_tokens, self.assumed_output) * (3 if reasoning else 1)
            r = Reply(text="(simulated)", cost=price(model, tin, tout), tokens_in=tin, tokens_out=tout)
        elif (hit := self.cache.get(k)) is not None:
            r = Reply(text=hit["text"], cost=0.0, tokens_in=hit["in"], tokens_out=hit["out"], cached=True)
        else:
            r = await self._call(model, messages, max_tokens, temperature, reasoning)
            self.cache.put(k, {"text": r.text, "in": r.tokens_in, "out": r.tokens_out, "cost": r.cost})
        if meter is not None:
            meter.add(r)
        return r

    async def _call(self, model, messages, max_tokens, temperature, reasoning) -> Reply:
        from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, GenerateConfig

        conv = {"system": ChatMessageSystem, "user": ChatMessageUser, "assistant": ChatMessageAssistant}
        msgs = [conv[m["role"]](content=m["content"]) for m in messages]
        cfg = GenerateConfig(max_tokens=max_tokens, temperature=temperature,
                             reasoning_effort="low" if reasoning else None)
        loop = id(asyncio.get_running_loop())
        sem = self._sems.setdefault(loop, asyncio.Semaphore(self.concurrency))
        async with sem:
            for attempt in range(4):
                try:
                    out = await self._model(model).generate(msgs, config=cfg)
                    break
                except Exception:  # transient provider errors: back off and retry
                    if attempt == 3:
                        raise
                    await asyncio.sleep(2 ** attempt * 2)
        u = out.usage
        tin = int(getattr(u, "input_tokens", 0) or 0)
        tout = int(getattr(u, "output_tokens", 0) or 0)
        return Reply(text=out.completion or "", cost=price(model, tin, tout), tokens_in=tin, tokens_out=tout)


_DEFAULT: LLM | None = None


def default_llm(**kw: Any) -> LLM:
    global _DEFAULT
    if _DEFAULT is None or kw:
        _DEFAULT = LLM(**kw)
    return _DEFAULT
