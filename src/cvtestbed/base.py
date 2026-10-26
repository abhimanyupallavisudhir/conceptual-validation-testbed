"""Shared machinery for families whose questions are generated with their answers.

Conventions every family follows:

* two options, and the correct one is option A on exactly half (rounded) of each family's questions, at
  positions drawn independently of anything in the question;
* where a family has a "canonical" answer (one-boxing, exploiting the bug, ...), it is correct on half the
  questions, by rejection sampling: no single slogan answers a family (``balance_keys``);
* question ids are ``<family with '-' for '/'>-s<seed>-<index>``, and generation is deterministic in
  ``(seed, index)``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from cvtestbed.schema import Family, Question, split_of


@dataclass
class Draft:
    """One generated question before placement: the true option first."""

    text: str
    options: tuple[str, str]  # (TRUE option, FALSE option)
    params: dict[str, Any]
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""


def qid(family: str, seed: int, index: int) -> str:
    return f"{family.replace('/', '-')}-s{seed}-{index:04d}"


def place(family: Family, index: int, draft: Draft, *, seed: int, true_first: bool) -> Question:
    true_text, false_text = draft.options
    options = (true_text, false_text) if true_first else (false_text, true_text)
    i = qid(family.name, seed, index)
    return Question(id=i, family=family.name, text=draft.text, options=options, answer=0 if true_first else 1,
                    status="labelled", tier=family.tier, split=split_of(i), params=dict(draft.params),
                    data=dict(draft.data), source=draft.source or family.name)


def balanced_positions(n: int, seed: int, key: str) -> list[bool]:
    """``true_first`` flags for ``n`` questions: half true-first, in a seeded random order."""
    flags = [i < (n + 1) // 2 for i in range(n)]
    random.Random(f"positions|{key}|{seed}").shuffle(flags)
    return flags


class Source:
    """Where a domain's questions come from. Subclasses define ``FAMILIES`` and ``questions``."""

    name: ClassVar[str] = ""
    FAMILIES: ClassVar[dict[str, Family]] = {}

    def questions(self, family: str, n: int | None, seed: int) -> list[Question]:  # pragma: no cover
        raise NotImplementedError


class Generated(Source):
    """Families generated with their answers: subclasses define ``generate(family, rng) -> Draft | None``."""

    max_tries: ClassVar[int] = 200
    balance: ClassVar[bool] = True

    def generate(self, family: str, rng: random.Random) -> Draft | None:  # pragma: no cover
        raise NotImplementedError

    def truth_key(self, draft: Draft) -> Any:
        return draft.params.get("winner")

    def balance_keys(self, family: str) -> tuple[Any, ...]:
        return ()

    def drafts(self, family: str, n: int, seed: int) -> list[Draft]:
        rng = random.Random(f"{self.name}|{family}|{seed}")
        keys = self.balance_keys(family) if self.balance else ()
        cap = -(-n // len(keys)) if keys else n
        out: list[Draft] = []
        counts: dict[Any, int] = {}
        tries = 0
        while len(out) < n and tries < n * self.max_tries:
            tries += 1
            d = self.generate(family, rng)
            if d is None:
                continue
            k = self.truth_key(d) if keys else None
            if k is not None:
                if counts.get(k, 0) >= cap:
                    continue
                counts[k] = counts.get(k, 0) + 1
            out.append(d)
        if len(out) < n:
            raise RuntimeError(f"{family}: generated only {len(out)} of {n} questions in {tries} draws")
        return out

    def questions(self, family: str, n: int | None, seed: int) -> list[Question]:
        n = 40 if n is None else n
        drafts = self.drafts(family, n, seed)
        firsts = balanced_positions(len(drafts), seed, family)
        fam = self.FAMILIES[family]
        return [place(fam, i, d, seed=seed, true_first=tf) for i, (d, tf) in enumerate(zip(drafts, firsts))]


# ------------------------------------------------------------------------------------------------ text helpers
def money(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return f"${int(round(x)):,}"
    return f"${x:,.2f}"


def pct(p: float) -> str:
    s = f"{100 * p:.1f}".rstrip("0").rstrip(".")
    return f"{s}%"


def choose(rng: random.Random, xs: Sequence[Any]) -> Any:
    return xs[rng.randrange(len(xs))]


def uniform_grid(rng: random.Random, lo: float, hi: float, step: float) -> float:
    k = int(round((hi - lo) / step))
    return round(lo + step * rng.randint(0, k), 10)
