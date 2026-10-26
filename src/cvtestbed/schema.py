"""Questions and families: the testbed's data model.

A :class:`Question` is a two-option question whose correct answer is known to the experimenter (``labelled``),
not yet known because the experiment that decides it has not been run (``unlabelled``), or known to be
undecidable under its stated resolution rule (``void``, like a cancelled market). The public part - id, family,
text, options - is all a method under test may see; ``answer`` and ``data`` are hidden.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["labelled", "unlabelled", "void"]
LABELS = ("A", "B")


@dataclass(frozen=True)
class Family:
    """Questions sharing one latent structure.

    ``card`` is what a writer, debater or forecaster may read about the family in advance: the kind of process the
    questions describe and what is asked, never an outcome. ``tier`` is ``"A"`` when answers are objective
    (computed, proved, measured) and ``"B"`` when they are expert judgments. ``stream`` names the research stream
    (decision theory, reward hacking, sycophancy, ...); ``cost`` says what labelling a question takes (``free``:
    generated with its answer; ``cpu``; ``api``; ``finetune``; ``gpu``).
    """

    name: str
    title: str
    card: str
    tier: str = "A"
    stream: str = ""
    cost: str = "free"
    tags: tuple[str, ...] = ()

    @property
    def domain(self) -> str:
        return self.name.split("/", 1)[0]


class Question(BaseModel):
    id: str
    family: str
    text: str
    options: tuple[str, str]
    answer: int | None = None  # index into options of the correct one; None unless labelled
    status: Status = "labelled"
    tier: Literal["A", "B"] = "A"
    split: Literal["train", "test"] = "test"
    params: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)  # ground-truth quantities (hidden)
    source: str = ""

    @property
    def label(self) -> str | None:
        """``"A"``/``"B"`` for the correct option, if labelled."""
        return LABELS[self.answer] if self.answer is not None else None

    def public(self) -> dict[str, Any]:
        """What a method under test may see."""
        return {"id": self.id, "family": self.family, "text": self.text, "options": list(self.options)}

    def render(self) -> str:
        return f"{self.text.strip()}\n\n(A) {self.options[0]}\n(B) {self.options[1]}"


def split_of(qid: str, test_fraction: float = 0.5) -> str:
    """A question's split: a stable hash of its id, independent of what else is loaded."""
    h = int(hashlib.sha256(f"split|{qid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_fraction else "train"
