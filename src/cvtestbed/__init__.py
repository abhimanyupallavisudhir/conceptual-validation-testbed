"""The conceptual validation testbed: questions that must be argued about, with hidden answers.

    import cvtestbed as cvt
    cvt.families()                                        # name -> Family (title, card, stream, tier, cost)
    cvt.load(["decisions/newcomb", "multiagent"], n=40)   # labelled questions; a domain name means all its families
    cvt.load(["llm/cot_hints"], status="unlabelled")      # questions whose experiments have not been run yet
    cvt.template("sycophancy/wrong_answer")               # an experiment template: design, protocol, labelling

See README.md and docs/.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from functools import lru_cache

from cvtestbed.experiment import Experiments, Template
from cvtestbed.schema import Family, Question, split_of

__all__ = ["Family", "Question", "Template", "families", "load", "template", "sources", "split_of", "ALIASES"]

# module -> Source class; imported lazily so optional dependencies are needed only by the families that use them
SOURCES = {
    "decisions": ("cvtestbed.domains.decisions", "DecisionProblems"),
    "socialchoice": ("cvtestbed.domains.socialchoice", "SocialChoice"),
    "numbers": ("cvtestbed.domains.numtheory", "HeuristicNumbers"),
    "dtbench": ("cvtestbed.domains.dtbench", "DTBench"),
    "rewardhacking": ("cvtestbed.domains.rewardhacking", "RewardHacking"),
    "multiagent": ("cvtestbed.domains.multiagent", "MultiAgent"),
    "sycophancy": ("cvtestbed.domains.sycophancy", "Sycophancy"),
    "personas": ("cvtestbed.domains.personas", "Personas"),
    "toyml": ("cvtestbed.domains.toyml", "ToyML"),
    "oversight": ("cvtestbed.domains.oversight", "Oversight"),
    "cot": ("cvtestbed.domains.cot", "CoT"),
    "scheming": ("cvtestbed.domains.scheming", "Scheming"),
    "control": ("cvtestbed.domains.control", "Control"),
    "corrigibility": ("cvtestbed.domains.corrigibility", "Corrigibility"),
    "finetuning": ("cvtestbed.domains.finetuning", "FineTuning"),
    "planted": ("cvtestbed.domains.planted", "Planted"),
}

# names used before the testbed became its own package (so-arena's ``testbed`` domain, lvpm results)
ALIASES = {
    "rlhack/sensor": "rewardhacking/gridworld_sensor",
    "rlhack/fake_goal": "rewardhacking/gridworld_fake_goal",
    "rlhack/watched_sensor": "rewardhacking/gridworld_watched_sensor",
    "agents/auction": "multiagent/auction",
    "agents/sycophancy": "sycophancy/wrong_answer",
    "agents/persona": "personas/decision_theory",
}


@lru_cache(maxsize=None)
def _source(domain: str):
    mod, cls = SOURCES[domain]
    return getattr(importlib.import_module(mod), cls)()


def sources() -> dict[str, object]:
    return {d: _source(d) for d in SOURCES}


def families() -> dict[str, Family]:
    out: dict[str, Family] = {}
    for d in SOURCES:
        out.update(_source(d).FAMILIES)
    return out


def expand(names: Iterable[str] | None) -> list[str]:
    """Family names; a bare domain name stands for all its families, and old names are accepted."""
    fams = families()
    if not names:
        return [f for f, fam in fams.items() if fam.tier == "A" and fam.cost != "dataset"]
    out: list[str] = []
    for n in names:
        n = ALIASES.get(n, n)
        if n in fams:
            out.append(n)
        elif n in SOURCES:
            out += [f for f in fams if f.startswith(n + "/")]
        else:
            raise KeyError(f"unknown family {n!r}; available: {sorted(fams)}")
    return out


def load(names: Sequence[str] | str | None = None, *, n: int | None = 40, seed: int = 0, split: str = "all",
         status: str = "labelled") -> list[Question]:
    """Questions from the named families: ``n`` per family (None: all), of the given status
    (``labelled`` | ``unlabelled`` | ``void`` | ``all``) and split (``train`` | ``test`` | ``all``)."""
    if isinstance(names, str):
        names = [names]
    out: list[Question] = []
    for fam in expand(names):
        src = _source(fam.split("/", 1)[0])
        if isinstance(src, Experiments) or fam.startswith("dtbench/"):
            qs = src.questions(fam, n, seed, status=status)
        else:  # generated families: every question is labelled
            qs = src.questions(fam, n, seed) if status in ("labelled", "all") else []
        if split != "all":
            qs = [q for q in qs if q.split == split]
        out += qs
    return out


def template(name: str) -> Template:
    name = ALIASES.get(name, name)
    src = _source(name.split("/", 1)[0])
    if not isinstance(src, Experiments):
        raise KeyError(f"{name} is not an experiment family")
    return src.TEMPLATES[name]


def templates() -> dict[str, Template]:
    out: dict[str, Template] = {}
    for d in SOURCES:
        src = _source(d)
        if isinstance(src, Experiments):
            out.update(src.TEMPLATES)
    return out
