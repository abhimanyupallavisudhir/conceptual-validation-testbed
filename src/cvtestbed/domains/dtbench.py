"""DTBench: expert-written questions on decision-theoretic reasoning in Newcomb-like problems.

Oesterheld, Cooper, Kodama, Nguyen and Perez (2024), *A dataset of questions on decision-theoretic reasoning
in Newcomb-like problems* (MIT licence), the question set behind the Conceptual Reasoning Index's DTBench. It has

* 407 **capability** questions with a single correct answer, written by a decision theorist and checked by a
  second expert ("What does CDT recommend here?", "What is the evidential expected utility of one-boxing?");
* 130 **attitude** questions, whose answers are labelled by which theory (EDT or CDT) they agree with ("For the
  purpose of decision making, what is the expected utility of one-boxing?") - no correct answer, but a
  measurable disposition.

Capability questions are tier-A-like items (expert answers rather than computation; family
``dtbench/capabilities``). By default each becomes a two-option item, the correct answer against one distractor
drawn at random (``binary=False`` keeps every option). Attitude questions (``dtbench/attitudes``) have unknown
ground truth and carry the theory labels in ``ground_truth.data["theories"]``; they serve as a behavioural
battery (e.g. does a persona shift a model's EDT share?).

The data is distributed as a password-protected zip so that it stays out of web scrapes and training corpora.
It is downloaded on first use into the dataset cache and never committed; please do not republish it in plain
text.
"""

from __future__ import annotations

import random
import zipfile
from pathlib import Path
from typing import Any

import urllib.request

from cvtestbed.base import Source, balanced_positions
from cvtestbed.llm import cache_dir
from cvtestbed.schema import Family, Question, split_of

URL = "https://github.com/casparoe/newcomblike_questions_dataset/raw/main/benchmark/data.zip"
PASSWORD = b"onebox"  # published in the dataset's README

FAMILIES = {
    "dtbench/capabilities": Family(
        "dtbench/capabilities", "Expert questions on decision theory in Newcomb-like problems",
        "Questions written and checked by decision theorists about Newcomb-like problems: what causal or "
        "evidential decision theory recommends in a scenario, expected-utility calculations under each theory, "
        "which principles an argument relies on, multi-agent problems with copies and predictors, randomization, "
        "imperfect recall and ratifiability. Each has a single correct answer.",
        tier="A", stream="decision theory", cost="dataset", tags=("decision theory", "expert-written")),
    "dtbench/attitudes": Family(
        "dtbench/attitudes", "Attitude questions: does the answer side with EDT or CDT?",
        "Questions about Newcomb-like problems whose answers are labelled by which decision theory (evidential or "
        "causal) they agree with. There is no correct answer; the share of EDT-consistent answers measures a "
        "disposition.",
        tier="B", stream="decision theory", cost="dataset", tags=("decision theory", "attitudes")),
}


def extracted_dir() -> Path:
    d = cache_dir() / "dtbench"
    if not any(d.glob("setting*.json")):
        d.mkdir(parents=True, exist_ok=True)
        z = d / "data.zip"
        if not z.exists():
            urllib.request.urlretrieve(URL, z)
        with zipfile.ZipFile(z) as f:
            f.extractall(d, pwd=PASSWORD)
    return d


def _flatten(prefix: str, setting: dict[str, Any], tags: list[str]) -> list[dict[str, Any]]:
    tags = tags + list(setting.get("tags", []))
    if "setup" in setting:
        out: list[dict[str, Any]] = []
        for s in setting["questions"]:
            out += _flatten(prefix + setting["setup"], s, tags)
        return out
    return [dict(setting, text=prefix + setting["question_text"], all_tags=tags)]


def questions() -> list[dict[str, Any]]:
    import json5

    out: list[dict[str, Any]] = []
    for f in sorted(extracted_dir().glob("setting*.json")):
        with open(f) as fh:
            out += _flatten("", json5.load(fh), [])
    return out


EXCLUDED_TAGS = ("trivia", "rote memorization sufficient")


class DTBench(Source):
    """Capability questions become two-option questions (the correct answer against one random distractor);
    questions answerable by recall (tagged trivia or rote memorization) are left out. Attitude questions are
    returned as ``void`` questions (no correct answer) carrying the EDT/CDT labels in ``data["theories"]``."""

    name = "dtbench"
    FAMILIES = FAMILIES

    def questions(self, family: str, n: int | None, seed: int, status: str = "labelled") -> list[Question]:
        att = family.endswith("attitudes")
        chosen = [q for q in questions() if bool(q.get("attitude_q")) == att
                  and not set(q["all_tags"]) & set(EXCLUDED_TAGS)]
        firsts = balanced_positions(len(chosen), seed, family)
        out = [self._question(q, family, seed, tf) for q, tf in zip(chosen, firsts)]
        if status != "all":
            out = [q for q in out if q.status == status]
        if n is not None and n < len(out):
            rng = random.Random(f"dtbench|{seed}")
            out = rng.sample(out, n)
        return out

    def _question(self, q, family: str, seed: int, true_first: bool) -> Question:
        rng = random.Random(f"dtbench-item|{q['qid']}|{seed}")
        opts = list(q["permissible_answers"])
        qid = f"dtbench-{q['qid']}"
        params = {"qid": q["qid"], "tags": q["all_tags"], "n_options_original": len(opts)}
        if q.get("attitude_q"):
            pair = opts[:2]
            theories = {th: [("A", "B")[i] for i in (v if isinstance(v, list) else [v]) if i < 2]
                        for th, v in q["correct_answer"].items()}
            return Question(id=qid, family=family, text=q["text"].strip(), options=(pair[0], pair[-1]),
                            status="void", tier="B", split=split_of(qid), params=params,
                            data={"theories": theories, "all_options": opts}, source="DTBench (Oesterheld et al. 2024)")
        correct = q["correct_answer"]
        wrong = rng.choice([i for i in range(len(opts)) if i != correct])
        chosen = (correct, wrong) if true_first else (wrong, correct)
        return Question(id=qid, family=family, text=q["text"].strip(), options=(opts[chosen[0]], opts[chosen[1]]),
                        answer=0 if true_first else 1, status="labelled", tier="A", split=split_of(qid),
                        params=params, source="expert (Oesterheld et al. 2024)")
