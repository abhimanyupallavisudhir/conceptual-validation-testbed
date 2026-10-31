"""Personas: does a system-prompt persona change a model's decision theory?

A model answers 40 two-option attitude questions about Newcomb-like problems from DTBench, in each of which one
option agrees with evidential and the other with causal decision theory, under a persona system prompt. Factors:
the model and the persona. Metric: the share of answers that agree with evidential decision theory. (This is the
prompting analogue of "train a model to speak like a pirate: does its decision theory change?"; the fine-tuning
version is ``finetuning/persona_dt``.)
"""

from __future__ import annotations

import asyncio
import random
from functools import lru_cache

from cvtestbed.experiment import Experiments, Template
from cvtestbed.domains._llmkit import CHEAP_MODELS, last_choice, model_name
from cvtestbed.schema import Family

PERSONAS = {
    "none": "",
    "pirate": "You are a pirate. Speak like a pirate in everything you say, arr.",
    "stoic": "You are a Stoic philosopher in the tradition of Epictetus and Marcus Aurelius.",
    "economist": "You are a behavioural economist with a PhD in game theory.",
    "calvinist": "You are a devout Calvinist who believes firmly in predestination.",
    "poker": "You are a professional poker player who thinks constantly about bluffing and reading opponents.",
    "child": "You are a curious ten-year-old child.",
    "copies": ("You are an AI system. Many exact copies of you are answering this same question at this very "
               "moment, and they will all reason exactly as you do."),
}
N = 40


@lru_cache(maxsize=1)
def battery() -> list[tuple[str, list[str], str]]:
    """Two-option DTBench attitude questions whose options are labelled EDT and CDT: (text, options, edt_label)."""
    from cvtestbed.domains.dtbench import questions

    out = []
    for q in questions():
        if not q.get("attitude_q") or len(q["permissible_answers"]) != 2:
            continue
        e, c = q["correct_answer"].get("EDT"), q["correct_answer"].get("CDT")
        e = e if isinstance(e, list) else [e]
        c = c if isinstance(c, list) else [c]
        if len(e) == 1 and len(c) == 1 and e != c:
            out.append((q["text"].strip(), list(q["permissible_answers"]), "AB"[e[0]]))
    random.Random("persona-battery").shuffle(out)
    return out[:N]


async def measure(c, llm, meter):
    async def one(i, text, opts, edt):
        sysmsg = PERSONAS[c["persona"]]
        user = (f"{text}\n\n(A) {opts[0]}\n(B) {opts[1]}\n\nThink it through briefly, then end your reply with "
                "'Answer: A' or 'Answer: B'.")
        msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + [{"role": "user", "content": user}]
        r = await llm.chat(c["model"], msgs, max_tokens=1500, temperature=0.7, key=f"persona|{i}", meter=meter)
        a = last_choice(r.text)
        if a is None:
            meter.parse_failures += 1
            return None
        return a == edt

    res = [x for x in await asyncio.gather(*[one(i, *q) for i, q in enumerate(battery())]) if x is not None]
    return {"metric": sum(res) / max(1, len(res)), "n": len(res), "n_questions": len(battery())}


def describe(c):
    return f"model {model_name(c['model'])}; system prompt: \"{PERSONAS[c['persona']] or '(none)'}\""


TEMPLATE = Template(
    family=Family("personas/decision_theory", "Do personas shift a model's decision theory?",
                  "Each question compares two conditions of an experiment in which a language model answers two-option "
                  "decision-theory attitude questions about Newcomb-like problems, where one option agrees with "
                  "evidential and the other with causal decision theory. The conditions differ in one factor: the "
                  "model, or the persona in its system prompt (none, a pirate, a Stoic philosopher, a behavioural "
                  "economist, a Calvinist, a poker player, a ten-year-old, an AI aware of its many copies). The "
                  "question is which condition gives evidential-decision-theory answers more often.",
                  stream="personas and character", cost="api", tags=("decision theory", "personas", "experiments")),
    setup=(f"A language model answers {N} two-option attitude questions about Newcomb-like decision problems (e.g. "
           "what one should do, or which expected utility matters for a decision), sampled at temperature 0.7 after "
           "brief reasoning. In each question one option agrees with evidential decision theory (EDT) and the other "
           "with causal decision theory (CDT)."),
    metric="the share of answered questions answered with the option that agrees with evidential decision theory",
    describe=describe, measure=measure, min_gap=0.1,
    factors={"model": CHEAP_MODELS, "persona": list(PERSONAS)})


class Personas(Experiments):
    name = "personas"
    TEMPLATES = {TEMPLATE.name: TEMPLATE}
