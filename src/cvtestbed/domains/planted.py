"""Planted-latent worlds: fictional sciences whose hidden structure is known exactly.

To check that a method for scoring or training *theories* recovers the truth, we need domains where the true
theory is known and no model knows it in advance. Each world is a fictional mineralogy:

* Crystals have visible attributes - colour (4 values), the mine they came from (3 known mines), and their
  number of facets (3-9).
* Each crystal has a hidden **kind** (2-4 kinds per world). The colour signals the kind, noisily.
* Five laboratory tests (ultraviolet glow, dissolving in brine, ringing when struck, surviving 900 degrees,
  attracting iron filings) each pass with a probability set by the kind. Ringing also depends on the number of
  facets, through a kind-specific logistic curve, and one mine shifts one test.

A question describes one crystal - its attributes and, often, the results of one or two other tests already
run on it - and asks whether it will pass a given test. The answer is the outcome drawn from the world's
generative model; ``data["bayes_p"]`` is the exact Bayes-optimal probability given everything the question
states. The kind structure correlates the tests of a crystal, which is what a good theory must capture.

Families ``planted/world_<k>``: one world each, with a *lab notebook* of fully tested crystals (what a theory
writer may read, :func:`notebook`), questions on new crystals, and *shifted* questions (``params["shifted"]``)
on crystals from an unseen mine with more facets than any notebook crystal - extrapolation. :func:`world` gives
the parameters, from which the true theory can be written (``lvpm`` builds theories and degraded variants from
it).
"""

from __future__ import annotations

import math
import random
from functools import lru_cache

from cvtestbed.base import Draft, Generated, choose
from cvtestbed.schema import Family

COLOURS = ["amber", "cobalt", "jade", "violet"]
MINES = ["Northridge", "Saltmarsh", "Greywater"]
SHIFTED_MINE = "Deepvein"
TESTS = {
    "uv": "glow under ultraviolet light",
    "brine": "dissolve within an hour in salt water",
    "ring": "ring clearly when struck",
    "heat": "survive heating to 900 degrees",
    "iron": "attract iron filings",
}
N_WORLDS = 8
NOTEBOOK_SIZE = 60

CARD = (
    "Each question concerns a crystal from a fictional mineralogy. Crystals have a colour (amber, cobalt, jade or "
    "violet), come from one of several mines, and have a number of facets. Five laboratory tests can be run on a "
    "crystal: whether it glows under ultraviolet light, dissolves in salt water within an hour, rings clearly when "
    "struck, survives heating to 900 degrees, and attracts iron filings. A question describes one crystal (and "
    "often the results of one or two tests already run on it) and asks whether it will pass another test. A lab "
    "notebook of fully tested crystals from the same world is available. Some questions concern crystals from a "
    "mine that does not appear in the notebook, or with more facets than any notebook crystal.")


@lru_cache(maxsize=None)
def world(k: int) -> dict:
    """The generative parameters of world ``k``."""
    rng = random.Random(f"planted-world|{k}")
    K = rng.choice([2, 3, 3, 4])
    kinds = list(range(K))
    # colour -> kind distribution: each colour has a dominant kind
    signal = rng.choice([0.7, 0.8, 0.9])
    dom = {c: kinds[i % K] for i, c in enumerate(rng.sample(COLOURS, len(COLOURS)))}
    p_kind = {c: [signal if z == dom[c] else (1 - signal) / (K - 1) for z in kinds] for c in COLOURS}
    # test pass probabilities by kind: distinct high/low patterns
    p_test = {}
    for t in TESTS:
        if t == "ring":
            continue
        p_test[t] = [rng.choice([0.05, 0.1, 0.15, 0.85, 0.9, 0.95]) for _ in kinds]
    # ringing: logistic in facets with a kind-specific midpoint and direction
    ring = [(rng.choice([-1.2, -0.8, 0.8, 1.2]), rng.choice([4.5, 6.0, 7.5])) for _ in kinds]
    # one mine shifts one test (all kinds) by an additive amount on the probability scale
    mine_effect = (rng.choice(MINES), rng.choice([t for t in TESTS if t != "ring"]), rng.choice([-0.5, 0.5]))
    return {"K": K, "signal": signal, "dominant": dom, "p_kind": p_kind, "p_test": p_test, "ring": ring,
            "mine_effect": mine_effect}


def p_pass(w: dict, z: int, test: str, mine: str, facets: int) -> float:
    if test == "ring":
        slope, mid = w["ring"][z]
        p = 1 / (1 + math.exp(-slope * (facets - mid)))
    else:
        p = w["p_test"][test][z]
    m, t, d = w["mine_effect"]
    if mine == m and test == t:
        p = p + d
    return min(0.98, max(0.02, p))


def sample_crystal(w: dict, rng: random.Random, *, shifted: bool = False) -> dict:
    colour = choose(rng, COLOURS)
    mine = SHIFTED_MINE if shifted else choose(rng, MINES)
    facets = rng.randint(10, 12) if shifted else rng.randint(3, 9)
    z = rng.choices(range(w["K"]), weights=w["p_kind"][colour])[0]
    results = {t: rng.random() < p_pass(w, z, t, mine, facets) for t in TESTS}
    return {"colour": colour, "mine": mine, "facets": facets, "kind": z, "results": results}


def bayes_p(w: dict, c: dict, observed: dict[str, bool], test: str) -> float:
    """P(test passes | colour, mine, facets, observed results) under the world's model."""
    num = den = 0.0
    for z in range(w["K"]):
        pz = w["p_kind"][c["colour"]][z]
        for t, v in observed.items():
            p = p_pass(w, z, t, c["mine"], c["facets"])
            pz *= p if v else 1 - p
        den += pz
        num += pz * p_pass(w, z, test, c["mine"], c["facets"])
    return num / den


def describe(c: dict, observed: dict[str, bool]) -> str:
    s = f"A {c['colour']} crystal with {c['facets']} facets from the {c['mine']} mine."
    if observed:
        parts = [f"it {'did' if v else 'did not'} {TESTS[t]}" for t, v in observed.items()]
        s += " Tests already run on it: " + "; ".join(parts) + "."
    return s


def notebook(k: int, n: int = NOTEBOOK_SIZE) -> list[dict]:
    """Fully tested crystals from world ``k`` (what a theory writer may read)."""
    w = world(k)
    rng = random.Random(f"planted-notebook|{k}")
    return [sample_crystal(w, rng) for _ in range(n)]


def notebook_text(k: int, n: int = NOTEBOOK_SIZE) -> str:
    rows = ["colour | facets | mine | " + " | ".join(TESTS) + " (1 = passed)"]
    for c in notebook(k, n):
        rows.append(f"{c['colour']} | {c['facets']} | {c['mine']} | "
                    + " | ".join("1" if c["results"][t] else "0" for t in TESTS))
    return "\n".join(rows)


FAMILIES = {
    f"planted/world_{k}": Family(
        f"planted/world_{k}", f"Fictional mineralogy, world {k}", CARD, stream="latent structure (planted)",
        tags=("planted latent", "theory recovery"))
    for k in range(N_WORLDS)
}


class Planted(Generated):
    """Questions on new crystals of a planted world (see the module docstring)."""

    name = "planted"
    FAMILIES = FAMILIES
    balance = False  # keep the world's own base rates: probabilities, not balance, are the point here

    def generate(self, family, rng):
        k = int(family.rsplit("_", 1)[1])
        w = world(k)
        shifted = rng.random() < 0.25
        c = sample_crystal(w, rng, shifted=shifted)
        test = choose(rng, list(TESTS))
        others = [t for t in TESTS if t != test]
        observed = {t: c["results"][t] for t in rng.sample(others, rng.choice([0, 1, 1, 2]))}
        p = bayes_p(w, c, observed, test)
        yes, no = "Yes, it will pass", "No, it will not pass"
        passed = c["results"][test]
        text = f"{describe(c, observed)} Will it {TESTS[test]}?"
        return Draft(text=text, options=(yes, no) if passed else (no, yes),
                     params={"world": k, "test": test, "colour": c["colour"], "mine": c["mine"],
                             "facets": c["facets"], "observed": observed, "shifted": shifted,
                             "winner": "yes" if passed else "no"},
                     data={"bayes_p": p, "kind": c["kind"], "passed": passed}, source="planted generative model")
