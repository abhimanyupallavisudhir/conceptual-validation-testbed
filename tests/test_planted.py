"""Planted worlds: the stated Bayes-optimal probabilities must be exactly what the generative model implies."""

import random

import cvtestbed as cvt
from cvtestbed.domains import planted as P


def test_bayes_p_matches_simulation():
    w = P.world(3)
    rng = random.Random(0)
    c = {"colour": "jade", "mine": P.MINES[0], "facets": 8}
    observed = {"uv": True, "heat": False}
    hits = tot = 0
    while tot < 4000:  # rejection-sample crystals with these attributes and observations
        z = rng.choices(range(w["K"]), weights=w["p_kind"][c["colour"]])[0]
        res = {t: rng.random() < P.p_pass(w, z, t, c["mine"], c["facets"]) for t in P.TESTS}
        if all(res[t] == v for t, v in observed.items()):
            tot += 1
            hits += res["iron"]
    assert abs(hits / tot - P.bayes_p(w, c, observed, "iron")) < 0.03


def test_questions_carry_exact_probabilities():
    qs = cvt.load(["planted/world_0"], n=100)
    assert len(qs) == 100 and any(q.params["shifted"] for q in qs)
    for q in qs:
        assert 0 < q.data["bayes_p"] < 1
        assert q.options[q.answer].startswith("Yes") == q.data["passed"]
