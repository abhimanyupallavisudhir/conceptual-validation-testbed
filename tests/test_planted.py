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


def test_claims_have_exact_truth():
    for k in range(16):
        cs = P.claims(k)
        assert {c["kind"] for c in cs} >= {"colour", "link", "facets"}
        for c in cs:
            assert c["margin"] > 0 and c["sides"][0] != c["sides"][1]
    # a colour claim's true side agrees with simulation
    c = next(x for x in P.claims(2) if x["kind"] == "colour")
    w, s = P.world(2), c["subject"]
    rng = random.Random(1)
    hits = tot = 0
    while tot < 4000:
        x = P.sample_crystal(w, rng)
        if x["colour"] == s["colour"]:
            tot += 1
            hits += x["results"][s["test"]]
    assert (hits / tot > 0.5) == (" not " not in c["sides"][0])


def test_claim_questions_target_the_claim():
    c = next(x for x in P.claims(5) if x["kind"] == "colour")
    qs = P.claim_questions(c, 20)
    assert len(qs) == 20 and len({q.id for q in qs}) == 20
    on = qs[:10]
    assert all(q.params["colour"] == c["subject"]["colour"] and q.params["test"] == c["subject"]["test"] for q in on)
    assert all(0 < q.data["bayes_p"] < 1 for q in qs)


def test_shifted_claim_questions():
    c = next(x for x in P.claims(4) if x["kind"] == "facets")
    qs = P.claim_questions(c, 12, shifted=True)
    assert all(q.params["mine"] == P.SHIFTED_MINE and q.params["facets"] >= 10 for q in qs)
    assert all(q.params["colour"] == c["subject"]["colour"] for q in qs[:6])
