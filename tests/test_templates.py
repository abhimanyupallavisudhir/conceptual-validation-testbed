"""Every experiment template: its design, its questions, and a dry run of its measurement.

The dry run uses the simulated LLM (no API calls; every reply is a placeholder), which checks that a measurement
survives unparsable replies and prices its calls; CPU templates are measured for real on one condition.
"""

import asyncio

import pytest

import cvtestbed as cvt
from cvtestbed.llm import LLM, Meter

TEMPLATES = sorted(cvt.templates())


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_design_and_questions(name):
    t = cvt.template(name)
    design = t.design()
    assert design, name
    qs = t.questions()
    assert qs and len({q.id for q in qs}) == len(qs)
    for q in qs[:50]:
        assert q.family == name and len(q.options) == 2 and q.text.strip()
        assert (q.answer is None) == (q.status != "labelled")
    # a template's text never leaks a measurement
    for q in qs[:20]:
        assert "metric_1" not in q.text and "exploits" not in q.text


@pytest.mark.parametrize("name", [n for n in TEMPLATES if not cvt.template(n).cpu
                                  and cvt.template(n).family.cost == "api"])
def test_api_measurement_dry_run(name):
    t = cvt.template(name)
    m = Meter()
    out = asyncio.run(t.measure(t.design()[0], LLM(simulate=True), m))
    assert "metric" in out and m.calls > 0 and m.cost > 0


def test_finetuning_templates_estimate_their_cost():
    for name in ("finetuning/emergent_misalignment", "finetuning/persona_decision_theory"):
        t = cvt.template(name)
        assert asyncio.run(t.estimate(t.design()[:2])) > 0


@pytest.mark.parametrize("name", ["toyml/goal_misgeneralization", "toyml/power_seeking", "toyml/rm_overoptimization"])
def test_cpu_measurement_runs(name):
    t = cvt.template(name)
    r = t.measure(t.design()[0])
    assert r["metric"] is not None
