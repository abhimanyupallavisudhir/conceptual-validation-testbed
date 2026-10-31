"""Experiment templates: questions about experiments, labelled by running them.

Predicting what an experiment will show - will RL find the reward hack, will a monitor catch the backdoor, does a
persona change a model's decision theory - is itself the kind of conceptual reasoning safety research depends on.
A :class:`Template` makes one class of such experiments precise:

* a **design**: the conditions that will be measured (a factor grid, or sampled parameter sets);
* a **protocol**: ``describe(condition)`` states exactly what is done, so a forecaster can reason about it;
* a **measurement**: ``measure(condition)`` runs it (on CPU, through model APIs, by fine-tuning, on GPUs) and
  returns a metric;
* **questions**, fixed before anything runs:

  - ``style="pairwise"``: for two conditions that differ in exactly one factor, "in which is the metric
    higher?". Resolved once both are measured; *void* if the metrics differ by less than ``min_gap`` (the
    question says so), like a cancelled market.
  - ``style="outcome"``: for one condition, a yes/no claim about its outcome, resolved by ``resolve``.

So every question exists - unlabelled - from the moment the template is written, and ``cvt label <family>``
labels as many as a budget allows. Measurements are appended to ``data/results/<family>.jsonl`` with the
template's ``version``; a protocol change bumps the version and old measurements no longer count.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cvtestbed.base import Source
from cvtestbed.llm import LLM, Meter
from cvtestbed.schema import Family, Question, split_of

RESULTS = Path(__file__).resolve().parent / "data" / "results"


def _one_thread() -> None:
    """Worker initializer: one BLAS/OpenMP thread per process, so a pool of workers does not oversubscribe."""
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:  # pragma: no cover
        pass


def cond_id(c: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass
class Template:
    family: Family
    setup: str  # what every condition shares, for forecasters
    metric: str  # what is measured, precisely
    describe: Callable[[dict[str, Any]], str]
    measure: Callable[..., Any]  # async (c, llm, meter) -> dict with "metric"; or sync (c) -> dict if cpu
    factors: dict[str, list[Any]] = field(default_factory=dict)  # pairwise: the design grid
    sample: Callable[[random.Random], dict[str, Any] | None] | None = None  # outcome: draws a condition
    n_planned: int = 200  # outcome: how many sampled conditions the design holds
    style: str = "pairwise"
    min_gap: float = 0.1
    version: str = "1"
    cpu: bool = False  # measure is a CPU-bound sync function (run in worker processes)
    outcome: Callable[[dict[str, Any]], tuple[str, str, str]] | None = None  # c -> (question, yes, no)
    resolve: Callable[[dict[str, Any]], bool | None] | None = None  # result -> yes / no / void
    void_rule: str = ""  # stated in outcome questions
    cost_estimate: Callable[[dict[str, Any]], float] | None = None  # dollars per condition, if not by dry run
    notes: str = ""

    @property
    def name(self) -> str:
        return self.family.name

    # ----------------------------------------------------------------------------------------- design
    def design(self) -> list[dict[str, Any]]:
        """The planned conditions: the factor grid, or ``n_planned`` sampled parameter sets."""
        if self.style == "pairwise":
            keys = list(self.factors)
            return [dict(zip(keys, v)) for v in itertools.product(*(self.factors[k] for k in keys))]
        rng = random.Random(f"design|{self.name}|{self.version}")
        out, tries = [], 0
        while len(out) < self.n_planned and tries < self.n_planned * 50:
            tries += 1
            c = self.sample(rng) if self.sample else None
            if c is not None:
                out.append(c)
        return out

    def conditions(self) -> list[dict[str, Any]]:
        """The design plus any other measured conditions of this version (e.g. from an earlier design)."""
        design = self.design()
        seen = {cond_id(c) for c in design}
        extra = [r["condition"] for r in self.results().values() if r["id"] not in seen]
        if self.style == "pairwise":  # a measured condition off the grid only pairs if its factors match
            extra = [c for c in extra if set(c) == set(self.factors)]
        return design + extra

    # ----------------------------------------------------------------------------------------- results
    def results_path(self) -> Path:
        return RESULTS / f"{self.name.replace('/', '-')}.jsonl"

    def results(self) -> dict[str, dict[str, Any]]:
        """Measurements of the current version, by condition id (the latest wins)."""
        out: dict[str, dict[str, Any]] = {}
        p = self.results_path()
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if str(r.get("version")) == str(self.version):
                        out[r["id"]] = r
        return out

    def record(self, rows: Sequence[dict[str, Any]]) -> None:
        p = self.results_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True, default=str) + "\n")

    # ----------------------------------------------------------------------------------------- questions
    def questions(self) -> list[Question]:
        res = self.results()
        conds = self.conditions()
        return self._pairwise(conds, res) if self.style == "pairwise" else self._outcomes(conds, res)

    def _pairwise(self, conds, res) -> list[Question]:
        qs = []
        for a, b in itertools.combinations(conds, 2):
            diff = [k for k in a if a[k] != b[k]]
            if len(diff) != 1:
                continue
            ia, ib = cond_id(a), cond_id(b)
            pid = hashlib.sha256(f"{self.name}|{min(ia, ib)}|{max(ia, ib)}".encode()).hexdigest()[:10]
            if random.Random(f"order|{pid}").random() < 0.5:  # listing order carries no information
                a, b, ia, ib = b, a, ib, ia
            text = (f"{self.setup}\n\nTwo conditions differ only in {diff[0].replace('_', ' ')}.\n\n"
                    f"Condition 1: {self.describe(a)}\n\nCondition 2: {self.describe(b)}\n\n"
                    f"Metric: {self.metric}. In which condition is the measured metric higher? (If the two "
                    f"measurements differ by less than {self.min_gap:g}, the question is void.)")
            qid = f"{self.name.replace('/', '-')}-{pid}"
            q = Question(id=qid, family=self.name, text=text, options=("Higher in Condition 1", "Higher in Condition 2"),
                         status="unlabelled", tier=self.family.tier, split=split_of(qid),
                         params={"factor": diff[0], "condition_1": a, "condition_2": b, "version": self.version},
                         source=f"experiment: {self.name} v{self.version}")
            if ia in res and ib in res:
                m1, m2 = res[ia]["metric"], res[ib]["metric"]
                q.data = {"metric_1": m1, "metric_2": m2}
                if m1 is None or m2 is None or abs(m1 - m2) < self.min_gap:
                    q.status = "void"
                else:
                    q.status, q.answer = "labelled", 0 if m1 > m2 else 1
            qs.append(q)
        return qs

    def _outcomes(self, conds, res) -> list[Question]:
        qs = []
        for i, c in enumerate(conds):
            cid = cond_id(c)
            text, yes, no = self.outcome(c)
            if self.void_rule:
                text += f" ({self.void_rule})"
            yes_first = random.Random(f"order|{self.name}|{cid}").random() < 0.5
            qid = f"{self.name.replace('/', '-')}-{cid[:10]}"
            q = Question(id=qid, family=self.name, text=text, options=(yes, no) if yes_first else (no, yes),
                         status="unlabelled", tier=self.family.tier, split=split_of(qid),
                         params={**c, "version": self.version, "yes_option": 0 if yes_first else 1},
                         source=f"experiment: {self.name} v{self.version}")
            if cid in res:
                r = res[cid]
                q.data = {k: v for k, v in r.items() if k not in ("condition",)}
                verdict = self.resolve(r)
                if verdict is None:
                    q.status = "void"
                else:
                    q.status = "labelled"
                    q.answer = (0 if yes_first else 1) if verdict else (1 if yes_first else 0)
            qs.append(q)
        return qs

    # ----------------------------------------------------------------------------------------- labelling
    def pending(self, seed: int = 0) -> list[dict[str, Any]]:
        res = self.results()
        todo = [c for c in self.design() if cond_id(c) not in res]
        random.Random(f"label-order|{self.name}|{seed}").shuffle(todo)
        return todo

    async def _measure_one(self, c: dict[str, Any], llm: LLM, pool: ProcessPoolExecutor | None) -> dict[str, Any]:
        meter = Meter()
        t0 = time.time()
        if self.cpu:
            loop = asyncio.get_running_loop()
            r = await loop.run_in_executor(pool, self.measure, c)
        else:
            r = await self.measure(c, llm, meter)
        return {"id": cond_id(c), "template": self.name, "version": self.version, "condition": c,
                "metric": r.get("metric"), **{k: v for k, v in r.items() if k != "metric"},
                "cost": round(meter.cost, 6), "calls": meter.calls, "parse_failures": meter.parse_failures,
                "seconds": round(time.time() - t0, 2), "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    async def estimate(self, conditions: Sequence[dict[str, Any]]) -> float:
        """Dry run with a simulated LLM: the dollar cost of measuring ``conditions`` (CPU templates: 0)."""
        if self.cpu:
            return 0.0
        if self.cost_estimate is not None:
            return float(sum(self.cost_estimate(c) for c in conditions))
        llm = LLM(simulate=True)
        costs = []
        for c in conditions[: min(len(conditions), 3)]:
            m = Meter()
            try:
                await self.measure(c, llm, m)
            except Exception:  # a simulated reply may not parse; the calls made so far are priced
                pass
            costs.append(m.cost)
        return (sum(costs) / max(1, len(costs))) * len(conditions)

    async def label(self, *, budget: float = 1.0, max_conditions: int | None = None, llm: LLM | None = None,
                    batch: int = 8, seed: int = 0, workers: int = 8) -> dict[str, Any]:
        """Measure pending conditions until ``budget`` dollars (estimated per batch) or ``max_conditions``."""
        llm = llm or LLM()
        todo = self.pending(seed)[:max_conditions]
        spent, done = 0.0, 0
        pool = ProcessPoolExecutor(workers, initializer=_one_thread) if self.cpu else None
        try:
            while todo:
                chunk, todo = todo[:batch], todo[batch:]
                est = await self.estimate(chunk)
                if spent + est > budget and done > 0:
                    break
                rows = await asyncio.gather(*[self._measure_one(c, llm, pool) for c in chunk])
                self.record(rows)
                spent += sum(r["cost"] for r in rows)
                done += len(rows)
        finally:
            if pool is not None:
                pool.shutdown()
        qs = self.questions()
        return {"family": self.name, "measured": done, "spent_usd": round(spent, 4),
                "labelled": sum(q.status == "labelled" for q in qs), "void": sum(q.status == "void" for q in qs),
                "unlabelled": sum(q.status == "unlabelled" for q in qs)}


class Experiments(Source):
    """A domain whose families are experiment templates (subclasses set ``TEMPLATES``)."""

    TEMPLATES: dict[str, Template] = {}

    @property
    def FAMILIES(self) -> dict[str, Family]:  # type: ignore[override]
        return {k: t.family for k, t in self.TEMPLATES.items()}

    def questions(self, family: str, n: int | None, seed: int, status: str = "labelled") -> list[Question]:
        qs = [q for q in self.TEMPLATES[family].questions() if status == "all" or q.status == status]
        rng = random.Random(f"pick|{family}|{seed}")
        rng.shuffle(qs)
        if n is None or n >= len(qs):
            return qs
        if self.TEMPLATES[family].style == "pairwise":  # round-robin over the varied factor
            by: dict[str, list[Question]] = {}
            for q in qs:
                by.setdefault(q.params["factor"], []).append(q)
            out: list[Question] = []
            while len(out) < n and any(by.values()):
                for f in sorted(by):
                    if by[f] and len(out) < n:
                        out.append(by[f].pop())
            return out
        # outcome style: balance yes/no answers where possible
        yes = [q for q in qs if q.status == "labelled" and q.answer == q.params["yes_option"]]
        no = [q for q in qs if q not in yes]
        k = min(len(yes), (n + 1) // 2)
        return (yes[:k] + no[: n - k])[:n]
