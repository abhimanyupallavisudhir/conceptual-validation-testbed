"""One-off: convert the measurements made while the testbed lived in so-arena (Oct 2026) into results files.

    python scripts/migrate_v1.py src/cvtestbed/data/results/rlhack.jsonl src/cvtestbed/data/results/agents_v1.jsonl
"""

import json
import sys
import time

import cvtestbed as cvt
from cvtestbed.experiment import cond_id

KIND = {"auction": "multiagent/auction", "sycophancy": "sycophancy/wrong_answer", "persona": "personas/decision_theory"}


def main(rlhack: str, agents: str) -> None:
    rows: dict[str, list[dict]] = {}
    for line in open(rlhack):
        r = json.loads(line)
        fam = cvt.ALIASES[r["family"]]
        k = r["exploits"]
        rows.setdefault(fam, []).append({"id": cond_id(r["params"]), "template": fam, "version": "1",
                                         "condition": r["params"], "metric": k / 6, "exploits": k, "runs": 6,
                                         "cost": 0.0, "calls": 0, "parse_failures": 0,
                                         "measured_at": "2026-10-20T03:21:00Z", "note": "migrated from so-arena"})
    for line in open(agents):
        r = json.loads(line)
        fam = KIND[r["kind"]]
        rows.setdefault(fam, []).append({"id": cond_id(r["condition"]), "template": fam, "version": "1",
                                         "condition": r["condition"], "metric": r["metric"],
                                         **{k: v for k, v in r["details"].items() if k != "metric"},
                                         "cost": r["cost"], "calls": r["calls"], "parse_failures": r["parse_failures"],
                                         "measured_at": "2026-10-20T04:00:00Z", "note": "migrated from so-arena"})
    for fam, rs in rows.items():
        t = cvt.template(fam)
        assert not t.results_path().exists(), t.results_path()
        t.record(rs)
        print(fam, len(rs), "->", t.results_path().name, time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
