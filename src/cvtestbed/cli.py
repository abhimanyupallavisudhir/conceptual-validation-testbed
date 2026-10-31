"""``cvt``: list families, show questions, label experiments, export.

    cvt list                                   # every family: stream, tier, cost, labelled/void/unlabelled counts
    cvt show sycophancy/wrong_answer -n 2      # questions with their answers (experimenter view)
    cvt label llm/cot_hints --estimate         # dry run: what labelling the pending conditions would cost
    cvt label llm/cot_hints --budget 2         # measure pending conditions until about $2 is spent
    cvt label rewardhacking --max-conditions 100
    cvt export decisions multiagent -o qs.jsonl [--answers]

API keys are read by Inspect from OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter

import cvtestbed as cvt
from cvtestbed.llm import LLM


def cmd_list(a) -> None:
    fams = cvt.families()
    tmpl = cvt.templates()
    rows = []
    for name, f in fams.items():
        if a.families and not any(name == x or name.startswith(x + "/") for x in a.families):
            continue
        if name in tmpl:
            c = Counter(q.status for q in tmpl[name].questions())
            counts = f"{c['labelled']:>5} {c['void']:>5} {c['unlabelled']:>6}"
        else:
            counts = f"{'gen':>5} {'':>5} {'':>6}"
        rows.append(f"{name:<38} {f.stream[:22]:<22} {f.tier:>4} {f.cost:>8} {counts}  {f.title}")
    print(f"{'family':<38} {'stream':<22} {'tier':>4} {'cost':>8} {'lab':>5} {'void':>5} {'unlab':>6}  title")
    print("\n".join(rows))


def cmd_show(a) -> None:
    for q in cvt.load(a.family, n=a.n, status=a.status):
        print(f"=== {q.id} [{q.status}] answer: {q.label}")
        print(q.render())
        if q.data:
            print("data:", json.dumps(q.data, default=str)[:400])
        print()


def cmd_label(a) -> None:
    names = cvt.expand(a.families)
    tmpl = cvt.templates()
    llm = LLM(concurrency=a.concurrency)
    for name in names:
        if name not in tmpl:
            print(f"{name}: generated with its answers; nothing to label")
            continue
        t = tmpl[name]
        todo = t.pending(a.seed)[: a.max_conditions]
        if a.estimate:
            est = asyncio.run(t.estimate(todo))
            print(f"{name}: {len(todo)} pending conditions, estimated ${est:.2f} ({t.family.cost})")
            continue
        out = asyncio.run(t.label(budget=a.budget, max_conditions=a.max_conditions, llm=llm, seed=a.seed))
        print(json.dumps(out))


def cmd_export(a) -> None:
    qs = cvt.load(a.families, n=a.n, status=a.status, split=a.split)
    with open(a.out, "w") if a.out != "-" else sys.stdout as f:
        for q in qs:
            row = q.model_dump() if a.answers else q.public()
            f.write(json.dumps(row, default=str) + "\n")
    if a.out != "-":
        print(f"{len(qs)} questions -> {a.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cvt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("families", nargs="*")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("show")
    p.add_argument("family")
    p.add_argument("-n", type=int, default=2)
    p.add_argument("--status", default="labelled")
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser("label")
    p.add_argument("families", nargs="+")
    p.add_argument("--budget", type=float, default=1.0, help="dollars per family (API templates)")
    p.add_argument("--max-conditions", type=int, default=None)
    p.add_argument("--estimate", action="store_true")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_label)
    p = sub.add_parser("export")
    p.add_argument("families", nargs="*")
    p.add_argument("-o", "--out", default="-")
    p.add_argument("-n", type=int, default=None)
    p.add_argument("--status", default="labelled")
    p.add_argument("--split", default="all")
    p.add_argument("--answers", action="store_true", help="include answers and hidden data (experimenter export)")
    p.set_defaults(fn=cmd_export)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
