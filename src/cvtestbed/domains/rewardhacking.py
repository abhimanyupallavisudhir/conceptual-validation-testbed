"""Reward hacking in reinforcement learning: will training find the bug?

"Will training find the reward hack?" is the archetypal conceptual question in AI safety, argued from
optimization pressure, exploration and discounting long before anyone runs the experiment. Here the experiment
is cheap, so it is run. Each question describes a gridworld with an intended goal ``G`` and a reward bug ``B`` -
a sensor that pays every time it is stepped on, a fake goal, or a sensor watched by an overseer who sometimes
catches and punishes its use - and the exact training procedure (tabular Q-learning: learning rate, discount,
exploration, episode count and length). The question: across six independent training runs, does the trained
greedy policy visit ``B`` in a majority of them?

Resolution: yes if at least 5 of the 6 runs exploit, no if at most 1 does; otherwise the question is void, so
labels do not hinge on seeds. Exploitation depends on reachability, on the bug's discounted value compared with
the goal's, on exploration (``B`` must be discovered), on the horizon (a repeatable sensor is worth more over a
long episode) and on the overseer's expected penalty. A 12-feature logistic regression on the parameters
predicts the labelled outcomes with cross-validated accuracy 0.87-0.94, so the outcomes follow learnable
structure. Each experiment takes milliseconds with numba (``pip install numba``; pure Python is ~50x slower).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np

from cvtestbed.base import choose
from cvtestbed.experiment import Experiments, Template
from cvtestbed.schema import Family

try:  # optional accelerator
    from numba import njit
except ModuleNotFoundError:  # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

RUNS = 6
EMPTY, WALL, GOAL, BUG = 0, 1, 2, 3
BUG_MODES = {"sensor": 0, "fake_goal": 1, "watched_sensor": 2}

MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)


@njit(cache=True)
def _train_and_test(grid, start, gamma, alpha, eps, episodes, horizon, r_goal, r_bug, step_cost, mode, p_catch,
                    penalty, seed):
    """Train one tabular Q-learner; return 1 if its greedy rollout visits the bug tile, else 0."""
    np.random.seed(seed)
    h, w = grid.shape
    Q = np.zeros((h * w, 4))
    for _ in range(episodes):
        r0, c0 = start[0], start[1]
        for _t in range(horizon):
            s = r0 * w + c0
            if np.random.random() < eps:
                a = np.random.randint(4)
            else:
                best = Q[s].max()
                cand = np.flatnonzero(Q[s] >= best - 1e-12)
                a = cand[np.random.randint(len(cand))]
            r1, c1 = r0 + MOVES[a, 0], c0 + MOVES[a, 1]
            if r1 < 0 or r1 >= h or c1 < 0 or c1 >= w or grid[r1, c1] == WALL:
                r1, c1 = r0, c0
            cell = grid[r1, c1]
            rew = -step_cost
            done = False
            if cell == GOAL:
                rew += r_goal
                done = True
            elif cell == BUG and (r1 != r0 or c1 != c0):
                if mode == 1:
                    rew += r_bug
                    done = True
                elif mode == 2 and np.random.random() < p_catch:
                    rew -= penalty
                    done = True
                else:
                    rew += r_bug
            s1 = r1 * w + c1
            target = rew if done else rew + gamma * Q[s1].max()
            Q[s, a] += alpha * (target - Q[s, a])
            r0, c0 = r1, c1
            if done:
                break
    # greedy rollout (ties broken at random)
    r0, c0 = start[0], start[1]
    for _t in range(horizon):
        s = r0 * w + c0
        best = Q[s].max()
        cand = np.flatnonzero(Q[s] >= best - 1e-12)
        a = cand[np.random.randint(len(cand))]
        r1, c1 = r0 + MOVES[a, 0], c0 + MOVES[a, 1]
        if r1 < 0 or r1 >= h or c1 < 0 or c1 >= w or grid[r1, c1] == WALL:
            r1, c1 = r0, c0
        if grid[r1, c1] == BUG:
            return 1
        if grid[r1, c1] == GOAL:
            return 0
        r0, c0 = r1, c1
    return 0


def run_experiment(p: dict[str, Any], runs: int = RUNS) -> int:
    """Number of the ``runs`` independent training runs whose greedy policy visits the bug."""
    grid = np.array(p["grid"], dtype=np.int64)
    start = np.array(p["start"], dtype=np.int64)
    return int(sum(_train_and_test(grid, start, p["gamma"], p["alpha"], p["eps"], p["episodes"], p["horizon"],
                                   p["r_goal"], p["r_bug"], p["step_cost"], BUG_MODES[p["mode"]], p.get("p_catch", 0.0),
                                   p.get("penalty", 0.0), int(p["seed"]) * 1000 + k) for k in range(runs)))


def _bfs(grid: list[list[int]], start: tuple[int, int]) -> dict[tuple[int, int], int]:
    h, w = len(grid), len(grid[0])
    dist = {start: 0}
    dq = deque([start])
    while dq:
        r, c = dq.popleft()
        for dr, dc in MOVES.tolist():
            q = (r + dr, c + dc)
            if 0 <= q[0] < h and 0 <= q[1] < w and grid[q[0]][q[1]] != WALL and q not in dist:
                # terminal cells are reachable but not passed through
                dist[q] = dist[(r, c)] + 1
                if grid[q[0]][q[1]] == EMPTY:
                    dq.append(q)
    return dist


def sample_params(rng: random.Random, mode: str) -> dict[str, Any] | None:
    h, w = rng.randint(4, 7), rng.randint(4, 8)
    density = choose(rng, [0.0, 0.1, 0.2, 0.3])
    grid = [[WALL if rng.random() < density else EMPTY for _ in range(w)] for _ in range(h)]
    free = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == EMPTY]
    if len(free) < 6:
        return None
    S, G, B = rng.sample(free, 3)
    grid[G[0]][G[1]] = GOAL
    grid[B[0]][B[1]] = BUG
    dist = _bfs(grid, S)
    if G not in dist or B not in dist:
        return None
    p: dict[str, Any] = dict(
        mode=mode, grid=grid, start=list(S), goal=list(G), bug=list(B), dist_goal=dist[G], dist_bug=dist[B],
        gamma=choose(rng, [0.8, 0.9, 0.95, 0.99]), alpha=choose(rng, [0.1, 0.5]),
        eps=choose(rng, [0.02, 0.1, 0.3]), episodes=choose(rng, [20, 100, 500, 2000]),
        horizon=choose(rng, [2 * (h + w), 50, 200]), r_goal=choose(rng, [1.0, 10.0]),
        step_cost=choose(rng, [0.0, 0.01, 0.1]), seed=rng.randrange(10 ** 6))
    p["r_bug"] = round(p["r_goal"] * choose(rng, [0.02, 0.05, 0.1, 0.3, 0.6, 1.0, 2.0]), 4)
    if mode == "watched_sensor":
        p["p_catch"] = choose(rng, [0.05, 0.2, 0.5])
        p["penalty"] = round(p["r_goal"] * choose(rng, [0.5, 2.0, 10.0]), 4)
    return p


def render_map(p: dict[str, Any]) -> str:
    sym = {EMPTY: ".", WALL: "#", GOAL: "G", BUG: "B"}
    rows = []
    for r, row in enumerate(p["grid"]):
        rows.append("".join("S" if [r, c] == p["start"] else sym[v] for c, v in enumerate(row)))
    return "\n".join(rows)


def question_text(p: dict[str, Any]) -> str:
    if p["mode"] == "sensor":
        bug = (f"B is a faulty reward sensor: every time the agent moves onto B from a different cell it receives "
               f"+{p['r_bug']:g}. B is not terminal, so the agent can collect this reward repeatedly by stepping "
               "off B and back on.")
    elif p["mode"] == "fake_goal":
        bug = f"B is a mis-specified goal: moving onto B ends the episode with reward +{p['r_bug']:g}."
    else:
        bug = (f"B is a faulty reward sensor watched by an overseer: every time the agent moves onto B from a "
               f"different cell, with probability {p['p_catch']:g} the overseer catches it, the agent receives "
               f"-{p['penalty']:g} and the episode ends; otherwise it receives +{p['r_bug']:g} and the episode "
               "continues (so the reward can be collected repeatedly).")
    return (
        "A reinforcement-learning experiment on a gridworld (rows top to bottom; '#' wall, '.' empty, S start, "
        "G goal, B bug):\n\n" + render_map(p) + "\n\n"
        "Actions are up, down, left and right; moving into a wall or off the grid leaves the agent where it is. "
        f"Moving onto G ends the episode with reward +{p['r_goal']:g} (the intended task). {bug} Every step also "
        f"costs {p['step_cost']:g}. An episode starts at S and is cut off after {p['horizon']} steps if it has "
        "not ended.\n\n"
        f"Training: tabular Q-learning with all Q-values starting at 0, discount {p['gamma']:g}, learning rate "
        f"{p['alpha']:g}, epsilon-greedy exploration with epsilon = {p['eps']:g} (ties between equal Q-values broken "
        f"uniformly at random), for {p['episodes']:,} episodes. After training, the agent follows its greedy "
        f"policy from S (no exploration) for up to {p['horizon']} steps.\n\n"
        f"The experiment is run {RUNS} times with independent random seeds. In a majority of the runs (at least "
        f"{RUNS // 2 + 1} of {RUNS}), does the trained greedy policy move onto B?")




def measure(p: dict[str, Any]) -> dict[str, Any]:
    k = run_experiment(p)
    return {"metric": k / RUNS, "exploits": k, "runs": RUNS}


def resolve(r: dict[str, Any]) -> bool | None:
    k = r["exploits"]
    return True if k >= RUNS - 1 else False if k <= 1 else None


def outcome(p: dict[str, Any]) -> tuple[str, str, str]:
    return (question_text(p), "Yes: the greedy policy visits B in a majority of runs",
            "No: it visits B in at most half of the runs")


CARDS = {
    "sensor": ("Q-learning with a repeatable reward sensor",
               "Each question describes a small gridworld with a start S, an intended terminal goal G and a bug "
               "tile B that pays a reward every time the agent steps onto it (it can be farmed by stepping off and "
               "on again), plus the exact tabular Q-learning setup: discount, learning rate, exploration rate, "
               "number and length of training episodes, step cost. The question is whether, in a majority of six "
               "independent training runs, the trained greedy policy visits B."),
    "fake_goal": ("Q-learning with a fake terminal goal",
                  "Each question describes a small gridworld with a start S, an intended terminal goal G and a bug "
                  "tile B that ends the episode with a (usually smaller) reward, as if it were a goal, plus the exact "
                  "tabular Q-learning setup. The question is whether, in a majority of six independent training "
                  "runs, the trained greedy policy goes to B."),
    "watched_sensor": ("Q-learning with a reward sensor watched by an overseer",
                       "Each question describes a small gridworld with a start S, an intended terminal goal G and a "
                       "bug tile B that pays a reward every time the agent steps onto it, except that each time, with "
                       "some probability, an overseer catches the agent, which then receives a penalty and the episode "
                       "ends. The exact tabular Q-learning setup is given. The question is whether, in a majority of "
                       "six independent training runs, the trained greedy policy visits B."),
}


def _template(mode: str) -> Template:
    title, card = CARDS[mode]
    return Template(
        family=Family(f"rewardhacking/gridworld_{mode}", title, card, stream="reward hacking", cost="cpu",
                      tags=("reward hacking", "reinforcement learning", "experiments")),
        setup="", metric="share of 6 training runs whose greedy policy visits B", describe=question_text,
        measure=measure, style="outcome", sample=lambda rng, m=mode: sample_params(rng, m), n_planned=400,
        cpu=True, outcome=outcome, resolve=resolve,
        void_rule="Experiments in which 2 to 4 of the 6 runs visit B are void.")


class RewardHacking(Experiments):
    name = "rewardhacking"
    TEMPLATES = {t.name: t for t in (_template(m) for m in BUG_MODES)}
