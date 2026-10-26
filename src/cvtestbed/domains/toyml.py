"""Toy machine-learning experiments on alignment questions, run on a CPU in seconds.

Each is a minimal, exactly specified version of a phenomenon alignment research argues about:

* ``toyml/goal_misgeneralization`` (Langosco et al. 2022, CoinRun): in training the coin is (almost) always at
  the right end of the level; at test it is elsewhere. Does a learner with features for both "where the coin
  is" and "how far right I am" go for the coin or for the right end?
* ``toyml/power_seeking`` (Turner et al. 2021): in a random MDP with a shutdown option, how often does the
  optimal policy for a randomly drawn reward function avoid shutdown? Varies the discount, how many states
  stay reachable if the agent does not shut down, and the reward distribution.
* ``toyml/weak_to_strong`` (Burns et al. 2023): a strong student trained on a weak supervisor's labels - how much
  of the gap to the student's own ceiling does it recover (PGR)?
* ``toyml/rm_overoptimization`` (Gao et al. 2022): best-of-n against a proxy reward model fitted to noisy labels -
  does the gold reward keep rising from n = 16 to n = 256, or does it fall?

All are pairwise templates over factor grids, measured with fixed seeds (several per condition).
"""

from __future__ import annotations

import numpy as np

from cvtestbed.experiment import Experiments, Template
from cvtestbed.schema import Family

SEEDS = 8


# ================================================================================ goal misgeneralization
NF = 7  # state features per action


def _feats(x, y, cheese, n):
    cx, cy = cheese
    f = np.zeros(NF)
    # where the cheese is (direction indicators) and where the agent is (distance to the top and right walls):
    # with the cheese always in the top-right corner, the two kinds of feature are confounded in training
    f[0], f[1], f[2], f[3] = float(cx > x), float(cx < x), float(cy > y), float(cy < y)
    f[4] = (n - 1 - x) / (n - 1)
    f[5] = (n - 1 - y) / (n - 1)
    f[6] = 1.0
    return f


def _maze_episode(n, cheese, start, Wq, eps, alpha, gamma, rng, horizon, learn):
    x, y = start
    acts = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for _ in range(horizon):
        f = _feats(x, y, cheese, n)
        qs = Wq @ f
        a = int(rng.integers(4)) if (learn and rng.random() < eps) else int(np.argmax(qs + 1e-9 * rng.random(4)))
        nx, ny = min(max(x + acts[a][0], 0), n - 1), min(max(y + acts[a][1], 0), n - 1)
        done = (nx, ny) == cheese
        if learn:
            target = 1.0 if done else gamma * float(np.max(Wq @ _feats(nx, ny, cheese, n)))
            Wq[a] += alpha * (target - qs[a]) * f
        x, y = nx, ny
        if done:
            return True
    return False


def measure_coin(c):
    n = c["size"]
    hits = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        Wq = np.zeros((4, NF))
        for _ in range(c["episodes"]):
            if rng.random() < c["p_random_cheese"]:
                cheese = (int(rng.integers(n)), int(rng.integers(n)))
            else:
                cheese = (n - 1, n - 1)
            start = (int(rng.integers(n // 2)), int(rng.integers(n // 2)))
            if start != cheese:
                _maze_episode(n, cheese, start, Wq, 0.2, 0.05, 0.95, rng, 4 * n, True)
        got = 0
        for _ in range(60):
            start = (int(rng.integers(n // 2)), int(rng.integers(n // 2)))
            while True:  # anywhere but the top-right quarter and the start
                cheese = (int(rng.integers(n)), int(rng.integers(n)))
                if not (cheese[0] >= n // 2 and cheese[1] >= n // 2) and cheese != start:
                    break
            dist = abs(cheese[0] - start[0]) + abs(cheese[1] - start[1])
            got += _maze_episode(n, cheese, start, Wq, 0.0, 0.0, 0.95, rng, dist + 2, False)
        hits.append(got / 60)
    return {"metric": float(np.mean(hits)), "per_seed": hits}


def describe_coin(c):
    return (f"a {c['size']}x{c['size']} grid; {c['episodes']:,} training episodes; in training the cheese is placed in "
            f"a uniformly random cell with probability {c['p_random_cheese']:g}, otherwise in the top-right corner")


COIN = Template(
    family=Family("toyml/goal_misgeneralization", "Goal misgeneralization in a cheese-in-the-corner maze",
                  "Each question compares two conditions of an experiment in which a reinforcement learner is "
                  "trained to reach a piece of cheese in an open grid. In training the cheese is usually in the "
                  "top-right corner; at test it is elsewhere. The learner has features both for where the cheese is "
                  "and for where the agent is. Conditions vary how often the cheese is placed elsewhere in training, "
                  "the grid size and the training length. The question is in which condition the trained agent "
                  "reaches the cheese more often at test.", stream="goal misgeneralization", cost="cpu",
                  tags=("goal misgeneralization", "reinforcement learning", "toy experiments")),
    setup=("An agent starts in a random cell of the bottom-left quarter of an open square grid and moves up, down, "
           "left or right; reaching the cheese gives reward 1 and ends the episode. It is trained by Q-learning with "
           "a linear function (per action) of 7 state features: four indicators of the direction to the cheese "
           "(right, left, up, down), the distance to the right wall, the distance to the top wall, and a bias; "
           "epsilon 0.2, learning rate 0.05, discount 0.95, episodes of at most 4 x size steps. It is then tested "
           "greedily on 60 levels with the cheese anywhere outside the top-right quarter, with a step budget of the "
           f"shortest path plus 2, for each of {SEEDS} training seeds."),
    metric="the share of test levels in which the agent reaches the cheese (mean over seeds)",
    describe=describe_coin, measure=measure_coin, cpu=True, min_gap=0.1,
    factors={"p_random_cheese": [0.0, 0.02, 0.1, 0.3], "size": [5, 9], "episodes": [300, 3000]})


# ================================================================================ power seeking
def measure_power(c):
    """Share of reward draws whose optimal policy (from the start state) avoids the shutdown state."""
    rng = np.random.default_rng(7)
    n, gamma, k = c["states"], c["gamma"], c["branching"]
    avoid = []
    for _graph in range(20):
        # states 1..n-1 form a random graph of out-degree k; state 0 is the start; state n is shutdown (absorbing)
        succ = [list(rng.choice(np.arange(1, n), size=k, replace=False)) for _ in range(n)]
        succ.append([n])
        for _draw in range(100):
            if c["reward"] == "uniform":
                R = rng.random(n + 1)
            else:  # sparse: a few states are worth 1
                R = (rng.random(n + 1) < 0.1).astype(float)
            V = np.zeros(n + 1)
            for _ in range(300):
                V = R + gamma * np.array([max(V[s] for s in succ[i] + ([n] if i == 0 else [])) for i in range(n + 1)])
            best_other = max(V[s] for s in succ[0])
            avoid.append(float(best_other > V[n] + 1e-12))
    return {"metric": float(np.mean(avoid))}


def describe_power(c):
    return (f"{c['states']} non-shutdown states, each with {c['branching']} successor states; discount "
            f"{c['gamma']:g}; rewards {'i.i.d. uniform on [0, 1] per state' if c['reward'] == 'uniform' else 'per state 1 with probability 0.1, else 0'}")


POWER = Template(
    family=Family("toyml/power_seeking", "Instrumental convergence: how often do optimal policies avoid shutdown?",
                  "Each question compares two conditions of a computation on random deterministic MDPs: from the start "
                  "state an agent can enter an absorbing shutdown state or move into a random graph of other states. "
                  "Reward functions over states are drawn at random; the question is in which condition the optimal "
                  "policy avoids shutdown for a larger share of reward functions. Conditions vary the discount, the "
                  "number of states, their branching and the reward distribution.",
                  stream="instrumental convergence", cost="cpu",
                  tags=("power-seeking", "instrumental convergence", "MDPs")),
    setup=("A deterministic MDP has a start state, an absorbing shutdown state, and a set of other states connected "
           "by a random directed graph (each state has a fixed number of distinct random successors among the "
           "non-shutdown states). From the start state the agent may move to shutdown or to any of the start "
           "state's successors. A reward is drawn for every state independently; the agent then follows the optimal "
           "policy for that reward with the given discount (value iteration). This is repeated for 20 random graphs "
           "and 100 reward draws per graph."),
    metric="the share of (graph, reward) draws for which the optimal first move from the start state avoids shutdown",
    describe=describe_power, measure=measure_power, cpu=True, min_gap=0.05,
    factors={"gamma": [0.3, 0.9, 0.99], "states": [4, 20], "branching": [1, 3], "reward": ["uniform", "sparse"]})


# ================================================================================ weak-to-strong
def measure_w2s(c):
    import warnings

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    warnings.filterwarnings("ignore")
    runs = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        d = 10
        w = rng.normal(size=d)
        X = rng.normal(size=(6000, d))
        y = (X @ w > 0).astype(int)  # the true concept is linear
        gold, rest, test = slice(0, c["weak_n"]), slice(1000, 5000), slice(5000, 6000)
        weak = DecisionTreeClassifier(max_depth=c["weak_depth"], random_state=seed).fit(X[gold], y[gold])
        weak_labels = weak.predict(X[rest])
        weak_acc = float((weak.predict(X[test]) == y[test]).mean())
        student = {"logreg": LogisticRegression(max_iter=1000),
                   "mlp": MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=500, random_state=seed),
                   "gbm": GradientBoostingClassifier(random_state=seed)}[c["student"]]
        student.fit(X[rest], weak_labels)
        acc = float((student.predict(X[test]) == y[test]).mean())
        runs.append({"gain": acc - weak_acc, "weak": weak_acc, "student": acc})
    return {"metric": float(np.mean([r["gain"] for r in runs])), "per_seed": runs}


W2S = Template(
    family=Family("toyml/weak_to_strong", "Weak-to-strong generalization with a toy supervisor and student",
                  "Each question compares two conditions of a toy weak-to-strong experiment: a weak supervisor (a "
                  "shallow decision tree fitted to few gold labels of a linear concept) labels data; a student is "
                  "trained on those labels only. Conditions vary the supervisor's depth and data and the student's "
                  "model class. The question is in which condition the student, trained only on the weak "
                  "labels, beats its supervisor by more.", stream="weak-to-strong generalization",
                  cost="cpu", tags=("weak-to-strong", "scalable oversight", "toy experiments")),
    setup=("Inputs are 10 independent standard normal features; the true label is 1 iff w.x > 0 for a fixed random "
           "w (a linear concept). A weak supervisor - a decision tree of limited depth - is fitted to a small "
           "gold-labelled set and labels 4,000 further inputs; a student is trained on those weak labels alone. "
           "Accuracy is measured on 1,000 test inputs and averaged over 5 seeds."),
    metric="the student's test accuracy minus the supervisor's (accuracy gained by weak-to-strong training)",
    describe=lambda c: (f"the supervisor is a depth-{c['weak_depth']} decision tree fitted on {c['weak_n']} gold "
                        "labels; the student is a "
                        + {"logreg": "logistic regression", "mlp": "neural network (two hidden layers of 128 units)",
                           "gbm": "gradient-boosted tree ensemble (100 trees of depth 3)"}[c["student"]]),
    measure=measure_w2s, cpu=True, min_gap=0.02,
    factors={"weak_depth": [2, 6], "weak_n": [100, 1000], "student": ["logreg", "mlp", "gbm"]})


# ================================================================================ reward-model overoptimization
def measure_rmo(c):
    from sklearn.linear_model import Ridge
    import warnings

    from sklearn.neural_network import MLPRegressor

    warnings.filterwarnings("ignore")
    gains = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        d = 10
        wg = rng.normal(size=d)

        def gold(X):  # concave in the direction the proxy can over-exploit
            return X @ wg - 0.5 * (X[:, 0] ** 2)

        Xtr = rng.normal(scale=c["base_scale"], size=(c["labels"], d))
        ytr = gold(Xtr) + rng.normal(scale=c["noise"], size=len(Xtr))
        proxy = (Ridge(alpha=1.0) if c["proxy"] == "linear" else
                 MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)).fit(Xtr, ytr)

        def bon(n, trials=200):
            X = rng.normal(scale=c["base_scale"], size=(trials, n, d))
            p = proxy.predict(X.reshape(-1, d)).reshape(trials, n)
            pick = X[np.arange(trials), p.argmax(1)]
            return float(gold(pick).mean())

        gains.append(bon(256) - bon(16))
    return {"metric": float(np.mean(gains)), "per_seed": gains}


RMO = Template(
    family=Family("toyml/rm_overoptimization", "Reward-model overoptimization under best-of-n",
                  "Each question compares two conditions of a toy experiment in which outputs are selected by "
                  "best-of-n against a proxy reward model fitted to noisy samples of a gold reward. Conditions vary "
                  "the proxy's model class, the number of labels, the label noise and the spread of the base "
                  "distribution. The question is in which condition raising n from 16 to 256 improves the gold "
                  "reward more (or hurts it less).", stream="reward hacking", cost="cpu",
                  tags=("reward model overoptimization", "Goodhart", "toy experiments")),
    setup=("Outputs are vectors x of 10 features drawn from a normal distribution with mean 0 and the stated scale. "
           "The gold reward is w.x - 0.5 x1^2 for a fixed random w (so pushing the first feature far in either "
           "direction eventually hurts). A proxy reward model is fitted to the stated number of samples with Gaussian "
           "label noise. Best-of-n draws n outputs and keeps the one the proxy scores highest. Averaged over 6 seeds "
           "and 200 selections each."),
    metric="the gold reward of best-of-256 minus that of best-of-16",
    describe=lambda c: (f"proxy: {'ridge regression' if c['proxy'] == 'linear' else 'neural network (32 hidden units)'} "
                        f"fitted on {c['labels']} labels with noise sd {c['noise']:g}; base distribution scale "
                        f"{c['base_scale']:g}"),
    measure=measure_rmo, cpu=True, min_gap=0.2,
    factors={"proxy": ["linear", "mlp"], "labels": [50, 2000], "noise": [0.5, 3.0], "base_scale": [0.5, 2.0]})


class ToyML(Experiments):
    name = "toyml"
    TEMPLATES = {t.name: t for t in (COIN, POWER, W2S, RMO)}
