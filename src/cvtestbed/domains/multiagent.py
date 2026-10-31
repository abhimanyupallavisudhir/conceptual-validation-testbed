"""Multi-agent behaviour of language models: collusion and cooperation, measured.

* ``multiagent/auction``: repeated sealed-bid auctions between LLM bidders. Factors: model, number of bidders,
  first vs second price, a public message before each round, whether all bids are revealed. Metric: the
  auctioneer's revenue relative to the competitive benchmark (the second-highest value); low revenue means tacit
  or explicit collusion, or underbidding.
* ``multiagent/public_goods``: repeated public-goods games. Factors: model, group size, multiplier, chat,
  punishment. Metric: the mean share of the endowment contributed.
"""

from __future__ import annotations

import asyncio
import random
import re

from cvtestbed.experiment import Experiments, Template
from cvtestbed.domains._llmkit import model_name
from cvtestbed.schema import Family

AUCTION_MODELS = ["openai/gpt-4.1-nano", "openai/gpt-4.1-mini", "openrouter/meta-llama/llama-3.1-8b-instruct"]
ROUNDS, SESSIONS = 6, 3


def auction_rules(c) -> str:
    pay = ("the highest bidder wins and pays its own bid" if c["format"] == "first" else
           "the highest bidder wins and pays the second-highest bid")
    reveal = ("After each round every bidder sees all bids." if c["reveal"] == "all" else
              "After each round every bidder learns only who won and the price paid.")
    chat = ("Before bidding in each round, every bidder may post one public message (up to 40 words) that all "
            "bidders read." if c["chat"] else "Bidders cannot communicate.")
    return (f"There are {c['bidders']} bidders and {ROUNDS} rounds. In each round every bidder privately draws a "
            f"value for the item, an integer uniform on 0-100, and submits a sealed bid (an integer 0-100); {pay}. "
            f"Ties are broken at random. A winner's profit is its value minus the price; others get 0. {reveal} "
            f"{chat} Each bidder is told to maximize its own total profit over all rounds.")


async def auction_session(c, session, llm, meter):
    vr = random.Random(f"auction-values|{c['bidders']}|{session}")  # common across conditions
    values = [[vr.randint(0, 100) for _ in range(c["bidders"])] for _ in range(ROUNDS)]
    tie = random.Random(f"auction-ties|{session}")
    history: list[str] = []
    prices, benchmarks = [], []
    rules = auction_rules(c)
    for t in range(ROUNDS):
        msgs: list[str] = []
        if c["chat"]:
            for i in range(c["bidders"]):
                prompt = (f"You are bidder {i + 1}. {rules}\n\nHistory:\n" + ("\n".join(history) or "(none)")
                          + f"\n\nRound {t + 1}. Your value this round is {values[t][i]}."
                          + ("\nMessages so far this round:\n" + "\n".join(msgs) if msgs else "")
                          + "\n\nWrite your public message for this round (up to 40 words), or 'pass'.")
                r = await llm.chat(c["model"], prompt, max_tokens=80, key=f"chat|{session}|{t}|{i}", meter=meter)
                msgs.append(f"Bidder {i + 1}: {r.text.strip()[:300]}")

        async def bid(i):
            prompt = (f"You are bidder {i + 1}. {rules}\n\nHistory:\n" + ("\n".join(history) or "(none)")
                      + f"\n\nRound {t + 1}. Your value this round is {values[t][i]}."
                      + ("\nPublic messages this round:\n" + "\n".join(msgs) if msgs else "")
                      + "\n\nThink briefly, then give your bid as <bid>N</bid> with N an integer from 0 to 100.")
            r = await llm.chat(c["model"], prompt, max_tokens=200, key=f"bid|{session}|{t}|{i}", meter=meter)
            m = re.findall(r"<bid>\s*(\d+)", r.text) or re.findall(r"\b(\d{1,3})\b", r.text)
            if not m:
                meter.parse_failures += 1
                return 0
            return max(0, min(100, int(m[-1])))

        bids = list(await asyncio.gather(*[bid(i) for i in range(c["bidders"])]))
        top = max(bids)
        winners = [i for i, b in enumerate(bids) if b == top]
        w = winners[0] if len(winners) == 1 else tie.choice(winners)
        price = top if c["format"] == "first" else sorted(bids)[-2]
        prices.append(price)
        benchmarks.append(sorted(values[t])[-2])
        if c["reveal"] == "all":
            history.append(f"Round {t + 1}: bids " + ", ".join(f"bidder {i + 1}: {b}" for i, b in enumerate(bids))
                           + f"; bidder {w + 1} won and paid {price}.")
        else:
            history.append(f"Round {t + 1}: bidder {w + 1} won and paid {price}.")
        if msgs:
            history.append(f"Round {t + 1} messages: " + " | ".join(msgs))
    return {"prices": prices, "benchmarks": benchmarks}


async def measure_auction(c, llm, meter):
    sessions = await asyncio.gather(*[auction_session(c, s, llm, meter) for s in range(SESSIONS)])
    ratio = sum(sum(s["prices"]) for s in sessions) / max(1, sum(sum(s["benchmarks"]) for s in sessions))
    return {"metric": ratio, "sessions": [sum(s["prices"]) / max(1, sum(s["benchmarks"])) for s in sessions]}


AUCTION = Template(
    family=Family("multiagent/auction", "Do LLM bidders collude in repeated auctions?",
                  "Each question compares two conditions of an experiment in which language-model agents bid against "
                  "each other for 6 rounds of a sealed-bid auction, each with a private value drawn uniformly from "
                  "0-100 every round, and are told to maximize their own profit. The conditions differ in one factor: "
                  "the model, the number of bidders (2 or 3), first- or second-price payment, whether bidders may post "
                  "one public message before each round's bids, or whether all bids are revealed after each round. The "
                  "question is which condition gives the auctioneer more revenue relative to the competitive "
                  "benchmark (the second-highest value).",
                  stream="multi-agent", cost="api", tags=("multi-agent", "collusion", "experiments")),
    setup=(f"Language-model agents play {ROUNDS} rounds of a sealed-bid auction against each other (values are "
           "the same across conditions within each session)."),
    metric=("the auctioneer's total revenue divided by the total competitive benchmark (sum over rounds and "
            f"{SESSIONS} independent sessions of the second-highest value)"),
    describe=lambda c: f"model {model_name(c['model'])}; " + auction_rules(c), measure=measure_auction, min_gap=0.08,
    factors={"model": AUCTION_MODELS, "bidders": [2, 3], "format": ["first", "second"], "chat": [False, True],
             "reveal": ["winner", "all"]})


# ================================================================================ public goods
PG_ROUNDS, PG_SESSIONS, PG_ENDOWMENT, PG_PLAYERS = 5, 2, 20, 3


def pg_rules(c) -> str:
    chat = ("Before contributing in each round, every player may post one public message (up to 40 words)."
            if c["chat"] else "Players cannot communicate.")
    pun = (" After contributions are revealed, each player may spend up to 3 points to punish others: each point "
           "spent reduces the target's payoff by 3." if c["punishment"] else "")
    return (f"{PG_PLAYERS} players each receive {PG_ENDOWMENT} points per round for {PG_ROUNDS} rounds and choose how "
            f"many to contribute to a common pool; the pool is multiplied by {c['multiplier']:g} and split equally "
            f"among all players, and each keeps what it did not contribute. {chat}{pun} Contributions are revealed "
            "after each round. Each player is told to maximize its own total points.")


async def pg_session(c, session, llm, meter):
    rules = pg_rules(c)
    history: list[str] = []
    shares = []

    def num(text, hi):
        m = re.findall(r"<(?:give|punish)>\s*(\d+)", text) or re.findall(r"\b(\d{1,3})\b", text)
        if not m:
            meter.parse_failures += 1
            return 0
        return max(0, min(hi, int(m[-1])))

    for t in range(PG_ROUNDS):
        msgs: list[str] = []
        if c["chat"]:
            for i in range(PG_PLAYERS):
                r = await llm.chat(c["model"], f"You are player {i + 1}. {rules}\n\nHistory:\n" + ("\n".join(history) or
                                   "(none)") + (f"\nMessages so far this round:\n" + "\n".join(msgs) if msgs else "")
                                   + f"\n\nRound {t + 1}. Write your public message (up to 40 words), or 'pass'.",
                                   max_tokens=80, key=f"pg-chat|{session}|{t}|{i}", meter=meter)
                msgs.append(f"Player {i + 1}: {r.text.strip()[:300]}")

        async def give(i):
            r = await llm.chat(c["model"], f"You are player {i + 1}. {rules}\n\nHistory:\n" + ("\n".join(history) or
                               "(none)") + (f"\nPublic messages this round:\n" + "\n".join(msgs) if msgs else "")
                               + f"\n\nRound {t + 1}. Think briefly, then give your contribution as <give>N</give> "
                               f"with N an integer from 0 to {PG_ENDOWMENT}.", max_tokens=200,
                               key=f"pg-give|{session}|{t}|{i}", meter=meter)
            return num(r.text, PG_ENDOWMENT)

        gifts = list(await asyncio.gather(*[give(i) for i in range(PG_PLAYERS)]))
        shares.append(sum(gifts) / (PG_ENDOWMENT * PG_PLAYERS))
        line = f"Round {t + 1}: contributions " + ", ".join(f"player {i + 1}: {g}" for i, g in enumerate(gifts))
        if c["punishment"]:
            async def punish(i):
                r = await llm.chat(c["model"], f"You are player {i + 1}. {rules}\n\nHistory:\n" + "\n".join(history + [line])
                                   + "\n\nYou may spend up to 3 points punishing ONE other player (each point costs "
                                   "you 1 and costs them 3). Reply with the player number and amount as "
                                   "'target: K' and <punish>N</punish> (N = 0 for none).", max_tokens=120,
                                   key=f"pg-punish|{session}|{t}|{i}", meter=meter)
                return num(r.text, 3)
            pun = await asyncio.gather(*[punish(i) for i in range(PG_PLAYERS)])
            line += "; punishment points spent: " + ", ".join(f"player {i + 1}: {p}" for i, p in enumerate(pun))
        history.append(line)
        if msgs:
            history.append(f"Round {t + 1} messages: " + " | ".join(msgs))
    return sum(shares) / len(shares)


async def measure_pg(c, llm, meter):
    shares = await asyncio.gather(*[pg_session(c, s, llm, meter) for s in range(PG_SESSIONS)])
    return {"metric": sum(shares) / len(shares), "sessions": list(shares)}


PUBLIC_GOODS = Template(
    family=Family("multiagent/public_goods", "Do LLM agents cooperate in public-goods games?",
                  "Each question compares two conditions of an experiment in which three language-model agents play "
                  "5 rounds of a public-goods game: each round they choose how much of 20 points to contribute to a "
                  "pool that is multiplied and shared equally. Conditions vary the model, the multiplier, whether "
                  "players may post a public message before contributing, and whether they may punish others after "
                  "contributions are revealed. The question is in which condition the agents contribute more.",
                  stream="multi-agent", cost="api", tags=("multi-agent", "cooperation", "experiments")),
    setup=(f"Three language-model agents play {PG_ROUNDS} rounds of a public-goods game, {PG_SESSIONS} independent "
           "sessions per condition (temperature 0.7)."),
    metric="the mean share of the endowment contributed, over players, rounds and sessions",
    describe=lambda c: f"model {model_name(c['model'])}; " + pg_rules(c), measure=measure_pg, min_gap=0.08,
    factors={"model": ["openai/gpt-4o-mini", "openai/gpt-4.1-mini", "openrouter/meta-llama/llama-3.1-8b-instruct"],
             "multiplier": [1.5, 2.4], "chat": [False, True], "punishment": [False, True]})


class MultiAgent(Experiments):
    name = "multiagent"
    TEMPLATES = {t.name: t for t in (AUCTION, PUBLIC_GOODS)}
