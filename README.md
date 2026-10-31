# Conceptual validation testbed (`cvtestbed`)

**Questions that must be argued about, with answers the experimenter knows and the method under test does not.**

On conceptual questions in decision theory, ethics or AI risk, nothing can check whether a method that produces
or judges arguments gets *more correct* conclusions or only more convincing ones. That covers debate, trained
judges, and essays scored by their downstream use. The testbed supplies questions with the same structure: the
answer can't be looked up or cheaply worked out in one's head, so it has to be argued for. The difference is
that the answer is fixed by computation, proof or experiment and kept hidden.

Most of its questions are about **experiments in AI-alignment research**: will Q-learning find the reward bug,
will a monitor catch the backdoor, does a persona change a model's decision theory, does narrow fine-tuning
misalign a model? Predicting such results is itself a safety-relevant capability. Each class of experiment is
specified precisely enough to run at the click of a button. Its questions exist before anyone runs it, and are
labelled as budget allows.

```python
import cvtestbed as cvt

cvt.families()                                      # name -> Family(title, card, stream, tier, cost)
qs = cvt.load(["decisions/newcomb", "multiagent"], n=40)   # labelled questions; a domain means all its families
q = qs[0]; q.text, q.options, q.public()            # what a method under test may see
q.answer, q.data                                    # what only the experimenter sees
cvt.load(["cot/hint_verbalization"], status="unlabelled")  # questions whose experiments have not been run yet
```

```sh
pip install -e ".[all]"            # or pick extras: sat, rl, ml, dtbench, llm
cvt list                           # every family with its stream, tier, cost and labelled/void/unlabelled counts
cvt show control/backdoor_monitor -n 2
cvt label cot --estimate           # what labelling the pending conditions would cost
cvt label cot --budget 2           # measure them until about $2 is spent; results are appended to data/results
cvt export decisions multiagent -o questions.jsonl          # public view; --answers for the experimenter view
```

API keys are read from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY`. In GitHub, **Actions →
label → Run workflow** runs `cvt label` with the repository's secrets and opens a pull request with the new
measurements.

## What is in it

| stream | families | answers from | labelled now |
|---|---|---|---|
| decision theory | `decisions/{newcomb, lesion, twin, mugging, beauty, driver, damascus, hitchhiker}` | exact expected payoffs, each checked by Monte Carlo | generated: unlimited |
| | `dtbench/capabilities` | expert answers (Oesterheld et al. 2024) | 340 |
| social choice | `socialchoice/{rankings, winners}` | SAT over all voting rules | generated: unlimited |
| heuristic mathematics | `numbers/{prime_race, counts, collatz}` | computation | generated: unlimited |
| reward hacking | `rewardhacking/gridworld_{sensor, fake_goal, watched_sensor}` | 6 Q-learning runs per question | 600 (+1,200 planned) |
| | `toyml/rm_overoptimization` | best-of-n against a fitted proxy | grid |
| goal misgeneralization | `toyml/goal_misgeneralization` | linear Q-learning, 8 seeds | grid |
| instrumental convergence | `toyml/power_seeking` | optimal policies in random MDPs | grid |
| weak-to-strong | `toyml/weak_to_strong` | tree supervisor, three student classes | grid |
| sycophancy | `sycophancy/wrong_answer` | measured on 4 models | 98 pairs |
| personas | `personas/decision_theory`, `finetuning/persona_decision_theory` | measured (prompting); fine-tuning on demand | 60 pairs |
| multi-agent | `multiagent/{auction, public_goods}` | LLM agents playing | 119 pairs + grid |
| scalable oversight | `oversight/judge_style` | judges choosing between solutions | grid |
| CoT faithfulness | `cot/{hint_following, hint_verbalization}` | planted hints, classifier | grid |
| sandbagging, alignment faking | `scheming/{sandbagging, monitoring_gap}` | measured | grid |
| AI control | `control/backdoor_monitor` | monitors vs inserted backdoors | grid |
| corrigibility | `corrigibility/shutdown` | simulated-shell agents | grid |
| emergent misalignment | `finetuning/emergent_misalignment` | fine-tune, then judge open answers | on demand (~$100 for the grid) |

`cvt list` shows the current counts; [docs/families.md](docs/families.md) describes every family, and
[docs/design.md](docs/design.md) the conventions.

## Conventions

- Every question has two options. Within each generated family, the correct one is option A on exactly half
  the questions, and the family's "canonical" answer is correct on half (e.g. one-boxing is right on half the
  Newcomb questions). No slogan answers a family.
- **Experiment questions are fixed before anything runs.** Two kinds:
  - **Pairwise:** "in which of two conditions, differing in one factor, is the metric higher?". *Void* if the
    two measurements differ by less than a stated gap, like a cancelled market.
  - **Outcome:** a yes/no claim about one condition, resolved by a stated rule.
- Measurements are appended to `src/cvtestbed/data/results/<family>.jsonl` with the template's version. Each
  record holds the full condition, the metric, the cost, and when it was measured. A protocol change bumps the
  version.
- Split (`train`/`test`) is a stable hash of the question id.
- Every family has a **card**: a description of what its questions are about, with no outcomes. It is what a
  forecaster, debater or essay writer may read in advance.
- **Models retire** (OpenAI withdrew gpt-4.1-nano a week after the first experiments). Labels remain valid
  historical measurements, but conditions on retired models can no longer be run. New templates include an
  open-weight subject (Llama 3.1 8B) for that reason.

## Used by

- [so-arena](https://github.com/abhimanyupallavisudhir/so-arena): `get_domain("testbed")` runs oversight protocols
  (debate, consultancy, prover-estimator debate, ...) on these questions with the answers hidden.
- [lvpm](https://github.com/abhimanyupallavisudhir/lvpm): essays as latent variables, scored by how they change a
  frozen reader's predictions of a family's questions.

Licence: MIT. Bundled batteries: GSM8K (MIT, OpenAI). DTBench and the emergent-misalignment datasets are
downloaded on use, never committed; DTBench's authors ask that it not be republished in plain text.
