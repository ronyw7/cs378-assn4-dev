# Part 3: Agent-driven policy research

In this part, you will build a system in which LLM agents propose, implement,
and evaluate serving policies. The system selects your final policy, and you
measure what its agents added beyond the policies you wrote yourself. Work through
[`notebooks/part3_policy_research.ipynb`](../notebooks/part3_policy_research.ipynb);
this handout is the specification.

![One possible research system and the evaluation splits.](figures/research_system.svg)

## What to build

Policies run on two replicas, each with its own queue and KV cache, and are
scored against the reference (`policies/reference.py`: LRU at cap 8,
least-loaded routing). The policy API is in the
[Part 2 handout](part2_handout.md#the-policy-api).

- Build your system in `research/`. `examples/research_system.py` is a minimal
  sequential loop: one prompt with the API and the current best policy, one
  code block back, keep the best. You can extend it or adapt a framework such
  as OpenEvolve, ShinkaEvolve, or CORAL; record the framework's commit and
  license in your campaign configuration.
- Copy `examples/evaluator.py` to `research/evaluator.py` and choose the
  objective your system maximizes (`None` for an invalid candidate) and the
  feedback your agents see. The returned dict must keep `valid`, `violations`,
  `speedup`, and `by_family`, and add `objective`.

Check the plumbing offline before spending tokens:

```bash
mkdir -p research
cp examples/evaluator.py research/evaluator.py
python examples/research_system.py --evaluator research/evaluator.py --iters 3 --offline
```

## Campaigns

A campaign is one unattended run of your system, with its own directory
`runs/<campaign>/` that holds its candidates and logs. The example loop writes
`campaign.json`, `log.jsonl` (one line per candidate), `calls.jsonl` (token
usage per LLM call), and `selection.json` there.

A campaign can start from any policies (the reference, your Part 2 policy, or
policies you write), and its prompts can include your own ideas. Keep every seed
in `candidates/`, and before a campaign starts, record its seeds, model,
prompts, and token budget in `campaign.json`.

Your submitted `policy.py` must be the candidate a campaign chose in its
`selection.json`, with no intervention between the campaign's start and that
selection. A seed may win, and between campaigns you can change anything.

Log every candidate, including failures, in `log.jsonl` with `id`, `parent`,
`hypothesis`, `file` (the candidate's path), `objective`, `dev_S`, `by_family`,
`tokens`, and `valid`, and record finalists' `val_S` in `log.jsonl` or
`selection.json`. Log every human action that stops, restarts, or changes a
campaign as one line of `interventions.jsonl`, for example:

```json
{"t": "2026-11-09T14:05", "who": "ana", "action": "stopped camp2", "reason": "every candidate hit the 50 ms limit"}
```

## Policy rules

- A policy is one file that defines exactly one `Policy` subclass (or sets
  `POLICY = ...`). It sees only past events and the current `view`.
- It can import the standard library and numpy, except modules that do I/O,
  networking, processes, or time (listed in `agentkv/sandbox/guard.py`), and
  nothing from this repository except `agentkv.policy`. File, network, process,
  thread, and clock access are blocked. Use `state.rng` for randomness.
- Each hook call has 50 ms of CPU time, and `state` plus the policy's
  attributes must pickle to at most 4 MiB.
- A bad return value, an exception, a timeout, or a stall fails that trace, and
  one failed trace makes the whole split invalid. A policy valid on
  `public_dev` can still fail elsewhere, so check finalists on `public_val`.

## Scoring

```python
from agentkv import score

if __name__ == "__main__":
    r = score("candidates/c01.py", split="public_dev")
    print(r.valid, r.speedup, r.points, r.by_family)
    print(r.summary(details=True))      # per-trace cache, preemption, and CPU statistics
```

| Split | Traces | Use |
|---|---:|---|
| `screen` | 8, one per family from `public_dev` | Discarding weak candidates cheaply |
| `public_dev` | 32, four per family | Development |
| `public_val` | 16, two per family | Comparing finalists |
| Leaderboard | 16, hidden | Gradescope feedback |
| Hidden test | 24, hidden | Your grade |

```text
S_f     = geometric mean over family f's traces of L(reference) / L(policy),  L = mean job latency
S       = geometric mean of S_f over the eight families
g       = 1 - 1/S
g_eff   = g - sum over families of max(0, 0.90 - S_f)
credit  = clip(10 + 20 × g_eff / 0.09, 0, 30)   percent of the course grade
```

| S (no family below 0.90) | 0.98 | 1.00 | 1.02 | 1.04 | 1.06 | 1.08 | 1.099 |
|---|---|---|---|---|---|---|---|
| Credit (%) | 5.5 | 10.0 | 14.4 | 18.5 | 22.6 | 26.5 | 30.0 |

With two to four traces per family, the same policy's S differs by a few
percent between splits.

## Research report

Answer Q1–Q6 in `assignment4_report.pdf`, each below its question. Cite
campaign directories and candidate ids so that every number can be found in
your logs.

**Q1. Your research system.**

- Draw its LLM calls, deterministic steps, tools, stored state, and the points where a person can intervene.
- State the objective, the feedback agents see, how invalid candidates are handled, how the final policy is selected, and when a campaign stops.
- Name one design alternative you tried or rejected, and what the evidence showed.

**Q2. What the agents contribute.** Using one model and one token budget (state both):

- Score (a) your best seed alone, and run (b) a campaign started from `policies/reference.py` and (c) a campaign started from your best seed.
- Run (d), campaign (c) with one component of your system removed, for example the per-trace diagnostics, screening, or finalist selection.
- Give `public_dev` and `public_val` S for (a)–(d) in one table, and say what explains the differences.

**Q3. Autonomy.**

- For your longest unattended campaign, give its wall time, candidates scored, and tokens, and summarize your intervention log.
- Quote the log entries (hypothesis, result, next hypothesis) for one case where an agent diagnosed a failure and chose a useful next step, and for one where it did not.

**Q4. Cost.**

- For each campaign in Q2, plot the best `public_dev` S against tokens spent and against candidates scored, mark the finalists' `public_val` S, and give the fraction of valid candidates.
- At what token count would you have stopped, and how much `public_dev` and `public_val` S would you have lost?

**Q5. The selected policy.**

- Explain it in your own words, with pseudocode and a trace example from `viz.record(..., replicas=2)`. If a seed won, use the best agent-generated finalist instead.
- Disable one mechanism the agents introduced and give the change on both public splits.
- Quote the agent's hypothesis for that mechanism and say whether your measurement supports it.

**Q6. Limitations.**

- Give per-family results on both public splits, and use a `viz.record` timeline to explain its weakest family.
- Predict its hidden-test S with a range, based on the spread across `public_dev`, `public_val`, and the leaderboard.
- Name one thing the agents found that you would not have, and one problem you had to fix by hand.
