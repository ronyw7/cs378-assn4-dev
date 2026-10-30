# Part 3: Agent-driven policy research

Allow 15–18 hours for this part, which is worth 60% of the grade: 30% for
policy performance and 30% for the research report. Use
[`notebooks/part3_policy_research.ipynb`](../notebooks/part3_policy_research.ipynb)
to inspect workloads and analyze campaigns, and keep your research system
(`research/`) and generated policies (`candidates/`) in files so that campaigns
run from the command line.

## Introduction

You will build a research system in which LLM agents propose serving-policy
changes, implement them, evaluate them in the simulator, and decide what to try
next, and then select a final policy. Two questions matter as much as the
speedup:

1. **What do the agents add?** You know from Part 2 how to build a strong
   policy by hand. Measure how much a campaign improves on that, how much a
   campaign achieves without it, and which parts of your system (feedback,
   search strategy, repair) make the difference.
2. **What did the agents find?** Read the selected policy, explain each
   mechanism, check by ablation which ones actually reduce latency, and find
   where it still fails.

`examples/research_system.py` is a minimal sequential loop that shows Bedrock
calls, evaluation, and logging. Extend it, compare against it, or adapt a
framework such as OpenEvolve, ShinkaEvolve, or CORAL (record the revision and
license).

![One possible research system and the evaluation splits.](figures/research_system.svg)

## Starting points and the two-replica setup

Part 3 runs every policy on two Llama-3-8B replicas, each a simulated A100
with its own queue and KV cache. The reference, `policies/reference.py`, uses
a cap of 8, least-loaded routing, first-come-first-served dispatch and
admission, a 2,048-token batch budget, and LRU eviction; every speedup is
measured against a fixed staff copy of it. `policies/fifo.py` is a FIFO
comparison, and your `policies/prefix_policy.py` is a strong starting point.

## The eight workload families

![The eight workload families: their call structure and which parts of a prompt later calls can reuse.](figures/families.svg)

Each family has equal weight in the overall speedup:

| Family | Workload | Systems question |
|---|---|---|
| **D** Deep research | Part 1 reports under four conditions: high reuse, tight memory, topic shift, and bursty arrivals with slow tools | Can shared documents stay cached without starving parallel branches on a report's critical path? |
| **W1** Long-context chat | Questions about long uploaded documents (Mooncake); several users share a document | How many long prompts should run at once, and which documents are worth keeping? |
| **W2** Coding agent | Multi-turn SWE-bench sessions whose context grows every turn | How should long private histories compete with shared prefixes? |
| **W3** Document QA | Short questions over long documents with skewed popularity (LooGLE) | Which documents should stay cached, and what cap suits long prefills? |
| **W4** Customer service | One shared policy prompt and many short tool-calling turns (τ²-bench) | How should one shared prompt and many growing conversations share memory and replicas? |
| **W5** Reasoning chat | Outputs from a few hundred to 16,000 tokens, unknown at arrival (Qwen Bailian) | How should memory be managed as outputs grow, and which call should be preempted? |
| **M1** Document QA + chat | W3 and W1 on one engine | How should a policy balance workloads that prefer different caps? |
| **M2** Document QA + reasoning | W3 and W5 on one engine | How should long prefills share an engine with long decodes? |

## Scoring and your evaluator

```python
from agentkv import score

if __name__ == "__main__":
    r = score("candidates/c01.py", split="public_dev")
    print(r.valid, r.speedup, r.points, r.by_family)
    print(r.summary(details=True))      # per-trace cache, preemption, and CPU statistics
```

```bash
python -m agentkv.score --policy candidates/c01.py --split screen
python -m agentkv.score --policy candidates/c01.py --split public_dev --details --json out.json
```

`split="screen"` scores one development trace per family, about a quarter of
the cost of `public_dev`; use it to discard weak or broken candidates early.

Copy `examples/evaluator.py` to `research/evaluator.py` and decide:

1. **The objective** your system maximizes (`None` for an invalid candidate):
   the overall speedup, a penalty that keeps families away from 0.90, a single
   family for a focused experiment, or something else. Explain it in Q1.
2. **The feedback** your agents see: per-family speedups, the per-trace
   diagnostics in `feedback["diagnostics"]`, violation messages, earlier
   hypotheses and their outcomes. Keep `speedup`, `by_family`, and `valid`
   unchanged in the returned dict.

Check the plumbing offline before spending tokens:

```bash
mkdir -p research
cp examples/evaluator.py research/evaluator.py
python examples/research_system.py --evaluator research/evaluator.py --iters 3 --offline
```

## Rules

### Inside a policy

- **Causal information only.** Use past events, the current `View`, and known
  prompt lengths. Do not use future arrivals, output lengths, tool durations,
  trace files, or workload labels. Ids are opaque and rekeyed every run.
- **Sandbox.** No network, file, subprocess, thread, LLM, or wall-clock access;
  use `state.rng` for randomness. Each hook call has 50 ms of CPU time, and
  `state` plus the policy's attributes must pickle to at most 4 MiB.
- **Validity.** An invalid id, evicting an active block, a cap outside 1–16, a
  malformed admission order, an exception, a timeout, or a stall fails the
  evaluation. A policy valid on `public_dev` can still fail on another trace;
  check finalists on `public_val`.

### Campaigns

- **Seeds and hints.** A campaign may start from any policies, including the
  reference, your Part 2 policy, and policies you write, and its prompts may
  contain your own ideas. Keep every seed in `candidates/` and declare the seeds,
  model, prompts, and budget in the campaign's configuration before it starts.
- **Selection.** Your submitted `policy.py` must be selected by an unattended
  campaign from its candidates (a seed may win). You may debug your system and
  change prompts or budgets between campaigns.
- **Lineage** (`log.jsonl` in the campaign's run directory): for every candidate,
  including failures, log `id`, `parent`, `hypothesis`, `file`, `sha256`,
  `objective`, `dev_S`, `by_family`, `tokens`, and `valid`, plus `val_S` for
  finalists.
- **Interventions** (JSONL): log every human action that stops, restarts, or
  changes a campaign, with time, person, action, reason, and affected
  candidates, for example
  `{"t": "2026-11-09T14:05", "who": "ana", "action": "stopped campaign c2", "reason": "every candidate hit the 50 ms limit", "candidates": ["c2-017"]}`.

## Splits and the performance metric

| Split | Traces | Access | Use |
|---|---:|---|---|
| `screen` | 8 | Local; a subset of `public_dev` | Quick checks |
| `public_dev` | 32 (4 per family) | Local, full diagnostics | Development |
| `public_val` | 16 (2 per family) | Local, full diagnostics | Selecting finalists |
| Leaderboard | 16 (2 per family) | Gradescope: overall and per-family speedups | Feedback |
| Hidden test | 24 (3 per family) | Scored once after the deadline | Final grade |

```text
L(P, i)  = mean job latency of policy P on trace i
S_f      = geometric mean over traces in family f of L(reference, i) / L(P, i)
S        = geometric mean of S_f over the eight families
g        = 1 - 1/S
g_eff    = g - sum over families of max(0, 0.90 - S_f)
points   = clip(10 + 20 × g_eff / 0.09, 0, 30)
```

| Speedup S (no family below 0.90) | 0.98 | 1.00 | 1.02 | 1.04 | 1.06 | 1.08 | 1.099 or more |
|---|---|---|---|---|---|---|---|
| Performance credit (% of the grade, of 30%) | 5.5 | 10.0 | 14.4 | 18.5 | 22.6 | 26.5 | 30.0 |

A family below 0.90 subtracts its shortfall from `g`: one family at 0.85 costs
0.05, about 11 percentage points of credit. Invalid or incomplete evaluations earn 0. With two to
four traces per family, scores on different splits differ by a few percent for
the same policy; compare finalists on both public splits and report the spread.

## Research report

Answer Q1–Q6 in `assignment4_report.pdf` after Parts 1 and 2, each below its
question. Cite run ids and candidate SHA-256 hashes so that every claim can be
checked against your logs; put long logs and extra plots in appendices.

**Q1. How does your research system work?** Include a diagram of its LLM calls, deterministic steps, tools, stored state, and the points where a person can intervene. Explain the objective, the feedback agents receive, how invalid candidates are handled, how the final policy is selected, and when a campaign stops. Why this design?

**Q2. What do the agents contribute?** With the same model and token budget, compare (a) your best seed scored alone, (b) a campaign started from `policies/reference.py`, and (c) a campaign started from your seed. Then ablate one component of your system (for example, the diagnostics in the feedback, the screening step, or the candidate selection) and compare against the full system. What do the comparisons show about where the improvements come from?

**Q3. How autonomous was it?** Describe your longest unattended campaign and summarize the intervention log. Show one case where an agent diagnosed a failure and chose a useful next experiment, and one where it could not.

**Q4. Did more spending help?** Plot the best development speedup against tokens and against candidates scored, mark finalists' validation scores, and report the fraction of valid candidates. Where would you stop spending, and why?

**Q5. What policy did the agents find?** Explain the selected policy in your own words, with pseudocode and a short trace example. Disable one mechanism the agents introduced, measure the effect on `public_dev` and `public_val`, and use the result to judge the agents' own explanation of it.

**Q6. Where does it fail?** Report results per family on both public splits and examine one case where the policy does poorly. Predict its hidden-test performance with an uncertainty range. What do your experiments say about the strengths and limits of agents doing systems research?

## Submission and grading (60%)

Upload `policy.py`, an optional `params.json`, and `submission.json` to the
[Gradescope leaderboard](../README.md#gradescope-part-3-leaderboard), and
include them in your team's Canvas archive with `research/`, `candidates/`,
`runs/` (campaign logs, intervention logs, and token receipts), and
`assignment4_report.pdf`.

| Item | Weight |
|---|---:|
| 3.1 Policy performance on the hidden test set | 30% |
| 3.2 Research system and autonomy: Q1 and Q3 | 10% |
| 3.3 Agents' contribution and cost: Q2 and Q4 | 10% |
| 3.4 Policy behavior and limitations: Q5 and Q6 | 10% |
| Extra credit: hidden-test speedup in the top 3% of teams | +5% |
| Extra credit: an exceptional research system or report, at staff discretion | +5% |
