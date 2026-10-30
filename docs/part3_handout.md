# Part 3: Agent-driven policy research

Allow about 14 hours for this part (70 points: 50 for policy performance, 20 for the research report). Use [`notebooks/part3_policy_research.ipynb`](../notebooks/part3_policy_research.ipynb) to inspect workloads and analyze campaign logs, and keep your research system (`research/`) and generated candidate policies (`candidates/`) in separate code files.

## Introduction

In this part, you will design and implement your own agentic research system in `research/` that uses LLM agents to propose hypotheses, write candidate serving policies, evaluate them on the simulator, and select a final policy. Your goal is not only to improve speedup, but also to understand:

1. **How your research system behaves:** Which architectural choices, diagnostic feedback, and search objectives help agents produce valid, higher-performing policies, and where agents fail or waste tokens.
2. **What policy your agents discovered and why it works:** What scheduling, routing, concurrency, and eviction mechanisms your agents invented, which mechanisms genuinely drive the speedup (verified by ablating them), and on which workloads the policy still struggles.

We provide a minimal sequential script (`examples/research_system.py`) to demonstrate Bedrock calls, evaluation, and logging, which you can extend or use as a comparison baseline. You may choose any architecture for your system in `research/`: a single coding agent with diagnostic feedback and repair, a multi-agent pipeline (e.g., planner/coder/critic), a search tree, or an evolutionary loop (including adapting open-source frameworks such as OpenEvolve, ShinkaEvolve, or CORAL; record the framework revision and license if used).

![One possible research system and the four evaluation splits.](figures/research_system.svg)

## Baseline policy and two-replica setup

Part 3 evaluates policies on **two independent replicas** of Llama-3-8B (each on a simulated NVIDIA A100 80 GB GPU with its own queue and 16-token KV cache; see the [README](../README.md#evaluator-configuration)).

Your completed Part 2 reference policy (`policies/reference.py`) serves as the starting policy for your agents: it uses `cap = 8` across both replicas, least-loaded routing, FCFS dispatch and admission, a 2,048-token batch budget, and LRU eviction in event delivery order. (Official speedups are always measured against fixed results from the staff reference implementation, so editing `policies/reference.py` does not change the scoring denominator; `policies/fifo.py` is also provided as a FIFO baseline.)

## The eight workload families

![One job from each workload family: deep research, long-context chat, a coding agent, document QA, customer-service agents, and reasoning chat, plus the two shared-engine mixes.](figures/families.svg)

Each of the eight workload families has equal weight in the overall speedup score `S`:

| Family | Workload pattern | Core systems question |
|---|---|---|
| **D** Deep research | Part 1 agent reports under four conditions: high reuse, tight memory, topic/corpus shift, and bursty arrivals with slow tools. | Can retaining shared documents avoid repeated prefill without starving parallel branches on a report's critical path? |
| **W1** Long-context chat | Users ask questions about long uploaded documents (Mooncake Kimi trace); several users share a document. | When long prompts saturate GPU memory, how many calls should run at once, and which documents are worth keeping? |
| **W2** Coding agent | Multi-step SWE-bench OpenHands sessions whose context grows every turn. | How should long private prefix chains compete with shared prefixes? |
| **W3** Document QA | Short questions over long documents with skewed popularity (LooGLE). | Which documents should stay cached, and what concurrency cap works best when every call is a long prefill? |
| **W4** Customer-service agents | Shared policy prompt with many short tool-calling turns (τ²-bench). | How should one shared system prompt and many growing conversations share memory and replicas? |
| **W5** Reasoning chat | Outputs ranging from a few hundred to 16,000 tokens (Qwen Bailian trace); output length is unknown at arrival. | How should the engine manage memory as outputs grow, and which call should it preempt under pressure? |
| **M1** Shared: Doc QA + Chat | `W3` document QA and `W1` long-context chat sharing one engine. | How should a policy balance workloads that prefer different concurrency caps? |
| **M2** Shared: Doc QA + Reasoning | `W3` document QA and `W5` long-output reasoning sharing one engine. | How should the engine balance long prefills against decodes that hold KV blocks for extended periods? |

## Scoring and evaluator design

Use `agentkv.score` from Python or the command line to evaluate a candidate in the sandbox:

```python
from agentkv import score

if __name__ == "__main__":
    r = score("candidates/candidate_01.py", split="public_dev", backend="sim")
    print(r.valid, r.speedup, r.points, r.by_family)
    print(r.summary())       # pass details=True (or --details on CLI) for per-trace diagnostics
```

```bash
python -m agentkv.score --policy candidates/candidate_01.py --split public_dev --backend sim --json out.json
python -m agentkv.score --policy candidates/candidate_01.py --split data/traces/public_dev/dev-W3-0.json
```

### Implement your research evaluator (`research/evaluator.py`)

Copy `examples/evaluator.py` to `research/evaluator.py` and customize `evaluate(policy_path, *, split="public_dev")`:

```python
from agentkv import score

def evaluate(policy_path, *, split="public_dev"):
    result = score(policy_path, split=split)
    feedback = result.to_json()
    feedback["objective"] = result.speedup if result.valid else None
    return feedback
```

Your evaluator controls two things:
1. **`objective` (`None` if `not result.valid`):** How your research system ranks valid candidates (e.g., overall `result.speedup`, a family-specific score like `result.by_family["W3"]`, or a penalty-adjusted objective that guards against the 20% per-family regression limit).
2. **Agent feedback:** Which per-family scores, cache/queueing diagnostics (`r.by_trace`), or violation messages your system feeds into the next LLM prompt. Keep the original `speedup`, `by_family`, and `valid` fields intact in the returned dict.

Test the starter pipeline offline before running a Bedrock campaign:

```bash
mkdir -p research
cp examples/evaluator.py research/evaluator.py
python examples/research_system.py --evaluator research/evaluator.py --iters 3 --offline
```

## Policy and campaign rules

### Sandbox rules for policies (`policy.py`)

While your outer research system in `research/` may inspect public traces, family labels, and diagnostics, any policy running inside `score()` is strictly sandboxed:

- **Causal information only:** Use only past events (`on_event`), current `View` state, and known prompt lengths. Do not read future arrivals, output lengths, tool delays, trace files, or workload labels. Request, job, stage, and prefix IDs are rekeyed every run.
- **Resource limits:** No network, file, subprocess, thread, LLM, or wall-clock access. Use `state.rng` for randomness. Each hook call has a **50 ms** CPU limit, and pickled state (`state` + policy attributes) must stay within **1 MiB**.
- **Strict validity:** Returning an invalid ID, evicting an active block, returning a concurrency cap outside `1..16`, returning an invalid `admission_order`, raising an exception, timing out, or stalling all ready calls fails evaluation immediately.

### Autonomy and provenance rules for campaigns

- **Unattended campaign:** A campaign is one run of your research system from a declared starting point, configuration, and budget to final candidate selection. Your submitted `policy.py` must be generated and selected during an unattended campaign. Hand-written policies may only be used as labeled baselines for comparison.
- **Candidate lineage (`log.jsonl`):** Retain every generated candidate in `candidates/` (including failed attempts) and log its `id`, `parent`, `hypothesis`, `file`, `sha256`, `objective`, `dev_S`, `by_family`, `tokens`, and `valid` status (plus `val_S` for finalists evaluated on `public_val`).
- **Intervention log:** You may debug your system and change prompts or budgets between campaigns. Log any human action that stops, restarts, or alters a campaign (`JSONL` with timestamp, person, action, reason, and affected candidates):
  `{"t": "2026-11-09T14:05", "who": "ana", "action": "stopped campaign c2", "reason": "every candidate hit the 50 ms CPU limit", "candidates": ["c2-017", "c2-018"]}`.

## Evaluation splits and grading formula

| Split | Size | Access | Purpose |
|---|---|---|---|
| `public_dev` | 32 traces (4/family) | Full local scoring and diagnostics | Develop and search for policies |
| `public_val` | 16 traces (2/family) | Full local scoring and diagnostics | Select finalists (account for selection bias) |
| Leaderboard | 16 traces (2/family) | Gradescope leaderboard (overall and per-family speedups; projected points in private feedback) | Check generalization on held-out feedback traces |
| Hidden test | 24 traces (3/family) | Scored once after the deadline on your submitted `policy.py` (combines conditions such as bursty arrivals, tight memory, and new workload mixes) | Final performance grade (50 points) |

```text
L(P, i) = mean job latency of policy P on trace i
S       = exp( mean over families ( mean over traces ( log( L(baseline, i) / L(P, i) ) ) ) )
g       = 1 - 1/S
points  = clip( 25 + 25 × g_eff / 0.10, 0, 50 )
```

| Speedup S | 0.95 | 1.00 (baseline) | 1.02 | 1.05 | 1.08 | 1.112 or more |
|---|---|---|---|---|---|---|
| Performance points | 11.8 | 25.0 | 29.9 | 36.9 | 43.5 | 50.0 |

Here, `g_eff = g`, except that if any family's mean job latency is more than 20% worse than the baseline's (`by_family[f] < 1 / 1.2 ≈ 0.8333`), `g_eff` is capped at `0` (at most 25 points). Invalid or incomplete runs receive 0 points. (The Strong benchmark row on the leaderboard is a comparison point, not the 50-point threshold; full credit requires `S >= 1 / 0.9 ≈ 1.1111×` with no >20% family regression.)

## Research report (20 points)

Answer Q1–Q6 in the notebook, then include each original question and your answer in `assignment4_report.pdf` after Part 1. Support your claims with run IDs, candidate SHA-256 hashes, tables, and plots (place long logs or extra plots in appendices):

**Q1. How does your agentic research system work?** Include a diagram showing its LLM calls, deterministic steps, tools, stored state, and points where a person can intervene. Explain your evaluator's objective, the feedback given to agents, how invalid candidates are handled, and how the system selects a final policy and decides when to stop. Explain why you chose this architecture.

**Q2. What did your ablation study show?** Ablate one or two components or design choices in your research system while holding the model and token budget fixed. Describe what you removed, replaced, or disabled, show the results, and explain what the comparison reveals. For example, remove evaluation feedback, disable shared memory, or replace a candidate-selection strategy. The supplied sequential system can serve as one comparison.

**Q3. How autonomous was it in practice?** Describe the longest unattended campaign and summarize the intervention log. Show one case where an agent diagnosed a failure and chose a useful follow-up experiment, or explain why it could not.

**Q4. Did more spending help?** Plot the best development speedup against tokens used and against candidates evaluated. Mark your finalists' validation scores, report the fraction of candidates that were valid, and explain where you would stop spending.

**Q5. What policy did the agents find?** Explain the policy in your own words, with pseudocode and a short trace example. Why should its decisions reduce mean job latency? Disable one discovered mechanism, measure the effect, and use the results to assess the agents' explanation.

**Q6. Where does it fail, and what does that say about agents doing systems research?** Report results for each workload category and examine one case where the policy performs poorly. Predict how it will perform on the hidden test set. What do your experiments show about the strengths and limitations of using agents for this research?

## Submission and grading (70 points)

Upload `policy.py`, optional `params.json`, and `submission.json` to the [Gradescope leaderboard](../README.md#gradescope-part-3-leaderboard), and include them in your team's Canvas archive alongside `research/`, `candidates/`, `runs/` (campaign logs, intervention logs, and Bedrock token receipts), and `assignment4_report.pdf`.

| Item | Points |
|---|---|
| Policy performance on the hidden test set | 50 |
| Report: Q1 and Q3 | 8 |
| Report: Q2 and Q4 | 6 |
| Report: Q5 and Q6 | 6 |
