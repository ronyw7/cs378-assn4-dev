# Assignment 4: Deep Research Agent and Block Management

**CS 378: Systems for Generative AI · Fall 2026**  
**Released:** November 3, 2026  
**Due:** November 16, 2026, at 11:59 PM CT  
Complete this assignment with your registered team of two.

- [Overview and setup](#part-0-overview-and-setup)
- [Part 1: Build a deep research agent](#part-1-build-a-deep-research-agent)
- [Part 2: Implement and evaluate LRU](#part-2-implement-and-evaluate-lru)
- [Part 3: Agent-driven policy research](#part-3-agent-driven-policy-research)
- [Evaluation and scoring](#evaluation-and-scoring)
- [Submission](#submission)
- [Grading](#grading)
- [Common questions](#common-questions)

## Part 0: Overview and setup

### Overview

In this assignment, you will build a deep research agent, run it, and
examine its workload from an inference engine's perspective. The agent
breaks a question into subtopics, researches them in parallel, and writes
a report with citations. Completing one report requires several dependent
LLM calls, often separated by tool calls and often sharing long prompt
prefixes.

In Assignment 3, request arrivals were fixed by the replay trace. Here, a
later call may become ready only after an earlier call or tool finishes, so
scheduling decisions affect when subsequent calls can begin. Caching also
matters: a cached prefix can save prefill work on a later call, but retaining
it uses memory that another call may need. Increasing concurrency can improve
batching while also increasing memory pressure. You will study these
trade-offs by implementing an LRU policy and comparing its behavior across
workloads, then build an LLM-driven research system to search for better policies.

By the end of the assignment, you should be able to explain how an agent's
dependencies and prefix reuse affect its completion time, implement a
policy through the supplied API, and evaluate whether a policy improvement
holds across workloads. You will also assess the agentic research system itself:
which feedback helped the agents, how much supervision they needed, and
whether additional experiments justified their token cost.

| Part | Main task | Estimated time | Points |
|---|---|---|---|
| 1 | Implement the research agent and analyze two runs | 4–5 hours | 15 |
| 2 | Implement LRU and study eviction and scheduling | 1 hour | 5 |
| 3 | Build an agentic research system for serving policies and evaluate its results | 14 hours | 70 |
| Individual check | Explain your implementation and results to staff | Scheduled separately | 10 |

Each part has a notebook with implementation guidance, experiments, and
plots. The linked handouts provide the full API contracts and report
requirements.

### Setup

The [README setup instructions](../README.md#getting-started) walk through the
one-time installation, baseline check, and steps for reopening a notebook.

Use Python 3.11 or newer on Linux or macOS and the same project environment for all three parts.
On Windows, use WSL2; the policy runner requires Unix process and pipe support.
From the starter repository's root, run these commands once in a terminal to
create the environment, install the dependencies, and register a notebook kernel:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[agent,notebook,dev]'
python -m ipykernel install --user --name cs378-assignment4 --display-name "CS 378 Assignment 4"
```

The last command registers this environment as **CS 378 Assignment 4** in
Jupyter. Check the installation with the completed FIFO example:

```bash
python -m agentkv.score --policy policies/fifo.py --split data/traces/public_dev/dev-W3-0.json
```

This runs one development trace and should report `Status: VALID`. Its speedup need
not equal 1.0000. The check works before you implement the Part 2 reference,
because the scorer includes fixed results from the staff implementation.

Whenever you open a new terminal to run assignment commands, activate the
environment again. From the repository root, open the first notebook with:

```bash
source .venv/bin/activate
jupyter lab notebooks/part1_deep_research.ipynb
```

Select **CS 378 Assignment 4** as the notebook's kernel before running cells.
Terminal activation does not change an already-open notebook's kernel. Run
`import sys; print(sys.executable)` in a cell to check that the notebook uses
this repository's `.venv/bin/python`.

The simulator runs locally on a CPU and requires no GPU. Part 2 uses one Llama-3-8B/A100 replica; Part 3 uses two independent replicas with separate caches. See the [evaluator configuration](../README.md#evaluator-configuration) for the profile, workload encoding, batch settings, and limitations. Part 1 also
provides a deterministic `FakeLLM` for development without model access.
For Bedrock access, use the credentials and model ids
provided by staff:

| Setting | Purpose |
|---|---|
| AWS credentials or `AWS_PROFILE`, and `AWS_REGION` | Authenticate Bedrock calls |
| `AGENTKV_BEDROCK_MODEL` | Smaller course model for the Part 1 agent |
| Search-provider key | Optional Part 1 web research; see the provider table in notebook section 6 |
| `AGENTKV_RESEARCH_MODEL` | Frontier model for the Part 3 agentic research system |

All required Part 1 work can use `FakeLLM`. Optional web research in section 6 uses a real LLM and a search provider. Part 3 requires actual LLM-generated experiments. The sequential example's
offline mode checks its execution and logging, but does not satisfy that
requirement. Use your team's assigned token budget and contact staff if
Bedrock access is still unavailable at the end of the first week.

### Files you edit

| File or directory | Purpose |
|---|---|
| `research_app/concurrency.py` | Part 1: concurrency cap |
| `research_app/agent.py` | Part 1: graph nodes and routing |
| `notebooks/part1_deep_research.ipynb` | Part 1: prefix-sharing calculation and analysis |
| `policies/reference.py` | Part 2: LRU event handling and eviction |
| `research/` | Part 3: agentic research system, evaluator, prompts, and configuration; create this directory |
| `candidates/` | Policy experiments; create this directory and retain every candidate |
| `runs/` | Agent outputs and research campaign logs |

Read `agentkv/policy.py` for the policy API and complete the Part 2 skeleton in
`policies/reference.py`. Keep the simulator, replay driver, sandbox,
public traces, reference data, and tests unchanged.

## Part 1: Build a deep research agent

**Notebook:** [part1_deep_research.ipynb](../notebooks/part1_deep_research.ipynb)  
**Specification:** [Part 1 handout](part1_handout.md)

Implement an agent that plans a report, researches two to four subtopics in
parallel, checks coverage, and synthesizes a cited response. If a subtopic has
no evidence, the agent performs one follow-up search branch. The supplied
citation checker validates the report, with at most one repair attempt.

![Research agent workflow with parallel branches, follow-up, and repair.](figures/research_workflow.svg)

### What to implement

Start with `AdjustableCap` in `research_app/concurrency.py`. All reports
share this cap on concurrent LLM calls. A retry must retain the original
slot, and lowering the limit must allow running calls to finish before
admitting new ones.

Then complete the graph and node functions in `research_app/agent.py`.
Use the supplied prompt builders and tracing client for every LLM and tool
operation. The client enforces per-report limits of 8 LLM calls, 5 searches,
15 fetches, and 3 attempts per operation. Follow the handout's operation-id
and dependency conventions so that the exported trace represents the
agent's actual dependencies.

```bash
pytest tests/test_part1_concurrency.py
pytest tests/test_part1_agent.py tests/test_part1_citations.py tests/test_part1_llm.py tests/test_part1_trace_export.py
```

### Run and analyze

Run all 16 public questions at C = 2 and C = 8, starting one report every
2 seconds. The notebook writes the results to `runs/c2/` and `runs/c8/`.
Use Bedrock if you have access; otherwise use `FakeLLM` and identify the
backend in your report.

Answer Part 1 Q1–Q3 in the PDF report, keeping each question above its answer. The questions cover:

1. The dependencies and critical path of one report.
2. The fraction of prompt tokens in prefixes seen by earlier calls, both
   overall and by stage. Complete `shared_prefix_share` in the notebook
   to calculate it.
3. The effect of the concurrency cap on latency, makespan, and waiting
   time, including the memory implications for a real GPU.

Support your explanation with measurements or plots. Retain all four files
from each run: `trace.jsonl`, `reports.json`, `summary.json`, and
`replay.json`. Parts 2 and 3 use fixed staff traces, so a Part 1 bug does
not affect your later performance score.

### Optional: research a topic on the web

The agent you have built can also investigate questions on the web. After completing Part 1's implementation and analysis, try a topic of your own in notebook section 6. Keep the same `await agent.research(topic)` interface and choose a supplied Exa, Tavily, Perplexity Search, or OpenAI web-search adapter. Exa and Tavily offer free credits; the [optional extension](part1_handout.md#optional-research-a-topic-on-the-web) lists current offers and setup instructions.

You can use the recorded trace to investigate search and model latency, token usage, repeated sources, and the quality of the resulting evidence. This extension has no required submission and does not affect your grade. The required C = 2 and C = 8 experiments use the fixed local corpus.


## Part 2: Implement and evaluate LRU

**Notebook:** [part2_lru_policy.ipynb](../notebooks/part2_lru_policy.ipynb)  
**Specification:** [Part 2 handout](part2_handout.md)

The engine retains released KV blocks for prefix reuse. Initialize the recency
state, then implement `on_event` and `choose_victim` in `policies/reference.py` so that allocations and
cache hits update recency, releases do not, and eviction removes the least
recently used candidate. Active blocks may appear in your tracking state,
but they are never eligible eviction candidates. Track event delivery order,
including accesses that share a timestamp. This file is also your starting
policy for Part 3; the default concurrency limit is eight calls.

```bash
pytest tests/test_part2_lru.py tests/test_kv_invariants.py
```

Once the tests pass, use the notebook to compare concurrency caps on
document QA and deep research traces. Then examine why evicting the root
of a prefix can prevent reuse of its remaining blocks, and compare
call-level dispatch with oldest-job-first dispatch. Put experimental
variants in `candidates/` so that `policies/reference.py` continues to implement
the specified LRU behavior.

Part 2 is graded on correctness. Its experiments prepare you to interpret
Part 3 results; your LRU policy does not need to beat the baseline.

## Part 3: Agent-driven policy research

**Notebook:** [part3_policy_research.ipynb](../notebooks/part3_policy_research.ipynb)  
**Specification:** [Part 3 handout](part3_handout.md)

Build an agentic research system that develops, evaluates, and selects serving policies. You may extend
`examples/research_system.py`, use an existing framework, or write your own.
At least one campaign must run unattended from a declared starting point
through selection of a final candidate, and the submitted policy must
come from such a campaign.

You may choose any architecture: a single coding agent, a fixed pipeline, parallel agents with different roles, a search tree, or an iterative loop. The system must use LLMs to develop policies, evaluate candidates, and select a final policy. A campaign is one run with a declared starting point, configuration, and budget.

Use your completed Part 2 `policies/reference.py` as a starting policy for your agents. It combines
LRU eviction with first-come, first-served dispatch and admission, a concurrency
limit of eight calls, least-loaded routing, and a 2,048-token batch budget.
`policies/fifo.py` provides a FIFO comparison with the same other settings.
Official speedups use the staff implementation of the same reference contract.
Editing your local policy cannot change the stored comparison results.

Implement an evaluator that calls `score()` and chooses how to use its overall
and per-category results. You may optimize overall speedup, focus an experiment
on one category, or use an objective that penalizes regressions. Start with
`examples/evaluator.py` and connect your version to the supplied sequential example through
`--evaluator research/evaluator.py`. Keep your objective separate from the original
scores, exclude invalid candidates, and report final results for all categories.
See the [Part 3 handout](part3_handout.md#design-your-evaluator) for the interface.

![Research campaign and evaluation splits.](figures/research_system.svg)

Your research system may inspect public traces and evaluation diagnostics, but the
policy being evaluated is restricted to the API's causal observations.
It may use past events, current state, and known input lengths; it may
not inspect future operations, trace files, or workload labels. Policies
run in a sandbox with a 50 ms CPU limit per hook and a 1 MiB state limit.
Network access, LLM calls, file access, subprocesses, threads, and
wall-clock decisions are prohibited inside a policy. Use `state.rng` for
randomness. Invalid decisions, exceptions, timeouts, or incomplete jobs
fail the evaluation.

Retain every candidate, including failures, along with its parent,
hypothesis, file hash, scores, and token usage. Log human interventions
with their time, reason, and affected candidates. You may debug the system
and revise its setup between campaigns; hand-edited policies may be used
only as labeled comparison points.

### Research report

Answer the handout's Q1–Q6 in order, keeping each original question above its answer:

1. **System design:** show the LLM calls, deterministic steps, tools, stored
   state, and human interventions. Explain your evaluator objective, feedback,
   candidate selection, and stopping rules.
2. **Ablation study:** remove, replace, or disable one or two components while
   holding the model and token budget fixed.
3. **Autonomy:** describe the longest unattended campaign and explain how
   the agents handled a failure, using the intervention log as evidence.
4. **Cost:** plot the best development score against token usage and
   candidates scored, mark finalists' validation results, and discuss
   the valid-candidate rate and when further spending stopped helping.
5. **Policy behavior:** explain the final policy in your own words, with
   pseudocode, a trace example, and an experiment disabling one mechanism.
6. **Limitations:** report per-family results, examine an unfavorable case,
   and predict how the policy will perform on the hidden test set.

Cite run ids and candidate hashes so that each claim can be checked
against your logs. Include additional plots and logs in appendices.

## Evaluation and scoring

**Job latency** is the time from its arrival until its final LLM or
tool operation completes. The score compares each trace's mean job latency
with the baseline, takes a geometric mean within each family, then takes a geometric
mean across the eight families. This gives every family equal weight.

```text
L(P, i) = mean job latency for policy P on trace i
S = exp(mean over families(mean over traces(log(L(baseline, i) / L(P, i)))))
g = 1 - 1/S
performance points = clip(25 + 25 × g_eff / 0.10, 0, 50)
```

S = 1 matches the baseline. A valid policy with `g_eff = 0` earns 25 performance
points, and `g_eff = 0.10` earns all 50. If any family's mean job latency is
more than 20% worse than the baseline's, `g_eff` is capped at 0. Invalid or incomplete
evaluations receive no performance credit. The current evaluator uses Vidur.

| Split | Traces | Use |
|---|---:|---|
| `public_dev` | 32 | Develop policies with unrestricted local scoring and diagnostics |
| `public_val` | 16 | Select finalists; account for selection bias when interpreting results |
| Leaderboard | 16 | Compare policies on the Gradescope leaderboard (hidden feedback traces) |
| Hidden test | 24 | Evaluate the final policy after the deadline |

## Submission

### Gradescope: Part 3 leaderboard

Upload the following at the top level:

```text
policy.py       Policy generated and selected by your campaign
params.json     Parameter object, if needed
submission.json   Policy path, parameters path, hash, Python version, and imports
```

For example:

```json
{
  "policy": "policy.py",
  "params": "params.json",
  "candidate_hash": "<SHA-256 of policy.py>",
  "python": "3.12",
  "imports": []
}
```

Set `params` to `null` if no parameter file is needed, record the Python
version you used, and list imports outside the standard library. Obtain
the policy hash from `r.policy_sha256` or `shasum -a 256 policy.py`.

Gradescope runs your policy on the 16 hidden feedback traces using the CPU
simulator. The leaderboard shows overall speedup and speedup for each of the
eight workload categories, rounded to four decimal places. Each trace's
speedup is the fixed reference policy's mean job latency divided by your
policy's mean job latency; category scores are geometric means across
traces, and the overall score gives each category equal weight. The reference
is the supplied `policies/reference.py`, rerun on the same traces: it scores 1.0000×
in every category. Higher is better.

Your submission feedback also shows projected performance points from 0 to
50, rounded to one decimal place; the reference earns 25.0. These points do
not appear on the public leaderboard. A valid policy can receive zero points
while still showing its measured speedups. Invalid policies
receive a failure message and no leaderboard entry. The leaderboard provides
feedback and does not assign your course grade. Include your selected final
policy and its submission metadata in the Canvas archive for final evaluation on the hidden test set.

Full credit requires at least a 10% latency reduction (`S >= 1 / 0.9`,
approximately `1.1111×`) and no category regression beyond the 20% limit.
The Strong benchmark is a comparison point, not the grading threshold.
Follow the [step-by-step submission instructions](../README.md#gradescope-part-3-leaderboard).

### Canvas: code, runs, and report

Submit one archive per team, named `assignment4_<team name>.zip`, containing:

```text
research_app/agent.py
research_app/concurrency.py
policies/reference.py
policy.py and submission.json
params.json                   Optional policy parameters
runs/c2/ and runs/c8/
research/                         Research system and evaluator source, configuration, and prompts
candidates/                   All candidate policies, including failures
runs/...                      Campaign logs and intervention records
assignment4_report.pdf
```

The PDF should include your group name, both team members' names and EIDs,
the pytest summary, Part 1 Q1–Q3 and Part 3 Q1–Q6 with their answers and
requested tables and plots, and token receipts.
Keep each question above its answer and follow the question order, as in
the previous assignments. There is no fixed page limit. Part 2's discussion
questions are practice and need not be included. Include input and output token counts for each LLM call and
your team's Bedrock usage total. Document model versions, reused code,
and any framework revisions needed to reproduce your work.

Both team members should be able to explain the implementation and
results during the individual check. The course late policy applies:
10% of the assignment's total points per day, including weekends, unless
you have made prior arrangements with the instructor.

## Grading

| Item | Points |
|---|---:|
| Part 1 implementation correctness | 9 |
| Part 1 workload analysis | 6 |
| Part 2 LRU and KV-cache invariant tests | 5 |
| Part 3 policy performance (hidden test) | 50 |
| Part 3 report: Q1 and Q3 | 8 |
| Part 3 report: Q2 and Q4 | 6 |
| Part 3 report: Q5 and Q6 | 6 |
| Individual check | 10 |
| **Total** | **100** |

## Common questions

**Do I need a local GPU?** No. Development scoring uses the CPU simulator,
using bundled predictions fitted to existing GPU measurements.

**Why do the tests raise `NotImplementedError`?** The starter includes
unfinished Part 1 and Part 2 functions. Run the relevant tests as you
complete each function; `pytest` runs the full public suite.

**Does changing my Part 1 agent change the Part 3 evaluation?** No. The
evaluation uses fixed staff traces, shared by all teams.

**Can I submit a policy that I edited by hand?** The final Part 3 policy
must be generated and selected in an unattended campaign. Hand-edited
policies are allowed as labeled controls in your experiments.

**Why does a scoring script need a main guard?** `score()` evaluates
traces in a process pool. Put top-level script calls under
`if __name__ == "__main__":`, or pass `workers=1`.

## Acknowledgments

The evaluator uses Vidur canary's vLLM V1 scheduler and cache manager. The Part 1
skeleton is adapted from LangChain's *Deep Research from Scratch*, and
its report-with-sources interface follows Stanford CS329a HW3. The
[repository README](../README.md#data-and-licenses) lists the public
datasets, source revisions, and licenses used in the assignment.
