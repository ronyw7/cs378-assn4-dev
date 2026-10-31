# Assignment 4: Deep Research Agent and Block Management

**Released:** 11/03/2026  
**Due:** 11/16/2026 11:59 PM CT

CS 378: Systems for Generative AI, Fall 2026. Complete this assignment with your registered team of two.

- [Overview](#overview)
- [Setup](#setup)
- [Part 1: Build and analyze a deep research agent](#part-1-build-and-analyze-a-deep-research-agent)
- [Part 2: KV-cache eviction and routing](#part-2-kv-cache-eviction-and-routing)
- [Part 3: Agent-driven policy research](#part-3-agent-driven-policy-research)
- [Submission](#submission)
- [Grading](#grading)
- [Acknowledgments](#acknowledgments)

## Overview

In Assignment 3, we studied how an LLM serving engine schedules individual requests. Agentic applications change the problem. One user task issues many dependent LLM and tool calls, some in parallel, and the goal is to finish the whole task quickly. These calls also repeat long prompt prefixes, so the engine can reuse cached KV blocks instead of recomputing them, but cached blocks take memory that running calls need.

In this assignment, you will:

1. Build a deep research agent and measure the workload it creates.
2. Write KV-cache eviction and routing policies for a simulated inference engine.
3. Build an LLM-driven system that searches for better policies, and measure what its agents add beyond your own designs.

Parts 2 and 3 run on a CPU simulator built on [Vidur](https://github.com/microsoft/vidur/tree/25e0082dbbfb206fb0477c3ebbededa7ead78949) that models Llama-3-8B on one or two A100 GPUs, each with its own queue and KV cache. A policy decides how many calls run at once, which ready call runs next and on which GPU, and which cached blocks to keep.

![Policy decisions in the application and inference engine.](docs/figures/system.svg)

## Setup

Use Python 3.11 or newer on Linux or macOS (on Windows, use WSL2). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[agent,notebook,dev]'
python -m ipykernel install --user --name cs378-assignment4 --display-name "CS 378 Assignment 4"
```

Check the simulator; it should print `Status: VALID`:

```bash
python -m agentkv.score --policy policies/fifo.py --split data/traces/public_dev/dev-W3-0.json
```

Open the notebooks with `jupyter lab` and select the **CS 378 Assignment 4** kernel. `pytest` runs the public tests, which all pass once every Part 1 and Part 2 TODO is complete.

### Model access

| Variable | Used by |
|---|---|
| `AWS_REGION` and AWS credentials (or `AWS_PROFILE`) | Bedrock calls in Parts 1 and 3 |
| `AGENTKV_BEDROCK_MODEL` | The Part 1 agent with `--backend bedrock` |
| `AGENTKV_RESEARCH_MODEL` | `examples/research_system.py` in Part 3 |

The values and your team's token budget are on the course page. Part 1 can run entirely on the offline `FakeLLM`; Part 3 needs Bedrock. Contact staff if Bedrock access does not work by the end of the first week.

## Part 1: Build and analyze a deep research agent

[Notebook](notebooks/part1_deep_research.ipynb) · [Handout](docs/part1_handout.md) · About 10–12 hours

In this part, you will build a LangGraph agent that splits a question into subtopics, researches them in parallel over a local Wikipedia corpus, and writes a cited report. You will then measure the workload it sends to the model: what determines when a report finishes, how much prompt content repeats, and how much KV memory concurrency costs.

![Research agent workflow with parallel branches, follow-up, and repair.](docs/figures/research_workflow.svg)

- **1.1** Implement the concurrency cap in `research_app/concurrency.py`.
- **1.2** Implement the research graph in `research_app/agent.py`.
- **1.3** Implement the workload analysis functions in `research_app/analysis.py`.
- **1.4** Run the agent on the 16 public questions at C = 2, at C = 8, and at C = 8 with tool failures, replay your C = 8 run in the simulator, and answer Q1–Q4.

Parts 2 and 3 use staff traces, so your Part 1 code affects only Part 1.

## Part 2: KV-cache eviction and routing

[Notebook](notebooks/part2_lru_policy.ipynb) · [Handout](docs/part2_handout.md) · About 7–8 hours

In this part, you will write cache policies for the simulated engine and measure them on eight workload families.

- **2.1** Implement LRU eviction in `policies/reference.py`. Part 3 speedups are measured against the staff version of this policy.
- **2.2** Implement eviction that keeps prefix trees reusable, in `policies/prefix_policy.py`.
- **2.3** Implement cache-affinity routing across two GPUs, in the same file.
- **2.4** Run the notebook's experiments and answer Q1–Q3.

## Part 3: Agent-driven policy research

[Notebook](notebooks/part3_policy_research.ipynb) · [Handout](docs/part3_handout.md) · About 15–18 hours

In this part, you will build a system in which LLM agents propose, implement, and evaluate serving policies, and you will submit the policy it selects.

![One possible research system and the evaluation splits.](docs/figures/research_system.svg)

1. Build your research system in `research/`. `examples/research_system.py` is a minimal loop to start from; you can also adapt a framework such as OpenEvolve, ShinkaEvolve, or CORAL.
2. Run campaigns. A campaign can start from any policies, including your Part 2 policy, and its prompts can include your own ideas.
3. Select a final policy with an unattended campaign and submit it to the Gradescope leaderboard.
4. Answer Q1–Q6. Q2 needs three campaigns with the same model and token budget (from the reference, from your best policy, and one ablation), so plan your budget for them.

Your selected policy is scored once on 24 hidden traces. Matching the reference earns 10% of the grade, and a speedup of 1.099 or more with no workload family below 0.90 earns the full 30%. A policy that fails on any hidden trace earns 0.

## Submission

### Gradescope: Part 3 leaderboard

Upload `policy.py` (the candidate your campaign selected, unmodified) and, if your policy takes parameters, `params.json`, at the top level rather than inside a folder. Use your team name as the leaderboard name. The leaderboard shows your overall and per-family speedups on 16 hidden traces next to staff benchmark policies, and your feedback shows projected performance credit. It does not determine your grade.

### Canvas: code, runs, and report

One team member submits `assignment4_<team name>.zip` containing:

```text
research_app/agent.py, research_app/concurrency.py, research_app/analysis.py
policies/reference.py, policies/prefix_policy.py
runs/c2/, runs/c8/, runs/c8-faults/
research/                              your research system
candidates/                            your seed policies
runs/<campaign>/                       each campaign's candidates (including failures), logs, and intervention log
policy.py, params.json (if used)       the files you submitted to Gradescope
assignment4_report.pdf
```

The `policy.py` in this archive is the one scored on the hidden test set. The report contains your team name, both members' names and EIDs, the `pytest` summary, your answers to Part 1 Q1–Q4, Part 2 Q1–Q3, and Part 3 Q1–Q6 in order, each below its question, the Bedrock model ids and total tokens you used, and the AI Use Disclosure required by the [course policies](https://utcs378.github.io/fall26/logistics).

## Grading

Total: 100% (+10% extra credit)

**Part 1 — Deep research agent and workload analysis: 20%**

- 1.1 Concurrency cap (correctness): 3%
- 1.2 Research graph (correctness): 7%
- 1.3 Workload analysis functions (correctness): 4%
- 1.4 Runs and answers to Q1–Q4: 6%

**Part 2 — KV-cache eviction and routing: 20%**

- 2.1 LRU eviction (correctness): 5%
- 2.2 Prefix-aware eviction (correctness): 4%
- 2.3 Cache-affinity routing (correctness): 3%
- 2.4 Experiments and answers to Q1–Q3: 8%

**Part 3 — Agent-driven policy research: 60%**

- 3.1 Policy performance on the hidden test set: 30%
- 3.2 Research system and autonomy (Q1, Q3): 10%
- 3.3 Agents' contribution and cost (Q2, Q4): 10%
- 3.4 Policy behavior and limitations (Q5, Q6): 10%

**Extra credit: up to +10%**

- Hidden-test speedup in the top 3% of teams: +5%
- An exceptional research system or report, at staff discretion: +5%

Correctness items are graded by staff tests of the same contracts as the public tests, on new inputs.

## Acknowledgments

The simulator adapts Vidur's vLLM V1 scheduler and cache manager. The Part 1 agent is adapted from LangChain's *Deep Research from Scratch* (MIT). The workload traces are built from public datasets; [data/README.md](data/README.md) lists their sources and licenses.
