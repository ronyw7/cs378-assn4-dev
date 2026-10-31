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

Previously in [Assignment 3](../assignment3/README.md), we studied how an LLM serving engine schedules requests efficiently. We implemented techniques such as chunked prefill, request prioritization, and multi-replica routing, and measured their effects on latency and throughput. Those techniques focused on scheduling **individual** LLM requests.

**Agentic applications introduce new challenges.** A single user task typically involves multiple LLM calls and tool invocations, with some running in parallel and others waiting on earlier results. For example, a deep research agent may first plan a task, investigate several subtopics concurrently, and then synthesize its findings. In this setting, optimizing individual calls in isolation is no longer enough—the goal is to minimize end-to-end completion time for the entire user task.

At the same time, agentic workflows create new opportunities for optimization. Related LLM calls often share long prompt prefixes, allowing the inference engine to reuse previously computed KV-cache blocks instead of recomputing prefill from scratch. Retaining cached blocks, however, consumes GPU memory that active requests also need. Serving agentic workloads efficiently therefore requires jointly managing call dependencies, concurrency, request scheduling, replica routing, and KV-cache eviction.

In this assignment, you will study these system tradeoffs first-hand and then explore an automated approach to designing serving policies. After building the research agent and implementing KV-cache eviction and routing policies, you will build an LLM-driven agentic research system that iteratively proposes, implements, and evaluates serving policies. Along the way, you will evaluate both the policies discovered by your agents and the reliability, cost, and generalizability of the automated research process itself.

Parts 2 and 3 run on a CPU simulator built on [Vidur](https://github.com/microsoft/vidur/tree/25e0082dbbfb206fb0477c3ebbededa7ead78949) that models Llama-3-8B on one or two A100 GPUs, each with its own request queue and KV cache. The figure below shows how calls move from the application into the inference engine, with blue labels marking the hooks your policy can implement:

![Policy decisions in the application and inference engine.](docs/figures/system.svg)

- **Concurrency and dispatch:** A **job** (such as generating a research report) consists of dependent LLM and tool calls. Once a call's dependencies finish, it enters **ready calls**. Your policy controls how many calls can be in flight (`target_concurrency`), which ready call is dispatched next (`dispatch_order`), and which GPU replica receives it (`route_call`).
- **Replica scheduling and KV-cache management:** Each replica maintains its own waiting queue, mixed prefill/decode batch, and KV cache. Your policy controls admission order (`admission_order`), batch token budget (`batch_token_budget`), preemption under memory pressure (`choose_preempted`), and cached block eviction (`choose_victim`), while observing system events through `on_event`.

## Setup

Use Python 3.11 or newer on Linux or macOS. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[agent,notebook,dev]'
python -m ipykernel install --user --name cs378-assignment4 --display-name "CS 378 Assignment 4"
```

Verify the simulator installation:

```bash
python -m agentkv.score --policy policies/fifo.py --split data/traces/public_dev/dev-W3-0.json

# Should print `Status: VALID`
```

Launch Jupyter (`jupyter lab`), open the notebooks, and select the **CS 378 Assignment 4** kernel. Run `pytest` to execute the public test suite. All public tests should pass once you complete the Part 1 and Part 2 `TODO`s.

### Model access

Parts 1 and 3 call models on Amazon Bedrock. Set `AWS_REGION` and your AWS credentials (or `AWS_PROFILE`), then set `AGENTKV_BEDROCK_MODEL` for the Part 1 agent and `AGENTKV_RESEARCH_MODEL` for your Part 3 research system. The model ids and your team's token budget are on the course page. Part 1 can also run entirely on the offline `FakeLLM`, but Part 3 needs Bedrock. Contact staff if Bedrock access does not work by the end of the first week.

## Part 1: Build and analyze a deep research agent

[Notebook](notebooks/part1_deep_research.ipynb) · [Handout](docs/part1_handout.md) · About 10–12 hours

In this part, you will build a LangGraph agent that splits a question into subtopics, researches them in parallel over a local Wikipedia corpus, and writes a report. You will then measure the workload it sends to the model: what determines when a report finishes, how much prompt content repeats, and how much KV memory concurrency costs.

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
- **2.3** Implement cache-aware routing across two GPUs, in the same file.
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
- 2.3 Cache-aware routing (correctness): 3%
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