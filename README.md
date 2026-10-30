# Assignment 4: Deep Research Agent and Block Management

**Released:** 11/03/2026  
**Due:** 11/16/2026 11:59 PM CT

CS 378: Systems for Generative AI, Fall 2026. Complete this assignment with your registered team of two.

- [Preparation](#preparation)
- [Part 0: Overview and Setup](#part-0-overview-and-setup)
- [Evaluator configuration](#evaluator-configuration)
- [Part 1: Build a deep research agent](#part-1-build-a-deep-research-agent)
- [Part 2: Write an LRU policy](#part-2-write-an-lru-policy)
- [Part 3: Agent-driven policy research](#part-3-agent-driven-policy-research)
- [How to test](#how-to-test)
- [What not to modify](#what-not-to-modify)
- [Submission](#submission)
- [Grading](#grading)
- [FAQ](#faq)
- [Data and licenses](#data-and-licenses)
- [Acknowledgments](#acknowledgments)



## Preparation

Use the starter repository provided for Assignment 4, and complete the [environment setup](#getting-started) below before running the notebooks or tests. Each part includes a Jupyter notebook that guides you through the implementation and experiments, along with a handout containing the full specification. A separate [assignment overview](docs/assignment4_overview.md) is also available.

## Part 0: Overview and Setup



### Overview

Previously in [Assignment 3](../assignment3/README.md), we studied how an LLM serving engine schedules requests efficiently. We implemented techniques such as chunked prefill, request prioritization, and multi-replica routing, and measured their effects on latency and throughput. Those techniques focused on scheduling **individual** LLM requests.

**Agentic applications introduce new challenges.** A single user task typically involves multiple LLM calls and tool invocations, with some running in parallel and others waiting on earlier results. For example, a deep research agent may first plan a task, investigate several subtopics concurrently, and then synthesize its findings. In this setting, optimizing individual calls in isolation is no longer enough—the goal is to minimize end-to-end completion time for the entire user task.

At the same time, agentic workflows create new opportunities for optimization. Related LLM calls often share long prompt prefixes, allowing the inference engine to reuse previously computed KV-cache blocks instead of recomputing prefill from scratch. Retaining cached blocks, however, consumes GPU memory that active requests also need. Serving agentic workloads efficiently therefore requires jointly managing call dependencies, concurrency, request scheduling, replica routing, and KV-cache eviction.

In this assignment, you will study these system tradeoffs first-hand and then explore an automated approach to designing serving policies. After building the research agent and implementing a baseline LRU cache policy, you will build an LLM-driven agentic research system that iteratively proposes, implements, and evaluates serving policies. Along the way, you will evaluate both the policies discovered by your agents and the reliability, cost, and generalizability of the automated research process itself.

The assignment has three parts:

- **Part 1: Build a deep research agent.** Implement a LangGraph agent that decomposes research questions into parallel tasks, calls search and fetch tools, and synthesizes cited reports. Analyze its execution DAG, prefix reuse, and concurrency tradeoffs.
- **Part 2: Understand cache management.** Implement LRU cache eviction in `policies/reference.py` and explore how eviction, concurrency caps, and dispatch order affect end-to-end job latency across workloads.
- **Part 3: Automate systems research.** Build an LLM-driven agentic research system that iteratively proposes, implements, and evaluates serving policies. Select a final policy from an unattended campaign and analyze both the discovered optimizations and the research process.

Your Part 1 implementation affects only your Part 1 grade; Parts 2 and 3 replay fixed staff traces.

The figure below illustrates how LLM calls flow from the application into the inference engine, with blue labels indicating the methods your policy can implement:

![Policy decisions in the application and inference engine.](docs/figures/system.svg)

- **Jobs and ready calls:** A **job** represents a complete user task (such as generating a research report) composed of multiple dependent LLM and tool calls. Once a call's dependencies finish, it enters the pool of **ready calls**.
- **Dispatch and routing:** Before calls reach a GPU replica, your policy controls how many calls can be in flight (`target_concurrency`), which ready call leaves the application next (`dispatch_order`), and which replica receives it (`route_call`).
- **Replica scheduling and KV-cache management:** Each replica maintains its own waiting queue, mixed prefill/decode batch, and KV cache. Within each replica, your policy decides which waiting calls are admitted (`admission_order`), how many tokens a batch may process (`batch_token_budget`), which running call to preempt when memory is tight (`choose_preempted`), and which cached KV block to evict (`choose_victim`).
- **System events:** As calls complete or cached blocks are accessed, the simulator emits **events** through `on_event`, allowing your policy to track system state and adapt future decisions.

All policies are evaluated on identical workloads and hardware settings using a local CPU simulator. See [Evaluator configuration](#evaluator-configuration) below for the model, hardware, and timing assumptions used in each part.

### Evaluator configuration

Parts 2 and 3 evaluate policies by replaying fixed workloads. Each workload specifies jobs, their LLM and tool calls, and the dependencies between calls. While prompts, output lengths, and tool delays remain fixed, your policy determines when and where the work runs.


| Setting            | Part 2: LRU implementation                                             | Part 3: Policy research                                                      |
| ------------------ | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Simulated model    | Llama-3-8B                                                             | Llama-3-8B on both replicas                                                  |
| Simulated hardware | One NVIDIA A100 80 GB                                                  | Two NVIDIA A100 80 GB GPUs                                                   |
| Model placement    | One model replica on one GPU                                           | One model replica on each GPU                                                |
| Model parallelism  | Tensor parallelism 1; pipeline parallelism 1                           | Tensor parallelism 1; pipeline parallelism 1 per replica                     |
| KV cache           | One cache, with 16-token blocks                                        | A separate cache on each replica, with 16-token blocks                       |
| Scheduling         | Vidur's vLLM V1 scheduler, which can batch prefill and decode together | The same scheduler on each replica, with a routing decision before admission |
| Timing             | Vidur predictions fitted to existing A100 profiling measurements       | The same profiling measurements for both replicas                            |


- **How replay works:** The simulator runs on your CPU without loading model weights, using [Vidur](https://github.com/microsoft/vidur/tree/25e0082dbbfb206fb0477c3ebbededa7ead78949) latency predictions fitted to [Llama-3-8B A100 profiling data](https://github.com/microsoft/vidur/tree/25e0082dbbfb206fb0477c3ebbededa7ead78949/data/profiling/compute/a100/meta-llama/Meta-Llama-3-8B) (metadata in [`agentkv/data/vidur/profile.json`](agentkv/data/vidur/profile.json)). Each evaluation case pairs a workload trace with fixed engine settings (such as KV-cache quota and arrival times) starting from an empty cache. As dependencies finish, the evaluator releases ready LLM calls, applies your policy's decisions, advances a simulated clock by each batch's predicted execution time, and triggers downstream tool or LLM calls. **Job latency** is the total time from arrival until a job's final operation completes (including execution, queueing, and tool waiting), and **speedup** compares your policy's mean job latency against the reference policy on the same case.
- **Single vs. multi-replica routing:** Part 2 (`viz.record`, or `score(..., replicas=1)`) uses a single replica. Part 3 (`score`, which defaults to `replicas=2`) uses two independent replicas—two complete copies of the model, each with its own request queue and KV cache. Because a prefix cached on one replica is unavailable on the other, routing introduces a key tradeoff: a call can wait for the replica that already holds its prefix, or go to a less busy replica and recompute the prompt.
- **Engine limits and Bedrock separation:** Across all replicas, `target_concurrency` can cap unfinished calls between 1 and 16 (and each replica admits at most 16 calls). Policies can choose a `batch_token_budget` of 512, 1024, 2048 (default), or 4096 tokens. Note that the simulated Llama-3-8B engine is separate from the live Bedrock models you call to generate reports in Part 1 or propose policy code in Part 3.

<a id="getting-started"></a>

### Setup

All three parts share a single local Python environment (**Python 3.11 or newer** on Linux, macOS, or Windows via WSL2). No GPU or cloud credentials are needed for initial setup.

**1. Create the virtual environment and register the Jupyter kernel (run once from the repository root):**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[agent,notebook,dev]'
python -m ipykernel install --user --name cs378-assignment4 --display-name "CS 378 Assignment 4"
```

**2. Verify the installation:**

Run the supplied FIFO policy on a single development trace to confirm the simulator works:

```bash
python -m agentkv.score --policy policies/fifo.py --split data/traces/public_dev/dev-W3-0.json
```

The output should finish with `Status: VALID`. (You will implement the LRU reference policy in `policies/reference.py` during Part 2; once complete, it will score `S = 1.0000` against the saved staff reference results and serve as the starting point for Part 3.)

**3. Launch Jupyter and select the kernel:**

Whenever you open a new terminal, activate `.venv` from the repository root before running commands or starting Jupyter:

```bash
source .venv/bin/activate
jupyter lab notebooks/part1_deep_research.ipynb
```

In Jupyter, select **CS 378 Assignment 4** as the notebook kernel (activating `.venv` in a terminal does not change the kernel of an already-open notebook). You can verify the active Python interpreter inside a notebook cell with:

```python
import sys
print(sys.executable)  # should end with .venv/bin/python
```

### Model access

For runs that call a live LLM on Amazon Bedrock, configure the following environment variables using the values posted on the course page:


| Variable                         | Used by                                  | Value                                              |
| -------------------------------- | ---------------------------------------- | -------------------------------------------------- |
| `AWS_REGION` and AWS credentials | Bedrock calls in Parts 1 and 3           | Course AWS account credentials (or `AWS_PROFILE`). |
| `AGENTKV_BEDROCK_MODEL`          | The Part 1 agent (`--backend bedrock`)   | Smaller course model ID.                           |
| `AGENTKV_RESEARCH_MODEL`         | `examples/research_system.py`            | Frontier model ID for Part 3 policy research.      |
| `EXA_API_KEY` (optional)         | Optional web search in Part 1 section 6  | API key for Exa (or another search provider).      |


All required Part 1 runs can use `FakeLLM` if needed, whereas the optional web-research extension in section 6 uses Bedrock and a live search provider. Part 3 requires Bedrock for your actual research campaign (`--offline` in `examples/research_system.py` only tests execution and logging). Check the course page for your team's LLM token budget, and contact staff if you do not have working Bedrock access by the end of the first week.

### Files you edit


| File or directory                     | Purpose                                                                                     |
| ------------------------------------- | ------------------------------------------------------------------------------------------- |
| `research_app/concurrency.py`         | Part 1: concurrency cap (`AdjustableCap`)                                                   |
| `research_app/agent.py`               | Part 1: research graph nodes and routing                                                    |
| `notebooks/part1_deep_research.ipynb` | Part 1: prefix-sharing calculation and workload analysis                                    |
| `policies/reference.py`               | Part 2: LRU cache eviction and event handling                                               |
| `research/`                           | Part 3: agentic research system, evaluator, prompts, and configuration (create this folder) |
| `candidates/`                         | Part 2 & 3: policy experiments and all campaign candidates (create this folder)             |
| `runs/`                               | Part 1 & 3: agent trace outputs and campaign logs                                           |


Read `agentkv/policy.py` for the policy API and complete the reference-policy skeleton in `policies/reference.py` during Part 2. The rest of `research_app/` provides the corpus, tools, tracing client, prompts, and citation checker; `examples/research_system.py` and `examples/evaluator.py` provide an example research system and evaluator; and `docs/trace_format.md` documents the replay trace format.

### Deliverables


| Part             | Deliverable                                                                      | Estimated time       | Points |
| ---------------- | -------------------------------------------------------------------------------- | -------------------- | ------ |
| 1                | Completed research agent, runs at C = 2 and C = 8, and answers to Q1–Q3          | 4–5 hours            | 15     |
| 2                | Correct LRU implementation in `policies/reference.py`                            | 1 hour               | 5      |
| 3                | Selected policy, agentic research system, campaign records, and answers to Q1–Q6 | 14 hours             | 70     |
| Individual check | Explain your implementation and results to staff                                 | Scheduled separately | 10     |


Submit your Part 3 policy to the Gradescope leaderboard for feedback, and submit your code, runs, final policy, and report to Canvas (see [Submission](#submission) for the required archive layout).

---



## Part 1: Build a deep research agent

**Notebook:** [part1_deep_research.ipynb](notebooks/part1_deep_research.ipynb) · **Full specification:** [Part 1 handout](docs/part1_handout.md)

Implement a LangGraph agent that decomposes a question into two to four subtopics, researches them in parallel, checks subtopic coverage (running one follow-up branch if needed), synthesizes a cited report, and validates citations (with at most one repair attempt).

![Research agent workflow with parallel branches, follow-up, and repair.](docs/figures/research_workflow.svg)

### What to implement

- **Concurrency cap (`research_app/concurrency.py`):** Implement `acquire`, `release`, and `set_limit` on `AdjustableCap`. All concurrent reports share this cap on in-flight LLM calls, and retries retain their original slot. Lowering the limit must let running calls finish while new calls wait until `in_flight < limit`.
- **Research graph (`research_app/agent.py`):** Implement the graph nodes, routing functions, and `build_graph`. Route every LLM and tool call through the supplied prompt builders and tracing client, which enforces per-report budgets of 8 LLM calls, 5 searches, 15 fetches, and 3 attempts per operation. Follow the handout's operation-ID and dependency rules so the exported trace accurately reflects the execution DAG.

```bash
pytest tests/test_part1_concurrency.py
pytest tests/test_part1_agent.py tests/test_part1_citations.py tests/test_part1_llm.py tests/test_part1_trace_export.py
```

### Run and analyze

Run all 16 public questions at C = 2 (`runs/c2/`) and C = 8 (`runs/c8/`), launching one report every 2 seconds. Use Bedrock if available; otherwise use `FakeLLM` and note which backend you used. Retain all four output files from each run (`trace.jsonl`, `reports.json`, `summary.json`, and `replay.json`) and answer Q1–Q3 in the [Part 1 handout](docs/part1_handout.md#task-4-workload-analysis-about-1-hour), covering:

1. **Execution DAG and critical path:** Which operations determine when a report finishes?
2. **Prefix sharing:** What fraction of prompt tokens belong to prefixes seen in earlier calls, both overall and by stage (using your `shared_prefix_share` implementation in the notebook)?
3. **Concurrency tradeoffs:** How does raising the cap from C = 2 to C = 8 affect report latency, makespan, waiting time, and GPU KV-cache memory footprint?

### Optional: research a topic on the web

After completing the required runs and analysis, you can use your agent to investigate a topic of your own on the live web in section 6 of the notebook. Keep the same `await agent.research(topic)` interface and swap `LocalSearch()` for one of the supplied web-search adapters (`ExaSearch`, `TavilySearch`, `PerplexitySearch`, or `OpenAISearch`). See the [Part 1 handout](docs/part1_handout.md#optional-research-a-topic-on-the-web) for setup details and free-credit offers. This extension is optional and does not affect your grade; use the fixed local corpus for the required C = 2 and C = 8 runs.

---



## Part 2: Write an LRU policy

**Notebook:** [part2_lru_policy.ipynb](notebooks/part2_lru_policy.ipynb) · **Full specification:** [Part 2 handout](docs/part2_handout.md)

### What to implement

When an LLM call finishes, the engine retains its KV-cache blocks so later calls can reuse shared prompt prefixes. When the cache fills up, your policy chooses which inactive block to evict (active blocks held by running calls cannot be evicted).

Complete the `[Part 2] TODO` sections in `policies/reference.py`:

- Initialize state to track block access recency.
- In `on_event`, update recency on `alloc` and `hit` events, and remove blocks on `evict` (`release` does not count as an access).
- In `choose_victim`, return the least recently used block ID among `candidates`.

```bash
pytest tests/test_part2_lru.py tests/test_kv_invariants.py
```

### Run and compare

Once the unit tests pass, work through the notebook to explore how concurrency caps, prefix structures, and dispatch order (call-readiness vs. oldest-job-first) affect cache hit rates and job latencies across document QA and deep research workloads. Save any experimental policy variants in `candidates/` so `policies/reference.py` remains a clean LRU implementation for grading.

---



## Part 3: Agent-driven policy research

**Notebook:** [part3_policy_research.ipynb](notebooks/part3_policy_research.ipynb) · **Full specification:** [Part 3 handout](docs/part3_handout.md)

### What to implement

Design and implement your own agentic research system in `research/` in which LLM agents iteratively propose policy hypotheses, implement candidate policies, evaluate them on the simulator, and decide what to try next. We provide a minimal sequential script in `examples/research_system.py` as a starting reference and comparison baseline, or you may adapt an existing framework (such as OpenEvolve, ShinkaEvolve, or CORAL) or build a custom multi-agent or tree-search system. At least one campaign must run unattended from a declared starting point to final candidate selection, and your submitted policy must come from such an unattended campaign.

Beyond achieving a strong speedup score, you must be able to explain **what works and what does not** in both your research system (verified through a system ablation study and failure analysis) and the final policy your agents discovered (verified by reading the generated code, walking through a trace, and ablating a key policy mechanism).

![One possible research system and the evaluation splits.](docs/figures/research_system.svg)

### Policy and campaign rules

While your outer research system may inspect public traces and evaluation diagnostics, any policy running inside the scorer must obey strict sandbox rules:

- **No future or external information:** Decisions may only use past events, current `View` state, and known input lengths. Do not access future arrivals, output lengths, tool durations, trace files, or workload labels. Request and block IDs are opaque and rekeyed every run; use `state.rng` for randomness.
- **Sandbox limits:** Policies run without network, file, subprocess, thread, LLM, or wall-clock access. Each hook call must complete within **50 ms** of CPU time, and policy state must stay within **1 MiB**.
- **Strict validity:** Returning invalid IDs, evicting active blocks, setting concurrency outside 1–16, returning invalid admission orders, raising exceptions, timing out, or stalling a workload fails evaluation immediately.
- **Provenance and logging:** Save every candidate policy (including failed attempts) in `candidates/` along with its parent, hypothesis, SHA-256 hash, scores, and token usage. Log any human interventions with their timestamp, reason, and affected candidates. Hand-written policies may only be used as labeled baselines, never as the final submitted policy.

### Run and compare

Develop policies on `public_dev` (32 traces) and select finalists on `public_val` (16 traces). Your completed Part 2 reference policy ([`policies/reference.py`](policies/reference.py)) uses `target_concurrency = 8` across both replicas, least-loaded replica routing, a 2048-token batch budget, first-come, first-served dispatch and admission, and LRU eviction in event delivery order. Pass `policies/reference.py` and `agentkv/policy.py` to your research agents as their starting point, and save generated variants as new files in `candidates/`.

Official speedups are always measured against fixed results from the staff reference implementation, so editing `policies/reference.py` will not alter the scoring denominator. For additional comparisons, [`policies/fifo.py`](policies/fifo.py) implements FIFO cache eviction with the same defaults.

```bash
python -m agentkv.score --policy candidates/candidate_01.py --split public_dev --backend sim --json out.json
```

You can also invoke `score()` from Python (wrap script calls in `if __name__ == "__main__":` or pass `workers=1`, since `score()` uses a process pool):

```python
from agentkv import score

if __name__ == "__main__":
    r = score("candidates/candidate_01.py", split="public_dev")
    print(r.summary())       # add details=True (or --details on CLI) for per-trace diagnostics
    print(r.by_family)       # speedups for D, W1–W5, M1, and M2
```

### Build your research evaluator

Implement an evaluator for your research system that calls `score()` and decides how to rank candidates and format feedback for your agents (for example, optimizing overall speedup, targeting a specific workload family like `result.by_family["W3"]`, or maximizing worst-family speedup `min(result.by_family.values())`). Explain your evaluator design in Q1, and note that final grading always uses the official metric across all eight families.

[`examples/evaluator.py`](examples/evaluator.py) provides a starter implementation whose `evaluate(policy_path, *, split)` function returns the score dictionary with an added `objective` field (`None` for invalid candidates). Copy it into `research/` and adapt it before running your campaign:

```bash
mkdir -p research
cp examples/evaluator.py research/evaluator.py
python examples/research_system.py --evaluator research/evaluator.py --iters 3 --offline
```

### Performance metric

**Job latency** is the elapsed time from a job's arrival until its final LLM or tool operation completes. For each trace, the evaluator compares your policy's mean job latency against `policies/reference.py`, computes the geometric mean speedup within each workload family, and then takes the geometric mean across all eight families (weighting each family equally):

```text
L(P, i) = mean job latency for policy P on trace i
S = exp(mean over families(mean over traces(log(L(baseline, i) / L(P, i)))))
g = 1 - 1/S
performance points = clip(25 + 25 × g_eff / 0.10, 0, 50)
```

Here, `S = 1.0000` matches the baseline and `S = 1.1000` means the baseline takes 1.1× as long as your policy. `g_eff` equals `g`, except that if any workload family's mean job latency is more than 20% worse than the baseline's, `g_eff` is capped at `0`. A valid policy with `g_eff = 0` earns 25/50 points, and `g_eff >= 0.10` (`S >= 1 / 0.9`, or about `1.1111×`) earns the full 50/50 points. Invalid or incomplete evaluations receive 0 points.


| Split           | Traces | Purpose                                                                 |
| --------------- | ------ | ----------------------------------------------------------------------- |
| `public_dev`    | 32     | Develop policies locally with full per-trace diagnostics                |
| `public_val`    | 16     | Select finalists; account for selection bias when interpreting results  |
| Leaderboard     | 16     | Compare policies on the Gradescope leaderboard (hidden feedback traces) |
| Hidden test     | 24     | Final policy evaluation after the deadline                              |


### Research report

Answer Q1–Q6 in the [Part 3 handout](docs/part3_handout.md#research-report), keeping each original question above its answer in order:

1. **System design:** Diagram and explain your system's LLM calls, deterministic steps, tools, stored state, evaluator objective, agent feedback, selection criteria, stopping rules, and human interventions.
2. **Ablation study:** Remove, replace, or disable one or two components in your research system while holding the model and token budget fixed.
3. **Autonomy:** Describe your longest unattended campaign and show how the system recovered from failures, citing your intervention log.
4. **Cost and scaling:** Plot best development score vs. token usage and candidates evaluated, mark validation scores of finalists, and analyze the valid-candidate rate and diminishing returns.
5. **Policy behavior:** Explain your selected policy with pseudocode, a concrete trace walkthrough, and an ablation disabling one key mechanism.
6. **Limitations and generalization:** Analyze per-family performance, diagnose an unfavorable case, predict how the policy will generalize to the hidden test set, and reflect on the strengths and limits of agent-driven systems research.

Cite run IDs and candidate hashes throughout so claims can be verified against your logs. Additional plots and logs may be placed in appendices.

---



## How to test

With `.venv` activated, run `pytest` from the repository root:

```bash
source .venv/bin/activate
pytest                                      # full public test suite
pytest tests/test_part1_concurrency.py       # single test module
pytest tests/test_part2_lru.py tests/test_kv_invariants.py
pytest -x                                   # stop at first failure
```

Unfinished `[Part 1] TODO` and `[Part 2] TODO` functions raise `NotImplementedError` until you implement them.

---



## What not to modify

Keep your changes within the files listed in [Files you edit](#files-you-edit). Do not modify:

- The rest of `research_app/` (corpus, tools, tracing client, prompt builders, LLM clients, and citation checker).
- `agentkv/policy.py`, `agentkv/replay.py`, `agentkv/engine/`, `agentkv/sandbox/`, or `agentkv/data/`.
- Public tests and datasets in `tests/` and `data/`.

Policies may influence execution only through the documented hooks in `agentkv/policy.py`. Do not alter prompts, output lengths, tool delays, stages, cache quotas, batch limits, or simulator internals.

---



<a id="submitting"></a>

## Submission

### Gradescope: Part 3 leaderboard

Submit the following files at the top level of the Gradescope leaderboard assignment (either uploaded directly or in a flat ZIP archive without a parent folder):

```text
policy.py         Policy generated and selected by your campaign
params.json       Optional parameter file (if used by policy.py)
submission.json   Metadata: policy path, params path, SHA-256 hash, Python version, and non-stdlib imports
```

Example `submission.json` (use `r.policy_sha256` or `shasum -a 256 policy.py` to compute the hash, and set `"params": null` if unused):

```json
{
  "policy": "policy.py",
  "params": "params.json",
  "candidate_hash": "<SHA-256 of policy.py>",
  "python": "3.12",
  "imports": []
}
```

Gradescope requires `policy.py` and a matching `submission.json` (use your registered team name as your leaderboard name). It evaluates your policy on the 16 hidden feedback traces using the CPU simulator and posts overall and per-family speedups (`policies/reference.py` scores `1.0000×`; benchmark rows like FIFO, Weak, and Strong are comparison points using the same reference denominator).

Your private submission page also displays projected performance points (0–50) based on the [Performance metric](#performance-metric); invalid policies receive an error message and no leaderboard entry. The leaderboard is for feedback only—you should still include your final `policy.py` and `submission.json` in your Canvas submission.

### Canvas: code, runs, and report

Submit one archive per team named `assignment4_<team name>.zip` containing:

```text
research_app/agent.py
research_app/concurrency.py
policies/reference.py
policy.py and submission.json
params.json                   Optional policy parameters
runs/c2/ and runs/c8/         Part 1 traces, reports, summaries, and replay files
research/                     Part 3 research system and evaluator source, configuration, and prompts
candidates/                   All generated candidate policies, including failures
runs/...                      Campaign logs and intervention records
assignment4_report.pdf        Answers and plots for Part 1 Q1–Q3 and Part 3 Q1–Q6
```

In `assignment4_report.pdf`, keep each original question above your answer in part and question order (Part 1 Q1–Q3 followed by Part 3 Q1–Q6; Part 2's notebook questions are for practice and do not need to be included). There is no fixed page limit. Your PDF must include:

- Group name, both team members' names, and EIDs
- `pytest` summary output
- Answers, tables, and plots for Part 1 Q1–Q3 and Part 3 Q1–Q6
- Token usage receipts (per-call input/output token counts and total team Bedrock usage) and model versions

Both team members must be prepared to explain all submitted code and analysis during the individual check. Late submissions incur a 10% penalty per day (including weekends) unless arranged in advance with the instructor.

---



## Grading


| Item                                    | Points  |
| --------------------------------------- | ------- |
| Part 1 implementation correctness       | 9       |
| Part 1 workload analysis                | 6       |
| Part 2 LRU and KV-cache invariant tests | 5       |
| Part 3 policy performance (hidden test) | 50      |
| Part 3 report: Q1 and Q3                | 8       |
| Part 3 report: Q2 and Q4                | 6       |
| Part 3 report: Q5 and Q6                | 6       |
| Individual check                        | 10      |
| **Total**                               | **100** |


---




## Data and licenses


| Data                       | Source                                                                                                    | License      |
| -------------------------- | --------------------------------------------------------------------------------------------------------- | ------------ |
| Part 1 corpus              | English Wikipedia, 48 articles, 2 pinned revisions each                                                   | CC BY-SA 4.0 |
| W1 long-context chat       | kvcache-ai/Mooncake conversation trace (FAST 2025): token counts and prefix hashes, with placeholder text | Apache-2.0   |
| W2 coding agent            | sammshen/lmcache-agentic-traces (SWE-bench sessions)                                                      | CC BY 4.0    |
| W3 document QA             | bigai-nlco/LooGLE short-dependency QA (ACL 2024)                                                          | CC BY-SA 4.0 |
| W4 customer-service agents | sierra-research/tau2-bench published conversations                                                        | MIT          |
| Bursty arrivals            | kvcache-ai/Mooncake tool-agent trace (timestamps only)                                                    | Apache-2.0   |
| Tokenizer                  | Qwen/Qwen3-0.6B tokenizer (shared by all Qwen3 sizes)                                                     | Apache-2.0   |
| Simulator                  | microsoft/vidur @ 25e0082dbbfb                                                                            | MIT          |


Each trace's `provenance` field names its exact sources and revisions.

---



## Acknowledgments

The evaluator uses Vidur canary's vLLM V1 scheduler and cache manager, with adapters for the assignment's policy hooks and application dependencies. The Part 1 skeleton is adapted from LangChain's *Deep Research from Scratch* (MIT). The workloads come from the public traces and datasets above. We thank their authors for releasing them.
