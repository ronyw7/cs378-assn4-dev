# Part 1: Build and analyze a deep research agent

Allow 10–12 hours for this part, which is worth 20 points. Work through
[`notebooks/part1_deep_research.ipynb`](../notebooks/part1_deep_research.ipynb)
and use this handout for the specifications.

## Introduction

You will build a deep research agent, run it, and measure its workload from an
inference engine's point of view. Later LLM calls wait for earlier calls and
tools, and many calls repeat long prompt prefixes: the same instructions, the
same documents, and the conversation so far. You will find which operations
determine when a report finishes, how much prompt content repeats and how long
it would have to stay cached, and what running more calls at once costs in GPU
memory, first from your own runs and then by replaying them on a simulated GPU.

Parts 2 and 3 use traces from a fixed staff implementation of the agent, so
your Part 1 code affects only your Part 1 grade.

## What you will build

![The deep research graph: plan, parallel research branches, coverage check, optional follow-up, synthesize, validate, optional repair.](figures/research_workflow.svg)

| Step | What happens |
|---|---|
| Plan | One LLM call splits the question into two to four subtopics. |
| Research each subtopic, in parallel | One search, up to three fetches, and one LLM call that extracts evidence quotes from the fetched documents. |
| Check coverage | Deterministic code finds the first subtopic with no evidence. If there is one, a single follow-up branch researches it again with a broader query. |
| Synthesize | One LLM call writes the report. Every claim cites evidence ids. |
| Validate | The supplied citation checker. If the report fails, one repair call fixes it, and the checker runs again. The checker verifies that each cited quote exists at its offsets in the cited snapshot; it does not check that the claim follows from the quote. |

Use this graph structure so that your traces can be compared with the
evaluation traces. Per report, the tracing client enforces at most **8 LLM
calls**, **5 searches**, **15 fetches**, and **3 attempts per operation**.

All 16 reports in a run share one cap **C** on concurrent LLM calls. A retry
retains its original slot, while tool calls do not acquire slots. If the cap
is lowered, running calls must be allowed to finish; new calls wait until
the number in flight falls below the limit.

In the code, `research_facet` is the node that researches one subtopic, and
`facet_id` identifies that subtopic.

### LangGraph concepts

The implementation uses five LangGraph concepts
([documentation](https://langchain-ai.github.io/langgraph/)):

- **State.** A dict that flows through the graph. Here it is `ReportState` in
  `research_app/agent.py`. A node is a function
  `async def node(state, config) -> dict` that returns only the keys it changes.
- **Edges.** `g.add_edge("synthesize", "validate")` runs `validate` after
  `synthesize`.
- **Conditional edges.** `g.add_conditional_edges("coverage", route_coverage, ["followup", "synthesize"])`
  calls `route_coverage(state)` and goes to the node it names.
- **Fan-out.** A routing function may return a list of
  `Send("research_facet", payload)`. LangGraph runs `research_facet` once per
  `Send`, all in parallel, each with its own payload. Their results are merged
  by the state key's reducer: `branches` is declared
  `Annotated[list[BranchResult], operator.add]`, so each node returns a
  one-element list and LangGraph concatenates them.
- **Join.** A plain edge out of the fanned-out node,
  `g.add_edge("research_facet", "coverage")`, runs `coverage` once, after
  every branch has finished.

## Files

| You edit | |
|---|---|
| `research_app/concurrency.py` | `AdjustableCap`: `acquire`, `release`, and `set_limit`. |
| `research_app/agent.py` | The nodes (`plan`, `research_facet` and its helper `research_branch`, `coverage`, `followup`, `synthesize`, `validate`, `repair`), the routing functions (`fan_out`, `route_coverage`, `route_validation`), and `build_graph`. The comment at each TODO states the required behavior. |
| `research_app/analysis.py` | `critical_path`, `shared_prefix_share`, `reuse_gaps`, and `kv_footprint` over a recorded run. |

| Supporting files | |
|---|---|
| `research_app/tracing.py` | The tracing client. Every LLM and tool call goes through it. It records the trace and enforces the budgets. |
| `research_app/prompts.py` | The prompts, laid out so that calls share prefixes: instructions first, then documents, then the part that varies. The system prompt is about 730 tokens, like the long system prompts of production agents, and every call starts with it. |
| `research_app/corpus.py` | 48 Wikipedia articles on six systems topics, each pinned at two revisions, behind `search()` and `fetch()`, plus a seeded tool-latency model. The required experiments use these fixed sources. |
| `research_app/llm.py` | `BedrockLLM` (the Converse API) and `FakeLLM` (deterministic and offline). |
| `research_app/artifacts.py`, `research_app/citations.py` | `Evidence`, `Claim`, `Report`, `BranchResult`, and the citation checker. |
| `research_app/data/questions.json` | The questions. The 16 marked `public` are the ones you run. |

## Task 1: The concurrency cap (about 1 hour)

Implement `acquire`, `release`, and `set_limit` in `research_app/concurrency.py`. The cap ranges from 1 to 8 here; the simulated engine of Parts 2 and 3 allows 1 to 16:

- `acquire()` waits until `in_flight < limit`, then reserves a slot and updates
  the counters that track current calls, total grants, and peak concurrency.
- `release()` returns a slot and wakes waiting calls so they can check whether
  there is room to start.
- `set_limit(n)` changes the limit and wakes waiting calls. Lowering the limit
  must allow running calls to finish; new calls wait until `in_flight < limit`.

The supplied `slot()` context manager calls `acquire()` and `release()` for you.
The tracing client uses it to hold a slot for an entire LLM operation,
including retries.

An `asyncio.Condition` can coordinate waiting calls. Check your implementation
with `pytest tests/test_part1_concurrency.py`, then run the notebook example
to observe how six simulated calls respond when the cap drops from 3 to 1.

## Task 2: The agent (3 to 4 hours)

Start with `build_graph`, leaving the node functions as stubs. As you implement
each node, use the notebook's single-report run to check your progress; it
will raise `NotImplementedError` at the first unfinished node it reaches.
Once the graph is complete, run:

```bash
pytest tests/test_part1_agent.py tests/test_part1_citations.py tests/test_part1_llm.py tests/test_part1_trace_export.py
```

### Calling the tracing client

Inside a node, `ctx = ctx_of(config)` gives you the report's context:
`ctx.client` (the tracing client), `ctx.rid` (the report id), and
`ctx.settings` (`AgentSettings`: subtopic counts, the fetch limit, and each stage's `max_tokens`).

```python
resp, op = await ctx.client.llm_call(
    "extract",
    messages,
    op_id=f"{ctx.rid}/{facet.facet_id}/extract",
    deps=fetch_op_ids,
    order=(1, index, 2),
    max_tokens=ctx.settings.extract_tokens,
)
```

Here, `messages` is the prompt from `prompts.extract_messages`, and
`fetch_op_ids` contains the ids of all fetches the branch waited for, including
failed ones. Only successfully fetched documents go into the prompt. Search, fetch,
and validation use `ctx.client.search`, `ctx.client.fetch`, and
`ctx.client.validate`; their signatures are in `research_app/tracing.py`.

- **Failed operations.** A call that fails permanently, either after 3 attempts
  or on a permanent error, returns `None`. Handle that result by returning
  a typed `BranchResult` rather than raising an exception.
- **`deps`** are the op ids this operation must wait for. They become
  the edges of the job's DAG in the replay trace.
- **`op_id`** names the operation's position in the report, not the order it
  happened to run in. The tests look operations up by these ids.
- **`order`** is the sort key of the recording. Every operation must sort after
  the operations it depends on.

Record an operation's id before awaiting it, and keep each fetched document
as soon as its call succeeds. A timeout must retain that completed work.
For follow-up and synthesis, use `_branch_tails` to collect dependencies from
all earlier branches, including parallel fetches in a branch that did not
reach extraction. A successful extraction can return no usable evidence;
coverage still identifies that subtopic for a follow-up. The
comments at the `research_branch` TODO specify how to assign each branch status.

Use the following id and ordering conventions. `R03` is the report id (`ctx.rid`), `f2` the subtopic's
`facet_id`, `i` its 0-based index in the plan, and `k` the 0-based fetch number.

| Operation | `op_id` | `order` |
|---|---|---|
| Plan | `R03/plan` | `(0,)` |
| Subtopic search | `R03/f2/search` | `(1, i, 0)` |
| Subtopic fetches, in hit order | `R03/f2/fetch1` … `R03/f2/fetch3` | `(1, i, 1, k)` |
| Subtopic extract | `R03/f2/extract` | `(1, i, 2)` |
| Follow-up branch | `R03/followup/search`, `.../fetch1` …, `.../extract` | `(2, 0, …)`, same steps |
| Synthesize | `R03/synthesize` | `(3,)` |
| Citation check | `R03/validate` | `(4,)` |
| Repair | `R03/repair` | `(5,)` |
| Check after repair | `R03/revalidate` | `(6,)` |

The follow-up branch researches the first uncovered subtopic again with a broader
query, for example the subtopic's title and query plus the question.

## Task 3: Three runs (30 minutes)

Run all 16 public questions three times, one report starting every 2 seconds:

```bash
python -m research_app.run --cap 2 --replay --out runs/c2
python -m research_app.run --cap 8 --replay --out runs/c8
python -m research_app.run --cap 8 --failure-rate 0.2 --replay --out runs/c8-faults
```

These use the offline `FakeLLM`. With Bedrock, set the model and add `--backend bedrock`:

```bash
export AWS_REGION=us-west-2
export AGENTKV_BEDROCK_MODEL="MODEL_ID_FROM_COURSE_PAGE"
python -m research_app.run --backend bedrock --cap 2 --replay --out runs/c2
```

`--failure-rate 0.2` makes each tool attempt fail transiently with probability
0.2, so the third run exercises retries, follow-up branches, and repairs. Each
run writes `trace.jsonl` (every LLM and tool call), `reports.json`,
`summary.json`, and `replay.json`, the same run as a trace that the Parts 2 and
3 simulator can replay.

Bedrock does not expose its KV cache, so `trace.jsonl` records `prefix_ids`
computed locally with the course tokenizer over 256-token blocks. They show
potential prefix reuse; the simulator in Parts 2 and 3 uses 16-token blocks.

## Task 4: Workload analysis (about 3 hours)

Implement the four functions in `research_app/analysis.py`. Each takes the
operations of a run loaded with `load_run("runs/c2")`; `Op` records an
operation's id, report, kind, stage, dependencies, start and end times, slot
request and grant times (LLM calls), token counts, and `prefix_ids`.

- **`critical_path(run, report)`** returns the operations that determined when
  the report finished. Start from its operation with the latest end (break ties
  by `op_id`), step to the dependency that finished last, and repeat until an
  operation has no dependencies. Each `Step` splits an operation's share of the
  latency into `gap` (from its blocking dependency's end, or the report's
  arrival, to its start), `wait` (for an LLM slot), and `service` (the rest of
  its duration, including retries and backoff). The steps add up to the
  report's latency.
- **`shared_prefix_share(calls)`** returns the fraction of prompt tokens in full
  256-token blocks whose prefix id appeared in an earlier call. The final
  partial block of a prompt never counts as shared.
- **`reuse_gaps(calls)`** records, for each reuse of a block, how long it sat
  unused: the reusing call's start minus the latest end of the earlier calls
  that held it, or 0 if one of them is still running. This is how long a cache
  would have had to keep the block for the reuse to hit.
- **`kv_footprint(calls)`** returns the KV-cache bytes held by the LLM calls
  that hold a slot, at every time that changes, both when each call stores its
  own prompt and output and when identical full prefix blocks are stored once.
  It assumes Llama-3-8B in bf16: 128 KiB per token (32 layers × 8 KV heads ×
  128 dimensions × 2 for K and V × 2 bytes).

```bash
pytest tests/test_part1_analysis.py
```

The notebook then plots one report's DAG with its critical path, every report's
critical-path breakdown, prefix reuse by stage, reuse gaps, and the KV
footprint of each run. Finally, it replays your C = 8 run on a simulated A100
with `viz.record(..., kv_tokens=...)` across concurrency caps of 1 to 16 and
KV-cache capacities of 12,288 to 393,216 tokens, so that you can measure the
trade-off between concurrency and memory that the hosted API hides.

## Task 5: Questions (2 to 3 hours)

Answer Q1–Q4 in `assignment4_report.pdf`, each below its question, with the
requested plots and tables. Support every answer with measurements from your
runs.

**Q1. Critical path.** For one report at C = 2, show its DAG and critical path. Which operations determine when it finishes, and how much of its latency is slot waiting, LLM service, and tool service? How does the breakdown change at C = 8, across all reports? Which single change to the agent or the cap would shorten the typical C = 8 critical path most, and by roughly how much?

**Q2. Prefix reuse and retention.** What fraction of prompt tokens are in previously seen blocks, overall and by stage, and what content does each stage reuse? Using the reuse gaps, how long must a cache keep each kind of prefix for its reuse to hit, and which prefixes would you never keep? Use both runs.

**Q3. Concurrency and memory.** Compare C = 2 and C = 8 in a table: mean report latency, makespan, mean slot wait, peak concurrent calls, and peak KV footprint with and without prefix sharing. Then use your replay sweep: at which KV capacity does a higher cap stop helping, and why? Assuming Llama-3-8B in bf16 on one 80 GB A100 (about 16 GB of weights), which cap would you choose for this workload, and how much memory headroom would remain?

**Q4. Failures.** In the failure run, where do retries, follow-ups, and repairs appear, and which of them land on critical paths? How do they change the median and worst report latency? An LLM call keeps its slot while it backs off between attempts; at C = 2, would releasing the slot during backoff help or hurt mean report latency, and why?

## Hints and common pitfalls

- **Use the stage names `plan`, `extract`, `synthesize`, and `repair`**, and
  build every prompt with the matching `prompts.*_messages` function. `FakeLLM`
  recognizes calls by stage, and the fixed prompt layout is what makes calls
  share prefixes.
- **An extract call depends on every fetch it waited for**, including a fetch
  that failed. Incorrect dependencies may still produce a valid report, but they
  change the exported DAG and can invalidate the critical-path analysis.
- **Keep `order` consistent with `deps`.** If an operation sorts before one it
  depends on, the replay export drops that dependency and inherits the other
  operation's own dependencies instead. The export may succeed despite the
  incorrect DAG, so follow the ordering table above.
- **Evidence offsets come from code**, never from the model: `artifacts.locate`
  finds each quote in its document. A quote it cannot find is discarded.
- **Raising the cap must wake waiters.** If `set_limit` raises the limit and
  nobody is notified, waiting calls sleep until some other call releases.

## Submission

Include in your team's Canvas archive (see the README):

- `research_app/agent.py`, `research_app/concurrency.py`, and `research_app/analysis.py`;
- the runs `runs/c2/`, `runs/c8/`, and `runs/c8-faults/`, and the backend you used;
- answers to Q1–Q4, with the requested plots and tables, in `assignment4_report.pdf`.

## Grading (20 points)

| Item | Points |
|---|---|
| Implementation correctness on staff tests with new inputs and failure conditions: concurrency cap (2), research graph (6), analysis functions (4) | 12 |
| Answers to Q1–Q4, with the requested plots and tables | 8 |

The skeleton is adapted from LangChain's *Deep Research from Scratch* (MIT).

## Optional: research a topic on the web

Once you have finished the required implementation and analysis above, you can also use your `ResearchAgent` to investigate any topic on the live web (notebook section 6). This extension is optional and does not affect your grade; use the fixed local corpus for the required C = 2 and C = 8 runs.

Keep the same `ResearchAgent` with `BedrockLLM()` and replace `LocalSearch()` with one of the web-search adapters supplied in `research_app/search.py` (`"exa"`, `"tavily"`, `"perplexity"`, or `"openai"`). See the [Artificial Analysis Search API leaderboard](https://artificialanalysis.ai/agents/search-api) for a comparison of search providers:

| Provider | Adapter and environment variable | Free access (checked October 30, 2026) |
|---|---|---|
| [Exa](https://exa.ai/pricing) | `ExaSearch`, `EXA_API_KEY` | $10 signup bonus plus $10/month; no card required. |
| [Tavily](https://www.tavily.com/pricing) | `TavilySearch`, `TAVILY_API_KEY` | 1,000 credits/month (or 4,000/month via [student offer](https://help.tavily.com/articles/6606514713-student-account)); no card required. |
| [Perplexity Search](https://docs.perplexity.ai/docs/getting-started/pricing) | `PerplexitySearch`, `PERPLEXITY_API_KEY` | Pay as you go. |
| [OpenAI web search](https://developers.openai.com/api/docs/guides/tools-web-search) | `OpenAISearch`, `OPENAI_API_KEY`, `AGENTKV_SEARCH_MODEL` | Pay as you go (also set `AGENTKV_SEARCH_MODEL` to a Responses model supporting `web_search`). |

Configure Bedrock ([Model access](../README.md#model-access)) and your chosen search provider's API key in the environment before launching Jupyter. Note how the adapters retrieve source passages: Exa and Tavily use search and page-extraction APIs; Perplexity returns extracted passages directly with search results; and OpenAI discovers URLs that the adapter downloads as HTML/text (generated search answers are never used as source passages).

```python
from research_app import ResearchAgent
from research_app.llm import BedrockLLM
from research_app.search import make_search

agent = ResearchAgent(llm=BedrockLLM(), search=make_search("exa"), cap=2)
result = await agent.research(
    "What are the tradeoffs between passkeys and passwords for account recovery?",
    out="runs/topic-passkeys",
)
print(result.markdown())
```

Each run directory (`out=...`) saves the report, archived source text (`sources.json`), trace (`trace.jsonl`), and token/search receipts (`summary.json`). You can also re-verify citations against the saved `SourceArchive` offline:

```python
import json
from pathlib import Path
from research_app.search import SourceArchive
from research_app.citations import check_report

sources = SourceArchive.load("runs/topic-passkeys/sources.json")
report = json.loads(Path("runs/topic-passkeys/report.json").read_text())
assert not check_report(report, sources)
```
