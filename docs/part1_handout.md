# Part 1: Build a deep research agent

Allow about 4–5 hours for this part, which is worth 15 points. Work through
[`notebooks/part1_deep_research.ipynb`](../notebooks/part1_deep_research.ipynb)
and use this handout for implementation requirements and submission details.

## Introduction

In this part, you will build a deep research agent, run it, and examine its
workload from an inference engine's perspective. In Assignment 3, request
arrivals were fixed by the replay trace. Here, later LLM calls may need to
wait for earlier calls or tools to finish. These calls also often repeat long
prefixes containing the same instructions, documents, and conversation history.

You will use recorded traces to identify parallel branches, shared prefixes,
and operations on a report's critical path. This analysis provides the basis
for Parts 2 and 3, where you will study scheduling and caching across several
workloads, including mixtures that share one engine.

The later parts use traces from a fixed staff implementation of the agent.
Your Part 1 implementation therefore affects only your Part 1 grade.

## What you will build

![The deep research graph: plan, parallel research branches, coverage check, optional follow-up, synthesize, validate, optional repair.](figures/research_workflow.svg)

| Step | What happens |
|---|---|
| Plan | One LLM call splits the question into two to four subtopics. |
| Research each subtopic, in parallel | One search, up to three fetches, and one LLM call that extracts evidence quotes from the fetched documents. |
| Check coverage | Deterministic code finds the first subtopic with no evidence. If there is one, a single follow-up branch researches it again with a broader query. |
| Synthesize | One LLM call writes the report. Every claim cites evidence ids. |
| Validate | The supplied citation checker. If the report fails, one repair call fixes it, and the checker runs again. |

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
| `research_app/agent.py` | The nodes (`plan`, `research_facet` and its helper `research_branch`, `coverage`, `followup`, `synthesize`, `validate`, `repair`), the routing functions (`fan_out`, `route_coverage`, `route_validation`), and `build_graph`. The comments at each TODO explain the required behavior and relevant helpers. |

| Supporting files | |
|---|---|
| `research_app/tracing.py` | The tracing client. Every LLM and tool call goes through it. It records the trace and enforces the budgets. |
| `research_app/prompts.py` | The prompts, laid out so that calls share prefixes: instructions first, then documents, then the part that varies. |
| `research_app/corpus.py` | 48 Wikipedia articles on six systems topics, each pinned at two revisions, behind `search()` and `fetch()`, plus a seeded tool-latency model. The required experiments use these fixed sources. |
| `research_app/llm.py` | `BedrockLLM` (the Converse API) and `FakeLLM` (deterministic and offline). |
| `research_app/artifacts.py`, `research_app/citations.py` | `Evidence`, `Claim`, `Report`, `BranchResult`, and the citation checker. |
| `research_app/data/questions.json` | The questions. The 16 marked `public` are the ones you run. |

## Task 1: The concurrency cap (about 30 minutes)

Implement `acquire`, `release`, and `set_limit` in `research_app/concurrency.py`:

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

## Task 2: The agent (2 to 3 hours)

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

## Task 3: Two runs (15 minutes)

Run all 16 public questions, one report starting every 2 seconds, at C = 2 and
at C = 8. Offline, with `FakeLLM` and its simulated latency, the C = 2 run takes
about a minute and a half and the C = 8 run under a minute:

```bash
python -m research_app.run --cap 2 --replay --out runs/c2
python -m research_app.run --cap 8 --replay --out runs/c8
```

On Amazon Bedrock, with the smaller course model, expect a few minutes per run:

```bash
export AWS_REGION=us-west-2
export AGENTKV_BEDROCK_MODEL="MODEL_ID_FROM_COURSE_PAGE"
python -m research_app.run --backend bedrock --cap 2 --replay --out runs/c2
python -m research_app.run --backend bedrock --cap 8 --replay --out runs/c8
```

Each run writes `trace.jsonl` (every LLM and tool call), `reports.json`, and
`summary.json`. `--replay` also writes `replay.json`, the same run as a trace
that the simulator of Parts 2 and 3 can replay.

Bedrock does not expose its KV cache, so a live run cannot directly report
cache hits. Instead, `trace.jsonl` records `prefix_ids` computed locally with
the course tokenizer over 256-token blocks so you can analyze potential prefix
reuse across calls. (When replaying traces in Parts 2 and 3, the Vidur
simulator subdivides prompts into 16-token KV-cache blocks.)

## Task 4: Workload analysis (about 1 hour)

Answer Q1–Q3 in the notebook, then include the questions and your answers in
`assignment4_report.pdf`. Keep each question above its answer and follow the
question order. Include the requested plots and table, and support your
conclusions with measurements from both runs. There is no fixed page limit.

**Q1.** Which operations are on the critical path of the report you examined? Include its dependency graph and explain which operations determine when it finishes. How much time does the report spend waiting for an LLM slot at C = 2?

**Q2.** What fraction of prompt tokens belong to prefix blocks that appeared in earlier calls? Use `prefix_ids` and `prompt_tokens`; only full 256-token blocks have prefix ids, so the final partial block of a prompt does not count as shared. Report the fraction by stage and explain what each stage reuses. Which prefixes would be useful to retain between calls, and for how long? Use measurements from both runs to support your answer.

**Q3.** How does increasing the concurrency cap from C = 2 to C = 8 affect mean report latency, makespan, waiting time, and peak concurrent calls? Include the timelines and a table comparing the two runs. Estimate the GPU memory needed at each cap, accounting for shared prefixes. For this estimate, assume Llama-3-8B in bf16 on one 80 GB A100, with approximately 16 GB of weights and 128 KiB of KV cache per token (32 layers × 8 KV heads × 128 dimensions × 2 for K and V × 2 bytes). Explain the tradeoff between memory use and completion time.

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

Your team submits, in its Canvas archive (see the README):

- `research_app/agent.py` and `research_app/concurrency.py`, passing the public tests;
- the two runs `runs/c2/` and `runs/c8/` (on Bedrock if you have access, otherwise with `FakeLLM`);
- answers to Q1–Q3, with the requested plots and table, in `assignment4_report.pdf`.

Include these files with Part 2's `policies/reference.py` in the Canvas archive.
Staff correctness tests check the same contracts as the public tests with
different inputs and failure conditions. Gradescope is used only for the
Part 3 policy leaderboard. Both team members must be able to explain the
implementation during the individual check.

## Grading (15 points)

| Item | Points |
|---|---|
| Implementation correctness: concurrency cap (2 points) and research graph (7 points), checked on hidden cases. Use the public tests during development. | 9 |
| Answers to Q1–Q3, with both controlled runs and the requested plots and table. | 6 |

The skeleton is adapted from LangChain's *Deep Research from Scratch* (MIT),
and the report-with-sources interface follows Stanford CS329a HW3.

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
