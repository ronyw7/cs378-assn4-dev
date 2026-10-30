# Part 2: KV-cache eviction and routing

Allow 7–8 hours for this part, which is worth 20% of the grade. Work through
[`notebooks/part2_lru_policy.ipynb`](../notebooks/part2_lru_policy.ipynb)
and use this handout for the policy API and the specifications.

## Introduction

The engine keeps released KV blocks so that later calls with the same prompt
prefix can skip their prefill. Agent workloads repeat instructions, documents,
and conversation history, so this saves a lot of work, but cached blocks
occupy memory that running calls also need. A serving policy decides which
cached blocks to overwrite, how many calls to run at once, in what order, and,
with two GPU replicas, where each call runs.

You will implement three policies through the API below: LRU eviction (the
reference that every Part 3 speedup is measured against), eviction that keeps
prefix trees reusable, and routing that trades cache affinity against load
balance. The notebook measures each on the eight workload families.

## Background: prefix caching

![Three prompts cut into 16-token blocks, and the tree of shared prefixes those blocks form in memory.](figures/kv_prefix.svg)

Memory is a pool of 16-token blocks. Each block is in one of three states:

- **active**: some call in flight holds it (its reference count is above zero);
- **cached**: no call holds it, but its contents are kept in case a later prompt
  starts the same way;
- **empty**: holding nothing reusable.

A new block takes an empty slot if there is one. When there is none, the
engine overwrites a cached block, and your policy's `choose_victim` picks
which. Active blocks are never candidates.

Each trace gives the engine far less KV memory than a dedicated GPU would: 736
to 6,656 blocks in the public traces, sized to that trace's working set so that
eviction decisions matter.

## The policy API

A policy controls concurrency, scheduling, and eviction through eight hooks.
Two are required, and the others have defaults. If an optional hook is
omitted, the engine uses first-come, first-served dispatch and Vidur's
vLLM V1 scheduling rules. The workload and hardware configuration remain fixed.

```python
from agentkv.policy import Policy

class MyPolicy(Policy):
    def on_event(self, event, state): ...                              # optional
    def target_concurrency(self, view, state) -> int: ...              # required
    def choose_victim(self, view, candidates, state) -> str: ...       # required
    def dispatch_order(self, view, ready, state) -> list[str]: ...     # optional
    def admission_order(self, view, waiting, state) -> list[str]: ...  # optional
    def route_call(self, view, call, state) -> int: ...                # optional
    def batch_token_budget(self, view, state) -> int: ...              # optional
    def choose_preempted(self, view, running, state) -> str: ...       # optional
```

![Policy decisions in the application and inference engine.](figures/system.svg)

Prefill, decode, and preemption have the same meanings as in Assignment 3.
Vidur can combine prefill and decode tokens in one batch. The default batch
budget is 2,048 tokens; policies can choose 512, 1,024, 2,048, or 4,096. The
API also uses the following terms:

- **dispatch**: the application (the replay driver that runs the jobs) sends
  a ready call, one whose dependencies have finished, to the engine, where it
  waits to be admitted;
- **admission**: the engine takes a waiting call, allocates its blocks, and
  starts its prefill;
- **job latency**: a job's time from arrival until its last step (LLM call or
  tool) finishes. Every score in this assignment is about mean job latency.

| Hook or input | Contract |
|---|---|
| `on_event` | Receives every event, in logical-time order with a fixed tie-breaker: `arrival`, `ready`, `dispatch`, `admit`, `hit`, `alloc`, `release`, `evict`, `preempt`, `complete` (with the call's queueing time and latency), `tool_start`, `tool_done`, and `job_done`. Their fields are in `agentkv.policy.EVENT_FIELDS`. |
| `target_concurrency` | The most calls in flight (dispatched and not complete), an int from 1 to 16 across all replicas. Lowering it never cancels calls in flight. |
| `choose_victim` | One id from `candidates`: the cached blocks on `view.replica`, oldest-released first. The engine uses empty blocks first; this hook only picks which cached block to overwrite. |
| `dispatch_order` | Ready call ids in priority order: any subset of `ready`, which is passed oldest first. The application dispatches from the front of your list until the cap is reached; the rest, and any you leave out, stay ready. Default: all of them, oldest first. |
| `admission_order` | `waiting` (dispatched calls the engine has not admitted) reordered. The engine admits from the front while calls fit in memory. Default: first come, first served. |
| `route_call` | Choose an available replica id for a ready call. Default: the replica with the fewest unfinished calls. Part 2 has replica 0; Part 3 has replicas 0 and 1. |
| `batch_token_budget` | Choose 512, 1024, 2048, or 4096 tokens for the next batch on `view.replica`. Default: 2048. |
| `choose_preempted` | When a decoding call needs a new block and every block is active, which of `running` gives its blocks back and is recomputed later. Default: the last one, `running[-1]`. |
| `view` | A read-only snapshot of the engine (below). All ids are opaque and rekeyed every run. |
| `state` | A dict for data your policy retains between events, plus `state.rng`, the only allowed source of randomness. `state` and the policy's own attributes must pickle to at most 4 MiB. Both are reset for each trace. |

What `view` holds (the fields are in `agentkv.policy.View`):

- memory counts: `total_blocks`, `empty_blocks`, `active_blocks`, `cached_blocks`, and `pressure` (the fraction of blocks that are not empty);
- queue counts: `waiting` counts calls not currently admitted, including preempted calls; `running` includes calls decoding or partway through prefill;
- `replicas`: per-replica queue and memory counts; `replica` identifies the target of an engine decision;
- `ready` and `calls`: each call's job, stage, known input length, prompt prefix identities, and assigned replica;
- `blocks`: physical residence ids, content identities (`prefix`), preceding content identities (`parent`), `depth`, `refs`, `hits`, `replica`, and last access. Multiple physical blocks can hold the same prefix;
- `jobs`: the jobs that have arrived and not finished.

The small `BlockHarness` examples teach the eviction contract. They do not
simulate GPU timing. Trace experiments use Vidur; see the
[configuration and limitations](../README.md#evaluator-configuration).

## Task 1: LRU (5%)

Complete `policies/reference.py`: LRU eviction among cached blocks, with a fixed
concurrency limit. `target_concurrency` already returns the `cap` parameter
(default 8); initialize the tracking state, then implement `on_event` and
`choose_victim`. The starter contains TODOs, not a completed reference.

- A block becomes recent when it is allocated (an `alloc` event) or reused (a
  `hit`). A `release` alone does not make it recent.
- Forget a block when you see its `evict` event.
- Follow event delivery order when accesses have the same timestamp. Reading
  block timestamps alone does not implement this ordering.
- `choose_victim` returns the least recently used id among `candidates`.

Local scoring compares candidates with saved results from the staff reference,
so it works before you finish this file. Once your implementation passes the
tests, its default configuration should reproduce the reference score of
1.0000. Changes to your local file cannot change the saved comparison results.

Check it with:

```bash
pytest tests/test_part2_lru.py tests/test_reference_results.py
```

The second test scores your file on both public splits and checks that it
reproduces the saved reference job latencies exactly; it takes a few minutes.

The public oracle uses one-token blocks and a memory of three blocks.
`agentkv.harness.BlockHarness` drives a policy through exactly these steps;
use it to try your own scenarios.

| Step | Expected result |
|---|---|
| Allocate A and B, then release both | A and B are cached, A older than B. One slot is still empty. |
| Allocate C, then release C | C takes the empty slot. Nothing is evicted. |
| Access A, then release A | An exact-prefix hit. Recency from oldest to newest is B, C, A. |
| Allocate D under pressure | LRU evicts B. A and C stay cached, and B's old lookup entry is removed. |
| Hold A active, then allocate E | A and D are active, so C is the only cached block, and it is evicted. |

## The teaching cache and the scored engine

`BlockHarness` drives a small block-level cache model (`agentkv.engine.kv`) with
the same events and hooks as the scored engine, so the oracles above can name
blocks with letters. Scoring always uses Vidur's cache manager through
`agentkv.engine.vidur_adapter`. Both follow the same contract:

- a block with active references is never a victim;
- a cached block keeps its contents until it is overwritten, and an
  exact-prefix match makes it active again;
- overwriting a block removes its old lookup entry first, so a stale entry
  cannot cause a false hit;
- the same tokens under a different parent form a different prefix, and
  partial blocks are never shared;
- if every block is active, the engine waits or preempts the call that
  `choose_preempted` names.

In Vidur, `view.blocks` is keyed by physical residence ids, and `Block.prefix`
and `Call.prefix` hold content ids; the same prefix can be resident on both
replicas. `view.cached_prefix_len(call, replica)` counts how many leading
blocks of a call's prompt are resident on a replica.

## Task 2: Prefix-aware eviction (4%)

![Evicting a root prevents reuse of its descendants; evicting a tail preserves the remaining prefix.](figures/evict_tail.svg)

A call touches its prompt's blocks from root to tail, so LRU considers a
prefix's root its least recently used block, and evicting it strands every
block after it. In `policies/prefix_policy.py`, implement `__init__`,
`on_event`, and `choose_victim`:

- A cached block is a **leaf** if no resident block on the same replica extends
  it, that is, has its prefix as `parent`.
- `choose_victim` returns the least recently used candidate that is a leaf, or
  the least recently used candidate if none is a leaf. With `tails=False` it
  is plain LRU.
- Maintain the prefix trees from events. `alloc` carries the block's `prefix`,
  `parent` (`None` for the first block of a prompt), and `replica`; `evict`
  carries only the block id, so remember what you need. Do not read
  `view.blocks` in this hook: the tests call `choose_victim` with `view=None`.

## Task 3: Cache-affinity routing (3%)

With two replicas, a prefix cached on one replica is useless to a call sent to
the other. In the same file, implement `route_call`: among replicas whose
`in_flight` is at most `slack` above the least loaded replica's, return the one
with the longest cached prefix of the call's prompt
(`view.cached_prefix_len`), breaking ties by lower `in_flight` and then lower
replica id. With `slack = -1`, return the least loaded replica, as the
reference does.

```bash
pytest tests/test_part2_prefix.py
```

With `tails=False` and `slack=-1`, `PrefixPolicy` makes the same decisions as
the reference; one test checks this on a real trace.

## Experiments and questions (8%)

The notebook compares concurrency caps on document QA and deep research
traces, sweeps the cap over all eight families on one and two replicas,
compares FIFO, LRU, and prefix-aware eviction, sweeps the routing `slack`, and
shows a customer-service trace at a small and a large slack. To score a policy
yourself:

```bash
python -m agentkv.score --policy policies/prefix_policy.py --split public_dev --params '{"slack": 2}'
python -m agentkv.score --policy policies/reference.py --split public_dev --replicas 1 --params '{"cap": 4}'
```

Answer Q1–Q3 in `assignment4_report.pdf`, after Part 1, each below its question.

**Q1. Concurrency on one and two replicas.** Using the two cap grids, which families prefer a low cap on one replica, and why (refer to memory per call, prefix reuse, and the structure of their jobs)? Why does the best cap change with two replicas? Name one family where you would expect a single fixed cap to be wrong for part of the trace.

**Q2. Eviction and prefix trees.** Report LRU and prefix-aware eviction on one and two replicas. Which families gain most, and what about their prompts explains it? Why is the gain smaller with two replicas? Name a workload for which evicting tails first would be the wrong choice.

**Q3. Affinity versus load.** Plot or tabulate S and the per-family speedups across `slack`. Explain the best slack and the failure at large slack, using the two W4 timelines as evidence. What would you change in `route_call` to make large slack safe?

## Hints and common pitfalls

- An `OrderedDict` with `move_to_end` gives you O(1) recency updates.
- Your recency order also contains active blocks, which are never in
  `candidates`. Return the least recent block that *is* a candidate.
- A block can be released and hit again many times. Only `alloc` and `hit`
  change its recency.
- Keep `policies/reference.py` and `policies/prefix_policy.py` exactly what the
  tests specify. Put experiments in copies under `candidates/`.
- `PrefixPolicy` must stay within the sandbox limits on the largest traces:
  50 ms of CPU per hook call and 4 MiB of pickled state. Keep per-block state small.

## Grading (20%)

| Item | Weight |
|---|---|
| 2.1 LRU (`policies/reference.py`) on staff access sequences | 5% |
| 2.2 Prefix-aware eviction on staff sequences | 4% |
| 2.3 Cache-affinity routing on staff views | 3% |
| 2.4 Sweeps and answers to Q1–Q3 | 8% |

Include `policies/reference.py` and `policies/prefix_policy.py` in the team's
Canvas archive.
