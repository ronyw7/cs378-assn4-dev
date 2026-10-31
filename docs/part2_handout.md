# Part 2: KV-cache eviction and routing

An engine can keep a finished call's KV blocks so that a later call with the
same prompt prefix skips prefill, but cached blocks take memory that running
calls need. You will implement LRU eviction (the reference for Part 3),
eviction that keeps prefix trees reusable, and routing that trades cache hits
against load on two GPUs. Work through
[`notebooks/part2_lru_policy.ipynb`](../notebooks/part2_lru_policy.ipynb);
this handout is the specification.

## The simulator

Each trace fixes a workload: 16 jobs, each a DAG of LLM calls and tool delays
with fixed prompts, output lengths, and arrival times. The simulator replays
it on Llama-3-8B on one or two A100s using Vidur's vLLM V1 scheduler,
which mixes prefill and decode in a batch, and Vidur's timing predictions from
A100 profiling data. Each GPU, or *replica*, has its own queue and KV cache.
The traces cover eight workload families:

![The eight workload families.](figures/families.svg)

A job's **latency** is the time from its arrival until its last call or tool
finishes. On a trace, a policy's **speedup** is the reference policy's mean job
latency divided by its own; `score()` combines speedups over the eight workload
families (see the [Part 3 handout](part3_handout.md#scoring)).

Memory is a pool of 16-token blocks, each in one of three states:

- *active*: a running call holds it;
- *cached*: no call holds it, but its contents are kept in case a later prompt starts the same way;
- *empty*.

![Three prompts cut into 16-token blocks, and the tree of shared prefixes those blocks form in memory.](figures/kv_prefix.svg)

A new block takes an empty slot if there is one; otherwise the engine
overwrites a cached block chosen by your policy. Each trace's memory is sized
to its working set, so eviction matters. A prefix block matches only if every
block before it matches too.

## The policy API

A policy is a class in one file. Two hooks are required; the others default to
the reference behavior.

| Hook | Returns | Default |
|---|---|---|
| `on_event(event, state)` | nothing; observes every event (fields in `agentkv.policy.EVENT_FIELDS`) | ignore |
| `target_concurrency(view, state)` | most calls in flight across all replicas, 1–16 | required |
| `choose_victim(view, candidates, state)` | one id from `candidates`, the cached blocks on `view.replica`, oldest-released first | required |
| `dispatch_order(view, ready, state)` | ready call ids to dispatch, in order (any subset) | oldest first |
| `route_call(view, call, state)` | the replica for a dispatched call | fewest calls in flight |
| `admission_order(view, waiting, state)` | `waiting` reordered; the engine admits from the front while calls fit | first come, first served |
| `batch_token_budget(view, state)` | 512, 1024, 2048, or 4096 tokens for the next batch on `view.replica` | 2048 |
| `choose_preempted(view, running, state)` | the running call that gives its blocks back when memory runs out | `running[-1]` |

`view` is a read-only snapshot of the engine (`agentkv.policy.View`): memory and
queue counts in total and per replica, the ready and in-flight calls with their
jobs, stages, prompt lengths, and prefix ids, every resident block with its
prefix, parent, replica, and reference count, and the live jobs.
`view.cached_prefix_len(call, r)` counts the leading blocks of a call's prompt
resident on replica `r`. `state` is a dict for your own data; `state.rng` is the
only allowed source of randomness. Ids are opaque and differ between traces.

Policies run in a sandbox with 50 ms of CPU per hook call and 4 MiB of state;
the [Part 3 handout](part3_handout.md#policy-rules) lists the rules.

```bash
python -m agentkv.score --policy policies/prefix_policy.py --split public_dev --params '{"slack": 2}'
python -m agentkv.score --policy policies/reference.py --split public_dev --replicas 1 --params '{"cap": 4}'
```

`score()` uses two replicas unless you pass `replicas=1`, and `viz.record` uses
one unless you pass `replicas=2`. `score()` runs traces in parallel on all but
one CPU (at most 12; set `AGENTKV_WORKERS` to change it); call it from a
notebook or from a script under `if __name__ == "__main__":`.

## 2.1 LRU

Complete `__init__`, `on_event`, and `choose_victim` in `policies/reference.py`.
An `alloc` or `hit` event makes a block the most recent; a `release` does not;
an `evict` removes it. `choose_victim` returns the least recent block among
`candidates`. Several events can share a timestamp, so track recency in event
order, not by time. `target_concurrency` already returns the `cap` parameter
(default 8).

```bash
pytest tests/test_part2_lru.py tests/test_reference_results.py
```

The second test checks, in a few minutes, that your file reproduces the staff
reference's job latencies exactly on both public splits. Scoring always
compares against the staff copy, so it works before you finish this file.

## 2.2 Prefix-aware eviction

![Evicting a root prevents reuse of its descendants; evicting a tail preserves the remaining prefix.](figures/evict_tail.svg)

A call touches its prompt's blocks from root to tail, so LRU sees a prefix's
root as least recent, and evicting it strands every block after it. Complete
`__init__`, `on_event`, and `choose_victim` in `policies/prefix_policy.py`:
return the least recently used candidate that is a *leaf*, or the least
recently used candidate if none is a leaf. With `tails=False`, it is plain LRU.
A block's `prefix` is its content id, and its `parent` is the `prefix` of the
block before it. A candidate is a leaf if no resident block on its replica has
`parent` equal to the candidate's `prefix`.

Track the trees from events: `alloc` carries the block's `prefix`, `parent`
(`None` for a prompt's first block), and `replica`, so store them; `evict`
carries only the block id. The tests call `choose_victim` with `view=None`, so
do not read `view.blocks`.

## 2.3 Cache-affinity routing

With two replicas, a prefix cached on one replica does not help a call sent to
the other. Implement `route_call` in the same file: among replicas whose
`in_flight` is at most `slack` above the least loaded one's, return the one
with the longest cached prefix of the call's prompt (`view.cached_prefix_len`),
breaking ties by lower `in_flight` and then lower replica id. With
`slack = -1`, return the least loaded replica, as the reference does.

```bash
pytest tests/test_part2_prefix.py
```

With `tails=False` and `slack=-1`, `PrefixPolicy` makes exactly the reference's
decisions.

## 2.4 Experiments and questions

The notebook sweeps the concurrency cap on one and two replicas, compares FIFO,
LRU, and prefix-aware eviction, sweeps the routing `slack`, and shows a
customer-service trace at slack 1 and 16. Run your own variants as copies
under `candidates/`, not by editing the two graded files.

**Q1. Concurrency.**

- From the two cap grids, which families prefer a low cap on one replica? Explain with their memory per call, prefix reuse, and job structure.
- Why does the best cap change with two replicas?
- Name one family whose best cap likely changes within a trace, and the trace feature that causes it.

**Q2. Eviction.**

- Give S and the per-family speedups, relative to LRU, of FIFO on one replica and of prefix-aware eviction on one and two replicas, in one table.
- Which families gain most, and what about their prompts explains it? Why is the gain smaller on two replicas?
- Describe a workload, real or hypothetical, in which evicting tails first does worse than LRU.

**Q3. Routing.**

- Include the slack grid and name the best slack.
- Which families lose at slack 16, and what in the two W4 timelines shows why?
- How would you change `route_call` so that it never does worse than least-loaded routing on W4?
