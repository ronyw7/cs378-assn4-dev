"""Replay application DAGs on one or two independent Vidur GPU replicas."""

from collections import OrderedDict
from dataclasses import replace
from functools import lru_cache
import hashlib
import heapq
import math

from .engine.vidur_adapter import PolicyScheduler, ReplayRequest
from .engine.vidur_profile import fingerprint, predictor
from .policy import MAX_CONCURRENCY
from .replay import Replay, Rekey, RunResult, _ARRIVAL, _TOOL_DONE
from .sandbox.client import InProcessClient, WorkerClient
from .sandbox.runtime import PolicyError
from .trace import EngineConfig, LLMNode, Memory, chain_hashes

_BATCH_DONE = 2


def prepare_trace(trace):
    """Keep the recorded workload and KV token quota; rebuild 16-token hashes.

    Token ids are workload identities, encoded by the supplied tokenizer.
    Vidur uses lengths and equality, without executing or generating text.
    """
    jobs = []
    for job in trace.jobs:
        nodes = []
        for node in job.nodes:
            if isinstance(node, LLMNode):
                if node.n_total > 32768 or node.n_prompt < 1 or node.n_output < 1:
                    raise ValueError(f"{trace.trace_id}/{node.id}: Vidur replay requires 1–32768 total tokens and a nonempty prompt/output")
                node = replace(node, hashes=tuple(chain_hashes(trace.tokens(node), 16)))
            nodes.append(node)
        jobs.append(replace(job, nodes=nodes))
    tokens = trace.memory.num_blocks * trace.memory.block_size
    if tokens % 16:
        raise ValueError("The KV token quota must be divisible by 16")
    return replace(trace, jobs=jobs, memory=Memory(tokens // 16, 16),
                   engine=EngineConfig(16, 4096), provenance={**trace.provenance,
                   "evaluator": "vidur-v1", "workload_encoding": "frozen-token-lengths-and-prefix-identities"})


def run_policy(trace, policy_path, *, params=None, seed=0, isolate=True, replicas=2):
    trace = prepare_trace(trace)
    client_type = WorkerClient if isolate else InProcessClient
    try:
        client = client_type(policy_path, params=params, seed=seed, block_size=16,
                             total_blocks=trace.memory.num_blocks * replicas)
    except PolicyError as exc:
        return RunResult(trace.trace_id, trace.family, [math.inf] * len(trace.jobs), str(exc))
    return VidurReplay(trace, client, seed=seed, replicas=replicas).run()


class VidurReplay(Replay):
    """Reuse the application's dependency handling with a parallel event loop."""
    def __init__(self, trace, client, *, seed=0, replicas=2):
        if replicas not in (1, 2):
            raise ValueError("The assignment supports one or two replicas")
        self.trace, self.client = trace, client
        self.rekey = Rekey(hashlib.blake2b(f"{trace.sha256}:{seed}".encode(), digest_size=16).digest())
        self.now = self.t = 0.0
        self.heap, self._tie = [], 0
        self.ready, self.inflight, self.jobs, self.flows = OrderedDict(), {}, {}, {}
        self.predictor = predictor()
        self.schedulers = [PolicyScheduler(self, r, self.predictor, trace.memory.num_blocks) for r in range(replicas)]
        self.cache_hit_tokens = self.tc_calls = self.cv_calls = self.steps = 0
        self.prefill_tokens = self.recomputed_tokens = self.preemptions = self.evictions = 0
        self.prefill_steps = self.decode_steps = self.mixed_steps = 0
        self.busy_time = self.prefill_time = 0.0
        self.replica_busy = [0.0] * replicas

    def _loop(self):
        if "prefill_first" in self.client.hooks:
            raise PolicyError("invalid", "prefill_first belongs to the previous engine. Vidur mixes prefill and decode; use batch_token_budget instead.")
        while len(self.flows) < len(self.trace.jobs):
            if not self.heap:
                raise PolicyError("stalled", "No progress is possible; dispatch_order may have omitted all ready calls")
            self.now = self.t = self.heap[0][0]
            while self.heap and self.heap[0][0] == self.t:
                _, _, kind, payload = heapq.heappop(self.heap)
                if kind == _ARRIVAL:
                    self._arrive(payload)
                elif kind == _TOOL_DONE:
                    job, node = payload
                    self.client.event(("tool_done", self.t, job.pid, self.rekey.stage(node.stage)))
                    self._node_done(job, node.id)
                else:
                    scheduler, batch = payload
                    batch.on_batch_end(self.t)
                    scheduler.pool.publish()
                    scheduler.on_batch_end(batch)
                    scheduler.busy = False
                    for request in batch.completed_requests:
                        self._complete_request(request)
            self._dispatch()
            for scheduler in self.schedulers:
                if not scheduler.busy and not scheduler.is_empty():
                    self._schedule(scheduler)

    def _schedule(self, scheduler):
        output = scheduler.on_schedule(self.t)
        batch = output.batch
        if batch is None:
            raise PolicyError("stalled", "Vidur could not schedule a call within this replica's memory quota")
        duration = self.predictor.get_batch_execution_time(batch, 0).total_time
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError("Invalid Vidur batch-time prediction")
        prefill = decode = 0
        for request, count in zip(batch.requests, batch.num_tokens):
            before = request.num_processed_tokens
            self.recomputed_tokens += max(0, min(count, request.high_water - before))
            request.high_water = max(request.high_water, before + count)
            if request.is_prefill_complete:
                decode += 1
            else:
                prefill += count
        self.prefill_tokens += prefill
        self.prefill_steps += bool(prefill)
        self.decode_steps += bool(decode)
        self.mixed_steps += bool(prefill and decode)
        self.prefill_time += duration if prefill else 0
        self.busy_time += duration
        self.replica_busy[scheduler.index] += duration
        self.steps += 1
        if self.steps > 500000:
            raise PolicyError("stalled", "Replay exceeded 500,000 batches, usually because calls repeatedly preempt one another")
        batch.on_schedule(self.t)
        scheduler.busy = True
        self._push(self.t + duration, _BATCH_DONE, (scheduler, batch))

    def _hdr(self, replica=None):
        rows = []
        for s in self.schedulers:
            pool = s.pool
            active = pool.num_gpu_blocks - pool.get_num_free_blocks()
            empty = len(pool.empty_ids)
            rows.append(dict(id=s.index, total_blocks=pool.num_gpu_blocks, empty_blocks=empty,
                active_blocks=active, cached_blocks=pool.num_gpu_blocks - active - empty,
                waiting=len(s._waiting_queue), running=len(s._running),
                in_flight=len(s._requests), busy=s.busy))
        return [self.t, sum(r["empty_blocks"] for r in rows), sum(r["active_blocks"] for r in rows),
                sum(r["waiting"] for r in rows), sum(r["running"] for r in rows), rows, replica]

    def memory_counts(self):
        hdr = self._hdr()
        empty, active = hdr[1:3]
        return active, sum(r["cached_blocks"] for r in hdr[5]), empty

    def _dispatch(self):
        if not self.ready or len(self.inflight) >= MAX_CONCURRENCY:
            return
        self.tc_calls += 1
        cap = self.client.target_concurrency(self._hdr())
        if type(cap) is not int or not 1 <= cap <= MAX_CONCURRENCY:
            raise PolicyError("invalid", f"target_concurrency must be an integer in 1..{MAX_CONCURRENCY}")
        by_pid = {c.pid: k for k, c in self.ready.items()}
        order = list(by_pid)
        if "dispatch_order" in self.client.hooks:
            order = self.client.dispatch_order(self._hdr(), order)
            if len(set(order)) != len(order) or not set(order) <= set(by_pid):
                raise PolicyError("invalid", "dispatch_order must return distinct ready call ids")
        for pid in order:
            if len(self.inflight) >= cap:
                break
            if "route_call" in self.client.hooks:
                replica = self.client.route_call(self._hdr(), pid)
            else:
                replica = min(range(len(self.schedulers)), key=lambda r: (len(self.schedulers[r]._requests), r))
            if type(replica) is not int or not 0 <= replica < len(self.schedulers):
                raise PolicyError("invalid", "route_call must return an available replica id")
            call = self.ready.pop(by_pid[pid])
            call.dispatched_at = self.t
            self.client.event(("dispatch", self.t, pid, replica))
            request = ReplayRequest(call, self, replica)
            self.inflight[request.id] = request
            self.schedulers[replica].add_request(request)

    def _complete_request(self, request):
        self.inflight.pop(request.id)
        call, node = request.call, request.call.node
        call.done_at = self.t
        self.client.event(("complete", self.t, call.pid, call.job.pid, call.stage, node.n_prompt,
            node.n_output, request.first_cached or 0, call.admitted_at - call.dispatched_at,
            self.t - call.dispatched_at))
        self._node_done(call.job, node.id)

    def _diag(self, stats, wall, flows):
        done = sorted(x for x in flows if math.isfinite(x))
        return dict(mean_flow=sum(done) / len(done) if len(done) == len(flows) else math.inf,
            p95_flow=done[min(len(done)-1, math.ceil(.95 * len(done))-1)] if done else math.inf,
            makespan=self.t - min(j.arrival for j in self.trace.jobs),
            prompt_tokens=sum(n.n_prompt for _, n in self.trace.llm_nodes()),
            cache_hit_tokens=self.cache_hit_tokens, prefill_tokens=self.prefill_tokens,
            recomputed_tokens=self.recomputed_tokens, preemptions=self.preemptions,
            evictions=self.evictions, prefill_steps=self.prefill_steps, decode_steps=self.decode_steps,
            mixed_steps=self.mixed_steps, busy_time=self.busy_time, prefill_time=self.prefill_time,
            replica_busy_time=self.replica_busy, policy_cpu_s=stats.get("cpu_s", getattr(self.client, "cpu", 0)),
            max_hook_cpu_s=stats.get("max_hook_cpu_s", 0), target_concurrency_calls=self.tc_calls,
            choose_victim_calls=self.cv_calls, events=stats.get("events", 0), wall_s=wall,
            evaluator="vidur-v1", sim_fingerprint=fingerprint(len(self.schedulers)),
            replicas=len(self.schedulers), block_size=16, kv_tokens_per_replica=self.trace.memory.num_blocks * 16,
            profile=self.predictor.metadata)
