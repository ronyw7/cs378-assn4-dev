"""Connect policy decisions and causal events to Vidur's V1 scheduler.

Vidur owns batching and KV allocation. These subclasses add the assignment's
policy hooks and record cache events. Newly computed blocks become visible at
batch completion. Frozen output hashes are never used to look ahead in a call.
"""

from collections import deque

from .._vendor.vidur.config import (CacheConfig, FCFSRequestQueueConfig,
    SyntheticRequestGeneratorConfig, VllmV1SchedulerConfig)
from .._vendor.vidur.entities import Replica, Request
from .._vendor.vidur.kv_cache.replica_kv_cache_manager import ReplicaKVCacheManager
from .._vendor.vidur.kv_cache.kv_cache_block_pool import BlockPool
from .._vendor.vidur.kv_cache.utils import BlockHashType
from .._vendor.vidur.scheduler.replica_scheduler.vllm_v1_replica_scheduler import VLLMV1ReplicaScheduler
from ..sandbox.runtime import PolicyError


class ReplayRequest(Request):
    def __init__(self, call, driver, replica):
        n = call.node
        super().__init__(driver.t, n.n_prompt, n.n_output, list(n.hashes), 16)
        self.call, self.driver, self.replica_index = call, driver, replica
        self.first_cached = None
        self.high_water = 0

    def on_cache_hit(self, count):
        super().on_cache_hit(count)
        if self.first_cached is None:
            self.first_cached = count
            self.call.admitted_at = self.driver.t
            self.driver.cache_hit_tokens += count
        self.driver.client.event(("admit", self.driver.t, self.call.pid, count))

    def restart(self):
        # Upstream assumes that a preempted request has finished prefill. Keep
        # the remaining original prompt when preempting a chunked prefill.
        prompt = max(self.call.node.n_prompt, self.num_processed_tokens)
        total = self.total_tokens
        super().restart()
        self._num_prefill_tokens, self._num_decode_tokens = prompt, total - prompt
        self.driver.preemptions += 1
        self.driver.client.event(("preempt", self.driver.t, self.call.pid))


class PolicyBlockPool(BlockPool):
    def __init__(self, count, driver, replica):
        super().__init__(count, True)
        self.driver, self.replica = driver, replica
        self.current_request = None
        self.published = {}  # slot -> opaque residence id
        self.residences = {}  # opaque residence id -> block
        self.generations = [0] * count
        self.empty_ids = set(range(count))
        self.pending = []

    def get_cached_block(self, block_hash):
        # A block scheduled in this batch has not been computed yet.
        return next((b for b in self.cached_block_hash_to_block.get(block_hash, {}).values()
                     if b.block_id in self.published), None)

    def get_new_blocks(self, count):
        result = []
        if count > self.get_num_free_blocks():
            raise RuntimeError("Vidur requested more free blocks than are available")
        for _ in range(count):
            if self.empty_ids:
                block = self.blocks[min(self.empty_ids)]
                self.empty_ids.remove(block.block_id)
            else:
                d = self.driver
                d.cv_calls += 1
                chosen = d.client.choose_victim(d._hdr(self.replica))
                block = self.residences.get(chosen)
                if block is None or block.ref_cnt != 0:
                    raise PolicyError("invalid", "choose_victim must return an eligible block on this replica")
                d.client.event(("evict", d.t, chosen))
                d.evictions += 1
                del self.residences[chosen]
                del self.published[block.block_id]
            self.free_block_queue.remove(block)
            self._maybe_evict_cached_block(block)
            self.generations[block.block_id] += 1
            block.incr_ref()
            result.append(block)
        return result

    def cache_full_blocks(self, **kwargs):
        super().cache_full_blocks(**kwargs)
        request = kwargs["request"]
        start, end = kwargs["num_cached_blocks"], kwargs["num_full_blocks"]
        for depth in range(start, end):
            block = kwargs["blocks"][depth]
            self.pending.append((block, request, depth))

    def publish(self):
        d = self.driver
        for block, request, depth in self.pending:
            slot = block.block_id
            pid = d.rekey._h("r", f"{self.replica}:{slot}:{self.generations[slot]}".encode())
            self.published[slot] = pid
            self.residences[pid] = block
            hashes = request.call.node.hashes
            d.client.event(("alloc", d.t, pid, d.rekey.block(hashes[depth - 1]) if depth else None,
                            depth, request.call.pid, d.rekey.block(hashes[depth]), self.replica))
        self.pending.clear()

    def touch(self, blocks):
        super().touch(blocks)
        for block in blocks:
            self.driver.client.event(("hit", self.driver.t, self.published[block.block_id],
                                      self.current_request.call.pid))

    def free_blocks(self, ordered_blocks):
        blocks = list(ordered_blocks)
        super().free_blocks(blocks)
        for block in blocks:
            if block.block_id in self.published:
                self.driver.client.event(("release", self.driver.t, self.published[block.block_id],
                                          self.current_request.call.pid, block.ref_cnt))
            elif block.ref_cnt == 0:
                self.empty_ids.add(block.block_id)


class PolicyCacheManager(ReplicaKVCacheManager):
    def __init__(self, count, driver, replica):
        super().__init__(block_size=16, num_gpu_blocks=count, enable_caching=True,
                         caching_hash_algo="sha256", num_preallocate_tokens=0)
        self.block_pool = PolicyBlockPool(count, driver, replica)

    def get_computed_blocks(self, request):
        hashes = self.req_to_block_hashes[request.id]
        if not hashes:
            hashes = [BlockHashType(h, ()) for h in request.block_hash_ids]
            self.req_to_block_hashes[request.id] = hashes
        blocks = []
        for h in hashes[:request.num_prefill_tokens // self.block_size]:
            block = self.block_pool.get_cached_block(h)
            if block is None:
                break
            blocks.append(block)
        return blocks, len(blocks) * self.block_size

    def allocate_slots(self, request, *args, **kwargs):
        self.block_pool.current_request = request
        return super().allocate_slots(request, *args, **kwargs)

    def free(self, request):
        self.block_pool.current_request = request
        return super().free(request)


class PolicyQueue:
    """FIFO queue with an optional policy permutation before each batch."""
    def __init__(self):
        self.queue = deque()

    def push(self, request):
        self.queue.append(request)

    def pop(self):
        return self.queue.popleft()

    def peek(self):
        return self.queue[0]

    def to_list(self):
        return list(self.queue)

    def __len__(self):
        return len(self.queue)

    def extend(self, requests):
        self.queue.extend(requests)


class PolicyScheduler(VLLMV1ReplicaScheduler):
    def __init__(self, driver, replica, predictor, count):
        config = predictor._replica_config
        super().__init__(config, VllmV1SchedulerConfig(batch_size_cap=16, chunk_size=2048),
            SyntheticRequestGeneratorConfig(), CacheConfig(block_size=16, num_blocks=count,
            enable_prefix_caching=True, num_preallocate_tokens=0),
            FCFSRequestQueueConfig(), Replica(config), predictor)
        self.driver, self.index = driver, replica
        self._kv_cache_manager = PolicyCacheManager(count, driver, replica)
        self._waiting_queue = PolicyQueue()
        self.busy = False

    @property
    def pool(self):
        return self._kv_cache_manager.block_pool

    def _get_next_batch(self, now):
        d = self.driver
        if "batch_token_budget" in d.client.hooks:
            budget = d.client.batch_token_budget(d._hdr(self.index))
            if type(budget) is not int or budget not in (512, 1024, 2048, 4096):
                raise PolicyError("invalid", "batch_token_budget must be 512, 1024, 2048, or 4096")
            self._config.chunk_size = budget
        if self._waiting_queue and "admission_order" in d.client.hooks:
            requests = {r.call.pid: r for r in self._waiting_queue.to_list()}
            order = d.client.admission_order(d._hdr(self.index), list(requests))
            if sorted(order) != sorted(requests):
                raise PolicyError("invalid", "admission_order must return every waiting call exactly once")
            self._waiting_queue.queue = deque(requests[p] for p in order)
        return super()._get_next_batch(now)

    def _select_preemption_victim(self):
        # Memory already reserved for this batch cannot be reclaimed while
        # that batch still references it.
        eligible = {r.call.pid: r for r in self._running if r.id not in self.scheduled_req_ids}
        if not eligible:
            raise RuntimeError("No safe Vidur preemption candidate")
        if "choose_preempted" in self.driver.client.hooks:
            chosen = self.driver.client.choose_preempted(self.driver._hdr(self.index), list(eligible))
            if chosen not in eligible:
                raise PolicyError("invalid", "choose_preempted must return an eligible running call")
        else:
            chosen = next(reversed(eligible))
        request = eligible[chosen]
        self._running.remove(request)
        return request
