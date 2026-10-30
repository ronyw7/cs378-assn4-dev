"""Implement the reference LRU policy (Part 2).

Evict the least recently used cached block (one that no call holds). A block
becomes recent when it is allocated or accessed (a hit); a release alone does
not make it recent. The concurrency limit defaults to eight unfinished LLM calls across all replicas.
Use this completed policy as the starting point for Part 3.

Run the public LRU and cache-invariant tests with:

    pytest tests/test_part2_lru.py tests/test_kv_invariants.py
"""

from collections import OrderedDict

from agentkv.policy import Policy


class ReferencePolicy(Policy):

    def __init__(self, params=None):
        """Initialize the policy and its recency state."""
        super().__init__(params)
        self.cap = int(self.params.get("cap", 8))
        # [Part 2] TODO: Set up the state used to track block accesses.
        # You need to retain the order of block ids from least to most recently
        # used across calls to on_event and choose_victim. An OrderedDict is
        # one option; allocation and access should be efficient to record.
        # The fixed concurrency cap above is already provided.
        # Use: OrderedDict, or another data structure stored on self.
        raise NotImplementedError("Part 2: fill __init__")
        # END OF YOUR CODE

    def on_event(self, event, state):
        """Update block recency from an engine event."""
        # [Part 2] TODO: Record allocations, cache hits, and evictions.
        # An alloc or hit makes event.block the most recently used block.
        # Follow event delivery order even when several events have the same
        # timestamp; the timestamp alone does not distinguish those accesses.
        # An evict removes that block from your tracking state. A release does
        # not count as an access, so it must not change recency. Ignore the
        # other event kinds. Your tracking state may include active blocks;
        # choose_victim will receive the subset eligible for eviction.
        # Use: event.kind, event.block, and the state initialized in __init__.
        raise NotImplementedError("Part 2: fill on_event")
        # END OF YOUR CODE

    def target_concurrency(self, view, state):
        return self.cap

    def choose_victim(self, view, candidates, state):
        """Return the least recently used block eligible for eviction."""
        # [Part 2] TODO: Choose the least recently used id in candidates.
        # Your recency state can also contain active blocks, which cannot be
        # evicted. Only consider ids in the supplied candidate list. Its order
        # is based on release time, so returning candidates[0] does not
        # implement LRU. Leave recency unchanged here; the engine sends an
        # evict event after it overwrites the chosen block.
        # Use: candidates and the recency state maintained by on_event.
        raise NotImplementedError("Part 2: fill choose_victim")
        # END OF YOUR CODE
