"""Prefix-aware eviction and cache-affinity routing (Part 2, Tasks 2 and 3).

choose_victim: evict the least recently used cached block that is a leaf of
its replica's prefix tree (no resident block on that replica extends it), and
fall back to plain LRU when no candidate is a leaf. Track the tree from alloc
and evict events; do not read view.blocks.

route_call: among replicas whose in_flight is at most `slack` above the least
loaded replica's, send the call to the one holding the longest cached prefix of
its prompt; break ties by lower in_flight, then lower replica id.

params: cap (default 8); tails (default True; False gives plain LRU); slack
(default 1; -1 routes to the least loaded replica, like the reference). With
tails=False and slack=-1 this policy makes the same decisions as the reference.

    pytest tests/test_part2_prefix.py
"""

from collections import OrderedDict

from agentkv.policy import Policy


class PrefixPolicy(Policy):

    def __init__(self, params=None):
        super().__init__(params)
        self.cap = int(self.params.get("cap", 8))
        self.tails = bool(self.params.get("tails", True))
        self.slack = int(self.params.get("slack", 1))
        # [Part 2] TODO: Set up the state for recency and for each replica's
        # prefix tree (which resident blocks extend which prefix).
        raise NotImplementedError("Part 2: fill PrefixPolicy.__init__")
        # END OF YOUR CODE

    def on_event(self, event, state):
        """Update recency and the prefix trees."""
        # [Part 2] TODO: alloc and hit make a block most recent; alloc adds it to
        # its replica's tree under event.parent (None for the first block of a
        # prompt); evict removes it.
        raise NotImplementedError("Part 2: fill PrefixPolicy.on_event")
        # END OF YOUR CODE

    def target_concurrency(self, view, state):
        return self.cap

    def choose_victim(self, view, candidates, state):
        """Return the least recently used leaf among candidates, or the least recently used candidate."""
        # [Part 2] TODO: With tails=False, behave exactly like LRU.
        raise NotImplementedError("Part 2: fill PrefixPolicy.choose_victim")
        # END OF YOUR CODE

    def route_call(self, view, call, state):
        """Choose a replica for a ready call."""
        # [Part 2] TODO: Use view.replicas[r].in_flight for load and
        # view.cached_prefix_len(call, r) for the cached prefix.
        raise NotImplementedError("Part 2: fill PrefixPolicy.route_call")
        # END OF YOUR CODE
