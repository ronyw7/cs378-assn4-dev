"""A fixed policy for infrastructure tests: evict the oldest-released block,
fixed cap (default 4). Independent of the Part 2 solution."""

from agentkv.policy import Policy


class FIFOPolicy(Policy):
    def target_concurrency(self, view, state):
        return int(self.params.get("cap", 4))

    def choose_victim(self, view, candidates, state):
        return candidates[0]
