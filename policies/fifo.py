"""FIFO comparison policy with a concurrency limit of eight LLM calls.

An eligible cache block enters the eviction queue when its last active
reference is released. Evict the block that has waited there longest.
Other decisions use the same defaults as policies/reference.py.
"""

from agentkv.policy import Policy


class FIFO(Policy):
    def target_concurrency(self, view, state):
        return int(self.params.get("cap", 8))

    def choose_victim(self, view, candidates, state):
        # The engine supplies eligible blocks in release order, oldest first.
        return candidates[0]
