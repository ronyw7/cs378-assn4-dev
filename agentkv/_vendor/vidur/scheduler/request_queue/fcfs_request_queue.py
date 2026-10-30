import heapq
from collections import deque
from typing import Deque, List

from agentkv._vendor.vidur.entities.request import Request
from agentkv._vendor.vidur.scheduler.request_queue.base_request_queue import BaseRequestQueue
from agentkv._vendor.vidur.scheduler.request_queue.prioritised_request import PrioritizedRequest


class FCFSRequestQueue(BaseRequestQueue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_queue: List[PrioritizedRequest] = []
        self._num_prefill_tokens = 0

    def _get_prioritized_request(self, request: Request) -> PrioritizedRequest:
        return PrioritizedRequest(request, request.arrived_at)

    def push(self, request):
        heapq.heappush(self._request_queue, self._get_prioritized_request(request))
        self._num_prefill_tokens += request.num_prefill_tokens

    def pop(self):
        request = heapq.heappop(self._request_queue).request
        self._num_prefill_tokens -= request.num_prefill_tokens
        return request

    def peek(self):
        return self._request_queue[0].request

    def to_list(self):
        return [
            prioritized_request.request for prioritized_request in self._request_queue
        ]

    def __len__(self):
        return len(self._request_queue)

    def get_num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    def sort(self, requests: Deque[Request]) -> Deque[Request]:
        return deque(sorted(requests, key=lambda x: (x.arrived_at, x.id)))
