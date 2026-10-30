from abc import ABC, abstractmethod
from typing import Deque, Iterable, List

from agentkv._vendor.vidur.config.config import BaseRequestQueueConfig
from agentkv._vendor.vidur.entities.request import Request
from agentkv._vendor.vidur.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)
from agentkv._vendor.vidur.scheduler.request_queue.prioritised_request import PrioritizedRequest


class BaseRequestQueue(ABC):
    def __init__(
        self,
        request_queue_config: BaseRequestQueueConfig,
        execution_time_predictor: BaseExecutionTimePredictor,
    ):
        self._config = request_queue_config
        self._execution_time_predictor = execution_time_predictor

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def peek(self) -> Request:
        pass

    @abstractmethod
    def pop(self) -> Request:
        pass

    @abstractmethod
    def push(self, request: Request):
        pass

    def extend(self, requests: Iterable[Request]):
        for request in requests:
            self.push(request)

    @abstractmethod
    def to_list(self) -> List[Request]:
        pass

    @abstractmethod
    def get_num_prefill_tokens(self) -> int:
        pass

    @abstractmethod
    def sort(self, requests: Iterable[Request]) -> Deque[Request]:
        pass
