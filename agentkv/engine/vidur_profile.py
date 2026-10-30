"""Vidur kernel-time predictions fitted to the bundled Llama-3-8B/A100 profile.

The tables contain numbers, not serialized executable model objects. Batch
features and timing aggregation use the pinned upstream implementation.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from .._vendor.vidur.config import CacheConfig, RandomForestExecutionTimePredictorConfig, ReplicaConfig
from .._vendor.vidur.execution_time_predictor.base_execution_time_predictor import BaseExecutionTimePredictor
from .._vendor.vidur.execution_time_predictor.prediction_methods import PredictionMethods

DATA = Path(__file__).resolve().parents[1] / "data/vidur"


class _Table:
    def __init__(self, name, values):
        self.name, self.values = name, values

    def __getitem__(self, key):
        if self.name == "attn_prefill":
            index = key[0] // 64, key[1] // 32 - 1
            valid = key[0] % 64 == 0 and key[1] >= 32 and key[1] % 32 == 0
        elif self.name == "attn_decode":
            index = key[0] - 1, key[1] // 64
            valid = key[0] >= 1 and key[1] % 64 == 0
        else:
            index, valid = key, key[0] >= 1
        if not valid or any(i < 0 or i >= s for i, s in zip(index, self.values.shape)):
            raise ValueError(f"Batch is outside the Vidur profile table: {self.name}{key}")
        return float(self.values[index])


class FrozenPredictor(PredictionMethods, BaseExecutionTimePredictor):
    def __init__(self):
        self.metadata = json.loads((DATA / "profile.json").read_text())
        artifact = DATA / "llama3_8b_a100.npz"
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != self.metadata["table_sha256"]:
            raise ValueError("Vidur timing table checksum does not match profile.json")
        replica = ReplicaConfig(model_name=self.metadata["model"], device="a100", network_device="a100_dgx",
                                tensor_parallel_size=1, num_pipeline_stages=1)
        config = RandomForestExecutionTimePredictorConfig(skip_cpu_overhead_modeling=True)
        super().__init__(config, replica, CacheConfig(block_size=16))
        for name in ("attention_prefill_batching_overhead_fraction", "attention_decode_batching_overhead_fraction"):
            setattr(self, "_" + name, self.metadata[name])
        with np.load(artifact, allow_pickle=False) as data:
            self._predictions = {name: _Table(name, data[name]) for name in data.files}


@lru_cache(maxsize=1)
def predictor():
    return FrozenPredictor()


def fingerprint(replicas=2):
    """Identify the code, profile, and fixed hardware configuration of an evaluation."""
    root = Path(__file__).resolve().parents[1]
    files = [DATA / "profile.json", *sorted((root / "_vendor/vidur").rglob("*.py")),
             Path(__file__), root / "vidur_replay.py", root / "engine/vidur_adapter.py"]
    h = hashlib.sha256(f"vidur-v1:replicas={replicas}:cap=16:block=16:budget=512,1024,2048,4096".encode())
    for path in files:
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return "vidur-" + h.hexdigest()[:24]
