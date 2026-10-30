"""GPU replay: agentkv's Scheduler and KVCacheManager driving the pinned
nano-vLLM ModelRunner (GeeeekExplorer/nano-vllm @ bb823b3e0698).

Only the model forward pass comes from nano-vLLM. Scheduling, block
allocation, prefix lookup and the policy hooks are the exact objects the CPU
simulator uses. Both backends expose the same event types and policy API.
Step times are measured here and estimated in the simulator, which can change
the order of events and the policy's subsequent decisions.

Forced decoding. Outputs are frozen in the trace, and every Seq already holds
prompt + output token ids. A step runs the real forward pass over the real
tokens; the sampler is replaced by an argmax that is recorded (for the smoke
test's correctness check) and otherwise ignored. Scheduler.postprocess then
advances the sequence by one position, so the next step's input is the
frozen token. nano-vLLM's SamplingParams (which refuses temperature 0) is
never used; agentkv's Seq carries temperature 1.0 for prepare_sample.

KV allocation. nano-vLLM sizes the KV cache from free GPU memory. Here
`allocate_kv_cache` allocates exactly the requested capacity and fails with
KVCapacityError if it does not fit. The capacity is the trace's num_blocks
(or AGENTKV_KV_CAPACITY if that is larger, to reuse one runner across a
split); agentkv's KVCacheManager only ever hands out block ids below the
trace's num_blocks, so a trace can never use more memory than it declares.

Nothing here imports torch at module import time: `make_engine` raises
GPUUnavailable with a clear message when the stack is missing.
"""

from __future__ import annotations

import atexit
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ..tokenizer import TOKENIZER_NAME, TOKENIZER_PATH
from .kv import EngineError, KVCacheManager
from .scheduler import Scheduler
from .seq import Seq
from .sim import Step

NANOVLLM_COMMIT = "bb823b3e06983d71485a8e1f23715ebd87d98ef8"
REPO = Path(__file__).resolve().parents[2]
NANOVLLM_PATH = REPO / "third_party" / "nano-vllm"
DEFAULT_MAX_MODEL_LEN = 40960          # max_position_embeddings of every Qwen3 dense size
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


class GPUUnavailable(RuntimeError):
    """The GPU stack (CUDA, torch, flash-attn, nano-vLLM, model weights) is missing."""


class KVCapacityError(RuntimeError):
    """The requested KV blocks do not fit on this GPU."""


# ------------------------------------------------------------------ pure helpers


def kv_block_bytes(num_layers: int, num_kv_heads: int, head_dim: int, block_size: int, dtype_bytes: int) -> int:
    """Compute the bytes needed for one block's keys and values across all model layers."""
    return 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_bytes


def kv_blocks_that_fit(total: int, free: int, peak: int, current: int, utilization: float, block_bytes: int) -> int:
    """nano-vLLM's sizing rule: what is left of `utilization * total` after the
    memory already in use and the activation peak seen during warmup."""
    used = total - free
    return max(0, int(total * utilization - used - peak + current) // block_bytes)


def check_tokenizer(model_path: str | Path) -> None:
    """Verify that the model uses the same frozen tokenizer as the traces."""
    path = Path(model_path) / "tokenizer.json"
    if not path.exists():
        raise GPUUnavailable(f"{path} not found; AGENTKV_MODEL_PATH must be a Qwen3 checkout (default Qwen3-8B)")
    want = hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest()
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise GPUUnavailable(
            f"tokenizer mismatch: {path} (sha256 {got[:12]}) is not the frozen {TOKENIZER_NAME} "
            f"tokenizer (sha256 {want[:12]}). Download Qwen/Qwen3-8B at revision b968826d (same tokenizer).")


def forced_next_token(seq: Seq) -> int:
    """Return the next recorded output token for the sequence."""
    assert seq.tokens is not None
    return seq.tokens[seq.num_tokens] if seq.num_tokens < len(seq.tokens) else -1


def prefill_inputs(seqs: list[Seq], block_size: int) -> dict:
    """Pure-Python transcription of ModelRunner.prepare_prefill (pinned), without tensors."""
    input_ids, positions, slots = [], [], []
    cu_q, cu_k = [0], [0]
    max_q = max_k = 0
    for seq in seqs:
        start = seq.num_cached_tokens
        seqlen_q = seq.num_scheduled_tokens
        end = start + seqlen_q
        seqlen_k = end
        input_ids.extend(seq[start:end])
        positions.extend(range(start, end))
        cu_q.append(cu_q[-1] + seqlen_q)
        cu_k.append(cu_k[-1] + seqlen_k)
        max_q, max_k = max(seqlen_q, max_q), max(seqlen_k, max_k)
        if not seq.block_table:
            continue
        start_block = start // block_size
        end_block = (end + block_size - 1) // block_size
        for i in range(start_block, end_block):
            slot_start = seq.block_table[i] * block_size
            if i == start_block:
                slot_start += start % block_size
            if i != end_block - 1:
                slot_end = seq.block_table[i] * block_size + block_size
            else:
                slot_end = seq.block_table[i] * block_size + end - i * block_size
            slots.extend(range(slot_start, slot_end))
    block_tables = _block_tables(seqs) if cu_k[-1] > cu_q[-1] else None
    return {"input_ids": input_ids, "positions": positions, "cu_seqlens_q": cu_q, "cu_seqlens_k": cu_k,
            "max_seqlen_q": max_q, "max_seqlen_k": max_k, "slot_mapping": slots, "block_tables": block_tables}


def decode_inputs(seqs: list[Seq], block_size: int) -> dict:
    """Pure-Python transcription of ModelRunner.prepare_decode (pinned), without tensors."""
    input_ids, positions, slots, context_lens = [], [], [], []
    for seq in seqs:
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        slots.append(seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1)
    return {"input_ids": input_ids, "positions": positions, "slot_mapping": slots,
            "context_lens": context_lens, "block_tables": _block_tables(seqs)}


def _block_tables(seqs: list[Seq]) -> list[list[int]]:
    width = max(len(s.block_table) for s in seqs)
    return [s.block_table + [-1] * (width - len(s.block_table)) for s in seqs]


# ------------------------------------------------------------------ the engine


class NanoVLLMEngine:
    """Same interface as SimEngine, but every step is one real model step and
    times are wall-clock seconds since `attach` (the replay's t0)."""

    realtime = True
    needs_tokens = True

    def __init__(self, trace, runner, *, step_log: str | Path | None = None, record_argmax: bool = False,
                 crosscheck: bool = False, verify: bool = True):
        self.trace_id = trace.trace_id
        self.memory = trace.memory
        self.config = trace.engine
        self.runner = runner
        self.verify = verify
        self.kv: KVCacheManager | None = None
        self.sched: Scheduler | None = None
        self.prefill_steps = 0
        self.decode_steps = 0
        self.prefill_tokens = 0
        self.busy_time = 0.0
        self.prefill_time = 0.0
        self.model_time = 0.0
        self.record_argmax = record_argmax
        # call key -> {position: argmax token}; positions are where the token would go
        self.argmax: dict[str, dict[int, int]] = {}
        self.finished_tokens: dict[str, list[int]] = {}
        self._log = open(step_log, "a", buffering=1) if step_log else None
        self._t0: float | None = None
        if crosscheck and hasattr(runner, "prepare_prefill"):
            install_crosscheck(runner, self.memory.block_size)

    def attach(self, *, choose_victim, emit_block=None, on_call=None, hooks=None) -> None:
        Seq.block_size = self.memory.block_size
        self.kv = KVCacheManager(self.memory.num_blocks, self.memory.block_size, choose_victim=choose_victim,
                                 emit=emit_block, verify=self.verify)
        self.sched = Scheduler(self.kv, max_num_seqs=self.config.max_num_seqs,
                               max_num_batched_tokens=self.config.max_num_batched_tokens, on_call=on_call,
                               hooks=hooks)
        self._t0 = time.perf_counter()

    def clock(self) -> float:
        if self._t0 is None:
            raise EngineError("engine used before attach()")
        return time.perf_counter() - self._t0

    def add(self, seq: Seq) -> None:
        if seq.tokens is None or len(seq.tokens) != seq.num_prompt_tokens + seq.max_tokens:
            raise EngineError("GPU replay needs prompt + frozen output tokens on every sequence")
        self.sched.add(seq)

    def has_work(self) -> bool:
        return not self.sched.is_finished()

    def begin_step(self, now: float, horizon: float = math.inf) -> Step:
        """Schedule and run exactly one model step. Scheduling time (including any
        victim choice by the policy) counts toward the step."""
        t_sched = self.clock()
        seqs, is_prefill = self.sched.schedule()
        if is_prefill:
            chunks = [(s.num_cached_tokens, s.num_scheduled_tokens) for s in seqs]
            self.prefill_tokens += sum(n for _, n in chunks)
            context = sum(c + n for c, n in chunks)
        else:
            chunks = None
            context = sum(s.num_tokens for s in seqs)
        t_run = self.clock()
        tokens = self.runner.run(seqs, is_prefill)   # .tolist() inside synchronizes the GPU
        t_end = self.clock()
        self.model_time += t_end - t_run
        if self.record_argmax:
            self._record(seqs, tokens, is_prefill)
        if self._log:
            self._log.write(json.dumps({
                "trace": self.trace_id, "t": round(t_run, 6), "prefill": is_prefill, "chunks": chunks,
                "batch": len(seqs), "context": context, "seconds": t_end - t_run,
                "schedule_seconds": t_run - t_sched}) + "\n")
        start = max(now, t_sched)
        return Step(seqs, is_prefill, start, max(0.0, t_end - start))

    def end_step(self, step: Step) -> list[Seq]:
        if step.is_prefill:
            self.prefill_steps += 1
            self.prefill_time += step.duration
        else:
            self.decode_steps += 1
        self.busy_time += step.duration
        finished = self.sched.postprocess(step.seqs, step.is_prefill)
        for s in finished:
            if s.num_tokens != s.num_prompt_tokens + s.max_tokens:
                raise EngineError("a sequence finished before producing its frozen output")
            self.finished_tokens[_key(s)] = s.tokens[:s.num_tokens].tolist()
        return finished

    def wait_until(self, t: float) -> None:
        while True:
            delay = t - self.clock()
            if delay <= 0:
                return
            time.sleep(min(delay, 0.05))

    def close(self) -> None:
        if self._log:
            self._log.close()
            self._log = None

    def _record(self, seqs: list[Seq], tokens: list[int], is_prefill: bool) -> None:
        for s, tok in zip(seqs, tokens):
            if is_prefill and s.num_cached_tokens + s.num_scheduled_tokens < s.num_tokens:
                continue   # a prefill chunk that does not reach the end predicts nothing
            self.argmax.setdefault(_key(s), {})[s.num_tokens] = int(tok)


def _key(seq: Seq) -> str:
    return getattr(seq.owner, "key", None) or str(seq.seq_id)


class ReferenceRunner:
    """A pure-Python stand-in for ModelRunner. It builds the same model inputs
    (checking them), returns the frozen next tokens, and optionally sleeps for a
    TimingModel's step time. Used by the CPU tests and for dry runs of the GPU path."""

    def __init__(self, block_size: int, capacity: int, timing=None):
        self.block_size = block_size
        self.capacity = capacity
        self.timing = timing
        self.steps = 0

    def run(self, seqs: list[Seq], is_prefill: bool) -> list[int]:
        bs = self.block_size
        inp = prefill_inputs(seqs, bs) if is_prefill else decode_inputs(seqs, bs)
        slots = inp["slot_mapping"]
        if len(set(slots)) != len(slots):
            raise EngineError("two tokens of one step write the same KV slot")
        if slots and (min(slots) < 0 or max(slots) >= self.capacity * bs):
            raise EngineError("KV slot outside the allocated cache")
        if len(slots) != len(inp["input_ids"]):
            raise EngineError("slot mapping does not cover every input token")
        self.steps += 1
        if self.timing is not None:
            if is_prefill:
                t = self.timing.prefill([(s.num_cached_tokens, s.num_scheduled_tokens) for s in seqs])
            else:
                t = self.timing.decode(len(seqs), sum(s.num_tokens for s in seqs))
            time.sleep(t)
        out = []
        for s in seqs:
            done = not is_prefill or s.num_cached_tokens + s.num_scheduled_tokens == s.num_tokens
            out.append(forced_next_token(s) if done else -1)
        return out


def install_crosscheck(runner, block_size: int) -> None:
    """Wrap a real ModelRunner so every prepared batch is compared with the
    pure-Python transcription (smoke test only; it costs a device sync)."""
    from nanovllm.utils.context import get_context

    orig_prefill, orig_decode = runner.prepare_prefill, runner.prepare_decode

    def check(ref: dict, input_ids, positions) -> None:
        ctx = get_context()
        got = {"input_ids": input_ids.tolist(), "positions": positions.tolist(),
               "slot_mapping": ctx.slot_mapping.tolist()}
        for k, v in got.items():
            if ref[k] != v:
                raise EngineError(f"crosscheck: {k} differs between ModelRunner and prefill/decode_inputs")

    def prepare_prefill(seqs):
        ids, pos = orig_prefill(seqs)
        check(prefill_inputs(seqs, block_size), ids, pos)
        return ids, pos

    def prepare_decode(seqs):
        ids, pos = orig_decode(seqs)
        check(decode_inputs(seqs, block_size), ids, pos)
        return ids, pos

    runner.prepare_prefill, runner.prepare_decode = prepare_prefill, prepare_decode


# ------------------------------------------------------------------ the real runner


@dataclass(frozen=True)
class RunnerSpec:
    model: str
    block_size: int
    max_num_seqs: int
    max_num_batched_tokens: int
    max_model_len: int
    enforce_eager: bool
    gpu_memory_utilization: float


_RUNNER: dict = {"spec": None, "runner": None}


def _import_stack():
    try:
        import torch
    except ImportError as exc:
        raise GPUUnavailable("GPU replay needs torch (pip install -e '.[gpu]' on the GPU host)") from exc
    if not torch.cuda.is_available():
        raise GPUUnavailable("GPU replay needs a CUDA GPU; torch.cuda.is_available() is False")
    try:
        import nanovllm  # noqa: F401
    except ImportError:
        if NANOVLLM_PATH.exists() and str(NANOVLLM_PATH) not in sys.path:
            sys.path.insert(0, str(NANOVLLM_PATH))
        try:
            import nanovllm  # noqa: F401
        except ImportError as exc:
            raise GPUUnavailable(f"cannot import nano-vLLM ({exc}); install third_party/nano-vllm at "
                                 f"{NANOVLLM_COMMIT[:12]} with flash-attn, triton, xxhash") from exc
    _check_pin()
    from nanovllm import config as cfg_mod
    from nanovllm.engine import model_runner as mr_mod
    from nanovllm.engine import sequence as seq_mod
    return torch, cfg_mod, mr_mod, seq_mod


def _check_pin() -> None:
    import nanovllm
    head = Path(nanovllm.__file__).resolve().parents[1] / ".git" / "HEAD"
    if head.exists():
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            p = head.parent / ref.split(None, 1)[1]
            ref = p.read_text().strip() if p.exists() else ref
        if ref != NANOVLLM_COMMIT:
            print(f"agentkv: warning: nano-vLLM checkout is at {ref[:12]}, not the pinned "
                  f"{NANOVLLM_COMMIT[:12]}", file=sys.stderr)


def _build_runner(spec: RunnerSpec, capacity: int):
    torch, cfg_mod, mr_mod, seq_mod = _import_stack()
    from torch import nn

    if spec.block_size % 256:
        raise GPUUnavailable(f"block_size {spec.block_size}: flash-attn paged KV needs a multiple of 256")
    try:
        config = cfg_mod.Config(spec.model, max_num_batched_tokens=spec.max_num_batched_tokens,
                                max_num_seqs=spec.max_num_seqs, max_model_len=spec.max_model_len,
                                gpu_memory_utilization=spec.gpu_memory_utilization, tensor_parallel_size=1,
                                enforce_eager=spec.enforce_eager, kvcache_block_size=spec.block_size,
                                num_kvcache_blocks=capacity)
    except AssertionError as exc:
        raise GPUUnavailable(f"nano-vLLM rejected the config (is {spec.model} a model directory?): {exc}") from exc
    hf = config.hf_config
    if getattr(hf, "dtype", None) is None or isinstance(getattr(hf, "dtype", None), str):
        dt = getattr(hf, "dtype", None) or getattr(hf, "torch_dtype", None) or "bfloat16"
        hf.dtype = getattr(torch, dt) if isinstance(dt, str) else dt
    seq_mod.Sequence.block_size = spec.block_size

    class ForcedSampler(nn.Module):
        """Greedy argmax. Its tokens are recorded, never fed back: outputs are frozen."""

        def forward(self, logits, temperatures):
            return logits.argmax(dim=-1)

    class AgentKVModelRunner(mr_mod.ModelRunner):

        def allocate_kv_cache(self):
            config = self.config
            hf = config.hf_config
            free, total = torch.cuda.mem_get_info()
            stats = torch.cuda.memory_stats()
            num_kv_heads = hf.num_key_value_heads // self.world_size
            head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
            block_bytes = kv_block_bytes(hf.num_hidden_layers, num_kv_heads, head_dim, self.block_size,
                                         hf.dtype.itemsize)
            fit = kv_blocks_that_fit(total, free, stats["allocated_bytes.all.peak"],
                                     stats["allocated_bytes.all.current"], config.gpu_memory_utilization,
                                     block_bytes)
            want = config.num_kvcache_blocks
            if want > fit:
                raise KVCapacityError(
                    f"{want} KV blocks of {block_bytes / 2**20:.1f} MiB do not fit: at most {fit} fit in "
                    f"{config.gpu_memory_utilization:.0%} of {total / 2**30:.1f} GiB. Use a larger GPU or "
                    f"raise AGENTKV_GPU_MEMORY_UTILIZATION.")
            self.kv_cache = torch.empty(2, hf.num_hidden_layers, want, self.block_size, num_kv_heads, head_dim)
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.kv_cache[0, layer_id]
                    module.v_cache = self.kv_cache[1, layer_id]
                    layer_id += 1

    original = mr_mod.Sampler
    mr_mod.Sampler = ForcedSampler     # ModelRunner.__init__ builds its sampler from this name
    try:
        runner = AgentKVModelRunner(config, 0, [])
    finally:
        mr_mod.Sampler = original
    runner.capacity = capacity
    return runner


def _teardown() -> None:
    runner = _RUNNER["runner"]
    if runner is None:
        return
    _RUNNER.update(spec=None, runner=None)
    try:
        runner.exit()   # frees CUDA graphs and the process group, so a new runner can start
    finally:
        del runner
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


atexit.register(_teardown)


def get_runner(spec: RunnerSpec, capacity: int):
    """One ModelRunner per process (nano-vLLM owns the default process group),
    rebuilt only when the spec changes or more KV blocks are needed."""
    runner = _RUNNER["runner"]
    if runner is not None and _RUNNER["spec"] == spec and runner.capacity >= capacity:
        return runner
    _teardown()
    runner = _build_runner(spec, capacity)
    _RUNNER.update(spec=spec, runner=runner)
    return runner


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def make_engine(trace, *, step_log: str | Path | None = None, record_argmax: bool = False,
                crosscheck: bool = False, enforce_eager: bool | None = None, runner=None) -> NanoVLLMEngine:
    """The GPU engine for one trace. `runner="reference"` (or AGENTKV_GPU_RUNNER=reference)
    uses ReferenceRunner instead of the model, which needs no GPU."""
    step_log = step_log or os.environ.get("AGENTKV_STEP_LOG") or None
    runner = runner or os.environ.get("AGENTKV_GPU_RUNNER") or "nanovllm"
    if runner == "reference":
        runner = ReferenceRunner(trace.memory.block_size, trace.memory.num_blocks)
    elif runner == "nanovllm":
        model = os.environ.get("AGENTKV_MODEL_PATH")
        if not model:
            raise GPUUnavailable("set AGENTKV_MODEL_PATH to a local Qwen3 checkout (default Qwen/Qwen3-8B at revision b968826d)")
        check_tokenizer(model)
        if enforce_eager is None:
            enforce_eager = _env_flag("AGENTKV_ENFORCE_EAGER")
        spec = RunnerSpec(
            model=str(model), block_size=trace.memory.block_size, max_num_seqs=trace.engine.max_num_seqs,
            max_num_batched_tokens=trace.engine.max_num_batched_tokens,
            max_model_len=int(os.environ.get("AGENTKV_MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN)),
            enforce_eager=bool(enforce_eager),
            gpu_memory_utilization=float(os.environ.get("AGENTKV_GPU_MEMORY_UTILIZATION",
                                                        DEFAULT_GPU_MEMORY_UTILIZATION)))
        longest = max((n.n_total for _, n in trace.llm_nodes()), default=0)
        if longest > spec.max_model_len:
            raise GPUUnavailable(f"{trace.trace_id}: a call has {longest} tokens, over max_model_len "
                                 f"{spec.max_model_len}")
        capacity = max(trace.memory.num_blocks, int(os.environ.get("AGENTKV_KV_CAPACITY", 0) or 0))
        runner = get_runner(spec, capacity)
        if runner.config.max_model_len < longest:
            raise GPUUnavailable(f"{trace.trace_id}: the model's max_model_len {runner.config.max_model_len} "
                                 f"is shorter than a {longest}-token call")
        vocab = runner.config.hf_config.vocab_size
        top = max((max(t) for t in trace.seg_tokens.values() if len(t)), default=0)
        if top >= vocab:
            raise GPUUnavailable(f"token id {top} is outside the model vocabulary ({vocab})")
    return NanoVLLMEngine(trace, runner, step_log=step_log, record_argmax=record_argmax, crosscheck=crosscheck)
