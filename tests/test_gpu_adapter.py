"""CPU-only tests of the GPU replay path (agentkv.engine.gpu, calibrate, fidelity).

The model forward pass needs a GPU; everything around it does not. These tests
run the pinned nano-vLLM input preparation code against agentkv's Seq, drive
the real-time engine with a pure-Python runner, and check calibration and
fidelity arithmetic. tests/staff/test_gpu_smoke.py covers the GPU itself.
"""

import ast
import importlib.util
import json
import math
import shutil
import sys
import types
from array import array
from pathlib import Path

import pytest
from conftest import FIFO, REPO, build_chat_trace

from agentkv.engine import calibrate, fidelity, gpu
from agentkv.engine.kv import EngineError, KVCacheManager
from agentkv.engine.seq import Seq
from agentkv.engine.timing import TimingModel
from agentkv.replay import Replay
from agentkv.sandbox.client import InProcessClient
from agentkv.tokenizer import TOKENIZER_PATH
from agentkv.trace import chain_hashes

NANO = REPO / "third_party" / "nano-vllm" / "nanovllm"
needs_nano = pytest.mark.skipif(not NANO.exists(), reason="third_party/nano-vllm checkout not present")


# ------------------------------------------------------------------ lazy import


def test_import_is_lazy_and_errors_are_clear(tmp_path, monkeypatch, chat_trace):
    monkeypatch.delenv("AGENTKV_GPU_RUNNER", raising=False)
    monkeypatch.delenv("AGENTKV_MODEL_PATH", raising=False)
    with pytest.raises(gpu.GPUUnavailable, match="AGENTKV_MODEL_PATH"):
        gpu.make_engine(chat_trace)
    model = tmp_path / "model"
    model.mkdir()
    shutil.copy(TOKENIZER_PATH, model / "tokenizer.json")
    monkeypatch.setenv("AGENTKV_MODEL_PATH", str(model))
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False
    if not has_torch:
        with pytest.raises(gpu.GPUUnavailable, match="torch"):
            gpu.make_engine(chat_trace)


def test_tokenizer_check(tmp_path):
    with pytest.raises(gpu.GPUUnavailable, match="not found"):
        gpu.check_tokenizer(tmp_path)
    shutil.copy(TOKENIZER_PATH, tmp_path / "tokenizer.json")
    gpu.check_tokenizer(tmp_path)
    with open(tmp_path / "tokenizer.json", "ab") as f:
        f.write(b" ")
    with pytest.raises(gpu.GPUUnavailable, match="mismatch"):
        gpu.check_tokenizer(tmp_path)


def test_kv_sizing():
    # Qwen3-0.6B: 28 layers, 8 KV heads of dim 128, bf16, 256-token blocks = 28 MiB
    bb = gpu.kv_block_bytes(28, 8, 128, 256, 2)
    assert bb == 29_360_128
    gib = 1 << 30
    # 24 GiB card, 2 GiB in use, 1 GiB activation peak above current: 0.9*24 - 2 - 1 = 18.6 GiB
    assert gpu.kv_blocks_that_fit(24 * gib, 22 * gib, 2 * gib, 1 * gib, 0.9, bb) == int(18.6 * gib) // bb
    assert gpu.kv_blocks_that_fit(gib, 0, gib, 0, 0.9, bb) == 0


# ------------------------------------------------------------------ duck typing vs the pinned source


def _load_pinned_sequence():
    pkg = types.ModuleType("_nano_stub")
    pkg.__path__ = []
    sys.modules["_nano_stub"] = pkg
    spec = importlib.util.spec_from_file_location("_nano_stub.sampling_params", NANO / "sampling_params.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    src = (NANO / "engine" / "sequence.py").read_text().replace(
        "from nanovllm.sampling_params import SamplingParams", "")
    ns = {"SamplingParams": sp.SamplingParams}
    exec(compile(src, "sequence.py", "exec"), ns)
    return ns["Sequence"]


@needs_nano
def test_seq_matches_pinned_sequence():
    Sequence = _load_pinned_sequence()
    for bs in (16, 256):
        Sequence.block_size = Seq.block_size = bs
        for n_prompt, n_out in ((1, 1), (15, 3), (16, 17), (300, 300), (513, 40)):
            toks = list(range(1000, 1000 + n_prompt + n_out))
            ours = Seq(n_prompt, n_out, tuple(chain_hashes(array("I", toks), bs)), tokens=array("I", toks))
            theirs = Sequence(toks[:n_prompt])
            for k in range(n_out):
                assert len(ours) == len(theirs)
                assert ours.num_blocks == theirs.num_blocks
                assert ours.last_block_num_tokens == theirs.last_block_num_tokens
                assert ours.last_token == theirs.last_token
                assert ours.num_completion_tokens == theirs.num_completion_tokens
                assert ours[0:len(ours)] == theirs[0:len(theirs)]
                assert ours[3:9] == theirs[3:9]
                # full blocks only: agentkv never reads a partial block (its Seq.block
                # would include frozen tokens past num_tokens there; nano-vLLM's would not)
                for i in range(ours.num_blocks - 1):
                    assert list(ours.block(i)) == theirs.block(i)
                nxt = gpu.forced_next_token(ours)
                assert nxt == toks[len(ours)]
                ours.append_token()
                theirs.append_token(nxt)


class _FakeTensor:
    def __init__(self, data):
        self.data = data

    def cuda(self, non_blocking=False):
        return self

    def tolist(self):
        return self.data


def _pinned_runner_methods():
    """Execute the pinned ModelRunner.prepare_prefill / prepare_decode /
    prepare_block_tables source with a stub torch, so the real index math runs
    on CPU against agentkv's Seq."""
    tree = ast.parse((NANO / "engine" / "model_runner.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner")
    wanted = [f for f in cls.body if isinstance(f, ast.FunctionDef)
              and f.name in ("prepare_prefill", "prepare_decode", "prepare_block_tables")]
    mod = ast.Module(body=[ast.ClassDef(name="Prep", bases=[], keywords=[], body=wanted, decorator_list=[])],
                     type_ignores=[])
    ctx = {}

    def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                    slot_mapping=None, context_lens=None, block_tables=None):
        ctx.clear()
        ctx.update(is_prefill=is_prefill, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                   max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, slot_mapping=slot_mapping,
                   context_lens=context_lens, block_tables=block_tables)

    fake_torch = types.SimpleNamespace(tensor=lambda data, dtype=None, pin_memory=False: _FakeTensor(data),
                                       int64="int64", int32="int32")
    ns = {"torch": fake_torch, "set_context": set_context, "Sequence": Seq}
    exec(compile(ast.fix_missing_locations(mod), "model_runner.py", "exec"), ns)
    return ns["Prep"], ctx


def _unwrap(v):
    return v.tolist() if isinstance(v, _FakeTensor) else v


@needs_nano
def test_pinned_input_preparation_runs_on_agentkv_seqs(pressure_trace):
    """Drive a full replay; at every step run the pinned prepare_* code on the
    scheduled agentkv Seqs and compare with gpu.prefill_inputs / decode_inputs."""
    Prep, ctx = _pinned_runner_methods()
    bs = pressure_trace.memory.block_size

    class PinnedRunner(gpu.ReferenceRunner):
        steps = 0

        def run(self, seqs, is_prefill):
            prep = Prep()
            prep.block_size = bs
            if is_prefill:
                ids, pos = prep.prepare_prefill(seqs)
                ref = gpu.prefill_inputs(seqs, bs)
                for k in ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"):
                    assert _unwrap(ctx[k]) == ref[k], k
            else:
                ids, pos = prep.prepare_decode(seqs)
                ref = gpu.decode_inputs(seqs, bs)
                assert _unwrap(ctx["context_lens"]) == ref["context_lens"]
            assert ids.tolist() == ref["input_ids"]
            assert pos.tolist() == ref["positions"]
            assert _unwrap(ctx["slot_mapping"]) == ref["slot_mapping"]
            assert _unwrap(ctx["block_tables"]) == ref["block_tables"]
            PinnedRunner.steps += 1
            return super().run(seqs, is_prefill)

    trace = pressure_trace
    engine = gpu.NanoVLLMEngine(trace, PinnedRunner(bs, trace.memory.num_blocks))
    client = InProcessClient(FIFO, params={"cap": 8}, block_size=bs, total_blocks=trace.memory.num_blocks)
    r = Replay(trace, client, engine=engine, check_every=1).run()
    assert r.valid, r.violation
    assert PinnedRunner.steps > 50
    assert r.diag["cache_hit_tokens"] > 0


# ------------------------------------------------------------------ the real-time engine


class Auditor(gpu.ReferenceRunner):
    """Checks, at every step, what a GPU would compute: inputs are the frozen
    tokens, KV writes go only to blocks this sequence owns alone, and every
    reused block holds exactly this sequence's prefix."""

    def __init__(self, bs, nb):
        super().__init__(bs, nb)
        self.engine = None
        self.seen: dict[int, set[int]] = {}
        self.prefix_steps = 0

    def run(self, seqs, is_prefill):
        kv: KVCacheManager = self.engine.kv
        bs = self.block_size
        inp = gpu.prefill_inputs(seqs, bs) if is_prefill else gpu.decode_inputs(seqs, bs)
        if is_prefill and inp["block_tables"] is not None:
            self.prefix_steps += 1
        k = 0
        for s in seqs:
            n = s.num_scheduled_tokens if is_prefill else 1
            start = s.num_cached_tokens if is_prefill else s.num_tokens - 1
            for p in range(start, start + n):
                assert inp["positions"][k] == p
                assert inp["input_ids"][k] == s.tokens[p], "input is not the frozen token"
                bid = s.block_table[p // bs]
                assert inp["slot_mapping"][k] == bid * bs + p % bs
                assert kv.blocks[bid].ref == 1, "KV written into a shared block"
                self.seen.setdefault(s.seq_id, set()).add(p)
                k += 1
            for i in range(s.num_cached_tokens // bs if is_prefill else 0):
                blk = kv.blocks[s.block_table[i]]
                if blk.hash is not None:
                    assert blk.hash == s.hashes[i] and blk.tokens == s.block(i), "reused block holds another prefix"
        assert k == len(inp["input_ids"])
        return super().run(seqs, is_prefill)


def test_engine_forced_decoding_under_pressure(tmp_path):
    trace = build_chat_trace(tmp_path / "rt.json", jobs=8, num_blocks=100, fanout=2, gap=0.0, think=(0.0, 0.01))
    runner = Auditor(trace.memory.block_size, trace.memory.num_blocks)
    log = tmp_path / "steps.jsonl"
    engine = gpu.NanoVLLMEngine(trace, runner, step_log=log, record_argmax=True)
    runner.engine = engine
    client = InProcessClient(FIFO, params={"cap": 8}, block_size=16, total_blocks=trace.memory.num_blocks)
    r = Replay(trace, client, engine=engine, check_every=1).run()
    engine.close()
    assert r.valid, r.violation
    assert r.diag["preemptions"] > 0 and r.diag["cache_hit_tokens"] > 0
    for job in trace.jobs:
        for node in job.nodes:
            if node.type == "llm":
                key = f"{job.id}/{node.id}"
                assert engine.finished_tokens[key] == trace.tokens(node).tolist()
                preds = engine.argmax[key]
                # one prediction per output position; the reference runner predicts the frozen token
                assert set(preds) == set(range(node.n_prompt, node.n_total))
                assert all(preds[p] == trace.tokens(node)[p] for p in preds)
    steps = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(steps) == r.diag["prefill_steps"] + r.diag["decode_steps"]
    assert all(s["seconds"] >= 0 and s["batch"] >= 1 for s in steps)
    assert any(s["prefill"] and any(c > 0 for c, _ in s["chunks"]) for s in steps)
    assert r.diag["busy_time"] <= max(r.flows) + 1.0
    assert all(0 < f < 60 for f in r.flows)


def test_engine_requires_tokens(chat_trace):
    engine = gpu.make_engine(chat_trace, runner="reference")
    engine.attach(choose_victim=lambda c: next(iter(c)))
    with pytest.raises(EngineError, match="frozen output tokens"):
        engine.add(Seq(10, 2, ()))


def test_reference_runner_rejects_slot_collisions():
    Seq.block_size = 4
    toks = array("I", range(12))
    a = Seq(6, 2, (), tokens=toks)
    b = Seq(6, 2, (), tokens=toks)
    a.block_table, b.block_table = [0, 1], [0, 2]
    a.num_scheduled_tokens = b.num_scheduled_tokens = 6
    with pytest.raises(EngineError, match="same KV slot"):
        gpu.ReferenceRunner(4, 8).run([a, b], True)


def test_cpu_dry_run_of_the_smoke_test():
    spec = importlib.util.spec_from_file_location("gpu_smoke", REPO / "tools" / "gpu_smoke.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    checks = smoke.run("reference", gap=0.2)
    assert checks["ok"], checks
    assert checks["hit_tokens"] >= 2048 and checks["warm_cold_agreement"] == 1.0


# ------------------------------------------------------------------ calibration


def test_fit_recovers_known_coefficients():
    truth = TimingModel(p_base=0.004, p_token=3e-5, p_attn=5e-9, d_base=0.007, d_seq=2e-4, d_ctx=2.5e-7)
    recs = calibrate.synthetic_log(truth, 600, seed=1)
    m = calibrate.fit(recs)
    for k in calibrate.PREFILL + calibrate.DECODE:
        assert getattr(m, k) == pytest.approx(getattr(truth, k), rel=1e-6), k
    noisy = calibrate.synthetic_log(truth, 3000, seed=2, noise=0.02)
    m = calibrate.fit(noisy, weight="relative")
    for k in ("p_token", "p_attn", "d_base", "d_ctx"):
        assert getattr(m, k) == pytest.approx(getattr(truth, k), rel=0.1), k
    rep = calibrate.report(m, noisy)
    assert abs(rep["prefill"]["total_error_pct"]) < 2 and abs(rep["decode"]["total_error_pct"]) < 2
    assert rep["decode"]["mean_abs_pct_error"] < 5


def test_nnls_is_non_negative_and_optimal():
    # y = 20 - x: the unconstrained slope is negative, so NNLS must zero it
    xs = [(1.0, float(i), 0.0) for i in range(10)]
    ys = [20.0 - i for i in range(10)]
    G = [[sum(x[i] * x[j] for x in xs) for j in range(3)] for i in range(3)]
    c = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(3)]
    b = calibrate.nnls_gram(G, c)
    assert all(v >= 0 for v in b)
    assert b[1] == 0.0 and b[0] == pytest.approx(sum(ys) / len(ys))
    # KKT: the gradient is zero on the support and non-negative off it
    grad = [2 * (sum(G[i][j] * b[j] for j in range(3)) - c[i]) for i in range(3)]
    for i in range(3):
        if b[i] > 0:
            assert abs(grad[i]) < 1e-6
        else:
            assert grad[i] >= -1e-6


def test_fit_keeps_base_for_missing_step_type():
    base = TimingModel()
    m = calibrate.fit([r for r in calibrate.synthetic_log(base, 200) if not r["prefill"]], base=base)
    assert m.p_token == base.p_token and m.p_attn == base.p_attn


def test_calibration_workload_is_valid(tmp_path):
    from agentkv.sandbox.loader import load_policy_class
    from agentkv.trace import load

    paths = calibrate.workload(tmp_path)
    traces = [load(p) for p in paths]
    assert {t.trace_id for t in traces} == {"calib-prefill", "calib-decode"}
    assert all(t.memory.block_size == 256 for t in traces)
    hits = [j for j in traces[0].jobs if j.meta.get("hit")]
    assert hits and all(len(j.nodes) == 3 for j in hits)
    load_policy_class((tmp_path / "calib_policy.py").read_text())
    # the decode sweep replays in the simulator (the same thing `run` does on a GPU)
    from agentkv.replay import run_policy
    r = run_policy(traces[1], str(tmp_path / "calib_policy.py"), params={"cap": 8}, isolate=False)
    assert r.valid and r.diag["preemptions"] == 0


def test_calibrate_cli_fit(tmp_path, capsys):
    log = tmp_path / "steps.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in calibrate.synthetic_log(TimingModel(), 100)) + "\n")
    assert calibrate.main(["fit", str(log), "--name", "test"]) == 0
    out = capsys.readouterr().out
    assert "fitted model" in out and "R^2" in out


# ------------------------------------------------------------------ fidelity


def _fake(flows):
    return {t: {"family": fam, "flows": {p: list(v) for p, v in pol.items()}} for t, (fam, pol) in flows.items()}


def test_compare_identical_backends():
    data = _fake({"t1": ("D1", {"ref": [10.0], "a": [8.0], "b": [12.0]}),
                  "t2": ("W1", {"ref": [5.0], "a": [5.0], "b": [4.0]})})
    rep = fidelity.compare(data, data, "ref")
    assert rep["summary"]["mean_abs_rel_error"] == 0
    assert rep["summary"]["max_abs_S_error"] == 0
    assert all(v == 1.0 for v in rep["kendall_tau"].values())
    assert rep["speedups"]["a"]["S_sim"] == pytest.approx(math.sqrt(10 / 8))
    assert "| t1 | D1 | a |" in fidelity.to_markdown(rep)


def test_compare_detects_error_and_rank_flips():
    sim = _fake({"t1": ("D1", {"ref": [10.0], "a": [8.0], "b": [9.0]})})
    gpu_ = _fake({"t1": ("D1", {"ref": [11.0, 10.0, 12.0], "a": [9.0, 10.0, 11.0], "b": [9.0, 9.5, 9.0]})})
    rep = fidelity.compare(sim, gpu_, "ref")
    row = next(r for r in rep["rows"] if r["policy"] == "a")
    assert row["gpu"] == 10.0 and row["rel_error"] == pytest.approx(-0.2)
    assert rep["kendall_tau"]["t1"] < 1.0
    assert rep["speedups"]["a"]["S_error"] == pytest.approx(10 / 8 - 11 / 10)


def test_kendall_tau_and_policy_specs():
    assert fidelity.kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0
    assert fidelity.kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0
    assert fidelity.parse_policy("lru=policies/reference.py") == ("lru", "policies/reference.py", None)
    assert fidelity.parse_policy('lru2=policies/reference.py:{"cap": 2}') == ("lru2", "policies/reference.py", {"cap": 2})


def test_collect_sim_vs_reference_gpu(tmp_path):
    trace = build_chat_trace(tmp_path / "f.json", jobs=4, turns=2, num_blocks=100, gap=0.0, think=(0.0, 0.005))
    pols = [("c1", FIFO, {"cap": 1}), ("c4", FIFO, {"cap": 4})]
    fast = TimingModel(p_base=0.002, p_token=1e-5, p_attn=2e-9, d_base=0.002, d_seq=5e-5, d_ctx=1e-7)
    sim = fidelity.collect(pols, [trace.path], "sim", isolate=False, timing=fast)
    rt = fidelity.collect(pols, [trace.path], "gpu", isolate=False, gpu_runner="reference", timing=fast)
    rep = fidelity.compare(sim, rt, "c1")
    assert len(rep["rows"]) == 2
    # the reference runner sleeps the model's step times, so only Python overhead and
    # sleep granularity separate the two backends
    assert rep["summary"]["max_abs_rel_error"] < 0.5
    assert math.isfinite(rep["speedups"]["c4"]["S_gpu"])
