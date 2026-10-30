"""GPU smoke test: confirm that the policy hooks drive real prefix reuse in nano-vLLM.

    AGENTKV_MODEL_PATH=/models/Qwen3-8B python tools/gpu_smoke.py
    python tools/gpu_smoke.py --runner reference      # test the replay interface without a GPU

Jobs J00 and J01 share a prefix of about 2,100 tokens. Job J02 has an
independent prefix of the same length. The test checks that:
  1. the replay is valid and the KV invariants (kv.check) hold after every step;
  2. J01, whose 2,048-token prefix is cached, prefills faster than the cold J02;
  3. every replayed sequence is exactly prompt + frozen output;
  4. the model's greedy predictions for J01 on the warm cache match a run of
     the same call on a cold cache, so the reused blocks hold the right KV;
  5. (real runner) ModelRunner's prepared inputs match agentkv's transcription.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agentkv import tokenizer  # noqa: E402
from agentkv.engine.gpu import ReferenceRunner, make_engine  # noqa: E402
from agentkv.engine.timing import TimingModel  # noqa: E402
from agentkv.replay import Replay  # noqa: E402
from agentkv.sandbox.client import InProcessClient  # noqa: E402
from agentkv.trace import HEADER, TraceBuilder, chat_segment, load, output_text  # noqa: E402

LRU = str(REPO / "policies" / "lru.py")
WORDS = ("cache block memory prefix token layer batch queue model request server data system "
         "state value index table page node graph store write read load time policy").split()


def text_of(n: int, tag: str) -> str:
    """Generate at least n tokens of text, determined by tag."""
    words, k = [f"[{tag}]"], 0
    while True:
        words.extend(" " + WORDS[(j * 7 + len(tag)) % len(WORDS)] for j in range(k, k + 64))
        k += 64
        s = "".join(words)
        if tokenizer.count(s) >= n:
            return s


def build(path: Path, *, gap: float = 3.0, warm: bool = True) -> object:
    """warm=True: J00 (shared prefix), J01 (same prefix), J02 (cold, same length).
    warm=False: J01's call alone, on an empty cache."""
    b = TraceBuilder("smoke-warm" if warm else "smoke-cold", "SMOKE", "smoke", num_blocks=64, block_size=256)
    shared = chat_segment("system", text_of(2100, "shared"))
    other = chat_segment("system", text_of(2100, "other"))
    q1 = chat_segment("user", "Which blocks of this prompt were already in the cache?")
    out = output_text(" ".join(WORDS[:12]))
    if warm:
        b.job(0.0).llm("chat", [shared, chat_segment("user", "Summarize the context."), HEADER], out)
        b.job(gap).llm("chat", [shared, q1, HEADER], out)
        b.job(2 * gap).llm("chat", [other, q1, HEADER], out)
    else:
        b.job(0.0, job_id="J01").llm("chat", [shared, q1, HEADER], out)
    b.save(path)
    return load(path)


def _replay(trace, runner, log: Path):
    engine = make_engine(trace, step_log=log, record_argmax=True, crosscheck=runner is None, runner=runner)
    client = InProcessClient(LRU, params={"cap": 1}, block_size=trace.memory.block_size,
                             total_blocks=trace.memory.num_blocks)
    result = Replay(trace, client, engine=engine, check_every=1).run()
    engine.close()
    return result, engine


def run(runner: str | None = None, *, gap: float = 3.0) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="agentkv-smoke-"))
    try:
        return _run(tmp, runner, gap)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run(tmp: Path, runner: str | None, gap: float) -> dict:
    warm = build(tmp / "warm.json", gap=gap)
    cold = build(tmp / "cold.json", warm=False)

    def make_runner(trace):
        if runner == "reference":   # sleeps the simulator's step times: same checks, no GPU
            return ReferenceRunner(trace.memory.block_size, trace.memory.num_blocks, TimingModel())
        return None

    log = tmp / "steps.jsonl"
    r_warm, e_warm = _replay(warm, make_runner(warm), log)
    r_cold, e_cold = _replay(cold, make_runner(cold), log)
    steps = [json.loads(x) for x in log.read_text().splitlines()]
    prefills = [s for s in steps if s["trace"] == "smoke-warm" and s["prefill"]]
    hit = [s for s in prefills if s["chunks"][0][0] >= 2048]
    miss = [s for s in prefills if s["chunks"][0][0] == 0]
    checks = {"valid": r_warm.valid and r_cold.valid, "violation": r_warm.violation or r_cold.violation}
    checks["hit_tokens"] = r_warm.diag.get("cache_hit_tokens", 0)
    if hit and miss:
        checks["hit_prefill_s"] = hit[-1]["seconds"]
        checks["cold_prefill_s"] = miss[-1]["seconds"]     # J02: the same length, no reuse
        checks["hit_is_faster"] = checks["hit_prefill_s"] < checks["cold_prefill_s"]
    else:
        checks["hit_is_faster"] = False
    frozen = True
    for trace, engine in ((warm, e_warm), (cold, e_cold)):
        for job in trace.jobs:
            for node in job.nodes:
                got = engine.finished_tokens.get(f"{job.id}/{node.id}")
                frozen &= got == trace.tokens(node).tolist()
    checks["outputs_are_frozen"] = frozen
    a, b = e_warm.argmax.get("J01/n0", {}), e_cold.argmax.get("J01/n0", {})
    common = sorted(set(a) & set(b))
    checks["argmax_positions"] = len(common)
    checks["warm_cold_agreement"] = sum(a[p] == b[p] for p in common) / len(common) if common else 0.0
    checks["ok"] = bool(checks["valid"] and checks["hit_tokens"] >= 2048 and checks["hit_is_faster"]
                        and frozen and checks["warm_cold_agreement"] >= 0.9)
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runner", choices=["nanovllm", "reference"], default="nanovllm")
    ap.add_argument("--gap", type=float, default=3.0, help="seconds between the three jobs")
    args = ap.parse_args(argv)
    checks = run(None if args.runner == "nanovllm" else "reference", gap=args.gap)
    print(json.dumps(checks, indent=2))
    print("SMOKE OK" if checks["ok"] else "SMOKE FAILED")
    return 0 if checks["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
