"""Fit the simulator's TimingModel to GPU step logs.

    # 1. on any machine: write the calibration workload (synthetic sweeps)
    python -m agentkv.engine.calibrate workload --out runs/calib
    # 2. on the GPU host: replay it, logging every step
    python -m agentkv.engine.calibrate run --traces runs/calib --log runs/calib/steps.jsonl
    #    (any replay can log too: AGENTKV_STEP_LOG=runs/steps.jsonl python -m agentkv.score ... --backend gpu)
    # 3. fit, inspect the error per step type, and save agentkv/data/timing.json
    python -m agentkv.engine.calibrate fit runs/calib/steps.jsonl --name qwen3-8b-a100 --write

The model is linear in its coefficients, so each step type is a 3-variable
non-negative least-squares problem; it is solved exactly (every active set of
the normal equations) in pure Python, with no numpy or scipy needed. After
writing new coefficients, bump SIM_VERSION in timing.py and regenerate the
baseline reference job latencies: every simulated score depends on them.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

from .timing import TIMING_PATH, TimingModel

PREFILL = ("p_base", "p_token", "p_attn")
DECODE = ("d_base", "d_seq", "d_ctx")


# ------------------------------------------------------------------ features


def prefill_features(chunks) -> tuple[float, float, float]:
    new = sum(n for _, n in chunks)
    pairs = sum(n * c + n * (n + 1) // 2 for c, n in chunks)
    return (1.0, float(new), float(pairs))


def decode_features(batch: int, context: int) -> tuple[float, float, float]:
    return (1.0, float(batch), float(context))


def features(rec: dict) -> tuple[str, tuple[float, float, float]]:
    if rec["prefill"]:
        return "prefill", prefill_features(rec["chunks"])
    return "decode", decode_features(rec["batch"], rec["context"])


def predict(model: TimingModel, rec: dict) -> float:
    kind, x = features(rec)
    names = PREFILL if kind == "prefill" else DECODE
    return sum(getattr(model, k) * v for k, v in zip(names, x))


# ------------------------------------------------------------------ NNLS


def _solve(A: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve a linear system by Gaussian elimination with partial pivoting, or return None if singular."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for k in range(col, n + 1):
                M[r][k] -= f * M[col][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x


def nnls_gram(G: list[list[float]], c: list[float]) -> list[float]:
    """argmin_{b >= 0} b'Gb - 2c'b for a small Gram matrix G = X'WX, c = X'Wy.
    The optimum is the unconstrained optimum on its own support, so trying
    every support and keeping the best feasible point is exact."""
    n = len(c)
    scale = [math.sqrt(G[i][i]) if G[i][i] > 0 else 0.0 for i in range(n)]
    live = [i for i in range(n) if scale[i] > 0]
    Gs = {(i, j): G[i][j] / (scale[i] * scale[j]) for i in live for j in live}
    cs = {i: c[i] / scale[i] for i in live}
    best, best_obj = {}, 0.0
    for k in range(1, len(live) + 1):
        for S in combinations(live, k):
            x = _solve([[Gs[i, j] for j in S] for i in S], [cs[i] for i in S])
            if x is None or min(x) < -1e-12:
                continue
            x = [max(v, 0.0) for v in x]
            obj = sum(x[a] * Gs[S[a], S[b]] * x[b] for a in range(k) for b in range(k)) \
                - 2 * sum(x[a] * cs[S[a]] for a in range(k))
            if obj < best_obj - 1e-15 * max(1.0, abs(best_obj)):
                best, best_obj = dict(zip(S, x)), obj
    return [best.get(i, 0.0) / scale[i] if scale[i] else 0.0 for i in range(n)]


# ------------------------------------------------------------------ fitting


def read_logs(paths) -> list[dict]:
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def fit(records: list[dict], *, weight: str = "absolute", name: str = "fitted",
        base: TimingModel | None = None) -> TimingModel:
    """Fit both step types. `weight="relative"` minimizes squared relative error
    instead of squared seconds. A step type with no records keeps `base`'s values."""
    base = base or TimingModel()
    coefs = asdict(base)
    for kind, names in (("prefill", PREFILL), ("decode", DECODE)):
        G = [[0.0] * 3 for _ in range(3)]
        c = [0.0] * 3
        n = 0
        for rec in records:
            k, x = features(rec)
            y = float(rec["seconds"])
            if k != kind or y <= 0:
                continue
            w = 1.0 / (y * y) if weight == "relative" else 1.0
            for i in range(3):
                c[i] += w * x[i] * y
                for j in range(3):
                    G[i][j] += w * x[i] * x[j]
            n += 1
        if n:
            for k, v in zip(names, nnls_gram(G, c)):
                coefs[k] = v
    coefs["name"] = name
    return TimingModel(**coefs)


def report(model: TimingModel, records: list[dict]) -> dict:
    out = {}
    for kind in ("prefill", "decode"):
        ys, ps = [], []
        for rec in records:
            if features(rec)[0] == kind and rec["seconds"] > 0:
                ys.append(float(rec["seconds"]))
                ps.append(predict(model, rec))
        if not ys:
            continue
        mean = sum(ys) / len(ys)
        ss_tot = sum((y - mean) ** 2 for y in ys)
        ss_res = sum((y - p) ** 2 for y, p in zip(ys, ps))
        out[kind] = {
            "steps": len(ys),
            "total_s": sum(ys),
            "predicted_total_s": sum(ps),
            "total_error_pct": 100 * (sum(ps) - sum(ys)) / sum(ys),
            "mean_abs_error_ms": 1000 * sum(abs(y - p) for y, p in zip(ys, ps)) / len(ys),
            "mean_abs_pct_error": 100 * sum(abs(y - p) / y for y, p in zip(ys, ps)) / len(ys),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        }
    return out


def synthetic_log(model: TimingModel, n: int = 400, *, seed: int = 0, noise: float = 0.0) -> list[dict]:
    """Generate step timings from model, with optional noise, to test coefficient fitting."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < 0.4:
            chunks = [(rng.choice([0, 0, 256, 1024, 4096, 8192]) + rng.randrange(0, 256),
                       rng.randint(1, 8192)) for _ in range(rng.randint(1, 4))]
            rec = {"prefill": True, "chunks": chunks, "batch": len(chunks),
                   "context": sum(c + k for c, k in chunks)}
        else:
            b = rng.randint(1, 16)
            rec = {"prefill": False, "chunks": None, "batch": b, "context": sum(rng.randint(64, 16384) for _ in range(b))}
        rec["seconds"] = predict(model, rec) * (1 + noise * rng.gauss(0, 1))
        out.append(rec)
    return out


# ------------------------------------------------------------------ workload


CALIB_POLICY = '''"""Calibration policy: a fixed cap and oldest-released eviction."""
from agentkv.policy import Policy


class CalibrationPolicy(Policy):
    def target_concurrency(self, view, state):
        return int(self.params.get("cap", 8))

    def choose_victim(self, view, candidates, state):
        return candidates[0]
'''

_WORDS = ("the of and to in is for on that with as by it at from this be are or an was which "
          "data system memory cache block time file server network model query table index page "
          "node graph value state queue thread lock write read store load batch token layer "
          "policy request").split()


def _text(n: int, tag: str, rng: random.Random) -> str:
    """About n tokens of filler; `tag` keeps different segments from sharing blocks."""
    return f"[{tag}]" + "".join(" " + rng.choice(_WORDS) for _ in range(max(n - 4, 1)))


def workload(out: str | Path, *, block_size: int = 256, seed: int = 0) -> list[Path]:
    """Build calibration traces covering prompt lengths, cache hits, and batch sizes.

    Space jobs far enough apart that measurements from different jobs do not overlap.
    """
    from ..trace import HEADER, TraceBuilder, output_text

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calib_policy.py").write_text(CALIB_POLICY)
    rng = random.Random(seed)
    paths = []

    # Prefill: lengths 256..16k at hit fractions 0, 0.5, 0.9 (long prompts are chunked).
    b = TraceBuilder("calib-prefill", "CAL", "calibration", num_blocks=200, block_size=block_size,
                     description="prefill sweep: prompt length x cache-hit fraction")
    t = 0.0
    for L in (256, 512, 1024, 2048, 4096, 8192, 16384):
        for h in (0.0, 0.5, 0.9):
            job = b.job(arrival=t, meta={"length": L, "hit": h})
            shared = _text(int(L * h), f"p{L}-{h}", rng) if h else None
            deps = []
            if shared:
                warm = job.llm("warm", [shared], output_text("ok"))
                deps = [job.tool("gap", 0.05, deps=[warm])]
            prompt = ([shared] if shared else []) + [_text(int(L * (1 - h)), f"q{L}-{h}", rng), HEADER]
            job.llm("test", prompt, output_text(" yes no"), deps=deps)
            t += 1.0 + L / 8000
    b.save(out / "calib-prefill.json")
    paths.append(out / "calib-prefill.json")

    # Decode: batch 1..16 (the engine limit) x context, with B * context within 64k tokens.
    b = TraceBuilder("calib-decode", "CAL", "calibration", num_blocks=300, block_size=block_size,
                     description="decode sweep: batch size x context")
    t = 0.0
    for B in (1, 2, 4, 8, 16):
        for C in (256, 1024, 4096, 8192, 16384):
            if B * (C + 64) > 65536:
                continue
            job = b.job(arrival=t, meta={"batch": B, "context": C})
            for i in range(B):
                job.llm("decode", [_text(C, f"d{B}-{C}-{i}", rng), HEADER],
                        output_text("".join(" " + rng.choice(_WORDS) for _ in range(64))))
            t += 1.0 + 64 * (0.006 + B * C * 4e-7) + B * C / 8000
    b.save(out / "calib-decode.json")
    paths.append(out / "calib-decode.json")
    return paths


def run_workload(trace_dir: str | Path, log: str | Path, *, cap: int = 8) -> None:
    """Replay every calibration trace on the GPU, appending each step to `log`."""
    from ..replay import run_policy
    from ..trace import load
    from .gpu import make_engine

    trace_dir = Path(trace_dir)
    policy = str(trace_dir / "calib_policy.py")
    for path in sorted(trace_dir.glob("*.json")):
        trace = load(path)
        engine = make_engine(trace, step_log=log)
        r = run_policy(trace, policy, params={"cap": cap}, isolate=False, engine=engine)
        engine.close()
        print(f"{trace.trace_id}: {'ok' if r.valid else r.violation}  mean flow {r.mean_flow:.2f} s")


# ------------------------------------------------------------------ CLI


def _print_report(model: TimingModel, rep: dict) -> None:
    print(json.dumps({k: v for k, v in asdict(model).items()}, indent=2))
    for kind, r in rep.items():
        print(f"{kind:>8}: {r['steps']} steps, total {r['total_s']:.2f} s (model {r['predicted_total_s']:.2f} s, "
              f"{r['total_error_pct']:+.1f}%), MAE {r['mean_abs_error_ms']:.2f} ms, "
              f"MAPE {r['mean_abs_pct_error']:.1f}%, R^2 {r['r2']:.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agentkv.engine.calibrate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("workload", help="write the calibration traces and policy")
    w.add_argument("--out", required=True)
    w.add_argument("--block-size", type=int, default=256)
    r = sub.add_parser("run", help="replay the calibration traces on the GPU (staff GPU host)")
    r.add_argument("--traces", required=True)
    r.add_argument("--log", required=True)
    f = sub.add_parser("fit", help="fit TimingModel coefficients to step logs")
    f.add_argument("logs", nargs="+")
    f.add_argument("--weight", choices=["absolute", "relative"], default="absolute")
    f.add_argument("--name", default="fitted")
    f.add_argument("--trace", action="append", help="only use steps from these trace ids")
    f.add_argument("--write", action="store_true", help=f"save to {TIMING_PATH}")
    args = ap.parse_args(argv)
    if args.cmd == "workload":
        for p in workload(args.out, block_size=args.block_size):
            print(p)
    elif args.cmd == "run":
        run_workload(args.traces, args.log)
    else:
        recs = read_logs(args.logs)
        if args.trace:
            recs = [x for x in recs if x.get("trace") in set(args.trace)]
        current = TimingModel.load()
        print("current model:")
        _print_report(current, report(current, recs))
        model = fit(recs, weight=args.weight, name=args.name, base=current)
        print("fitted model:")
        _print_report(model, report(model, recs))
        if args.write:
            model.save()
            print(f"Wrote {TIMING_PATH}; update SIM_VERSION and regenerate the baseline reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
