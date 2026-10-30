"""Compare simulated job latencies and policy rankings with measured GPU replays.

    python -m agentkv.engine.fidelity --split public_val --reps 3 \\
        --reference baseline=<baseline policy file> \\
        --policy lru2='policies/reference.py:{"cap": 2}' --policy lru8='policies/reference.py:{"cap": 8}' \\
        --out runs/fidelity.json --md runs/fidelity.md

For every (trace, policy) it reports the simulated and replayed mean job latency
and their relative error; for every trace, whether the two backends rank the
policies the same way (Kendall tau); and for every policy, the speedup S over
the reference policy under each backend and the difference.
`--gpu-runner reference` swaps the model for agentkv.engine.gpu.ReferenceRunner
to test the replay interface without a GPU. This checks execution, not the
accuracy of the simulator's timing estimates.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from .. import metric
from ..replay import run_policy
from ..trace import load
from .timing import TimingModel


def parse_policy(spec: str) -> tuple[str, str, dict | None]:
    """Parse a policy specification: name=path or name=path:{JSON parameters}."""
    name, _, rest = spec.partition("=")
    if not rest:
        raise ValueError(f"policy spec {spec!r} must look like name=path[:params-json]")
    path, sep, params = rest.partition(":")
    if sep and params.strip().startswith("{"):
        return name, path, json.loads(params)
    return name, rest, None


def collect(policies: list[tuple[str, str, dict | None]], trace_paths: list[str], backend: str, *,
            reps: int = 1, isolate: bool = True, gpu_runner: str = "nanovllm",
            timing: TimingModel | None = None) -> dict:
    """Evaluate each policy on every trace and collect its mean job latencys.

    Return {trace_id: {"family": family, "flows": {policy: [times per repetition]}}}.
    """
    out: dict = {}
    for p in trace_paths:
        trace = load(p)
        entry = out.setdefault(trace.trace_id, {"family": trace.family, "flows": {}})
        for name, path, params in policies:
            for rep in range(reps if backend == "gpu" else 1):
                engine = None
                if backend == "gpu":
                    from .gpu import ReferenceRunner, make_engine
                    runner = (ReferenceRunner(trace.memory.block_size, trace.memory.num_blocks,
                                              timing or TimingModel.load())
                              if gpu_runner == "reference" else gpu_runner)
                    engine = make_engine(trace, runner=runner)
                r = run_policy(trace, path, params=params, seed=rep, isolate=isolate, timing=timing,
                               engine=engine)
                if not r.valid:
                    raise RuntimeError(f"{name} invalid on {trace.trace_id}: {r.violation}")
                entry["flows"].setdefault(name, []).append(r.mean_flow)
    return out


def kendall_tau(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 1.0
    s = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                continue
            pairs += 1
            s += 1 if da * db > 0 else -1 if da * db < 0 else 0
    return s / pairs if pairs else 1.0


def compare(sim: dict, gpu: dict, reference: str) -> dict:
    """Compare job latencies, policy rankings, and speedups from two collect() results."""
    rows, taus = [], {}
    names = None
    for tid in sorted(sim):
        if tid not in gpu:
            continue
        fs, fg = sim[tid]["flows"], gpu[tid]["flows"]
        names = sorted(set(fs) & set(fg))
        for n in names:
            s, g = statistics.median(fs[n]), statistics.median(fg[n])
            rows.append({"trace": tid, "family": sim[tid]["family"], "policy": n, "sim": s, "gpu": g,
                         "rel_error": (s - g) / g if g else math.inf,
                         "gpu_spread": (max(fg[n]) - min(fg[n])) / g if g and len(fg[n]) > 1 else 0.0})
        taus[tid] = kendall_tau([statistics.median(fs[n]) for n in names],
                                [statistics.median(fg[n]) for n in names])
    speedups = {}
    for n in sorted({r["policy"] for r in rows} - {reference}):
        per = {}
        for backend in ("sim", "gpu"):
            src = sim if backend == "sim" else gpu
            ratios = [{"family": src[t]["family"],
                       "ratio": statistics.median(src[t]["flows"][reference]) / statistics.median(src[t]["flows"][n]),
                       "mean_flow": statistics.median(src[t]["flows"][n]),
                       "b0_mean_flow": statistics.median(src[t]["flows"][reference])}
                      for t in src if t in sim and t in gpu and n in src[t]["flows"] and reference in src[t]["flows"]]
            per[backend] = metric.summarize(ratios, backend=backend)["speedup"] if ratios else math.nan
        speedups[n] = {"S_sim": per["sim"], "S_gpu": per["gpu"], "S_error": per["sim"] - per["gpu"]}
    errs = [abs(r["rel_error"]) for r in rows if math.isfinite(r["rel_error"])]
    return {
        "reference": reference,
        "rows": rows,
        "kendall_tau": taus,
        "speedups": speedups,
        "summary": {
            "mean_abs_rel_error": statistics.fmean(errs) if errs else math.nan,
            "max_abs_rel_error": max(errs) if errs else math.nan,
            "mean_kendall_tau": statistics.fmean(taus.values()) if taus else math.nan,
            "max_abs_S_error": max((abs(v["S_error"]) for v in speedups.values()), default=math.nan),
        },
    }


def to_markdown(rep: dict) -> str:
    s = rep["summary"]
    lines = ["# Simulator fidelity", "",
             f"Reference policy: `{rep['reference']}`. Mean |flow error| {100 * s['mean_abs_rel_error']:.1f}%, "
             f"max {100 * s['max_abs_rel_error']:.1f}%; mean Kendall tau {s['mean_kendall_tau']:.2f}; "
             f"max |S_sim - S_gpu| {s['max_abs_S_error']:.3f}.", "",
             "| trace | family | policy | sim flow (s) | GPU flow (s) | error | GPU rep spread |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in rep["rows"]:
        lines.append(f"| {r['trace']} | {r['family']} | {r['policy']} | {r['sim']:.2f} | {r['gpu']:.2f} | "
                     f"{100 * r['rel_error']:+.1f}% | {100 * r['gpu_spread']:.1f}% |")
    lines += ["", "| policy | S (sim) | S (GPU) | difference |", "|---|---:|---:|---:|"]
    for n, v in rep["speedups"].items():
        lines.append(f"| {n} | {v['S_sim']:.4f} | {v['S_gpu']:.4f} | {v['S_error']:+.4f} |")
    lines += ["", "| trace | Kendall tau |", "|---|---:|"]
    lines += [f"| {t} | {v:.2f} |" for t, v in rep["kendall_tau"].items()]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    from ..score import split_paths

    ap = argparse.ArgumentParser(prog="python -m agentkv.engine.fidelity")
    ap.add_argument("--split", default="public_val")
    ap.add_argument("--reference", required=True, help="name=path[:params] of the policy S is measured against")
    ap.add_argument("--policy", action="append", default=[], help="name=path[:params]; repeat")
    ap.add_argument("--reps", type=int, default=3, help="GPU repetitions per (trace, policy)")
    ap.add_argument("--gpu-runner", choices=["nanovllm", "reference"], default="nanovllm")
    ap.add_argument("--out", help="write the full report as JSON")
    ap.add_argument("--md", help="write the publishable Markdown table")
    args = ap.parse_args(argv)
    ref = parse_policy(args.reference)
    policies = [ref] + [parse_policy(p) for p in args.policy]
    paths = [str(p) for p in split_paths(args.split)]
    sim = collect(policies, paths, "sim")
    gpu = collect(policies, paths, "gpu", reps=args.reps, gpu_runner=args.gpu_runner)
    rep = compare(sim, gpu, ref[0])
    rep["raw"] = {"sim": sim, "gpu": gpu}
    md = to_markdown(rep)
    print(md)
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
    if args.md:
        Path(args.md).write_text(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
