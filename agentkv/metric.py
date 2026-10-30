"""The grading metric.

    L(P, i) = mean job latency of policy P on trace i
    S       = exp( mean over families ( mean over traces ( log( L(baseline, i) / L(P, i) ) ) ) )
    g       = 1 - 1/S                    # fractional latency reduction
    P_perf  = clip( 25 + 25 * g_eff / g_target, 0, 50 )

Gates: an invalid or incomplete trace gives 0 performance points; if any
family's mean job latency regresses by more than 20%, g_eff is capped at 0
(parity). On GPU replay the per-trace ratio is the median of paired
repetitions and |g| <= 2% counts as parity.
"""

from __future__ import annotations

import math
import statistics

G_TARGET = 0.10          # g that earns full performance points; set by staff from their pilot campaigns
FAMILY_REGRESSION = 1.20
GPU_DEADBAND = 0.02
MAX_POINTS = 50.0
FAMILY_ORDER = ("D", "W1", "W2", "W3", "W4", "W5", "M1", "M2")


def family_key(f: str) -> tuple:
    return (FAMILY_ORDER.index(f), f) if f in FAMILY_ORDER else (len(FAMILY_ORDER), f)


def speedup(log_ratios: dict[str, list[float]]) -> float:
    """Compute family-balanced speedup from per-trace log ratios grouped by family."""
    fams = [statistics.fmean(v) for v in log_ratios.values() if v]
    return math.exp(statistics.fmean(fams)) if fams else 1.0


def points(g_eff: float, g_target: float = G_TARGET) -> float:
    return min(max(25.0 + 25.0 * g_eff / g_target, 0.0), MAX_POINTS)


def summarize(rows: list[dict], *, backend: str = "sim", g_target: float = G_TARGET) -> dict:
    """Compute the score and performance points from the per-trace results.

    Each row supplies family, ratio (baseline latency divided by candidate
    latency), mean_flow, and b0_mean_flow (the baseline mean job latency).
    """
    logs: dict[str, list[float]] = {}
    flows: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        logs.setdefault(r["family"], []).append(math.log(r["ratio"]))
        flows.setdefault(r["family"], []).append((r["mean_flow"], r["b0_mean_flow"]))
    s = speedup(logs)
    g = 1.0 - 1.0 / s
    by_family = {f: math.exp(statistics.fmean(v)) for f, v in sorted(logs.items(), key=lambda kv: family_key(kv[0]))}
    regressed = [f for f, pairs in flows.items()
                 if statistics.fmean(p for p, _ in pairs) > FAMILY_REGRESSION * statistics.fmean(b for _, b in pairs)]
    g_eff = g
    if backend == "gpu" and abs(g_eff) <= GPU_DEADBAND:
        g_eff = 0.0
    if regressed:
        g_eff = min(g_eff, 0.0)
    return {"speedup": s, "g": g, "g_eff": g_eff, "points": points(g_eff, g_target),
            "by_family": by_family, "capped": bool(regressed), "regressed_families": sorted(regressed, key=family_key)}
