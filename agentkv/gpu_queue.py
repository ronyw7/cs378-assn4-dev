"""Client for the course GPU queue: replay a candidate on nano-vLLM from your laptop.

`score(..., backend="gpu")` uses this when there is no local GPU. You can also
call it directly:

    from agentkv import gpu_queue
    exp = gpu_queue.submit("candidates/p17.py", "public_val")
    result = gpu_queue.wait(exp, timeout=3600)   # the ScoreResult as a dict
    gpu_queue.balance()                          # GPU-seconds left for your team

Settings come from the environment:

    AGENTKV_QUEUE_URL    the course GPU queue, e.g. https://grader.cs378.example.edu
    AGENTKV_TEAM_TOKEN   your team's token (from course staff; keep it private)

Each submission reserves an estimate of GPU time from your team's budget and
is settled to the measured runtime when it finishes. Only the public splits
(public_dev, public_val) can be replayed here. Use backend="sim" for search:
it is local, fast, and unlimited.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

SPLITS = ("public_dev", "public_val")
FINISHED = ("done", "candidate_failed", "infra_failed")


_NONFINITE = {"inf": float("inf"), "-inf": float("-inf"), "nan": float("nan")}


def _floats(obj):
    """Undo the queue's strict-JSON encoding of non-finite floats."""
    if isinstance(obj, str):
        return _NONFINITE.get(obj, obj)
    if isinstance(obj, dict):
        return {k: _floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_floats(v) for v in obj]
    return obj


class QueueError(RuntimeError):
    """The queue refused or failed a request; the message says why."""


def _config(url: str | None, token: str | None) -> tuple[str, str]:
    url = (url or os.environ.get("AGENTKV_QUEUE_URL") or "").rstrip("/")
    token = token or os.environ.get("AGENTKV_TEAM_TOKEN") or ""
    if not url:
        raise QueueError("set AGENTKV_QUEUE_URL to the course GPU queue (or pass url=)")
    if not token:
        raise QueueError("set AGENTKV_TEAM_TOKEN to your team's token (or pass token=)")
    return url, token


def _call(method: str, path: str, body: dict | None, url: str | None, token: str | None,
          timeout: float = 60.0) -> dict:
    base, tok = _config(url, token)
    req = urllib.request.Request(
        base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            detail = json.loads(raw).get("detail", raw.decode(errors="replace"))
        except ValueError:
            detail = raw.decode(errors="replace")
        raise QueueError(f"GPU queue refused the request ({e.code}): {detail}") from None
    except urllib.error.URLError as e:
        raise QueueError(f"cannot reach the GPU queue at {base}: {e.reason}") from None


def submit(policy_path: str | os.PathLike, split: str, params: dict | None = None, *,
           url: str | None = None, token: str | None = None) -> str:
    """Queue one replay of `policy_path` on a public split. Returns the experiment id."""
    if split not in SPLITS:
        raise QueueError(f"split must be one of {', '.join(SPLITS)}")
    data = Path(policy_path).read_bytes()
    body = {"policy": data.decode("utf-8"), "sha256": hashlib.sha256(data).hexdigest(), "split": split,
            "params": params}
    return _call("POST", "/experiments", body, url, token)["id"]


def status(exp_id: str, *, url: str | None = None, token: str | None = None) -> dict:
    """Return the experiment status, GPU time, and result if evaluation has finished."""
    return _call("GET", f"/experiments/{exp_id}", None, url, token)


def wait(exp_id: str, timeout: float = 3600.0, *, url: str | None = None, token: str | None = None,
         poll: float = 10.0) -> dict:
    """Block until the experiment finishes and return its ScoreResult as a dict
    (`valid`, `violations`, `speedup`, `points`, `by_family`, `by_trace`, ...)."""
    deadline = time.monotonic() + timeout
    while True:
        rec = status(exp_id, url=url, token=token)
        if rec["status"] == "infra_failed":
            raise QueueError(rec.get("error") or "infrastructure failure; resubmit")
        if rec["status"] in FINISHED:
            return _floats(rec["result"])
        if time.monotonic() + poll > deadline:
            raise TimeoutError(f"experiment {exp_id} is still {rec['status']} after {timeout:.0f} s")
        time.sleep(poll)


def balance(*, url: str | None = None, token: str | None = None) -> dict:
    """Return the team's GPU-time budget, usage, reservations, and remaining balance."""
    return _call("GET", "/gpu-balance", None, url, token)


def experiments(*, url: str | None = None, token: str | None = None) -> list[dict]:
    """List the team's experiments from oldest to newest; use status() for results."""
    return _call("GET", "/experiments", None, url, token)["experiments"]
