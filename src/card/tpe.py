from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


def load_reward_plugin(plugin_path: str) -> Callable[..., Tuple[float, Dict[str, float]]]:
    mod = importlib.import_module(plugin_path)
    fn = getattr(mod, "compute_reward", None)
    if not callable(fn):
        raise ValueError(f"Plugin '{plugin_path}' must define compute_reward(...)")
    return fn


def preference_accuracy(success_avg: List[float], fail_avg: List[float]) -> float:
    if not success_avg or not fail_avg:
        return 0.0
    s = np.asarray(success_avg, dtype=np.float32)
    f = np.asarray(fail_avg, dtype=np.float32)
    ok = (s.reshape(-1, 1) > f.reshape(1, -1)).sum()
    total = s.size * f.size
    return float(ok) / float(total)


@dataclass
class TpeResult:
    passed: bool
    accuracy: float
    n_success: int
    n_fail: int


def eval_tpe_from_npz_dir(
    *,
    traj_dir: str | Path,
    reward_plugin: str,
    threshold: float = 0.8,
    plugin_kwargs: Dict[str, Any] | None = None,
) -> TpeResult:
    """Re-evaluates a reward on stored trajectories and checks order-preserving.

    Paper Eq.(1) is strict, but their implementation uses an accuracy-based
    threshold (Table B.7). We mirror that: compute the fraction of (success, fail)
    pairs where avg_return_per_step(success) > avg_return_per_step(fail).
    """

    fn = load_reward_plugin(reward_plugin)
    plugin_kwargs = plugin_kwargs or {}

    success_avg: List[float] = []
    fail_avg: List[float] = []

    traj_dir = Path(traj_dir)
    for p in sorted(traj_dir.glob("*.npz")):
        data = np.load(p, allow_pickle=True)
        meta = json.loads(str(data["meta"].tolist()))
        won = bool(meta.get("battle_won", False))

        states = data["states"]
        actions = data["actions"]
        terminated = data["terminated"]

        G = 0.0
        T = int(states.shape[0]) if getattr(states, "ndim", 0) > 0 else 0
        for t in range(T):
            info = {"battle_won": won} if bool(terminated[t]) else {}
            r, _ = fn(
                state=states[t],
                obs=None,
                actions=actions[t],
                raw_reward=0.0,
                terminated=bool(terminated[t]),
                info=info,
                env=None,
                **plugin_kwargs,
            )
            G += float(r)

        avg = (G / float(T)) if T > 0 else 0.0
        if won:
            success_avg.append(avg)
        else:
            fail_avg.append(avg)

    acc = preference_accuracy(success_avg, fail_avg)
    passed = (acc >= float(threshold)) and (len(success_avg) > 0) and (len(fail_avg) > 0)
    return TpeResult(passed=passed, accuracy=acc, n_success=len(success_avg), n_fail=len(fail_avg))
