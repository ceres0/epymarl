from __future__ import annotations

from typing import Any, Dict, Tuple


def _to_float_reward(raw_reward: Any) -> float:
    try:
        return float(raw_reward)
    except Exception:
        try:
            return float(raw_reward.item())
        except Exception:
            return float(sum(raw_reward))


def compute_reward(
    *,
    state: Any,
    obs: Any,
    actions: Any,
    raw_reward: Any,
    terminated: bool,
    info: Dict[str, Any],
    env: Any = None,
    **kwargs,
) -> Tuple[float, Dict[str, float]]:
    """Baseline plugin: returns the environment reward unchanged."""

    r_env = _to_float_reward(raw_reward)
    return r_env, {"env": r_env}
