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
    win_bonus: float = 200.0,
    lose_penalty: float = 0.0,
    env_scale: float = 1.0,
    **kwargs,
) -> Tuple[float, Dict[str, float]]:
    """A simple shaped reward that is compatible with offline TPE.

    - Starts from SMAC env reward (scaled)
    - Adds a terminal bonus when `battle_won` is True
    - Optionally adds a terminal penalty on loss

    This is intentionally minimal so it can be re-evaluated on stored
    trajectories without needing the live environment.
    """

    r_env = float(env_scale) * _to_float_reward(raw_reward)
    won = bool(info.get("battle_won", False))
    r_win = float(win_bonus) if (terminated and won) else 0.0
    r_lose = float(-abs(lose_penalty)) if (terminated and (not won) and lose_penalty) else 0.0
    r_total = r_env + r_win + r_lose
    return r_total, {"env": r_env, "win_bonus": r_win, "lose_penalty": r_lose}
