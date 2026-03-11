from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class NpzTrajectoryWriter:
    """Writes compact episode trajectories to .npz for offline TPE/analysis."""

    out_dir: Path
    max_episodes: int = 100
    sample_interval: int = 1
    state_dim_cap: Optional[int] = None

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def can_write_next(self) -> bool:
        return self._count < int(self.max_episodes)

    def start_episode(self) -> Dict[str, Any]:
        return {
            "states": [],
            "actions": [],
            "rewards": [],
            "terminated": [],
            "components": {},
            "t": 0,
        }

    def add_step(
        self,
        buf: Dict[str, Any],
        *,
        state: Any,
        actions: Any,
        reward: float,
        terminated: bool,
        components: Dict[str, float] | None,
    ) -> None:
        t = int(buf["t"])
        buf["t"] = t + 1

        if self.sample_interval > 1 and (t % self.sample_interval) != 0:
            return

        s = np.asarray(state, dtype=np.float32)
        if self.state_dim_cap is not None:
            s = s[: int(self.state_dim_cap)]

        a = np.asarray(actions, dtype=np.int64)

        buf["states"].append(s)
        buf["actions"].append(a)
        buf["rewards"].append(float(reward))
        buf["terminated"].append(bool(terminated))

        if components:
            comps: Dict[str, List[float]] = buf["components"]
            for k, v in components.items():
                comps.setdefault(str(k), []).append(float(v))

    def end_episode(self, buf: Dict[str, Any], *, meta: Dict[str, Any]) -> Path:
        ep_idx = self._count
        self._count += 1

        states = np.stack(buf["states"], axis=0) if buf["states"] else np.zeros((0,))
        actions = np.stack(buf["actions"], axis=0) if buf["actions"] else np.zeros((0,))
        rewards = np.asarray(buf["rewards"], dtype=np.float32)
        terminated = np.asarray(buf["terminated"], dtype=np.bool_)

        out = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "terminated": terminated,
            "meta": np.asarray(json.dumps(meta), dtype=object),
        }

        for k, seq in buf["components"].items():
            out[f"comp_{k}"] = np.asarray(seq, dtype=np.float32)

        path = self.out_dir / f"episode_{ep_idx:04d}.npz"
        np.savez_compressed(path, **out)
        return path
