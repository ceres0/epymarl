from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

ComputeRewardFn = Callable[..., Tuple[float, Dict[str, float]]]


@dataclass
class CardRewardConfig:
    enabled: bool = False
    plugin: str = "card.rewards.smac_env_reward"
    kwargs: Dict[str, Any] | None = None


class CardRewardShaper:
    """Loads a reward plugin and applies it at env.step().

    Plugin contract:
      - module exposes `compute_reward(...) -> (reward: float, components: dict)`
      - kwargs are provided via config.

    The shaper is intentionally defensive: on exceptions it falls back to the
    raw env reward and records an error string in `info`.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        plugin: str,
        plugin_kwargs: Mapping[str, Any] | None = None,
    ):
        self.enabled = bool(enabled)
        self.plugin_path = str(plugin)
        self.plugin_kwargs = dict(plugin_kwargs or {})

        self._module = None
        self._fn: ComputeRewardFn | None = None

        if self.enabled:
            self._load()

    def _load(self) -> None:
        mod = importlib.import_module(self.plugin_path)
        fn = getattr(mod, "compute_reward", None)
        if not callable(fn):
            raise ValueError(
                f"CARD reward plugin '{self.plugin_path}' must define callable compute_reward(...)"
            )
        self._module = mod
        self._fn = fn

    def compute(
        self,
        *,
        state: Any,
        obs: Any,
        actions: Any,
        raw_reward: Any,
        terminated: bool,
        info: Dict[str, Any],
        env: Any = None,
    ) -> Tuple[float, Dict[str, float]]:
        if not self.enabled or self._fn is None:
            return raw_reward, {}

        try:
            shaped, components = self._fn(
                state=state,
                obs=obs,
                actions=actions,
                raw_reward=raw_reward,
                terminated=bool(terminated),
                info=info,
                env=env,
                **self.plugin_kwargs,
            )
            components = components or {}
            clean = {str(k): float(v) for k, v in components.items()}
            return float(shaped), clean
        except Exception as e:
            info["card_reward_error"] = f"{type(e).__name__}: {e}"
            return raw_reward, {}
