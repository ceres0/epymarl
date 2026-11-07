# Modified from EPyMARL for OA-QMIX and opponent-profile research.
import os
import sys
import warnings

from .multiagentenv import MultiAgentEnv
from .gymma import GymmaWrapper

try:
    from .smaclite_wrapper import SMACliteWrapper
except ImportError:
    SMACliteWrapper = None
    warnings.warn(
        "SMAClite is not installed, so the smaclite environment will not be available!"
    )


if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


def __check_and_prepare_smac_kwargs(kwargs):
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    card_cfg = kwargs.get("card", {}) or {}
    if not kwargs["common_reward"]:
        assert card_cfg.get("enabled", False) and card_cfg.get(
            "reward_mode", "team"
        ) == "individual", (
            "SMAC only supports common reward natively. To run with common_reward=False, "
            "enable the LLM-MARL reward module and use an individual reward override."
        )
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    if "card" in kwargs:
        del kwargs["card"]
    assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return kwargs


def smaclite_fn(**kwargs) -> MultiAgentEnv:
    if SMACliteWrapper is None:
        raise ImportError(
            "SMAClite is not installed. Install smaclite to use env='smaclite'."
        )
    kwargs = __check_and_prepare_smac_kwargs(kwargs)
    return SMACliteWrapper(**kwargs)


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


REGISTRY = {}
REGISTRY["smaclite"] = smaclite_fn
REGISTRY["gymma"] = gymma_fn


# registering both smac and smacv2 causes a pysc2 error
# --> dynamically register the needed env
def register_smac():
    from .smac_wrapper import SMACWrapper

    def smac_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACWrapper(**kwargs)

    REGISTRY["sc2"] = smac_fn


def register_smacv2():
    from .smacv2_wrapper import SMACv2Wrapper

    def smacv2_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACv2Wrapper(**kwargs)

    REGISTRY["sc2v2"] = smacv2_fn
