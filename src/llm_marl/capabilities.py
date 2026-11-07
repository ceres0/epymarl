# Modified from EPyMARL for OA-QMIX and opponent-profile research.
from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmCapability:
    name: str
    supports_common_reward: bool
    supports_individual_reward: bool
    preferred_runner: str
    notes: str = ""


def infer_algorithm_capability(args):
    learner = getattr(args, "learner", "")
    mixer = getattr(args, "mixer", "")

    if learner == "q_learner" and mixer == "qmix":
        return AlgorithmCapability(
            name="qmix",
            supports_common_reward=True,
            supports_individual_reward=False,
            preferred_runner="episode",
            notes="Value decomposition relies on a shared team reward.",
        )

    if learner == "q_learner" and mixer == "vdn":
        return AlgorithmCapability(
            name="vdn",
            supports_common_reward=True,
            supports_individual_reward=False,
            preferred_runner="episode",
            notes="VDN consumes a common reward signal.",
        )

    if learner == "ppo_learner":
        return AlgorithmCapability(
            name="mappo",
            supports_common_reward=True,
            supports_individual_reward=True,
            preferred_runner="parallel",
            notes="Actor-critic path can consume common or per-agent rewards.",
        )

    if learner in {"actor_critic_learner", "actor_critic_pac_learner"}:
        return AlgorithmCapability(
            name=learner,
            supports_common_reward=True,
            supports_individual_reward=True,
            preferred_runner=getattr(args, "runner", "parallel"),
        )

    return AlgorithmCapability(
        name=learner or "unknown",
        supports_common_reward=True,
        supports_individual_reward=not getattr(args, "common_reward", True),
        preferred_runner=getattr(args, "runner", "episode"),
        notes="Capability inferred conservatively from the configured learner.",
    )
