"""CARD-style reward shaping + TPE utilities.

This repo is based on PyMARL. CARD (KBS 2025) is an outer-loop method for
iteratively improving reward code using feedback and Trajectory Preference
Evaluation (TPE). Here we add the pieces we need to validate the idea in SMAC.
"""

__all__ = [
    "reward_shaper",
    "tpe",
    "trajectory",
]
