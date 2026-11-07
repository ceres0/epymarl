# Modified from EPyMARL for OA-QMIX and opponent-profile research.
"""Backward-compatible import shim for the renamed llm_marl package.

New code should import from ``llm_marl``. This package remains only so older
experiment scripts and result-replay commands keep working.
"""

from llm_marl import *  # noqa: F401,F403
