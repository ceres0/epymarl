# LLM-MARL Architecture

`LLM-MARL` is implemented as a runner-side reward/profile module for EPyMARL.

## Core rules

- Learners remain unchanged and continue to consume `batch["reward"]`.
- Reward shaping lives between `env.step(...)` and `EpisodeBatch.update(...)`.
- The same reward spec format is reused across QMIX and MAPPO.
- LLM-MARL artifacts are written outside the replay buffer:
  - reward spec snapshot
  - episode trajectory dumps
  - process / trajectory / preference feedback
  - TPE summaries

## Main modules

- `RewardSpec` / `RewardSpecStore`: JSON reward artifacts with versioning helpers.
- `RewardExecutor`: validates and executes reward code against a restricted context.
- `TrajectoryStore`: dumps step-wise and episode-wise reward traces.
- `TPEEvaluator`: checks outcome-consistent ordering over offline trajectories.
- `FeedbackBuilder`: builds the three CARD feedback views.
- `CardOrchestrator`: legacy class name that stores per-run metadata and reward-version history.
- `CardRuntime`: legacy class name that binds the above pieces to EPyMARL runners.

## Integration points

- `run.py`: creates the runtime and writes final LLM-MARL summaries.
- `episode_runner.py`: applies reward override in the single-env path used by QMIX.
- `parallel_runner.py`: applies reward override in the multi-env path used by MAPPO.
- `smac_wrapper.py` / `smacv2_wrapper.py`: expose battle summaries for reward code.

## Current scope

- Team reward path is implemented and shared by QMIX and MAPPO.
- Individual reward schema is validated at runtime, but SMAC/SMACv2 still require
  LLM-MARL reward override to be active if `common_reward=False`.
- LLM querying is intentionally separated behind provider/config gates; artifacts
  are produced so another process or model can refine reward specs or analyze
  opponent behavior.
