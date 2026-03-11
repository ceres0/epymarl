# CARD -> PyMARL (SMAC) validation notes

This branch adds a minimal, testable subset of CARD (KBS 2025) to epymarl:

- Pluggable reward code ("CARD reward plugin") for SMAC via `env_args.card_reward`.
- Optional sub-reward logging and trajectory dumping to `results/card/trajectories/...`.
- Offline TPE checker that re-evaluates candidate reward code on stored trajectories.

## 1) Run SMAC with a CARD reward plugin

Example (QMIX on 3m) using the env reward as-is:

```bash
python src/main.py --config=qmix --env-config=sc2_card \
  env_args.map_name=3m \
  env_args.card_reward.plugin=card.rewards.smac_env_reward
```

Switch to a simple shaped reward (adds terminal win bonus):

```bash
python src/main.py --config=qmix --env-config=sc2_card \
  env_args.map_name=3m \
  env_args.card_reward.plugin=card.rewards.smac_dense_v1
```

You can pass JSON kwargs to the plugin (SACRED CLI override syntax):

```bash
python src/main.py --config=qmix --env-config=sc2_card \
  env_args.map_name=3m \
  env_args.card_reward.plugin=card.rewards.smac_dense_v1 \
  env_args.card_reward.kwargs='{"win_bonus": 50.0, "env_scale": 1.0}'
```

## 2) Collect trajectories for offline TPE

```bash
python src/main.py --config=qmix --env-config=sc2_card \
  env_args.map_name=3m \
  card_collect_trajectories=True \
  card_run_id=my_card_run
```

Trajectories will be written under:

`results/card/trajectories/my_card_run/test/*.npz`

Each `.npz` stores `states`, `actions`, `terminated`, plus a JSON `meta`.

## 3) Run offline TPE (order-preserving) check

```bash
python src/card/cli_tpe.py \
  --traj-dir results/card/trajectories/my_card_run/test \
  --reward-plugin card.rewards.smac_dense_v1 \
  --threshold 0.8
```

It prints a JSON summary with `passed`, `accuracy`, `n_success`, `n_fail`.

Notes:
- This is the infrastructure needed for Stage A (no LLM): test a handful of reward plugins/weights and validate that feedback + TPE plumbing works in SMAC.
- Stage B (LLM-driven reward code generation/improvement) can be built on top by generating new reward modules and swapping `env_args.card_reward.plugin` each iteration.

