# MAGPIE Smoke Commands

Run from `epymarl/`.

## Baseline QMIX Matrix Game

```powershell
python src/main.py --config=qmix --env-config=gymma with env_args.key=matrixgames:penalty-100-nostate-v0 t_max=50 use_cuda=False use_tensorboard=True use_wandb=True wandb_mode=offline
```

## Historical Reward Override Path

Provide an existing reward spec path:

```powershell
python src/main.py --config=qmix --env-config=gymma with env_args.key=matrixgames:penalty-100-nostate-v0 t_max=50 use_cuda=False card.enabled=True card.reward_spec_path=<reward_spec.json> use_tensorboard=True use_wandb=True wandb_mode=offline
```

## MAGPIE Profile Extraction From Existing Trajectories

```powershell
python scripts/magpie_extract_profiles.py --input results/card/iterative_group_100k_v1 --output results/magpie/offline_profile_iterative_group_100k_v1
```

## MAGPIE Opponent-Aware Critic Smoke

```powershell
python src/main.py --config=oa_qmix --env-config=gymma with env_args.key=matrixgames:penalty-100-nostate-v0 t_max=50 use_cuda=False use_tensorboard=True use_wandb=True wandb_mode=offline
```

## Collect Run Index

```powershell
python scripts/magpie_collect_results.py --results-root results --output results/runs_index.csv
```
