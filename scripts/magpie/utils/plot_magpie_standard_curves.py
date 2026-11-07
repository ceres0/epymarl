import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    "qmix_baseline",
    "magpie_critic_only",
    "magpie_full",
    "llm_opponent_summary_reward",
    "tpe_llm_opponent_reward",
    "tpe_llm_reward_only",
]

METHOD_LABELS = {
    "qmix_baseline": "QMIX",
    "magpie_critic_only": "MAGPIE critic",
    "magpie_full": "MAGPIE full",
    "llm_opponent_summary_reward": "LLM opponent summary reward",
    "tpe_llm_opponent_reward": "TPE+LLM opponent reward",
    "tpe_llm_reward_only": "TPE+LLM reward only",
}

MAPS = ["8m", "2s3z"]
METRIC_RE = re.compile(r"\btest_battle_won_mean:\s*([-+]?\d+(?:\.\d+)?)")
T_ENV_RE = re.compile(r"\bt_env:\s*(\d+)\s*/")
SEED_RE = re.compile(r"_seed(\d+)\.log$")


def parse_log_name(path):
    name = path.name
    seed_match = SEED_RE.search(name)
    if not seed_match:
        return None

    stem = name[: seed_match.start()]
    parts = stem.split("_", 1)
    if len(parts) != 2:
        return None
    body = parts[1]
    if not body.startswith("standard_smac_"):
        return None
    body = body[len("standard_smac_") :]

    for map_name in MAPS:
        prefix = f"{map_name}_"
        if body.startswith(prefix):
            method = body[len(prefix) :]
            if method in METHODS:
                return map_name, method, int(seed_match.group(1))
    return None


def parse_curve(path):
    points = []
    pending_metric = None
    current_t_env = None
    last_t_env = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("wandb:"):
                continue

            metric_match = METRIC_RE.search(line)
            if metric_match:
                value = float(metric_match.group(1))
                if current_t_env is None or current_t_env == last_t_env:
                    pending_metric = value
                else:
                    points.append((current_t_env, value))
                    last_t_env = current_t_env
                continue

            t_env_match = T_ENV_RE.search(line)
            if t_env_match:
                current_t_env = int(t_env_match.group(1))
                if pending_metric is not None and current_t_env != last_t_env:
                    points.append((current_t_env, pending_metric))
                    last_t_env = current_t_env
                    pending_metric = None

    deduped = {}
    for t_env, value in points:
        deduped[t_env] = value
    return sorted(deduped.items())


def collect_curves(log_dirs):
    curves = {}
    for log_dir in log_dirs:
        for path in Path(log_dir).glob("*.log"):
            parsed = parse_log_name(path)
            if not parsed:
                continue
            map_name, method, seed = parsed
            curve = parse_curve(path)
            if curve:
                curves[(map_name, method, seed)] = curve
    return curves


def interpolate_group(curves, map_name, method, grid):
    seed_curves = [
        curve
        for (curve_map, curve_method, _seed), curve in curves.items()
        if curve_map == map_name and curve_method == method
    ]
    series = []
    for curve in seed_curves:
        xs = np.array([point[0] for point in curve], dtype=float)
        ys = np.array([point[1] for point in curve], dtype=float)
        if len(xs) < 2:
            continue
        values = np.interp(grid, xs, ys, left=np.nan, right=ys[-1])
        series.append(values)
    if not series:
        return None
    stacked = np.vstack(series)
    mean = np.full(stacked.shape[1], np.nan)
    std = np.full(stacked.shape[1], np.nan)
    valid = ~np.all(np.isnan(stacked), axis=0)
    mean[valid] = np.nanmean(stacked[:, valid], axis=0)
    std[valid] = np.nanstd(stacked[:, valid], axis=0)
    return {
        "n": len(series),
        "mean": mean,
        "std": std,
    }


def write_summary_csv(path, curves, grid):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["map", "method", "t_env", "seed_count", "mean", "std"])
        for map_name in MAPS:
            for method in METHODS:
                group = interpolate_group(curves, map_name, method, grid)
                if group is None:
                    continue
                for t_env, mean, std in zip(grid, group["mean"], group["std"]):
                    if math.isnan(mean):
                        continue
                    writer.writerow([map_name, method, int(t_env), group["n"], mean, std])


def plot(curves, output_path, csv_path):
    grid = np.arange(50_000, 2_000_001, 50_000, dtype=float)
    write_summary_csv(csv_path, curves, grid)

    colors = {
        "qmix_baseline": "#4C78A8",
        "magpie_critic_only": "#F58518",
        "magpie_full": "#E45756",
        "llm_opponent_summary_reward": "#72B7B2",
        "tpe_llm_opponent_reward": "#54A24B",
        "tpe_llm_reward_only": "#B279A2",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, map_name in zip(axes, MAPS):
        for method in METHODS:
            group = interpolate_group(curves, map_name, method, grid)
            if group is None:
                continue
            x = grid / 1_000_000
            mean = group["mean"]
            std = group["std"]
            label = f"{METHOD_LABELS[method]} (n={group['n']})"
            ax.plot(x, mean, label=label, color=colors[method], linewidth=2)
            ax.fill_between(
                x,
                np.clip(mean - std, 0, 1),
                np.clip(mean + std, 0, 1),
                color=colors[method],
                alpha=0.12,
                linewidth=0,
            )
        ax.set_title(f"SMAC {map_name}")
        ax.set_xlabel("Environment steps (millions)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Test battle won mean")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("Standard SMAC Training Curves")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()

    curves = collect_curves(args.log_dir)
    plot(curves, Path(args.output), Path(args.csv_output))
    print(f"curves={len(curves)}")
    print(args.output)
    print(args.csv_output)


if __name__ == "__main__":
    main()
