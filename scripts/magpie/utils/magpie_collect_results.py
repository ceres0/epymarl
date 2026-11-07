import argparse
import csv
import json
import os
from datetime import datetime


FIELDNAMES = [
    "run_id",
    "method_name",
    "method_group",
    "env_name",
    "map_name",
    "seed",
    "t_max",
    "sacred_path",
    "tb_path",
    "wandb_path",
    "profile_path",
    "magpie_pe_path",
    "reward_spec_path",
    "collected_at",
]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _walk_files(root, filename):
    matches = []
    if not os.path.isdir(root):
        return matches
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            matches.append(os.path.join(dirpath, filename))
    return sorted(matches)


def _walk_dirs(root, predicate):
    matches = []
    if not os.path.isdir(root):
        return matches
    for dirpath, dirnames, _ in os.walk(root):
        for dirname in dirnames:
            path = os.path.join(dirpath, dirname)
            if predicate(path):
                matches.append(path)
    return sorted(matches)


def _map_name(config):
    env_args = config.get("env_args", {}) or {}
    return env_args.get("map_name") or env_args.get("key") or ""


def _method_group(config):
    name = config.get("name", "")
    card = config.get("card", {}) or {}
    opponent = card.get("opponent", {}) or {}
    if name.startswith("magpie"):
        return "magpie_ablation" if "only" in name or "random" in name else "magpie_core"
    if card.get("enabled"):
        return "card"
    if opponent.get("enabled"):
        return "magpie_core"
    return "baseline"


def _find_by_prefix(paths, prefix):
    for path in paths:
        if os.path.basename(path).startswith(prefix):
            return path
    return ""


def _find_file_under(path, filename):
    if not path:
        return ""
    for dirpath, _, filenames in os.walk(path):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return ""


def collect_rows(results_root):
    results_root = os.path.abspath(results_root)
    tb_dirs = _walk_dirs(
        os.path.join(results_root, "tb_logs"),
        lambda path: os.path.isfile(
            os.path.join(path, next((name for name in os.listdir(path) if name.startswith("events.out.tfevents")), ""))
        )
        if os.path.isdir(path)
        else False,
    )
    wandb_files = _walk_files(os.path.join(results_root, "wandb"), "run-")
    wandb_dirs = _walk_dirs(
        os.path.join(results_root, "wandb"),
        lambda path: os.path.basename(path).startswith("offline-run-"),
    )
    card_dirs = _walk_dirs(
        os.path.join(results_root, "card"),
        lambda path: os.path.isfile(os.path.join(path, "opponent_profile.json"))
        or os.path.isfile(os.path.join(path, "card_summary.json")),
    )

    rows = []
    for config_path in _walk_files(os.path.join(results_root, "sacred"), "config.json"):
        sacred_path = os.path.dirname(config_path)
        config = _read_json(config_path)
        method_name = config.get("name", os.path.basename(sacred_path))
        map_name = _map_name(config)
        seed = config.get("seed", "")
        token_prefix = f"{method_name}_seed{seed}_"
        tb_path = _find_by_prefix(tb_dirs, token_prefix)
        card_path = _find_by_prefix(card_dirs, token_prefix)
        profile_path = _find_file_under(card_path, "opponent_profile.json")
        summary_path = _find_file_under(card_path, "card_summary.json")
        reward_spec_path = _find_file_under(card_path, "current_reward_spec.json")

        rows.append(
            {
                "run_id": f"{method_name}:{os.path.basename(sacred_path)}",
                "method_name": method_name,
                "method_group": _method_group(config),
                "env_name": config.get("env", ""),
                "map_name": map_name,
                "seed": seed,
                "t_max": config.get("t_max", ""),
                "sacred_path": sacred_path,
                "tb_path": tb_path,
                "wandb_path": "",
                "profile_path": profile_path,
                "magpie_pe_path": summary_path,
                "reward_spec_path": reward_spec_path,
                "collected_at": datetime.utcnow().isoformat() + "Z",
            }
        )

    # Add W&B offline directories as rows when Sacred matching is not available.
    sacred_names = {row["method_name"] for row in rows}
    for wandb_dir in wandb_dirs:
        name = os.path.basename(wandb_dir)
        if any(method in name for method in sacred_names):
            continue
        rows.append(
            {
                "run_id": name,
                "method_name": name,
                "method_group": "wandb",
                "env_name": "",
                "map_name": "",
                "seed": "",
                "t_max": "",
                "sacred_path": "",
                "tb_path": "",
                "wandb_path": wandb_dir,
                "profile_path": "",
                "magpie_pe_path": "",
                "reward_spec_path": "",
                "collected_at": datetime.utcnow().isoformat() + "Z",
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Collect MAGPIE/EPyMARL run artifacts into runs_index.csv."
    )
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = collect_rows(args.results_root)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
