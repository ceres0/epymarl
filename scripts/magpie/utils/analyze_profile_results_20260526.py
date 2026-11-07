from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any


RUN_RE = re.compile(
    r"standard_smac_(?P<map>.+?)_"
    r"(?P<tag>zero_profile_ablation_6maps_wheshell_20260526|llm_profile_restore_4maps_wheshell_20260526)"
    r"(?P<llm>_deepseek_v4_flash_default)?_seed(?P<seed>\d+)_"
    r"(?P<map2>.+?)_magpie_(?P<method>zero_profile|llm_profile_critic_only)$"
)

METRICS = [
    "test_battle_won_mean",
    "test_return_mean",
    "battle_won_mean",
    "return_mean",
    "card_win_rate",
    "test_card_win_rate",
    "llm_calls",
    "llm_total_tokens",
    "llm_mean_latency_seconds",
]

PRIMARY_METRICS = ["test_battle_won_mean", "test_return_mean"]

METHOD_LABELS = {
    "zero_profile": "Zero profile",
    "llm_profile": "LLM profile",
}

MAP_ORDER = ["8m", "2s3z", "3m", "3s5z", "1c3s5z", "MMM"]
LLM_MAP_ORDER = ["3m", "3s5z", "1c3s5z", "MMM"]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_run_name(name: str) -> dict[str, Any] | None:
    match = RUN_RE.match(name)
    if not match:
        return None
    groups = match.groupdict()
    method = "zero_profile" if groups["method"] == "zero_profile" else "llm_profile"
    return {
        "run_name": name,
        "map": groups["map"],
        "seed": int(groups["seed"]),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "dataset": groups["tag"],
    }


def scalar_events_from_tb(run_dir: Path, metric: str) -> list[tuple[int, float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        pass
    return scalar_events_from_tfrecord(run_dir, metric)


def scalar_events_from_tfrecord(run_dir: Path, metric: str) -> list[tuple[int, float]]:
    try:
        from tensorboard_logger.tf_protobuf import event_pb2
    except Exception:
        return []
    points: list[tuple[int, float]] = []
    for event_file in sorted(run_dir.glob("events.out.tfevents.*")):
        try:
            with event_file.open("rb") as handle:
                while True:
                    header = handle.read(8)
                    if not header:
                        break
                    if len(header) != 8:
                        break
                    length = struct.unpack("<Q", header)[0]
                    handle.read(4)  # length CRC
                    data = handle.read(length)
                    if len(data) != length:
                        break
                    handle.read(4)  # data CRC
                    event = event_pb2.Event()
                    event.ParseFromString(data)
                    if not event.HasField("summary"):
                        continue
                    for value in event.summary.value:
                        if value.tag != metric:
                            continue
                        if value.HasField("simple_value"):
                            points.append((int(event.step), float(value.simple_value)))
        except Exception:
            continue
    return dedupe(points)
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return []
    try:
        acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        acc.Reload()
        tags = set(acc.Tags().get("scalars", []))
        if metric not in tags:
            return []
        return [(int(event.step), float(event.value)) for event in acc.Scalars(metric)]
    except Exception:
        return []


def sacred_metrics_by_run(results_root: Path) -> dict[str, dict[str, Any]]:
    by_run: dict[str, dict[str, Any]] = {}
    sacred_root = results_root / "sacred"
    if not sacred_root.is_dir():
        return by_run
    for config_path in sacred_root.glob("*/*/*/config.json"):
        try:
            config = read_json(config_path)
        except Exception:
            continue
        run_name = str(config.get("unique_token") or "")
        if not run_name:
            continue
        metrics_path = config_path.parent / "metrics.json"
        run_path = config_path.parent / "run.json"
        if not metrics_path.is_file():
            continue
        try:
            metrics = read_json(metrics_path)
        except Exception:
            continue
        status = ""
        if run_path.is_file():
            try:
                status = str(read_json(run_path).get("status", ""))
            except Exception:
                status = ""
        entry = {
            "metrics": metrics,
            "status": status,
            "sacred_path": str(config_path.parent),
            "max_step": sacred_max_step(metrics),
        }
        previous = by_run.get(run_name)
        if previous is None or sacred_entry_rank(entry) > sacred_entry_rank(previous):
            by_run[run_name] = entry
    return by_run


def sacred_max_step(metrics: dict[str, Any]) -> int:
    max_step = 0
    for metric in PRIMARY_METRICS + ["battle_won_mean", "return_mean"]:
        item = metrics.get(metric)
        if not isinstance(item, dict):
            continue
        for step in item.get("steps") or []:
            try:
                max_step = max(max_step, int(step))
            except Exception:
                continue
    return max_step


def sacred_entry_rank(entry: dict[str, Any]) -> tuple[int, int]:
    status = str(entry.get("status", ""))
    status_rank = 1 if status == "COMPLETED" else 0
    return int(entry.get("max_step") or 0), status_rank


def scalar_events_from_sacred(sacred_entry: dict[str, Any] | None, metric: str) -> list[tuple[int, float]]:
    if not sacred_entry:
        return []
    item = (sacred_entry.get("metrics") or {}).get(metric)
    if not isinstance(item, dict):
        return []
    steps = item.get("steps") or []
    values = item.get("values") or []
    points: list[tuple[int, float]] = []
    for step, value in zip(steps, values):
        try:
            points.append((int(step), float(value)))
        except Exception:
            continue
    return points


def collect_runs(results_root: Path) -> list[dict[str, Any]]:
    tb_root = results_root / "tb_logs"
    sacred_by_run = sacred_metrics_by_run(results_root)
    rows: list[dict[str, Any]] = []
    if not tb_root.is_dir():
        return rows
    for run_dir in sorted(path for path in tb_root.iterdir() if path.is_dir()):
        parsed = parse_run_name(run_dir.name)
        if parsed is None:
            continue
        sacred_entry = sacred_by_run.get(run_dir.name)
        parsed["tb_path"] = str(run_dir)
        parsed["sacred_path"] = sacred_entry.get("sacred_path", "") if sacred_entry else ""
        parsed["status"] = sacred_entry.get("status", "") if sacred_entry else ""
        parsed["metrics_source"] = "tb"
        parsed["curves"] = {}
        for metric in METRICS:
            points = scalar_events_from_tb(run_dir, metric)
            if not points:
                points = scalar_events_from_sacred(sacred_entry, metric)
                if points and parsed["metrics_source"] == "tb":
                    parsed["metrics_source"] = "mixed_or_sacred"
            parsed["curves"][metric] = dedupe(points)
        rows.append(parsed)
    return rows


def dedupe(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    by_step: dict[int, float] = {}
    for step, value in points:
        by_step[int(step)] = float(value)
    return sorted(by_step.items())


def final_value(curve: list[tuple[int, float]]) -> tuple[int | None, float | None]:
    if not curve:
        return None, None
    return curve[-1][0], curve[-1][1]


def write_per_seed(path: Path, runs: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "map",
        "seed",
        "status",
        "metrics_source",
        "latest_t_env",
        "test_battle_won_mean",
        "test_return_mean",
        "battle_won_mean",
        "return_mean",
        "llm_calls",
        "llm_total_tokens",
        "llm_mean_latency_seconds",
        "tb_path",
        "sacred_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in sorted(runs, key=lambda r: (r["method"], MAP_ORDER.index(r["map"]) if r["map"] in MAP_ORDER else 99, r["seed"])):
            latest_t, win = final_value(run["curves"].get("test_battle_won_mean", []))
            _, ret = final_value(run["curves"].get("test_return_mean", []))
            _, train_win = final_value(run["curves"].get("battle_won_mean", []))
            _, train_ret = final_value(run["curves"].get("return_mean", []))
            _, llm_calls = final_value(run["curves"].get("llm_calls", []))
            _, llm_tokens = final_value(run["curves"].get("llm_total_tokens", []))
            _, llm_latency = final_value(run["curves"].get("llm_mean_latency_seconds", []))
            writer.writerow(
                {
                    "method": run["method"],
                    "map": run["map"],
                    "seed": run["seed"],
                    "status": run["status"],
                    "metrics_source": run["metrics_source"],
                    "latest_t_env": latest_t or "",
                    "test_battle_won_mean": fmt(win),
                    "test_return_mean": fmt(ret),
                    "battle_won_mean": fmt(train_win),
                    "return_mean": fmt(train_ret),
                    "llm_calls": fmt(llm_calls),
                    "llm_total_tokens": fmt(llm_tokens),
                    "llm_mean_latency_seconds": fmt(llm_latency),
                    "tb_path": run["tb_path"],
                    "sacred_path": run["sacred_path"],
                }
            )


def aggregate_final(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["map"], run["method"])].append(run)
    rows: list[dict[str, Any]] = []
    for (map_name, method), group in sorted(grouped.items(), key=lambda item: (MAP_ORDER.index(item[0][0]) if item[0][0] in MAP_ORDER else 99, item[0][1])):
        win_values: list[float] = []
        return_values: list[float] = []
        latest_steps: list[int] = []
        for run in group:
            step, win = final_value(run["curves"].get("test_battle_won_mean", []))
            _, ret = final_value(run["curves"].get("test_return_mean", []))
            if step is not None:
                latest_steps.append(step)
            if win is not None:
                win_values.append(win)
            if ret is not None:
                return_values.append(ret)
        rows.append(
            {
                "map": map_name,
                "method": method,
                "seeds": len(group),
                "mean_latest_t_env": mean_or_none(latest_steps),
                "min_latest_t_env": min(latest_steps) if latest_steps else "",
                "mean_test_battle_won_mean": mean_or_none(win_values),
                "std_test_battle_won_mean": std_or_none(win_values),
                "mean_test_return_mean": mean_or_none(return_values),
                "std_test_return_mean": std_or_none(return_values),
            }
        )
    return rows


def write_aggregate(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "map",
        "method",
        "seeds",
        "mean_latest_t_env",
        "min_latest_t_env",
        "mean_test_battle_won_mean",
        "std_test_battle_won_mean",
        "mean_test_return_mean",
        "std_test_return_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})


def interpolate(curve: list[tuple[int, float]], grid: list[int]) -> list[float]:
    if not curve:
        return [math.nan for _ in grid]
    xs = [point[0] for point in curve]
    ys = [point[1] for point in curve]
    values: list[float] = []
    idx = 0
    for x in grid:
        while idx + 1 < len(xs) and xs[idx + 1] <= x:
            idx += 1
        if x < xs[0]:
            values.append(math.nan)
        elif idx + 1 >= len(xs):
            values.append(ys[-1])
        else:
            x0, x1 = xs[idx], xs[idx + 1]
            y0, y1 = ys[idx], ys[idx + 1]
            values.append(y1 if x1 == x0 else y0 + (y1 - y0) * ((x - x0) / (x1 - x0)))
    return values


def curve_rows(runs: list[dict[str, Any]], metric: str, grid_step: int) -> list[dict[str, Any]]:
    max_step = max((final_value(run["curves"].get(metric, []))[0] or 0 for run in runs), default=0)
    max_step = max(2_000_000, int(math.ceil(max_step / grid_step) * grid_step))
    grid = list(range(grid_step, max_step + 1, grid_step))
    grouped: dict[tuple[str, str], list[list[tuple[int, float]]]] = defaultdict(list)
    for run in runs:
        curve = run["curves"].get(metric, [])
        if curve:
            grouped[(run["map"], run["method"])].append(curve)
    rows: list[dict[str, Any]] = []
    for (map_name, method), curves in grouped.items():
        series = [interpolate(curve, grid) for curve in curves]
        for col, step in enumerate(grid):
            values = [items[col] for items in series if not math.isnan(items[col])]
            if not values:
                continue
            rows.append(
                {
                    "map": map_name,
                    "method": method,
                    "metric": metric,
                    "t_env": step,
                    "seed_count": len(values),
                    "mean": sum(values) / len(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
            )
    return sorted(rows, key=lambda r: (MAP_ORDER.index(r["map"]) if r["map"] in MAP_ORDER else 99, r["method"], r["t_env"]))


def write_curve_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["map", "method", "metric", "t_env", "seed_count", "mean", "std"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})


def plot_metric(path: Path, rows: list[dict[str, Any]], metric: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    metric_rows = [row for row in rows if row["metric"] == metric]
    if not metric_rows:
        return False
    maps = [m for m in MAP_ORDER if any(row["map"] == m for row in metric_rows)]
    cols = 3
    rows_n = math.ceil(len(maps) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(15, 4.2 * rows_n), squeeze=False)
    colors = {"zero_profile": "#4C78A8", "llm_profile": "#F58518"}
    for ax, map_name in zip(axes.flatten(), maps):
        for method in ["zero_profile", "llm_profile"]:
            group = [row for row in metric_rows if row["map"] == map_name and row["method"] == method]
            if not group:
                continue
            xs = [row["t_env"] / 1_000_000 for row in group]
            means = [row["mean"] for row in group]
            stds = [row["std"] for row in group]
            lower = [max(0.0, m - s) if "win" in metric else m - s for m, s in zip(means, stds)]
            upper = [min(1.0, m + s) if "win" in metric else m + s for m, s in zip(means, stds)]
            seed_count = max(row["seed_count"] for row in group)
            ax.plot(xs, means, color=colors[method], label=f"{METHOD_LABELS[method]} (n={seed_count})", linewidth=2)
            ax.fill_between(xs, lower, upper, color=colors[method], alpha=0.15)
        ax.set_title(map_name)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Environment steps (M)")
        ax.set_ylabel(metric)
        if "win" in metric:
            ax.set_ylim(-0.03, 1.03)
    for ax in axes.flatten()[len(maps) :]:
        ax.axis("off")
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"MAGPIE profile comparison: {metric}", y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def write_report(path: Path, runs: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], plotted: list[str]) -> None:
    by_method = defaultdict(int)
    for run in runs:
        by_method[run["method"]] += 1
    lines = [
        "# MAGPIE Profile Result Comparison",
        "",
        f"Runs indexed from `tb_logs`: {len(runs)}",
        f"- zero-profile: {by_method['zero_profile']}",
        f"- llm-profile: {by_method['llm_profile']}",
        "",
        "Primary comparison metric is `test_battle_won_mean`; `test_return_mean` is reported as secondary.",
        "For `3m`, both queues contain intentionally stopped partial runs, so final values are checkpoint/partial-run values rather than full 2M training results.",
        "",
        "## Aggregate Final Metrics",
        "",
        "| map | method | seeds | mean latest t_env | win mean | win std | return mean | return std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {map} | {method} | {seeds} | {mean_latest_t_env} | {mean_test_battle_won_mean} | {std_test_battle_won_mean} | {mean_test_return_mean} | {std_test_return_mean} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Curve Artifacts",
            "",
        ]
    )
    for artifact in plotted:
        lines.append(f"- `{artifact}`")
    lines.extend(
        [
            "",
            "## Generated Tables",
            "",
            "- `per_seed_metrics.csv`",
            "- `aggregate_metrics.csv`",
            "- `test_battle_won_mean_curves.csv`",
            "- `test_return_mean_curves.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean_or_none(values: list[float] | list[int]) -> float | str:
    return sum(values) / len(values) if values else ""


def std_or_none(values: list[float]) -> float | str:
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 20260526 wheshell MAGPIE profile results.")
    parser.add_argument("--archive-root", required=True, help="Path containing results/{tb_logs,sacred,card,hpc_jobs}.")
    parser.add_argument("--out-dir", default="", help="Output directory. Defaults to archive-root/analysis.")
    parser.add_argument("--grid-step", type=int, default=50_000)
    args = parser.parse_args()

    archive_root = Path(args.archive_root).resolve()
    results_root = archive_root / "results"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else archive_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(results_root)
    if not runs:
        raise SystemExit(f"No matching runs found under {results_root / 'tb_logs'}")

    write_per_seed(out_dir / "per_seed_metrics.csv", runs)
    aggregate_rows = aggregate_final(runs)
    write_aggregate(out_dir / "aggregate_metrics.csv", aggregate_rows)

    plotted: list[str] = []
    for metric in PRIMARY_METRICS:
        rows = curve_rows(runs, metric, args.grid_step)
        write_curve_csv(out_dir / f"{metric}_curves.csv", rows)
        png_path = out_dir / f"{metric}_curves.png"
        if plot_metric(png_path, rows, metric):
            plotted.append(png_path.name)

    write_report(out_dir / "analysis_report.md", runs, aggregate_rows, plotted)
    print(f"runs={len(runs)}")
    print(f"out_dir={out_dir}")
    print(f"plots={','.join(plotted) if plotted else 'none'}")


if __name__ == "__main__":
    main()
