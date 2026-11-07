from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path


METRIC_PATTERNS = {
    "test_battle_won_mean": re.compile(r"test_battle_won_mean\s*[:=]\s*([-+0-9.eE]+)"),
    "test_return_mean": re.compile(r"test_return_mean\s*[:=]\s*([-+0-9.eE]+)"),
    "card_llm_calls": re.compile(r"card_llm_calls\s*[:=]\s*([-+0-9.eE]+)"),
    "card_llm_total_tokens": re.compile(r"card_llm_total_tokens\s*[:=]\s*([-+0-9.eE]+)"),
    "card_llm_mean_latency_seconds": re.compile(
        r"card_llm_mean_latency_seconds\s*[:=]\s*([-+0-9.eE]+)"
    ),
}


def parse_job_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    row = {
        "job_log": str(path),
        "status": infer_status(text),
        "run_name": "",
        "stage": "",
        "map": "",
        "seed": "",
        "t_env": "",
        "t_max": "",
    }
    command_match = re.search(r"\bname=([^\s]+)", text)
    if command_match:
        row["run_name"] = command_match.group(1)
    map_match = re.search(r"env_args\.map_name=([^\s]+)", text)
    if map_match:
        row["map"] = map_match.group(1)
    seed_match = re.search(r"\bseed=(\d+)\b", text)
    if seed_match:
        row["seed"] = seed_match.group(1)
    stage_match = re.match(r"\d+_([^_]+(?:_[^_]+)*?)_magpie_llm_profile", path.name)
    if stage_match:
        row["stage"] = stage_match.group(1)
    t_env_matches = re.findall(r"t_env:\s*(\d+)\s*/\s*(\d+)", text)
    if t_env_matches:
        row["t_env"], row["t_max"] = t_env_matches[-1]
    for key, pattern in METRIC_PATTERNS.items():
        matches = pattern.findall(text)
        row[key] = matches[-1] if matches else ""
    return row


def infer_status(text: str) -> str:
    tail = text[-12000:]
    if "[queue] DONE" in tail or "Finished Training" in tail or "Finished after" in tail:
        return "DONE"
    if any(token in tail for token in ("Traceback", "[queue] FAILED", "RuntimeError", "ConnectError")):
        return "FAILED"
    return "RUNNING_OR_PARTIAL"


def find_usage_for_run(results_root: Path, run_name: str) -> dict:
    if not run_name or not results_root.exists():
        return {}
    matches = []
    for path in results_root.glob(f"*{run_name}*/llm_usage.json"):
        try:
            matches.append((path.stat().st_mtime, path))
        except OSError:
            continue
    if not matches:
        return {}
    path = sorted(matches)[-1][1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"llm_usage_path": str(path), "llm_usage_error": "unreadable"}
    total = payload.get("total", {}) if isinstance(payload, dict) else {}
    by_purpose = payload.get("by_purpose", {}) if isinstance(payload, dict) else {}
    opponent = by_purpose.get("opponent_profile", {}) if isinstance(by_purpose, dict) else {}
    return {
        "llm_usage_path": str(path),
        "llm_calls": total.get("calls", ""),
        "llm_total_tokens": total.get("total_tokens", ""),
        "llm_estimated_calls": total.get("estimated_calls", ""),
        "llm_latency_seconds": total.get("latency_seconds", ""),
        "llm_mean_latency_seconds": total.get("mean_latency_seconds", ""),
        "opponent_profile_calls": opponent.get("calls", ""),
        "opponent_profile_total_tokens": opponent.get("total_tokens", ""),
        "opponent_profile_latency_seconds": opponent.get("latency_seconds", ""),
        "opponent_profile_mean_latency_seconds": opponent.get("mean_latency_seconds", ""),
    }


def summarize(rows: list[dict]) -> dict:
    completed = [row for row in rows if row.get("status") == "DONE"]
    return {
        "jobs": len(rows),
        "done": len(completed),
        "failed": sum(1 for row in rows if row.get("status") == "FAILED"),
        "running_or_partial": sum(
            1 for row in rows if row.get("status") == "RUNNING_OR_PARTIAL"
        ),
        "mean_test_battle_won_mean": mean_float(completed, "test_battle_won_mean"),
        "mean_test_return_mean": mean_float(completed, "test_return_mean"),
        "mean_llm_calls": mean_float(completed, "llm_calls"),
        "mean_llm_total_tokens": mean_float(completed, "llm_total_tokens"),
        "mean_llm_latency_seconds": mean_float(completed, "llm_latency_seconds"),
        "mean_llm_mean_latency_seconds": mean_float(completed, "llm_mean_latency_seconds"),
    }


def mean_float(rows: list[dict], key: str):
    values = []
    for row in rows:
        try:
            if row.get(key) not in ("", None):
                values.append(float(row[key]))
        except (TypeError, ValueError):
            continue
    if not values:
        return ""
    return sum(values) / len(values)


def write_outputs(output_prefix: Path, rows: list[dict], summary: dict) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    output_prefix.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# LLM Profile Training Queue Summary",
        "",
        json.dumps(summary, indent=2, sort_keys=True),
        "",
        "| run | status | map | seed | win | return | llm calls | llm latency s | mean call s |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run} | {status} | {map} | {seed} | {win} | {ret} | {calls} | {lat} | {mean_lat} |".format(
                run=row.get("run_name", ""),
                status=row.get("status", ""),
                map=row.get("map", ""),
                seed=row.get("seed", ""),
                win=row.get("test_battle_won_mean", ""),
                ret=row.get("test_return_mean", ""),
                calls=row.get("llm_calls", ""),
                lat=row.get("llm_latency_seconds", ""),
                mean_lat=row.get("llm_mean_latency_seconds", ""),
            )
        )
    output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MAGPIE LLM-profile training queue results.")
    parser.add_argument("--job-log-dir", required=True)
    parser.add_argument("--results-root", default="results/card")
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    job_log_dir = Path(args.job_log_dir)
    results_root = Path(args.results_root)
    rows = []
    for path in sorted(job_log_dir.glob("*.log")):
        row = parse_job_log(path)
        row.update(find_usage_for_run(results_root, row.get("run_name", "")))
        rows.append(row)
    summary = summarize(rows)
    write_outputs(Path(args.output_prefix), rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
