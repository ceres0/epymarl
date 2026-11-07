import argparse
import concurrent.futures
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
from datetime import datetime


METHODS = {
    "qmix_baseline": {
        "alg": "qmix",
        "group": "baseline",
        "overrides": [],
    },
    "magpie_profile_only": {
        "alg": "qmix",
        "group": "magpie_core",
        "overrides": [
            "card.opponent.enabled=True",
            "card.opponent.profile.enabled=True",
            "card.opponent.profile.mode=heuristic",
            "card.opponent.feature_extraction.enabled=True",
        ],
    },
    "magpie_critic_only": {
        "alg": "oa_qmix",
        "group": "magpie_ablation",
        "overrides": [
            "card.opponent.enabled=True",
            "card.opponent.profile.enabled=True",
            "card.opponent.profile.mode=heuristic",
            "card.opponent.critic.enabled=True",
        ],
    },
    "magpie_llm_profile_critic_only": {
        "alg": "oa_qmix",
        "group": "magpie_ablation",
        "overrides": [
            "card.llm.env_file=${MAGPIE_LLM_ENV_FILE}",
            "card.opponent.enabled=True",
            "card.opponent.feature_extraction.enabled=True",
            "card.opponent.profile.enabled=True",
            "card.opponent.profile.mode=llm",
            "card.opponent.profile.min_episodes=${MAGPIE_OPPONENT_PROFILE_MIN_EPISODES}",
            "card.opponent.profile.refresh_episodes=${MAGPIE_OPPONENT_PROFILE_REFRESH_EPISODES}",
            "card.opponent.critic.enabled=True",
        ],
    },
    "magpie_critic_aux": {
        "alg": "oa_qmix",
        "group": "magpie_ablation",
        "overrides": [
            "card.opponent.enabled=True",
            "card.opponent.profile.enabled=True",
            "card.opponent.profile.mode=heuristic",
            "card.opponent.critic.enabled=True",
            "card.opponent.aux.enabled=True",
            "card.opponent.aux.loss_weight=0.05",
        ],
    },
    "magpie_aux_only": {
        "alg": "oa_qmix",
        "group": "legacy_alias",
        "overrides": [
            "card.opponent.enabled=True",
            "card.opponent.profile.enabled=True",
            "card.opponent.profile.mode=heuristic",
            "card.opponent.critic.enabled=True",
            "card.opponent.aux.enabled=True",
            "card.opponent.aux.loss_weight=0.05",
        ],
    },
    "magpie_random_profile": {
        "alg": "oa_qmix",
        "group": "magpie_ablation",
        "overrides": [
            "card.opponent.enabled=True",
            "card.opponent.profile.enabled=True",
            "card.opponent.profile.mode=random",
            "card.opponent.critic.enabled=True",
        ],
    },
    "magpie_zero_profile": {
        "alg": "oa_qmix",
        "group": "magpie_ablation",
        "overrides": [
            "card.opponent.enabled=True",
            "card.opponent.profile.enabled=True",
            "card.opponent.profile.mode=zero",
            "card.opponent.critic.enabled=True",
        ],
    },
    "tpe_llm_reward_only": {
        "alg": "qmix",
        "group": "magpie_reward_core",
        "overrides": [
            "card.enabled=True",
            "card.reward_spec_path=${MAGPIE_BOOTSTRAP_REWARD_SPEC}",
            "card.skip_train_on_tpe_fail=False",
            "card.feedback_t_env_interval=${MAGPIE_FEEDBACK_T_ENV_INTERVAL}",
            "card.llm.env_file=${MAGPIE_LLM_ENV_FILE}",
            "card.generation.enabled=True",
            "card.generation.mode=iterative",
            "card.generation.auto_generate_after_feedback=True",
            "card.generation.activate_generated_spec=True",
            "card.generation.generate_on_final_feedback=False",
        ],
    },
    "llm_opponent_summary_reward": {
        "alg": "qmix",
        "group": "magpie_reward_core",
        "overrides": [
            "card.enabled=True",
            "card.reward_spec_path=${MAGPIE_BOOTSTRAP_REWARD_SPEC}",
            "card.skip_train_on_tpe_fail=False",
            "card.feedback_t_env_interval=${MAGPIE_FEEDBACK_T_ENV_INTERVAL}",
            "card.llm.env_file=${MAGPIE_LLM_ENV_FILE}",
            "card.generation.enabled=True",
            "card.generation.mode=iterative",
            "card.generation.auto_generate_after_feedback=True",
            "card.generation.activate_generated_spec=True",
            "card.generation.generate_on_final_feedback=False",
            "card.opponent.enabled=True",
            "card.opponent.feature_extraction.enabled=True",
            "card.opponent.profile.enabled=True",
            "card.opponent.profile.mode=llm",
            "card.opponent.profile.min_episodes=${MAGPIE_OPPONENT_PROFILE_MIN_EPISODES}",
            "card.opponent.profile.refresh_episodes=${MAGPIE_OPPONENT_PROFILE_REFRESH_EPISODES}",
            "card.opponent.reward_conditioning.enabled=True",
        ],
    },
    "tpe_llm_opponent_reward": {
        "alg": "qmix",
        "group": "magpie_reward_core",
        "overrides": [
            "card.enabled=True",
            "card.reward_spec_path=${MAGPIE_BOOTSTRAP_REWARD_SPEC}",
            "card.skip_train_on_tpe_fail=False",
            "card.feedback_t_env_interval=${MAGPIE_FEEDBACK_T_ENV_INTERVAL}",
            "card.llm.env_file=${MAGPIE_LLM_ENV_FILE}",
            "card.generation.enabled=True",
            "card.generation.mode=iterative",
            "card.generation.auto_generate_after_feedback=True",
            "card.generation.activate_generated_spec=True",
            "card.generation.generate_on_final_feedback=False",
            "card.opponent.enabled=True",
            "card.opponent.feature_extraction.enabled=True",
            "card.opponent.profile.enabled=True",
            "card.opponent.profile.mode=llm",
            "card.opponent.profile.min_episodes=${MAGPIE_OPPONENT_PROFILE_MIN_EPISODES}",
            "card.opponent.profile.refresh_episodes=${MAGPIE_OPPONENT_PROFILE_REFRESH_EPISODES}",
            "card.opponent.tpe.enabled=True",
            "card.opponent.reward_conditioning.enabled=True",
        ],
    },
}


STAGES = {
    "smoke": {
        "seeds": [1],
        "t_max": 50,
        "env_config": "gymma",
        "env_overrides": ["env_args.key=matrixgames:penalty-100-nostate-v0"],
        "run_overrides": [
            "batch_size=1",
            "buffer_size=1",
            "test_nepisode=1",
            "test_interval=50",
            "log_interval=50",
            "runner_log_interval=50",
            "learner_log_interval=50",
        ],
        "methods": ["qmix_baseline", "magpie_profile_only", "magpie_critic_only"],
    },
    "pilot_smac_3m": {
        "seeds": [1, 2, 3],
        "t_max": 10000,
        "env_config": "sc2",
        "env_overrides": ["env_args.map_name=3m"],
        "run_overrides": [],
        "methods": [
            "qmix_baseline",
            "magpie_profile_only",
            "magpie_critic_only",
            "magpie_critic_aux",
            "magpie_random_profile",
        ],
    },
    "pilot_smac_8m": {
        "seeds": [1, 2, 3],
        "t_max": 10000,
        "env_config": "sc2",
        "env_overrides": ["env_args.map_name=8m"],
        "run_overrides": [],
        "methods": [
            "qmix_baseline",
            "magpie_critic_only",
            "magpie_critic_aux",
            "magpie_random_profile",
        ],
    },
    "pilot_smac_2s3z": {
        "seeds": [1, 2, 3],
        "t_max": 10000,
        "env_config": "sc2",
        "env_overrides": ["env_args.map_name=2s3z"],
        "run_overrides": [],
        "methods": [
            "qmix_baseline",
            "magpie_critic_only",
            "magpie_critic_aux",
            "magpie_random_profile",
        ],
    },
    "pilot_smac_5m_vs_6m": {
        "seeds": [1, 2, 3],
        "t_max": 10000,
        "env_config": "sc2",
        "env_overrides": ["env_args.map_name=5m_vs_6m"],
        "run_overrides": [],
        "methods": [
            "qmix_baseline",
            "magpie_critic_only",
            "magpie_critic_aux",
            "magpie_random_profile",
        ],
    },
    "pilot_smac_3s5z": {
        "seeds": [1, 2, 3],
        "t_max": 10000,
        "env_config": "sc2",
        "env_overrides": ["env_args.map_name=3s5z"],
        "run_overrides": [],
        "methods": [
            "qmix_baseline",
            "magpie_critic_only",
            "magpie_critic_aux",
            "magpie_random_profile",
        ],
    },
    "standard_smac_8m": {
        "seeds": [1, 2, 3, 4, 5],
        "t_max": 2000000,
        "env_config": "sc2",
        "env_overrides": ["env_args.map_name=8m"],
        "run_overrides": [],
        "methods": [
            "qmix_baseline",
            "magpie_critic_only",
            "magpie_critic_aux",
        ],
    },
    "standard_smac_2s3z": {
        "seeds": [1, 2, 3, 4, 5],
        "t_max": 2000000,
        "env_config": "sc2",
        "env_overrides": ["env_args.map_name=2s3z"],
        "run_overrides": [],
        "methods": [
            "qmix_baseline",
            "magpie_critic_only",
            "magpie_critic_aux",
        ],
    },
    "hard_smac_5m_vs_6m": {
        "seeds": [1, 2, 3],
        "t_max": 5000000,
        "env_config": "sc2",
        "env_overrides": ["env_args.map_name=5m_vs_6m"],
        "run_overrides": [],
        "methods": [
            "qmix_baseline",
            "magpie_critic_only",
            "magpie_critic_aux",
        ],
    },
    "hard_smac_3s5z": {
        "seeds": [1, 2, 3],
        "t_max": 5000000,
        "env_config": "sc2",
        "env_overrides": ["env_args.map_name=3s5z"],
        "run_overrides": [],
        "methods": [
            "qmix_baseline",
            "magpie_critic_only",
            "magpie_critic_aux",
        ],
    },
    "standard_smacv2_terran_5_vs_5": {
        "seeds": [1, 2, 3],
        "t_max": 10000000,
        "env_config": "sc2v2",
        "env_overrides": ["env_args.map_name=terran_5_vs_5"],
        "run_overrides": [],
        "methods": [
            "qmix_baseline",
            "magpie_critic_only",
            "magpie_critic_aux",
        ],
    },
    "standard_smacv2_zerg_5_vs_5": {
        "seeds": [1, 2, 3],
        "t_max": 10000000,
        "env_config": "sc2v2",
        "env_overrides": ["env_args.map_name=zerg_5_vs_5"],
        "run_overrides": [],
        "methods": [
            "qmix_baseline",
            "magpie_critic_only",
            "magpie_critic_aux",
        ],
    },
    "standard_smacv2_protoss_5_vs_5": {
        "seeds": [1, 2, 3],
        "t_max": 10000000,
        "env_config": "sc2v2",
        "env_overrides": ["env_args.map_name=protoss_5_vs_5"],
        "run_overrides": [],
        "methods": [
            "qmix_baseline",
            "magpie_critic_only",
            "magpie_critic_aux",
        ],
    },
    "standard_smacv2_terran_10_vs_10": {
        "seeds": [1, 2, 3],
        "t_max": 10000000,
        "env_config": "sc2v2",
        "env_overrides": ["env_args.map_name=terran_10_vs_10"],
        "run_overrides": [],
        "methods": [
            "qmix_baseline",
            "magpie_critic_only",
            "magpie_critic_aux",
        ],
    },
    "pilot_smaclite": {
        "seeds": [1, 2, 3],
        "t_max": 10000,
        "env_config": "smaclite",
        "env_overrides": ["env_args.map_name=2s3z"],
        "run_overrides": [],
        "methods": ["qmix_baseline", "magpie_critic_only"],
    },
}


JOB_LOG_TAIL_BYTES = 64 * 1024
JOB_COMPLETION_PATTERNS = (
    ("queue_done", re.compile(r"^\[queue\] DONE\b", re.MULTILINE)),
    ("finished_training", re.compile(r"\bFinished Training\b")),
    ("completed_after", re.compile(r"\bCompleted after\b")),
    ("finished_after", re.compile(r"\bFinished after\b")),
)
JOB_FAILURE_PATTERNS = (
    re.compile(r"\bTraceback \(most recent call last\):"),
    re.compile(r"\bConnectError\b"),
    re.compile(r"\bConnection rejected\b"),
    re.compile(r"\bRuntimeError\b"),
    re.compile(r"\benv\.reset failed\b"),
    re.compile(r"^\[queue\] FAILED\b", re.MULTILINE),
)
TRANSIENT_SC2_JOB_FAILURE_PATTERNS = (
    re.compile(r"\bWebsocket timed out\b", re.IGNORECASE),
    re.compile(r"\bError during join_game\b", re.IGNORECASE),
    re.compile(r"\bConnection rejected\b", re.IGNORECASE),
    re.compile(r"\bConnectError\b", re.IGNORECASE),
    re.compile(r"\benv\.reset failed\b", re.IGNORECASE),
)

ENV_DEFAULTS = {
    "MAGPIE_BOOTSTRAP_REWARD_SPEC": "/pymarl/reward_specs/bootstrap_smac_team_battle_v1.json",
    "MAGPIE_LLM_ENV_FILE": "config/llm/.env.llm.deepseek",
    "MAGPIE_FEEDBACK_T_ENV_INTERVAL": "500000",
    "MAGPIE_FEEDBACK_ITERATIONS": "5",
    "MAGPIE_FEEDBACK_TRIGGER_RATIOS": "0.05,0.125,0.25,0.5,0.75",
    "MAGPIE_OPPONENT_PROFILE_MIN_EPISODES": "128",
    "MAGPIE_OPPONENT_PROFILE_REFRESH_EPISODES": "128",
}


def quote_cmd(parts):
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return " ".join(shlex.quote(part) for part in parts)


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def expand_override(value):
    text = str(value)

    def replace_braced(match):
        name = match.group(1)
        return os.environ.get(name, ENV_DEFAULTS.get(name, match.group(0)))

    def replace_plain(match):
        name = match.group(1)
        return os.environ.get(name, ENV_DEFAULTS.get(name, match.group(0)))

    text = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace_braced, text)
    return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", replace_plain, text)


def env_value(name):
    return os.environ.get(name, ENV_DEFAULTS.get(name, ""))


def feedback_t_env_points(t_max):
    explicit = os.environ.get("MAGPIE_FEEDBACK_T_ENV_POINTS", "").strip()
    if explicit:
        return explicit

    ratios = []
    for item in env_value("MAGPIE_FEEDBACK_TRIGGER_RATIOS").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ratio = float(item)
        except ValueError:
            continue
        if ratio > 0:
            ratios.append(ratio)

    try:
        iterations = int(env_value("MAGPIE_FEEDBACK_ITERATIONS") or len(ratios))
    except ValueError:
        iterations = len(ratios)
    if iterations > 0:
        ratios = ratios[:iterations]

    points = []
    for ratio in ratios:
        point = int(round(float(t_max) * ratio))
        if point > 0:
            points.append(min(point, int(t_max)))
    return ",".join(str(point) for point in sorted(set(points)))


def build_main_args(method_name, stage, seed, output_tag, extra_overrides=None):
    method = METHODS[method_name]
    parts = [
        "src/main.py",
        f"--config={method['alg']}",
        f"--env-config={stage['env_config']}",
        "with",
        f"seed={seed}",
        f"t_max={stage['t_max']}",
        "use_tensorboard=True",
        "use_wandb=True",
        "wandb_mode=offline",
        "wandb_team=local",
        "wandb_project=epymarl",
        f"name={method_name}_{output_tag}",
    ]
    if stage["env_config"] in ("sc2", "sc2v2"):
        parts.append(f"env_args.seed={seed}")
    parts.extend(expand_override(item) for item in stage["env_overrides"])
    parts.extend(expand_override(item) for item in stage.get("run_overrides", []))
    parts.extend(expand_override(item) for item in method["overrides"])
    if method.get("group") == "magpie_reward_core":
        points = feedback_t_env_points(stage["t_max"])
        if points:
            parts.append(f"card.feedback_t_env_points={points}")
    parts.extend(expand_override(item) for item in (extra_overrides or []))
    return parts


def run_local(main_args, dry_run, log_path=None):
    cmd = [sys.executable] + main_args
    print(quote_cmd(cmd))
    if not dry_run:
        run_subprocess(cmd, log_path)


def run_docker(main_args, dry_run, monitored, service, log_path=None):
    service = service or ("train-monitored" if monitored else "train")
    ensure_docker_user_env()
    profile = ["--profile", "monitor"] if monitored else []
    command = ["python3"] + main_args
    cmd = [
        "docker",
        "compose",
        *profile,
        "run",
        "--rm",
        service,
        *command,
    ]
    print(quote_cmd(cmd))
    if not dry_run:
        run_subprocess(cmd, log_path)


def run_subprocess(cmd, log_path=None):
    if log_path is None:
        subprocess.run(cmd, check=True)
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(quote_cmd(cmd) + "\n")
        handle.flush()
        subprocess.run(
            cmd,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def ensure_docker_user_env():
    if os.environ.get("EPYMARL_DOCKER_USER"):
        return
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None or getgid is None:
        return
    os.environ["EPYMARL_DOCKER_USER"] = f"{getuid()}:{getgid()}"


def build_jobs(stage_name, methods, stage, output_tag, extra_overrides):
    jobs = []
    for seed in stage["seeds"]:
        for method_name in methods:
            tagged_output = f"{stage_name}_{output_tag}"
            jobs.append(
                {
                    "stage": stage_name,
                    "stage_config": stage,
                    "seed": seed,
                    "method": method_name,
                    "main_args": build_main_args(
                        method_name, stage, seed, tagged_output, extra_overrides
                    ),
                }
            )
    return jobs


def parse_stage_methods(items):
    stage_methods = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"Invalid --stage-methods item {item!r}; expected STAGE=METHOD[,METHOD...]"
            )
        stage_name, raw_methods = item.split("=", 1)
        stage_name = stage_name.strip()
        if stage_name not in STAGES:
            raise ValueError(f"Unknown stage in --stage-methods: {stage_name}")
        methods = [method.strip() for method in raw_methods.split(",") if method.strip()]
        if not methods:
            raise ValueError(f"No methods provided for stage {stage_name}")
        unknown = [method for method in methods if method not in METHODS]
        if unknown:
            raise ValueError(
                f"Unknown method(s) for stage {stage_name}: {', '.join(unknown)}"
            )
        stage_methods[stage_name] = methods
    return stage_methods


def job_log_path(job_log_dir, job):
    if not job_log_dir:
        return None
    name = (
        f"{job['queue_index']:03d}_{safe_name(job['stage'])}_"
        f"{safe_name(job['method'])}_seed{job['seed']}.log"
    )
    return os.path.join(job_log_dir, name)


def read_log_tail(log_path, max_bytes=JOB_LOG_TAIL_BYTES):
    if not log_path or not os.path.isfile(log_path):
        return ""
    try:
        with open(log_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - max_bytes, 0))
            return handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""


def completed_job_marker(log_path):
    tail = read_log_tail(log_path)
    if not tail:
        return None
    if any(pattern.search(tail) for pattern in JOB_FAILURE_PATTERNS):
        return None
    for marker_name, pattern in JOB_COMPLETION_PATTERNS:
        if pattern.search(tail):
            return marker_name
    return None


def transient_job_failure_marker(log_path):
    tail = read_log_tail(log_path, max_bytes=16000)
    if not tail:
        return None
    for pattern in TRANSIENT_SC2_JOB_FAILURE_PATTERNS:
        match = pattern.search(tail)
        if match:
            return match.group(0)
    return None


def archive_failed_attempt_log(log_path, attempt):
    if not log_path or not os.path.exists(log_path):
        return ""
    archive_path = f"{log_path}.attempt{attempt}.failed.log"
    shutil.move(log_path, archive_path)
    return archive_path


def filter_completed_jobs(jobs, args):
    if not args.resume:
        return jobs

    pending = []
    skipped = []
    for job in jobs:
        log_path = job_log_path(args.job_log_dir, job)
        marker = completed_job_marker(log_path)
        if marker:
            skipped.append((job, log_path, marker))
            continue
        pending.append(job)

    for job, log_path, marker in skipped:
        print(
            f"[queue] SKIP DONE {job['queue_index']}: {job['stage']} {job['method']} "
            f"seed={job['seed']} via {marker} ({log_path})",
            flush=True,
        )

    if skipped:
        print(
            f"[queue] Resume filtered {len(skipped)} completed jobs; {len(pending)} remaining.",
            flush=True,
        )
    return pending


def run_job(index, job, args, log_path=None):
    label = f"{index}: {job['stage']} {job['method']} seed={job['seed']}"
    job_retries = int(getattr(args, "job_retries", 0) or 0)
    job_retry_delay = float(getattr(args, "job_retry_delay", 30.0) or 0.0)
    for attempt in range(job_retries + 1):
        attempt_label = label
        if attempt:
            attempt_label = f"{label} retry={attempt}/{job_retries}"
        print(f"[queue] START {attempt_label}", flush=True)
        try:
            if args.executor == "local":
                run_local(job["main_args"], args.dry_run, log_path=log_path)
            else:
                docker_service = args.docker_service
                if (
                    not docker_service
                    and job["stage_config"]["env_config"] in ("sc2", "sc2v2")
                ):
                    docker_service = "train"
                run_docker(
                    job["main_args"],
                    args.dry_run,
                    args.monitored,
                    docker_service,
                    log_path=log_path,
                )
            print(f"[queue] DONE {label}", flush=True)
            return
        except subprocess.CalledProcessError:
            marker = transient_job_failure_marker(log_path)
            if marker is None or attempt >= job_retries:
                raise
            archive_path = archive_failed_attempt_log(log_path, attempt + 1)
            archive_note = f" archived_log={archive_path}" if archive_path else ""
            print(
                f"[queue] RETRY transient SC2 failure for {label}: "
                f"{marker}; next_attempt={attempt + 2}/{job_retries + 1}"
                f"{archive_note}",
                flush=True,
            )
            if job_retry_delay > 0:
                time.sleep(job_retry_delay * (attempt + 1))


def run_jobs(jobs, args):
    jobs = filter_completed_jobs(jobs, args)
    if not jobs:
        print("[queue] No jobs left to run after resume filtering.", flush=True)
        return

    if args.max_concurrent <= 1 or args.dry_run:
        for job in jobs:
            run_job(
                job["queue_index"],
                job,
                args,
                log_path=job_log_path(args.job_log_dir, job),
            )
        return

    print(
        f"[queue] Running {len(jobs)} jobs with max_concurrent={args.max_concurrent}",
        flush=True,
    )
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
        future_to_job = {}
        for job in jobs:
            future = pool.submit(
                run_job,
                job["queue_index"],
                job,
                args,
                job_log_path(args.job_log_dir, job),
            )
            future_to_job[future] = job
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((job, exc))
                print(
                    f"[queue] FAILED {job['queue_index']}: {job['stage']} {job['method']} seed={job['seed']}: {exc}",
                    flush=True,
                )
    if failures:
        failed = ", ".join(
            f"{job['queue_index']}:{job['stage']}:{job['method']}:seed{job['seed']}"
            for job, _ in failures
        )
        raise RuntimeError(f"{len(failures)} queued jobs failed: {failed}")


def main():
    parser = argparse.ArgumentParser(description="Run MAGPIE experiment stages locally or through docker compose.")
    parser.add_argument("--executor", choices=["local", "docker"], default="docker")
    parser.add_argument("--stage", choices=sorted(STAGES), default="smoke")
    parser.add_argument(
        "--stages",
        nargs="*",
        choices=sorted(STAGES),
        default=None,
        help="Optional multi-stage queue. When provided, --stage is ignored.",
    )
    parser.add_argument("--methods", nargs="*", choices=sorted(METHODS), default=None)
    parser.add_argument(
        "--stage-methods",
        nargs="*",
        default=None,
        help=(
            "Optional per-stage method override, e.g. "
            "standard_smac_8m=magpie_llm_profile_critic_only "
            "hard_smac_5m_vs_6m=qmix_baseline,magpie_critic_only. "
            "Takes precedence over --methods for matching stages."
        ),
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--t-max", type=int, default=None)
    parser.add_argument(
        "--env-parallelism",
        type=int,
        default=1,
        help="Number of environments per training run. Values >1 use runner=parallel.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--monitored", action="store_true", help="Use train-monitored with TensorBoard/W&B local services.")
    parser.add_argument(
        "--docker-service",
        default="",
        help="Override docker compose service, e.g. train, train-sc2, or train-monitored. Default SC2 runs use train with mounted assets.",
    )
    parser.add_argument(
        "--extra-overrides",
        nargs="*",
        default=None,
        help="Additional Sacred overrides appended to every run, e.g. log_interval=10000 test_interval=20000.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Run up to this many method/seed jobs concurrently. Default preserves sequential behavior.",
    )
    parser.add_argument(
        "--job-log-dir",
        default="",
        help="Optional directory for one log file per queued job. Useful when --max-concurrent > 1.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs whose existing per-job log already contains a terminal completion marker.",
    )
    parser.add_argument(
        "--job-retries",
        type=int,
        default=int(os.environ.get("EPYMARL_JOB_RETRIES", "0")),
        help=(
            "Retry a failed job when its log matches a transient SC2 startup/connection "
            "failure such as join_game timeout. Failed attempt logs are archived."
        ),
    )
    parser.add_argument(
        "--job-retry-delay",
        type=float,
        default=float(os.environ.get("EPYMARL_JOB_RETRY_DELAY", "30")),
        help="Base seconds to wait before retrying a transient failed job.",
    )
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()

    if args.resume and not args.job_log_dir:
        parser.error("--resume requires --job-log-dir so completed jobs can be proven from per-job logs.")

    stage_names = args.stages or [args.stage]
    output_tag = args.output_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    extra_overrides = args.extra_overrides or []
    try:
        stage_methods = parse_stage_methods(args.stage_methods)
    except ValueError as exc:
        parser.error(str(exc))

    jobs = []
    for stage_name in stage_names:
        stage = dict(STAGES[stage_name])
        if args.seeds:
            stage["seeds"] = args.seeds
        if args.t_max is not None:
            stage["t_max"] = args.t_max
        if args.env_parallelism > 1:
            stage["run_overrides"] = list(stage.get("run_overrides", [])) + [
                "runner=parallel",
                f"batch_size_run={args.env_parallelism}",
                f"test_nepisode={max(args.env_parallelism, stage.get('test_nepisode', 100) if isinstance(stage.get('test_nepisode'), int) else 100)}",
            ]
        methods = stage_methods.get(stage_name) or args.methods or stage["methods"]
        jobs.extend(build_jobs(stage_name, methods, stage, output_tag, extra_overrides))

    for index, job in enumerate(jobs, start=1):
        job["queue_index"] = index

    print(
        f"executor={args.executor} stages={','.join(stage_names)} dry_run={args.dry_run}"
    )
    run_jobs(jobs, args)


if __name__ == "__main__":
    main()
