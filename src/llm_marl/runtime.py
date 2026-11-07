# Modified from EPyMARL for OA-QMIX and opponent-profile research.
import json
import os
from copy import deepcopy

from .capabilities import infer_algorithm_capability
from .executor import CardRewardError, RewardExecutor
from .feedback import FeedbackBuilder
from .generator import CardRewardGenerationError, CardRewardGenerator
from .llm_client import CardLLMError, OpenAICompatibleLLMClient, extract_json_payload
from .llm_config import load_llm_config
from .llm_module import CardLLMModule
from .orchestrator import CardOrchestrator
from .opponent_profile import (
    OpponentProfileGenerator,
    OpponentProfileStore,
    style_label_for,
)
from .reward_spec import RewardSpecStore
from .tpe import TPEEvaluator
from .trajectory import TrajectoryStore


DEFAULT_CARD_CONFIG = {
    "enabled": False,
    "iterations": 1,
    "reward_mode": "team",
    "reward_spec_path": "",
    "skip_train_on_tpe_fail": True,
    "feedback_interval": 1,
    "feedback_t_env_interval": 0,
    "feedback_t_env_points": "",
    "trajectory_dump_dir": "",
    "save_episodes": True,
    "allowed_context_keys": [],
    "llm": {"model": "", "env_file": ""},
    "generation": {
        "enabled": False,
        "mode": "",
        "auto_generate_after_feedback": False,
        "activate_generated_spec": False,
        "generate_on_final_feedback": True,
    },
    "tpe": {"threshold": 0.75},
    "opponent": {
        "enabled": False,
        "feature_extraction": {"enabled": False},
        "profile": {
            "enabled": False,
            "mode": "heuristic",
            "path": "",
            "vector_dim": 8,
            "refresh_episodes": 1,
            "min_episodes": 0,
            "record_calls": True,
        },
        "reward_conditioning": {"enabled": False},
        "tpe": {"enabled": False, "behavior_weight": 0.5},
        "critic": {"enabled": False, "inject_to": "mixer"},
        "aux": {"enabled": False, "loss_weight": 0.0},
        "ablation": {
            "profile_mode": "none",
            "disable_behavior_pe": False,
            "disable_hacking_check": False,
        },
        "cost_tracking": {"enabled": False},
    },
}


def merge_dict(base, override):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


class CardRuntime:
    def __init__(self, args, logger, env_info):
        self.args = args
        self.logger = logger
        self.env_info = env_info
        self.card_cfg = merge_dict(DEFAULT_CARD_CONFIG, getattr(args, "card", {}))
        self.reward_enabled = bool(self.card_cfg.get("enabled", False))
        self.opponent_enabled = bool(
            self.card_cfg.get("opponent", {}).get("enabled", False)
        )
        self.enabled = self.reward_enabled or self.opponent_enabled

        self.capability = infer_algorithm_capability(args)
        self.output_dir = None
        self.reward_spec = None
        self.executor = None
        self.trajectory_store = None
        self.feedback_builder = None
        self.tpe_evaluator = None
        self.orchestrator = None
        self.llm_config = None
        self.reward_generator = None
        self.llm_module = None
        self.opponent_profile_store = None
        self.opponent_profile_generator = None
        self.opponent_profile = None
        self._latest_tpe = None
        self._latest_env_descriptor = None
        self._latest_initial_context = None
        self._latest_generated_candidate = None
        self._last_feedback_t_env = 0
        self._completed_feedback_points = set()
        self._last_profile_refresh_episode_count = -1
        self._llm_profile_call_counter = 0
        self.llm_usage_totals = self._empty_llm_usage()
        self.llm_usage_by_purpose = {}

        if not self.enabled:
            return

        reward_spec_path = self.card_cfg.get("reward_spec_path")
        if self.reward_enabled and not reward_spec_path:
            raise CardRewardError(
                "LLM-MARL reward generation is enabled but card.reward_spec_path is empty."
            )
        if reward_spec_path:
            reward_spec_path = os.path.abspath(reward_spec_path)
            self.reward_spec = RewardSpecStore.load(reward_spec_path)
            self._validate_runtime_config()
            self.executor = RewardExecutor(self.reward_spec)
        self.feedback_builder = FeedbackBuilder()
        opponent_cfg = self.card_cfg.get("opponent", {})
        tpe_cfg = opponent_cfg.get("tpe", {})
        ablation_cfg = opponent_cfg.get("ablation", {})
        self.tpe_evaluator = TPEEvaluator(
            self.card_cfg["tpe"]["threshold"],
            behavior_weight=tpe_cfg.get("behavior_weight", 0.5),
            behavior_enabled=bool(tpe_cfg.get("enabled", False))
            and not bool(ablation_cfg.get("disable_behavior_pe", False)),
            hacking_check=not bool(ablation_cfg.get("disable_hacking_check", False)),
        )

        output_root = self.card_cfg.get("trajectory_dump_dir") or os.path.join(
            args.local_results_path,
            "card",
            args.unique_token,
        )
        output_root = os.path.abspath(output_root)
        self.output_dir = output_root
        self.trajectory_store = TrajectoryStore(
            output_root,
            save_episodes=bool(self.card_cfg.get("save_episodes", True)),
        )
        if self.reward_spec is not None:
            self.orchestrator = CardOrchestrator(output_root, self.reward_spec)
        self._maybe_init_opponent_profile()
        self._maybe_init_generation()

        self._record_run(
            {
                "event": "card_runtime_started",
                "algorithm": self.capability.name,
                "reward_mode": self.reward_spec.reward_mode
                if self.reward_spec is not None
                else "env",
                "reward_enabled": self.reward_enabled,
                "opponent_enabled": self.opponent_enabled,
                "unique_token": args.unique_token,
            }
        )
        self._prime_tpe_from_existing_trajectories()

    def _validate_runtime_config(self):
        if self.reward_spec.reward_mode == "team" and not self.args.common_reward:
            raise CardRewardError(
                "Team reward spec requires common_reward=True in the EPyMARL scheme."
            )
        if self.reward_spec.reward_mode == "individual":
            if self.args.common_reward:
                raise CardRewardError(
                    "Individual reward spec requires common_reward=False."
                )
            if not self.capability.supports_individual_reward:
                raise CardRewardError(
                    f"Algorithm '{self.capability.name}' does not support individual rewards."
                )

    def set_latest_tpe(self, tpe_result):
        self._latest_tpe = tpe_result

    def should_skip_training(self):
        if not self.enabled:
            return False
        if not self.card_cfg.get("skip_train_on_tpe_fail", True):
            return False
        if self._latest_tpe is None:
            return False
        return not self._latest_tpe.passed

    def start_episode(self, env_id, metadata, initial_context, test_mode):
        if not self.enabled:
            return
        if env_id == 0:
            self._latest_env_descriptor = dict(metadata or {})
            self._latest_initial_context = dict(initial_context or {})
        self.trajectory_store.start_episode(env_id, metadata, initial_context, test_mode)

    def compute_reward(
        self,
        env_id,
        step_index,
        actions,
        env_reward,
        pre_context,
        post_context,
        terminated,
        truncated,
        env_info,
    ):
        if not self.enabled:
            return env_reward

        if self.executor is None:
            self.trajectory_store.record_step(
                env_id=env_id,
                step_index=step_index,
                actions=actions,
                env_reward=env_reward,
                shaped_reward=env_reward,
                subrewards={},
                aux_stats={},
                prev_context=pre_context,
                curr_context=post_context,
                terminated=terminated,
                truncated=truncated,
                env_info=env_info,
            )
            return env_reward

        context = {
            "actions": actions,
            "agent_count": self.env_info["n_agents"],
            "base_reward": env_reward,
            "env_info": env_info,
            "episode_limit": self.env_info["episode_limit"],
            "step_index": step_index,
            "terminated": terminated,
            "truncated": truncated,
            "prev": pre_context,
            "curr": post_context,
        }
        if self.opponent_profile is not None:
            context["opponent_profile"] = self.opponent_profile.to_dict()
        result = self.executor.compute(context, expected_agents=self.env_info["n_agents"])
        reward_value = result["reward"]

        self.trajectory_store.record_step(
            env_id=env_id,
            step_index=step_index,
            actions=actions,
            env_reward=env_reward,
            shaped_reward=reward_value,
            subrewards=result["subrewards"],
            aux_stats=result["aux_stats"],
            prev_context=pre_context,
            curr_context=post_context,
            terminated=terminated,
            truncated=truncated,
            env_info=env_info,
        )
        return reward_value

    def finish_episode(self, env_id, env_info):
        if not self.enabled:
            return None
        outcome = self._infer_outcome(env_info)
        summary = self.trajectory_store.finish_episode(env_id, outcome, env_info)
        if (
            self._should_refresh_profile()
            and self.opponent_profile_generator is not None
        ):
            self.refresh_opponent_profile()
        return summary

    def _infer_outcome(self, env_info):
        if env_info.get("battle_won", False):
            return "win"
        if env_info.get("episode_limit", False):
            return "timeout"
        return "loss"

    def log_runner_stats(self, logger, prefix, t_env, test_mode):
        if not self.enabled:
            return
        summaries = self.trajectory_store.consume_log_summaries(test_mode=test_mode)
        if not summaries:
            return

        count = len(summaries)
        logger.log_stat(
            prefix + "card_win_rate",
            sum(1 for row in summaries if row["outcome"] == "win") / count,
            t_env,
        )
        logger.log_stat(
            prefix + "card_normalized_return_mean",
            sum(row["normalized_team_return"] for row in summaries) / count,
            t_env,
        )

        subreward_totals = {}
        for row in summaries:
            for key, value in row["subreward_totals"].items():
                subreward_totals[key] = subreward_totals.get(key, 0.0) + value
        for key, value in subreward_totals.items():
            logger.log_stat(prefix + f"card_{key}_mean", value / count, t_env)

    def maybe_build_feedback_artifacts(self, t_env):
        if not self.enabled:
            return None
        current_t_env = int(t_env or 0)
        if current_t_env <= 0:
            return None

        train_summaries = self.trajectory_store.completed_summaries(test_mode=False)
        if not train_summaries:
            return None

        feedback_points = self._configured_feedback_points()
        if feedback_points:
            pending_points = [
                point
                for point in feedback_points
                if point not in self._completed_feedback_points
                and current_t_env >= point
            ]
            if not pending_points:
                return None
            feedback_point = min(pending_points)
            self._completed_feedback_points.add(feedback_point)
            self._last_feedback_t_env = current_t_env
            return self.build_feedback_artifacts(
                event="feedback_point",
                t_env=current_t_env,
                feedback_point=feedback_point,
            )

        interval = int(self.card_cfg.get("feedback_t_env_interval", 0) or 0)
        if interval <= 0:
            return None
        if current_t_env - self._last_feedback_t_env < interval:
            return None
        self._last_feedback_t_env = current_t_env
        return self.build_feedback_artifacts(
            event="feedback_interval", t_env=current_t_env
        )

    def _configured_feedback_points(self):
        raw_points = self.card_cfg.get("feedback_t_env_points", "")
        if raw_points is None or raw_points == "":
            return []
        if isinstance(raw_points, str):
            values = [item.strip() for item in raw_points.split(",") if item.strip()]
        elif isinstance(raw_points, (list, tuple)):
            values = list(raw_points)
        else:
            values = [raw_points]
        points = []
        for value in values:
            try:
                point = int(float(value))
            except (TypeError, ValueError):
                continue
            if point > 0:
                points.append(point)
        return sorted(set(points))

    def build_feedback_artifacts(
        self, event="run_complete", t_env=None, feedback_point=None
    ):
        if not self.enabled:
            return None
        train_summaries = self.trajectory_store.completed_summaries(test_mode=False)
        if self._should_refresh_profile():
            self.refresh_opponent_profile()
        process_feedback = self.feedback_builder.build_process_feedback(train_summaries)
        trajectory_feedback = self.feedback_builder.build_trajectory_feedback(
            train_summaries
        )
        if train_summaries:
            tpe_result = self.tpe_evaluator.evaluate(
                train_summaries,
                opponent_profile=self.opponent_profile.to_dict()
                if self.opponent_profile is not None
                else None,
            )
            self.set_latest_tpe(tpe_result)
        else:
            tpe_result = self._latest_tpe or self.tpe_evaluator.evaluate([])
        preference_feedback = self.feedback_builder.build_preference_feedback(tpe_result)

        feedback_dir = os.path.join(self.output_dir, "feedback")
        os.makedirs(feedback_dir, exist_ok=True)
        payloads = {
            "process_feedback.txt": process_feedback["text"],
            "trajectory_feedback.txt": trajectory_feedback["text"],
            "preference_feedback.txt": preference_feedback["text"],
        }
        for name, text in payloads.items():
            with open(os.path.join(feedback_dir, name), "w", encoding="utf-8") as handle:
                handle.write(text)

        summary_path = os.path.join(self.output_dir, "card_summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "process_feedback": process_feedback["metrics"],
                    "trajectory_feedback": {
                        "best": trajectory_feedback["best"],
                        "worst": trajectory_feedback["worst"],
                    },
                    "tpe": tpe_result.to_dict(),
                    "magpie_pe": tpe_result.to_dict(),
                    "opponent_profile": self.opponent_profile.to_dict()
                    if self.opponent_profile is not None
                    else None,
                    "llm_usage": self._llm_usage_summary(),
                    "reward_spec": self.reward_spec.to_dict()
                    if self.reward_spec is not None
                    else None,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        self._record_run(
            {
                "event": event,
                "unique_token": self.args.unique_token,
                "t_env": t_env,
                "feedback_point": feedback_point,
                "tpe": tpe_result.to_dict(),
                "process_metrics": process_feedback["metrics"],
            }
        )
        generation_cfg = self.card_cfg.get("generation", {})
        should_generate = generation_cfg.get("auto_generate_after_feedback", False)
        if event == "run_complete" and not generation_cfg.get(
            "generate_on_final_feedback", True
        ):
            should_generate = False
        if should_generate:
            try:
                self.generate_reward_candidate(
                    mode=generation_cfg.get("mode") or None,
                    activate=generation_cfg.get("activate_generated_spec", False),
                )
            except CardRewardGenerationError as exc:
                self._record_run(
                    {
                        "event": "llm_candidate_generation_failed",
                        "unique_token": self.args.unique_token,
                        "mode": self.card_cfg["generation"].get("mode") or None,
                        "error": str(exc),
                        "reward_spec_version": self.reward_spec.version
                        if self.reward_spec is not None
                        else None,
                        "reward_spec_name": self.reward_spec.name
                        if self.reward_spec is not None
                        else None,
                    }
                )
                self.logger.console_logger.warning(
                    f"CARD iterative generation failed; keeping current reward spec. Error: {exc}"
                )
        return tpe_result

    def _maybe_init_opponent_profile(self):
        opponent_cfg = self.card_cfg.get("opponent", {})
        if not opponent_cfg.get("enabled", False):
            return
        profile_cfg = opponent_cfg.get("profile", {})
        if not profile_cfg.get("enabled", False):
            return
        mode = profile_cfg.get("mode", "heuristic")
        ablation_mode = opponent_cfg.get("ablation", {}).get("profile_mode", "none")
        if ablation_mode in ("rule", "random", "zero", "none"):
            mode = {"rule": "heuristic", "none": mode}.get(
                ablation_mode, ablation_mode
            )
        self.opponent_profile_store = OpponentProfileStore(self.output_dir)
        self.opponent_profile_generator = OpponentProfileGenerator(
            vector_dim=profile_cfg.get("vector_dim", 8),
            mode=mode,
            profile_path=profile_cfg.get("path", ""),
        )
        if mode != "llm" or self._should_refresh_profile():
            self.refresh_opponent_profile()

    def _should_refresh_profile(self):
        if self.opponent_profile_generator is None:
            return False
        profile_cfg = self.card_cfg.get("opponent", {}).get("profile", {})
        train_count = len(self.trajectory_store.completed_summaries(test_mode=False))
        min_episodes = int(profile_cfg.get("min_episodes", 0) or 0)
        if train_count < min_episodes:
            return False
        if self.opponent_profile is None:
            return True
        interval = max(1, int(profile_cfg.get("refresh_episodes", 1) or 1))
        return train_count - self._last_profile_refresh_episode_count >= interval

    def refresh_opponent_profile(self):
        summaries = self.trajectory_store.completed_summaries(test_mode=False)
        feature_summary = self.trajectory_store.aggregate_opponent_features(
            test_mode=False
        )
        source_episodes = [row.get("episode_path", "") for row in summaries[-10:]]
        if self.opponent_profile_generator.mode == "llm":
            self.opponent_profile = self._build_llm_opponent_profile(
                feature_summary=feature_summary,
                source_episodes=source_episodes,
            )
        else:
            self.opponent_profile = self.opponent_profile_generator.build(
                feature_summary=feature_summary,
                source_episodes=source_episodes,
            )
        self._last_profile_refresh_episode_count = len(summaries)
        if self.opponent_profile_store is not None:
            self.opponent_profile_store.save(self.opponent_profile)
        return self.opponent_profile

    def _build_llm_opponent_profile(self, feature_summary, source_episodes):
        heuristic_profile = OpponentProfileGenerator(
            vector_dim=self.card_cfg.get("opponent", {})
            .get("profile", {})
            .get("vector_dim", 8),
            mode="heuristic",
        ).build(feature_summary=feature_summary, source_episodes=source_episodes)
        system_prompt = ""
        user_prompt = ""
        llm_config = None
        result = None
        parsed_payload = None
        try:
            llm_config = self.llm_config or load_llm_config(
                self.card_cfg.get("llm", {}).get("env_file", "") or None
            )
            model_override = self.card_cfg.get("llm", {}).get("model", "").strip()
            if model_override:
                llm_config.model = model_override
            client = OpenAICompatibleLLMClient(llm_config)
            system_prompt = (
                "You summarize opponent strategy for multi-agent reinforcement learning. "
                "Return one valid JSON object only. Do not include chain-of-thought."
            )
            behavior_vector_schema = (
                "behavior_vector MUST contain exactly 8 numeric values in this order: "
                "[focus_fire, low_hp_chase, target_switching, kiting_distance, "
                "ally_damage_share, survival_pressure, grouping, retreat_bias]. "
                "Do not shorten, omit, group, rename, or reorder these values."
            )
            user_prompt = "\n\n".join(
                [
                    "Opponent feature summary:",
                    json.dumps(feature_summary or {}, indent=2, sort_keys=True),
                    "Heuristic profile draft:",
                    json.dumps(heuristic_profile.to_dict(), indent=2, sort_keys=True),
                    "Required JSON schema:",
                    json.dumps(
                        {
                            "opponent_style": "short style label",
                            "behavior_vector": [
                                "focus_fire_score",
                                "low_hp_chase_score",
                                "target_switching_score",
                                "kiting_distance_score",
                                "ally_damage_share_score",
                                "survival_pressure_score",
                                "grouping_score",
                                "retreat_bias_score",
                            ],
                            "counter_strategy": "short counter-strategy text",
                            "confidence": 0.5,
                            "profile_version": "magpie-profile-v1",
                        },
                        indent=2,
                    ),
                    behavior_vector_schema,
                    (
                        "opponent_style MUST be exactly one of: unknown, "
                        "aggressive_focus_fire, evasive_kiting, "
                        "scattered_target_switching, balanced. Do not invent "
                        "new style labels such as passive, coordinated, deathball, "
                        "or high_target_switching."
                    ),
                    (
                        "Replace every behavior_vector placeholder with a numeric score "
                        "from 0.0 to 1.0 inferred from the feature summary. Do not copy "
                        "placeholder strings. Do not use all zeros unless the feature "
                        "summary contains no behavioral evidence."
                    ),
                    "Task:",
                    "Infer the opponent style and counter-strategy. Return keys: "
                    "opponent_style, behavior_vector, counter_strategy, confidence, "
                    "profile_version. Return only the JSON object matching the schema.",
                ]
            )
            result = client.complete_json(system_prompt, user_prompt)
            self._record_llm_usage(
                "opponent_profile",
                result.token_usage,
                metadata={"model": llm_config.model, "provider": llm_config.provider},
                latency_seconds=result.latency_seconds,
            )
            parsed_payload = extract_json_payload(result.content)
            parsed_payload.setdefault("feature_summary", feature_summary or {})
            parsed_payload.setdefault("source_episodes", source_episodes or [])
            profile = self.opponent_profile_generator._profile_from_payload(
                parsed_payload, feature_summary, source_episodes
            )
            profile.generation_mode = "llm"
            profile.llm_call_count = int(profile.llm_call_count or 0) + 1
            profile.token_estimate = int(result.token_usage.get("total_tokens", 0) or 0)
            self._record_llm_profile_call(
                status="ok",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response=result.content,
                raw_json=result.raw_json,
                parsed_response=parsed_payload,
                feature_summary=feature_summary,
                heuristic_profile=heuristic_profile.to_dict(),
                source_episodes=source_episodes,
                token_usage=result.token_usage,
                latency_seconds=result.latency_seconds,
                provider=getattr(llm_config, "provider", ""),
                model=getattr(llm_config, "model", ""),
            )
            return profile
        except (CardLLMError, ValueError, KeyError) as exc:
            self._record_llm_profile_call(
                status="error",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response=result.content if result is not None else "",
                raw_json=result.raw_json if result is not None else {},
                parsed_response=parsed_payload,
                feature_summary=feature_summary,
                heuristic_profile=heuristic_profile.to_dict(),
                source_episodes=source_episodes,
                token_usage=result.token_usage if result is not None else {},
                latency_seconds=result.latency_seconds if result is not None else None,
                provider=getattr(llm_config, "provider", "") if llm_config else "",
                model=getattr(llm_config, "model", "") if llm_config else "",
                error=str(exc),
            )
            heuristic_profile.generation_mode = "llm_fallback_heuristic"
            heuristic_profile.failure_reason = str(exc)
            self._record_run(
                {
                    "event": "llm_opponent_profile_failed",
                    "unique_token": self.args.unique_token,
                    "error": str(exc),
                }
            )
            return heuristic_profile

    def _record_llm_profile_call(
        self,
        *,
        status,
        system_prompt,
        user_prompt,
        raw_response,
        raw_json,
        parsed_response,
        feature_summary,
        heuristic_profile,
        source_episodes,
        token_usage,
        latency_seconds,
        provider,
        model,
        error="",
    ):
        profile_cfg = self.card_cfg.get("opponent", {}).get("profile", {})
        if not profile_cfg.get("record_calls", True):
            return None
        if not self.output_dir:
            return None

        self._llm_profile_call_counter = int(
            getattr(self, "_llm_profile_call_counter", 0) or 0
        ) + 1
        call_id = f"profile_call_{self._llm_profile_call_counter:06d}"
        artifacts_dir = os.path.join(self.output_dir, "llm_profile_calls", call_id)
        os.makedirs(artifacts_dir, exist_ok=True)

        text_payloads = {
            "system_prompt.txt": system_prompt or "",
            "user_prompt.txt": user_prompt or "",
            "raw_response.txt": raw_response or "",
        }
        for filename, content in text_payloads.items():
            with open(os.path.join(artifacts_dir, filename), "w", encoding="utf-8") as handle:
                handle.write(content)

        json_payloads = {
            "raw_response.json": raw_json or {},
            "parsed_response.json": parsed_response or {},
            "feature_summary.json": feature_summary or {},
            "heuristic_profile.json": heuristic_profile or {},
            "source_episodes.json": source_episodes or [],
            "token_usage.json": token_usage or {},
            "latency.json": {"latency_seconds": float(latency_seconds or 0.0)},
        }
        for filename, payload in json_payloads.items():
            with open(os.path.join(artifacts_dir, filename), "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
                handle.write("\n")

        record = {
            "call_id": call_id,
            "status": status,
            "unique_token": self.args.unique_token,
            "provider": provider,
            "model": model,
            "artifacts_dir": artifacts_dir,
            "source_episodes": source_episodes or [],
            "feature_summary": feature_summary or {},
            "heuristic_profile": heuristic_profile or {},
            "parsed_response": parsed_response,
            "token_usage": token_usage or {},
            "latency_seconds": float(latency_seconds or 0.0),
            "error": error or "",
        }
        index_path = os.path.join(self.output_dir, "llm_profile_calls.jsonl")
        with open(index_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def current_opponent_profile_vector(self):
        vector_dim = (
            self.card_cfg.get("opponent", {})
            .get("profile", {})
            .get("vector_dim", 8)
        )
        if self.opponent_profile is None:
            return [0.0] * int(vector_dim or 8)
        return list(self.opponent_profile.behavior_vector)

    def current_opponent_aux_labels(self):
        if self.opponent_profile is None:
            return {
                "opp_style_label": (0,),
                "enemy_focus_target": (0,),
                "danger_window": (0.0,),
                "opp_label_mask": [0.0, 0.0, 0.0],
            }
        return {
            "opp_style_label": (style_label_for(self.opponent_profile.opponent_style),),
            "enemy_focus_target": (0,),
            "danger_window": (0.0,),
            "opp_label_mask": [1.0, 0.0, 0.0],
        }

    def _maybe_init_generation(self):
        generation_cfg = self.card_cfg.get("generation", {})
        if not generation_cfg.get("enabled", False):
            return
        if self.reward_spec is None:
            raise CardRewardGenerationError(
                "LLM reward generation requires card.enabled=True and a reward spec."
            )

        env_file = self.card_cfg.get("llm", {}).get("env_file", "")
        model_override = self.card_cfg.get("llm", {}).get("model", "").strip()
        preview_cfg = load_llm_config(env_file or None)
        if model_override:
            preview_cfg.model = model_override
        self.llm_config = preview_cfg
        generated_root = self.llm_config.generated_spec_dir
        if not os.path.isabs(generated_root):
            generated_root = os.path.abspath(os.path.join(self.output_dir, generated_root))
        self.llm_module = CardLLMModule.from_env_file(
            env_file=env_file or None,
            output_dir=generated_root,
            model_override=model_override,
        )
        self.reward_generator = self.llm_module.generator

    def generate_reward_candidate(self, mode=None, activate=False):
        if not self.enabled:
            raise CardRewardGenerationError("CARD runtime is disabled.")
        if self.reward_spec is None:
            raise CardRewardGenerationError(
                "Reward generation requires an active reward spec."
            )
        if self.reward_generator is None:
            raise CardRewardGenerationError(
                "LLM generation is not configured. Enable card.generation.enabled and provide an env file."
            )

        generation_mode = mode or self.card_cfg.get("generation", {}).get("mode") or self.llm_config.generation_mode
        feedback_bundle = self._collect_feedback_bundle() if generation_mode == "iterative" else {}
        version = self.reward_spec.version + 1
        generation_result = self.llm_module.generate_reward_spec(
            base_spec=self.reward_spec,
            env_descriptor=self._build_env_descriptor_for_generation(),
            env_info=self.env_info,
            mode=generation_mode,
            version=version,
            feedback_bundle=feedback_bundle,
        )
        spec = generation_result.spec
        artifacts_dir = generation_result.artifacts_dir
        parsed_payload = generation_result.parsed_payload
        token_usage = generation_result.token_usage
        self._record_llm_usage(
            "reward_generation",
            token_usage,
            metadata={
                "mode": generation_mode,
                "model": self.llm_config.model,
                "provider": self.llm_config.provider,
            },
            latency_seconds=generation_result.latency_seconds,
        )
        spec_root = os.path.join(self.output_dir, "spec_history")
        saved_path = RewardSpecStore.save_version(spec, spec_root)
        self._record_run(
            {
                "event": "llm_candidate_generated",
                "mode": generation_mode,
                "artifacts_dir": artifacts_dir,
                "spec_path": saved_path,
                "spec_id": spec.spec_id,
                "version": spec.version,
                "name": spec.name,
                "metadata": spec.metadata,
                "token_usage": token_usage,
                "parsed_payload_keys": sorted(parsed_payload.keys()),
            }
        )
        activation_info = None
        if activate:
            self.reward_spec = spec
            self.executor = RewardExecutor(spec)
            activation_info = self.orchestrator.activate_spec(spec)
        self._latest_generated_candidate = {
            "spec": spec,
            "spec_path": saved_path,
            "artifacts_dir": artifacts_dir,
            "activated": bool(activate),
            "activation_info": activation_info,
            "token_usage": token_usage,
        }
        return {
            "spec": spec,
            "spec_path": saved_path,
            "artifacts_dir": artifacts_dir,
            "activated": bool(activate),
            "activation_info": activation_info,
            "token_usage": token_usage,
        }

    @staticmethod
    def _empty_llm_usage():
        return {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_calls": 0,
            "latency_seconds": 0.0,
            "latency_observations": 0,
            "mean_latency_seconds": 0.0,
        }

    def _record_llm_usage(self, purpose, token_usage, metadata=None, latency_seconds=None):
        usage = self._normalize_llm_usage(token_usage)
        if latency_seconds is not None:
            try:
                usage["latency_seconds"] = float(latency_seconds)
            except (TypeError, ValueError):
                usage["latency_seconds"] = 0.0
        usage["purpose"] = purpose
        if metadata:
            usage["metadata"] = dict(metadata)

        self._accumulate_llm_usage(self.llm_usage_totals, usage)
        purpose_totals = self.llm_usage_by_purpose.setdefault(
            purpose, self._empty_llm_usage()
        )
        self._accumulate_llm_usage(purpose_totals, usage)
        self._write_llm_usage_summary()
        self._record_run(
            {
                "event": "llm_token_usage",
                "unique_token": self.args.unique_token,
                "usage": usage,
                "totals": self._llm_usage_summary(),
            }
        )
        self._log_llm_usage_stats()
        return usage

    def _normalize_llm_usage(self, token_usage):
        token_usage = token_usage or {}
        return {
            "calls": 1,
            "prompt_tokens": int(token_usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(token_usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(token_usage.get("total_tokens", 0) or 0),
            "estimated": bool(token_usage.get("estimated", False)),
            "latency_seconds": float(token_usage.get("latency_seconds", 0.0) or 0.0),
        }

    @staticmethod
    def _accumulate_llm_usage(target, usage):
        target["calls"] += int(usage.get("calls", 1) or 1)
        target["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        target["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        target["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        if usage.get("estimated", False):
            target["estimated_calls"] += 1
        latency = float(usage.get("latency_seconds", 0.0) or 0.0)
        if latency > 0.0:
            target["latency_seconds"] += latency
            target["latency_observations"] += 1
            target["mean_latency_seconds"] = (
                target["latency_seconds"] / target["latency_observations"]
            )

    def _llm_usage_summary(self):
        return {
            "total": dict(self.llm_usage_totals),
            "by_purpose": {
                purpose: dict(values)
                for purpose, values in sorted(self.llm_usage_by_purpose.items())
            },
        }

    def _write_llm_usage_summary(self):
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "llm_usage.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._llm_usage_summary(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _log_llm_usage_stats(self):
        if self.logger is None:
            return
        t_env = int(getattr(self, "_last_feedback_t_env", 0) or 0)
        totals = self.llm_usage_totals
        for key in (
            "calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "latency_seconds",
            "mean_latency_seconds",
        ):
            self.logger.log_stat(f"llm_{key}", totals[key], t_env)

    def current_reward_spec_path(self):
        if not self.enabled or self.orchestrator is None:
            return None
        return self.orchestrator.current_spec_path

    def latest_generated_candidate(self):
        return self._latest_generated_candidate

    def _build_env_descriptor_for_generation(self):
        descriptor = dict(self._latest_env_descriptor or {})
        if "episode_limit" not in descriptor:
            descriptor["episode_limit"] = self.env_info.get("episode_limit")
        if "n_agents" not in descriptor:
            descriptor["n_agents"] = self.env_info.get("n_agents")
        if "n_actions" not in descriptor:
            descriptor["n_actions"] = self.env_info.get("n_actions")
        if self._latest_initial_context:
            descriptor["initial_context"] = dict(self._latest_initial_context)
        return descriptor

    def _collect_feedback_bundle(self):
        feedback_dir = os.path.join(self.output_dir, "feedback")
        bundle = {}
        for key, filename in (
            ("process_feedback", "process_feedback.txt"),
            ("trajectory_feedback", "trajectory_feedback.txt"),
            ("preference_feedback", "preference_feedback.txt"),
        ):
            path = os.path.join(feedback_dir, filename)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as handle:
                    bundle[key] = handle.read()
        if (
            self.card_cfg.get("opponent", {})
            .get("reward_conditioning", {})
            .get("enabled", False)
            and self.opponent_profile is not None
        ):
            bundle["opponent_profile"] = json.dumps(
                self.opponent_profile.to_dict(), indent=2, sort_keys=True
            )
        return bundle

    def _prime_tpe_from_existing_trajectories(self):
        episode_dir = os.path.join(self.output_dir, "episodes", "train")
        if not os.path.isdir(episode_dir) or self.executor is None:
            return

        offline_summaries = []
        for filename in sorted(os.listdir(episode_dir)):
            if not filename.endswith(".json"):
                continue
            episode_path = os.path.join(episode_dir, filename)
            with open(episode_path, "r", encoding="utf-8") as handle:
                episode = json.load(handle)
            offline_summaries.append(self._recompute_episode_summary(episode))

        if not offline_summaries:
            return

        tpe_result = self.tpe_evaluator.evaluate(
            offline_summaries,
            opponent_profile=self.opponent_profile.to_dict()
            if self.opponent_profile is not None
            else None,
        )
        self.set_latest_tpe(tpe_result)

        feedback_dir = os.path.join(self.output_dir, "feedback")
        os.makedirs(feedback_dir, exist_ok=True)
        preference_feedback = self.feedback_builder.build_preference_feedback(tpe_result)
        with open(
            os.path.join(feedback_dir, "offline_preference_feedback.txt"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(preference_feedback["text"])

        self._record_run(
            {
                "event": "offline_tpe",
                "unique_token": self.args.unique_token,
                "tpe": tpe_result.to_dict(),
                "episodes_evaluated": len(offline_summaries),
            }
        )

    def _recompute_episode_summary(self, episode):
        steps = episode.get("steps", [])
        team_return = 0.0
        subreward_totals = {key: 0.0 for key in self.reward_spec.subreward_keys}
        last_curr_context = {}

        for step in steps:
            result = self.executor.compute(
                {
                    "actions": step.get("actions", []),
                    "agent_count": self.env_info["n_agents"],
                    "base_reward": step.get("env_reward", 0.0),
                    "env_info": step.get("env_info", {}),
                    "episode_limit": self.env_info["episode_limit"],
                    "step_index": step.get("step_index", 0),
                    "terminated": step.get("terminated", False),
                    "truncated": step.get("truncated", False),
                    "prev": step.get("prev_context", {}),
                    "curr": step.get("curr_context", {}),
                },
                expected_agents=self.env_info["n_agents"],
            )
            shaped_reward = result["reward"]
            if isinstance(shaped_reward, list):
                team_return += float(sum(shaped_reward))
            else:
                team_return += float(shaped_reward)
            for key, value in result["subrewards"].items():
                subreward_totals[key] = subreward_totals.get(key, 0.0) + float(value)
            last_curr_context = step.get("curr_context", {})

        summary = dict(episode.get("summary", {}))
        episode_length = len(steps)
        summary["team_return"] = team_return
        summary["normalized_team_return"] = team_return / max(1, episode_length)
        summary["subreward_totals"] = subreward_totals
        summary["combat_summary"] = {
            "ally_alive": last_curr_context.get("ally_alive"),
            "enemy_alive": last_curr_context.get("enemy_alive"),
            "ally_health_total": last_curr_context.get("ally_health_total"),
            "enemy_health_total": last_curr_context.get("enemy_health_total"),
            "battle_won": bool(summary.get("env_info", {}).get("battle_won", False)),
        }
        summary["opponent_features"] = episode.get("opponent_features") or episode.get(
            "summary", {}
        ).get("opponent_features", {})
        summary["episode_path"] = episode.get("summary", {}).get("episode_path", "")
        return summary

    def _record_run(self, payload):
        if self.orchestrator is not None:
            self.orchestrator.record_run(payload)
            return
        os.makedirs(self.output_dir, exist_ok=True)
        history_path = os.path.join(self.output_dir, "card_history.jsonl")
        with open(history_path, "a", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")


def build_card_runtime(args, logger, env_info):
    card_cfg = merge_dict(DEFAULT_CARD_CONFIG, getattr(args, "card", {}))
    if not card_cfg.get("enabled", False) and not card_cfg.get("opponent", {}).get(
        "enabled", False
    ):
        return None
    return CardRuntime(args, logger, env_info)
