# Modified from EPyMARL for OA-QMIX and opponent-profile research.
from __future__ import annotations

import json
import os
from datetime import datetime

from .executor import RewardExecutor
from .llm_client import CardLLMError, OpenAICompatibleLLMClient, extract_json_payload
from .prompt_builder import RewardPromptBuilder
from .reward_spec import RewardSpec, RewardSpecStore


class CardRewardGenerationError(RuntimeError):
    pass


class CardRewardGenerator:
    def __init__(self, llm_config, output_dir):
        self.llm_config = llm_config
        self.output_dir = os.path.abspath(output_dir)
        self.prompt_builder = RewardPromptBuilder(llm_config)
        if not llm_config.is_openai_compatible:
            raise CardRewardGenerationError(
                f"Unsupported LLM provider '{llm_config.provider}'."
            )
        self.client = OpenAICompatibleLLMClient(llm_config)

    def generate(
        self,
        base_spec,
        env_descriptor,
        env_info,
        mode,
        version,
        feedback_bundle=None,
    ):
        system_prompt, user_prompt = self.prompt_builder.build(
            mode=mode,
            reward_spec=base_spec,
            env_descriptor=env_descriptor,
            env_info=env_info,
            feedback_bundle=feedback_bundle or {},
        )
        payload_max_retries = max(
            0, int(getattr(self.llm_config, "payload_max_retries", 0) or 0)
        )
        payload_errors = []
        result = None
        payload = None
        spec = None
        final_user_prompt = user_prompt
        payload_attempt = 0
        for payload_attempt in range(payload_max_retries + 1):
            final_user_prompt = self._retry_user_prompt(user_prompt, payload_errors)
            try:
                result = self.client.complete_json(system_prompt, final_user_prompt)
            except Exception as exc:  # noqa: BLE001
                raise CardRewardGenerationError(
                    f"LLM completion failed: {exc}"
                ) from exc
            try:
                payload = extract_json_payload(result.content)
                spec = self._build_spec(base_spec, payload, version=version)
                spec.metadata["llm_token_usage"] = dict(result.token_usage)
                spec.metadata["llm_latency_seconds"] = float(
                    getattr(result, "latency_seconds", 0.0) or 0.0
                )
                spec.metadata["payload_retry_attempts"] = payload_attempt
                self._validate_spec(spec, env_info)
                break
            except CardRewardGenerationError as exc:
                payload_errors.append(f"attempt {payload_attempt + 1}: {exc}")
            except Exception as exc:  # noqa: BLE001
                payload_errors.append(f"attempt {payload_attempt + 1}: {exc}")
            if payload_attempt >= payload_max_retries:
                raise CardRewardGenerationError(
                    "LLM reward payload was not a usable reward spec after "
                    f"{payload_attempt + 1} attempts: {' | '.join(payload_errors)}"
                )
        artifacts_dir = self._write_generation_artifacts(
            system_prompt=system_prompt,
            user_prompt=final_user_prompt,
            raw_response=result.content,
            parsed_payload=payload,
            spec=spec,
            token_usage=result.token_usage,
            latency_seconds=float(getattr(result, "latency_seconds", 0.0) or 0.0),
        )
        token_usage = dict(result.token_usage)
        token_usage["latency_seconds"] = float(
            getattr(result, "latency_seconds", 0.0) or 0.0
        )
        return spec, artifacts_dir, payload, token_usage

    @staticmethod
    def _retry_user_prompt(user_prompt, payload_errors):
        if not payload_errors:
            return user_prompt
        return "\n\n".join(
            [
                user_prompt,
                "Retry instruction:",
                "The previous response could not be parsed or validated as a reward spec. "
                "Regenerate the complete reward spec as one valid JSON object only, with no markdown fences, no trailing prose, and no truncated strings.",
                "Previous validation errors:",
                "\n".join(payload_errors[-3:]),
            ]
        )

    def _build_spec(self, base_spec, payload, version):
        strategy = payload.get("strategy", {})
        if "type" not in strategy:
            strategy["type"] = "python_function"
        if "entrypoint" not in strategy:
            strategy["entrypoint"] = "compute_reward"
        if "code" not in strategy and payload.get("code"):
            strategy["code"] = payload["code"]

        spec_payload = {
            "format_version": int(payload.get("format_version", base_spec.format_version)),
            "spec_id": str(
                payload.get("spec_id", self.llm_config.output_spec_id_prefix or base_spec.spec_id)
            ),
            "version": int(version),
            "name": str(
                payload.get("name", f"{self.llm_config.output_name_prefix}_v{version}")
            ),
            "reward_mode": payload.get("reward_mode", base_spec.reward_mode),
            "description": payload.get("description", base_spec.description),
            "subreward_keys": list(
                payload.get("subreward_keys", list(base_spec.subreward_keys))
            ),
            "allowed_context_keys": list(
                payload.get("allowed_context_keys", list(base_spec.allowed_context_keys))
            ),
            "strategy": strategy,
            "metadata": {
                **dict(base_spec.metadata),
                **dict(payload.get("metadata", {})),
                "generator_provider": self.llm_config.provider,
                "generator_model": self.llm_config.model,
                "generation_mode": self.llm_config.generation_mode,
            },
        }
        return RewardSpec.from_dict(spec_payload)

    def _validate_spec(self, spec, env_info):
        executor = RewardExecutor(spec)
        dummy_context = {
            "actions": [0] * env_info["n_agents"],
            "agent_count": env_info["n_agents"],
            "base_reward": 0.0,
            "env_info": {"battle_won": False},
            "episode_limit": env_info["episode_limit"],
            "step_index": 0,
            "terminated": False,
            "truncated": False,
            "prev": {
                "ally_alive": float(env_info["n_agents"]),
                "enemy_alive": float(env_info["n_agents"]),
                "ally_health_total": 100.0,
                "enemy_health_total": 100.0,
            },
            "curr": {
                "ally_alive": float(env_info["n_agents"]),
                "enemy_alive": float(env_info["n_agents"]),
                "ally_health_total": 100.0,
                "enemy_health_total": 100.0,
            },
            "opponent_profile": {
                "opponent_style": "unknown",
                "behavior_vector": [0.0] * 8,
                "counter_strategy": "use outcome-return baseline until behavior evidence is sufficient",
                "confidence": 0.0,
            },
        }
        try:
            executor.compute(dummy_context, expected_agents=env_info["n_agents"])
        except Exception as exc:  # noqa: BLE001
            raise CardRewardGenerationError(
                f"Generated reward spec failed executor validation: {exc}"
            ) from exc

    def _write_generation_artifacts(
        self,
        system_prompt,
        user_prompt,
        raw_response,
        parsed_payload,
        spec,
        token_usage,
        latency_seconds=0.0,
    ):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_dir = os.path.join(self.output_dir, "llm_generations", stamp)
        os.makedirs(artifacts_dir, exist_ok=True)

        if self.llm_config.save_raw_prompt:
            with open(os.path.join(artifacts_dir, "system_prompt.txt"), "w", encoding="utf-8") as handle:
                handle.write(system_prompt)
            with open(os.path.join(artifacts_dir, "user_prompt.txt"), "w", encoding="utf-8") as handle:
                handle.write(user_prompt)

        if self.llm_config.save_raw_response:
            with open(os.path.join(artifacts_dir, "raw_response.txt"), "w", encoding="utf-8") as handle:
                handle.write(raw_response)

        if self.llm_config.save_parsed_spec:
            with open(os.path.join(artifacts_dir, "parsed_payload.json"), "w", encoding="utf-8") as handle:
                json.dump(parsed_payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
            RewardSpecStore.save(spec, os.path.join(artifacts_dir, "reward_spec.json"))

        with open(os.path.join(artifacts_dir, "token_usage.json"), "w", encoding="utf-8") as handle:
            json.dump(token_usage, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with open(os.path.join(artifacts_dir, "latency.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {"latency_seconds": float(latency_seconds or 0.0)},
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        return artifacts_dir
