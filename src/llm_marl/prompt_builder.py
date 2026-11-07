# Modified from EPyMARL for OA-QMIX and opponent-profile research.
from __future__ import annotations

import json


class RewardPromptBuilder:
    def __init__(self, llm_config):
        self.llm_config = llm_config

    def build(self, mode, reward_spec, env_descriptor, env_info, feedback_bundle=None):
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            mode=mode,
            reward_spec=reward_spec,
            env_descriptor=env_descriptor,
            env_info=env_info,
            feedback_bundle=feedback_bundle or {},
        )
        return system_prompt, user_prompt

    def _build_system_prompt(self):
        expected_keys = ", ".join(self.llm_config.expected_keys)
        return (
            "You are generating a reward specification for multi-agent reinforcement learning. "
            "Return a single valid JSON object only. "
            "Do not include markdown fences. "
            "The JSON must define these top-level keys: "
            f"{expected_keys}. "
            "The strategy must contain Python code for a function named compute_reward(context). "
            "The function must return team_reward, subreward_dict, and aux_stats."
        )

    def _build_user_prompt(
        self,
        mode,
        reward_spec,
        env_descriptor,
        env_info,
        feedback_bundle,
    ):
        sections = [
            f"Generation mode: {mode}",
            "Current reward spec:",
            json.dumps(reward_spec.to_dict(), ensure_ascii=False, indent=2),
        ]

        if self.llm_config.include_env_descriptor:
            sections.extend(
                [
                    "Environment descriptor:",
                    json.dumps(env_descriptor or {}, ensure_ascii=False, indent=2),
                    "Environment info:",
                    json.dumps(env_info or {}, ensure_ascii=False, indent=2),
                ]
            )

        if self.llm_config.include_reward_schema_rules:
            sections.extend(
                [
                    "Reward schema rules:",
                    "- reward_mode should remain 'team' unless there is a strong reason to change it.",
                    "- allowed_context_keys should be preserved unless an allowed key must be removed.",
                    "- subreward_keys should align with the keys actually returned by subreward_dict.",
                    "- strategy.type must be 'python_function'.",
                    "- strategy.entrypoint must be 'compute_reward'.",
                    "- Avoid private attribute access and unsupported Python features.",
                ]
            )

        if self.llm_config.include_previous_reward_code:
            sections.extend(
                [
                    "Current reward code:",
                    reward_spec.strategy.get("code", ""),
                ]
            )

        if feedback_bundle:
            if self.llm_config.include_process_feedback and feedback_bundle.get(
                "process_feedback"
            ):
                sections.extend(
                    [
                        "Process feedback:",
                        feedback_bundle["process_feedback"],
                    ]
                )
            if self.llm_config.include_trajectory_feedback and feedback_bundle.get(
                "trajectory_feedback"
            ):
                sections.extend(
                    [
                        "Trajectory feedback:",
                        feedback_bundle["trajectory_feedback"],
                    ]
                )
            if self.llm_config.include_preference_feedback and feedback_bundle.get(
                "preference_feedback"
            ):
                sections.extend(
                    [
                        "Preference feedback:",
                        feedback_bundle["preference_feedback"],
                    ]
                )
            if feedback_bundle.get("opponent_profile"):
                sections.extend(
                    [
                        "MAGPIE opponent profile:",
                        feedback_bundle["opponent_profile"],
                        "Reward conditioning instruction:",
                        "Ground reward changes in the opponent profile. Avoid shaped reward that encourages stalling, isolated low-HP deaths, or behavior unrelated to test return.",
                    ]
                )

        sections.extend(
            [
                "Task:",
                "Generate one improved reward specification. Stay close to the current reward unless the feedback strongly suggests a change. "
                "Preserve compatibility with the existing executor contract.",
            ]
        )
        return "\n\n".join(sections)
