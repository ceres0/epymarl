# Modified from EPyMARL for OA-QMIX and opponent-profile research.
from __future__ import annotations

from dataclasses import dataclass

from .generator import CardRewardGenerationError, CardRewardGenerator
from .llm_client import CardLLMError
from .llm_config import CardLLMConfig, load_llm_config


@dataclass
class CardLLMGenerationResult:
    spec: object
    artifacts_dir: str
    parsed_payload: dict
    token_usage: dict
    latency_seconds: float = 0.0


class CardLLMModule:
    def __init__(self, config: CardLLMConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self.generator = CardRewardGenerator(config, output_dir)

    @classmethod
    def from_env_file(cls, env_file: str | None, output_dir: str, model_override: str = ""):
        config = load_llm_config(env_file or None)
        if model_override.strip():
            config.model = model_override.strip()
        return cls(config=config, output_dir=output_dir)

    def generate_reward_spec(
        self,
        *,
        base_spec,
        env_descriptor,
        env_info,
        mode,
        version,
        feedback_bundle=None,
    ) -> CardLLMGenerationResult:
        spec, artifacts_dir, parsed_payload, token_usage = self.generator.generate(
            base_spec=base_spec,
            env_descriptor=env_descriptor,
            env_info=env_info,
            mode=mode,
            version=version,
            feedback_bundle=feedback_bundle or {},
        )
        return CardLLMGenerationResult(
            spec=spec,
            artifacts_dir=artifacts_dir,
            parsed_payload=parsed_payload,
            token_usage=token_usage,
            latency_seconds=float(token_usage.get("latency_seconds", 0.0) or 0.0),
        )


__all__ = [
    "CardLLMError",
    "CardLLMGenerationResult",
    "CardLLMModule",
    "CardRewardGenerationError",
]
