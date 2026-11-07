# Modified from EPyMARL for OA-QMIX and opponent-profile research.
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE_CANDIDATES = [
    PROJECT_ROOT / "config/llm/.env.llm.local",
    PROJECT_ROOT / "config/llm/.env.llm.deepseek",
    PROJECT_ROOT / "config/llm/.env.llm.modelscope",
    PROJECT_ROOT / "config/llm/.env.llm",
]


def _parse_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None or value.strip() == "":
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _parse_float(value: str, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _env_value(values: dict[str, str], key: str, default: str = "") -> str:
    return values.get(key, default)


def _process_env_value(key: str, default: str = "") -> str:
    value = os.environ.get(key, "").strip()
    if value:
        return value
    return default.strip()


@dataclass
class CardLLMConfig:
    source_path: str
    provider: str
    base_url: str
    api_key: str
    model: str
    organization: str
    project: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    max_retries: int
    payload_max_retries: int
    retry_backoff_seconds: int
    enable_thinking: bool | None
    thinking_mode: str
    ollama_keep_alive: str
    ollama_num_ctx: int
    use_response_format: bool
    response_format: str
    expected_keys: list[str]
    generation_mode: str
    generation_enabled: bool
    include_architecture_summary: bool
    include_env_descriptor: bool
    include_reward_schema_rules: bool
    include_previous_reward_code: bool
    include_process_feedback: bool
    include_trajectory_feedback: bool
    include_preference_feedback: bool
    output_spec_id_prefix: str
    output_name_prefix: str
    save_raw_prompt: bool
    save_raw_response: bool
    save_parsed_spec: bool
    require_schema_validation: bool
    require_executor_validation: bool
    require_tpe_gate: bool
    generated_spec_dir: str

    @property
    def is_openai_compatible(self) -> bool:
        return self.provider in {"openai_compatible", "openai", "modelscope", "ollama"}


def resolve_env_file(explicit_path: str | None = None) -> Path:
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.is_absolute():
            return candidate
        if candidate.exists():
            return candidate.resolve()
        project_relative = PROJECT_ROOT / explicit_path
        if project_relative.exists():
            return project_relative
        return project_relative

    env_override = _process_env_value("LLM_ENV_FILE")
    if env_override:
        candidate = Path(env_override)
        if candidate.is_absolute():
            return candidate
        if candidate.exists():
            return candidate.resolve()
        project_relative = PROJECT_ROOT / env_override
        if project_relative.exists():
            return project_relative
        return project_relative

    for candidate in DEFAULT_ENV_FILE_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_ENV_FILE_CANDIDATES[0]


def load_llm_config(explicit_path: str | None = None) -> CardLLMConfig:
    path = resolve_env_file(explicit_path)
    values = parse_env_file(path)

    provider = _env_value(values, "LLM_PROVIDER", "openai_compatible").strip()
    config = CardLLMConfig(
        source_path=str(path),
        provider=provider,
        base_url=_env_value(values, "LLM_BASE_URL").strip(),
        api_key=_env_value(values, "LLM_API_KEY").strip(),
        model=_env_value(values, "LLM_MODEL").strip(),
        organization=_env_value(values, "LLM_ORGANIZATION").strip(),
        project=_env_value(values, "LLM_PROJECT").strip(),
        temperature=_parse_float(_env_value(values, "LLM_TEMPERATURE"), 0.2),
        max_tokens=_parse_int(_env_value(values, "LLM_MAX_TOKENS"), 4000),
        timeout_seconds=_parse_int(_env_value(values, "LLM_TIMEOUT_SECONDS"), 120),
        max_retries=_parse_int(_env_value(values, "LLM_MAX_RETRIES"), 3),
        payload_max_retries=_parse_int(_env_value(values, "LLM_PAYLOAD_MAX_RETRIES"), 2),
        retry_backoff_seconds=_parse_int(
            _env_value(values, "LLM_RETRY_BACKOFF_SECONDS"), 2
        ),
        enable_thinking=_parse_optional_bool(_env_value(values, "LLM_ENABLE_THINKING")),
        thinking_mode=_env_value(values, "LLM_THINKING_MODE").strip().lower(),
        ollama_keep_alive=_env_value(values, "LLM_OLLAMA_KEEP_ALIVE", "10m").strip(),
        ollama_num_ctx=_parse_int(_env_value(values, "LLM_OLLAMA_NUM_CTX"), 4096),
        use_response_format=_parse_bool(
            _env_value(values, "LLM_USE_RESPONSE_FORMAT"), default=False
        ),
        response_format=_env_value(values, "LLM_RESPONSE_FORMAT", "json_object").strip(),
        expected_keys=[
            item.strip()
            for item in _env_value(
                values,
                "LLM_EXPECTED_KEYS",
                "name,description,reward_mode,subreward_keys,allowed_context_keys,strategy,metadata",
            ).split(",")
            if item.strip()
        ],
        generation_mode=_env_value(values, "LLM_GENERATION_MODE", "iterative").strip(),
        generation_enabled=_parse_bool(
            _env_value(values, "LLM_GENERATION_ENABLED"),
            default=True,
        ),
        include_architecture_summary=_parse_bool(
            _env_value(values, "LLM_INCLUDE_ARCHITECTURE_SUMMARY"),
            default=True,
        ),
        include_env_descriptor=_parse_bool(
            _env_value(values, "LLM_INCLUDE_ENV_DESCRIPTOR"),
            default=True,
        ),
        include_reward_schema_rules=_parse_bool(
            _env_value(values, "LLM_INCLUDE_REWARD_SCHEMA_RULES"),
            default=True,
        ),
        include_previous_reward_code=_parse_bool(
            _env_value(values, "LLM_INCLUDE_PREVIOUS_REWARD_CODE"),
            default=True,
        ),
        include_process_feedback=_parse_bool(
            _env_value(values, "LLM_INCLUDE_PROCESS_FEEDBACK"),
            default=True,
        ),
        include_trajectory_feedback=_parse_bool(
            _env_value(values, "LLM_INCLUDE_TRAJECTORY_FEEDBACK"),
            default=True,
        ),
        include_preference_feedback=_parse_bool(
            _env_value(values, "LLM_INCLUDE_PREFERENCE_FEEDBACK"),
            default=True,
        ),
        output_spec_id_prefix=_env_value(
            values,
            "LLM_OUTPUT_SPEC_ID_PREFIX",
            "smac_team_battle_llm",
        ).strip(),
        output_name_prefix=_env_value(
            values,
            "LLM_OUTPUT_NAME_PREFIX",
            "smac_team_battle_llm",
        ).strip(),
        save_raw_prompt=_parse_bool(
            _env_value(values, "LLM_SAVE_RAW_PROMPT"),
            default=True,
        ),
        save_raw_response=_parse_bool(
            _env_value(values, "LLM_SAVE_RAW_RESPONSE"),
            default=True,
        ),
        save_parsed_spec=_parse_bool(
            _env_value(values, "LLM_SAVE_PARSED_SPEC"),
            default=True,
        ),
        require_schema_validation=_parse_bool(
            _env_value(values, "LLM_REQUIRE_SCHEMA_VALIDATION"),
            default=True,
        ),
        require_executor_validation=_parse_bool(
            _env_value(values, "LLM_REQUIRE_EXECUTOR_VALIDATION"),
            default=True,
        ),
        require_tpe_gate=_parse_bool(
            _env_value(values, "LLM_REQUIRE_TPE_GATE"),
            default=True,
        ),
        generated_spec_dir=_env_value(
            values,
            "LLM_GENERATED_SPEC_DIR",
            "./results/llm/generated_specs",
        ).strip(),
    )
    return config
