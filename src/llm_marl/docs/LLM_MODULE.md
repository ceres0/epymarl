# LLM-MARL Module

## Current call path

The project now uses a dedicated `CardLLMModule` entrypoint:

1. `CardRuntime._maybe_init_generation(...)`
2. `CardLLMModule.from_env_file(...)`
3. `CardLLMModule.generate_reward_spec(...)`
4. `CardRewardGenerator.generate(...)`
5. `RewardPromptBuilder.build(...)`
6. `OpenAICompatibleLLMClient.complete_json(...)`
7. reward spec validation and artifact persistence
8. token usage persistence and runtime aggregation

## Main files

- `llm_marl/llm_config.py`
  Loads `config/llm/.env.llm.*` files and parses provider/runtime settings.
- `llm_marl/llm_client.py`
  Sends OpenAI-compatible chat completion requests, handles compatibility fallbacks,
  and normalizes provider token usage.
- `llm_marl/prompt_builder.py`
  Builds generation prompts from reward spec, env context, and feedback.
- `llm_marl/generator.py`
  Converts LLM output into a validated reward spec and saves generation artifacts.
- `llm_marl/llm_module.py`
  Reusable high-level module that wraps config loading and reward generation.

## Reusable module API

```python
from llm_marl.llm_module import CardLLMModule

module = CardLLMModule.from_env_file(
    env_file="../config/llm/.env.llm.modelscope",
    output_dir="../results/llm/generated_specs",
)

result = module.generate_reward_spec(
    base_spec=reward_spec,
    env_descriptor=env_descriptor,
    env_info=env_info,
    mode="iterative",
    version=2,
    feedback_bundle={
        "process_feedback": "...",
        "trajectory_feedback": "...",
        "preference_feedback": "...",
    },
)
```

`result` contains:

- `result.spec`
- `result.artifacts_dir`
- `result.parsed_payload`
- `result.token_usage`

## Token usage outputs

Each OpenAI-compatible response is normalized into:

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `estimated`

Provider-reported `usage` values are used when present. If the provider omits
usage, the module records a character-based estimate and marks `estimated=true`.

Reward generation writes `token_usage.json` under each `llm_generations/*`
artifact directory and stores the same usage in `result.token_usage` plus reward
spec metadata. Runtime LLM-MARL/MAGPIE execution also aggregates LLM calls into
`llm_usage.json` under the run output directory and logs cumulative
`llm_calls`, `llm_prompt_tokens`, `llm_completion_tokens`, and
`llm_total_tokens`.

## Intended reuse boundary

If another project needs the same LLM reward-generation flow, the recommended reuse surface is:

- `CardLLMModule`
- `CardLLMConfig`
- `config/llm/.env.llm.*` configuration format

This avoids coupling external callers directly to runtime, runner, or trajectory internals.
