import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch as th

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
MAGPIE_UTILS = os.path.join(SCRIPTS, "magpie", "utils")
if MAGPIE_UTILS not in sys.path:
    sys.path.insert(0, MAGPIE_UTILS)

from llm_marl.opponent_features import OpponentFeatureExtractor
from llm_marl.opponent_profile import (
    OpponentProfileGenerator,
    OpponentProfileStore,
    normalize_opponent_style,
    style_label_for,
)
from llm_marl.generator import CardRewardGenerator
from llm_marl.llm_client import CardLLMError, extract_json_payload, extract_token_usage
from llm_marl.llm_client import LLMCompletionResult
from llm_marl.llm_config import load_llm_config
from llm_marl.reward_spec import RewardSpec
import llm_marl.runtime as runtime_module
from llm_marl.runtime import CardRuntime
from llm_marl.tpe import TPEEvaluator
from modules.critics.opponent_aux import OpponentAuxHead, masked_aux_loss
from modules.mixers.oa_qmix import OAQMixer
from components.action_selectors import (
    REGISTRY as action_REGISTRY,
    _sample_available_actions,
)
from runners.parallel_runner import _live_env_indices, _unavailable_action_report
import run_magpie_experiments


def synthetic_episode(actions=None, ally_hp=0.2, enemy_drop=5.0):
    return {
        "steps": [
            {
                "actions": actions or [6, 6, 7],
                "prev_context": {
                    "ally_health_fraction": 0.5,
                    "enemy_health_fraction": 1.0,
                    "enemy_health_total": 30.0,
                    "ally_alive": 3,
                },
                "curr_context": {
                    "ally_health_fraction": ally_hp,
                    "enemy_health_fraction": 0.8,
                    "enemy_health_total": 30.0 - enemy_drop,
                    "ally_alive": 2,
                },
                "env_info": {},
            }
        ],
        "summary": {},
    }


class MagpieCoreTest(unittest.TestCase):
    def test_parallel_runner_reports_unavailable_selected_actions(self):
        actions = [[1, 10], [0, 1]]
        avail_actions = [
            [[0, 1, 0] + [0] * 8, [1, 0, 0] + [0] * 8],
            [[1, 0, 0] + [0] * 8, [0, 1, 0] + [0] * 8],
        ]

        report = _unavailable_action_report(actions, avail_actions)

        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["env_position"], 0)
        self.assertEqual(report[0]["agent_id"], 1)
        self.assertEqual(report[0]["action"], 10)
        self.assertEqual(report[0]["available"], 0)

    def test_parallel_runner_recomputes_live_envs_before_selecting_actions(self):
        self.assertEqual(_live_env_indices([False, True, False, True]), [0, 2])

    def test_available_action_sampler_respects_singleton_masks(self):
        avail_actions = th.ones(32, 3, 9)
        avail_actions[3, 0] = th.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for _ in range(1000):
            actions = _sample_available_actions(avail_actions)
            self.assertEqual(int(actions[3, 0]), 0)
            selected = avail_actions.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            self.assertTrue(th.all(selected == 1.0))

    def test_available_action_sampler_skips_leading_unavailable_actions_on_zero_draw(self):
        avail_actions = th.tensor([[[0.0, 1.0, 1.0, 0.0]]])
        original_rand_like = th.rand_like

        try:
            th.rand_like = lambda input_tensor: th.zeros_like(input_tensor)
            actions = _sample_available_actions(avail_actions)
        finally:
            th.rand_like = original_rand_like

        self.assertEqual(int(actions[0, 0]), 1)

    def test_load_llm_config_prefers_llm_prefix(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(
                "\n".join(
                    [
                        "LLM_PROVIDER=ollama",
                        "LLM_BASE_URL=http://platform-ollama:11434",
                        "LLM_MODEL=qwen3:0.6b",
                        "LLM_ENABLE_THINKING=false",
                        "LLM_GENERATION_MODE=iterative",
                        "LLM_OUTPUT_SPEC_ID_PREFIX=platform_llm",
                        "LLM_SAVE_RAW_RESPONSE=false",
                        "LLM_REQUIRE_TPE_GATE=false",
                        "LLM_GENERATED_SPEC_DIR=./results/llm/generated_specs",
                    ]
                )
            )
            path = handle.name
        try:
            config = load_llm_config(path)
        finally:
            os.unlink(path)

        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.base_url, "http://platform-ollama:11434")
        self.assertEqual(config.model, "qwen3:0.6b")
        self.assertFalse(config.enable_thinking)
        self.assertEqual(config.output_spec_id_prefix, "platform_llm")
        self.assertFalse(config.save_raw_response)
        self.assertFalse(config.require_tpe_gate)
        self.assertEqual(config.generated_spec_dir, "./results/llm/generated_specs")

    def test_runtime_records_llm_usage_summary_and_stats(self):
        class FakeLogger:
            def __init__(self):
                self.stats = []

            def log_stat(self, key, value, t_env):
                self.stats.append((key, value, t_env))

        runtime = object.__new__(CardRuntime)
        runtime.args = SimpleNamespace(unique_token="token-test")
        runtime.logger = FakeLogger()
        runtime.orchestrator = None
        runtime._last_feedback_t_env = 123
        runtime.llm_usage_totals = runtime._empty_llm_usage()
        runtime.llm_usage_by_purpose = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime.output_dir = tmpdir
            runtime._record_llm_usage(
                "reward_generation",
                {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                    "estimated": False,
                    "latency_seconds": 1.25,
                },
            )
            usage_path = os.path.join(tmpdir, "llm_usage.json")
            self.assertTrue(os.path.exists(usage_path))
            with open(usage_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(payload["total"]["calls"], 1)
        self.assertEqual(payload["total"]["total_tokens"], 18)
        self.assertEqual(payload["total"]["latency_observations"], 1)
        self.assertAlmostEqual(payload["total"]["latency_seconds"], 1.25)
        self.assertAlmostEqual(payload["total"]["mean_latency_seconds"], 1.25)
        self.assertEqual(
            payload["by_purpose"]["reward_generation"]["prompt_tokens"], 11
        )
        self.assertIn(("llm_total_tokens", 18, 123), runtime.logger.stats)
        self.assertIn(("llm_mean_latency_seconds", 1.25, 123), runtime.logger.stats)

    def test_extract_token_usage_uses_provider_usage(self):
        usage = extract_token_usage(
            {
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                }
            },
            system_prompt="system",
            user_prompt="user",
            content="response",
        )
        self.assertEqual(usage["prompt_tokens"], 11)
        self.assertEqual(usage["completion_tokens"], 7)
        self.assertEqual(usage["total_tokens"], 18)
        self.assertFalse(usage["estimated"])

    def test_extract_token_usage_estimates_when_provider_usage_missing(self):
        usage = extract_token_usage(
            {},
            system_prompt="12345678",
            user_prompt="12345678",
            content="12345678",
        )
        self.assertEqual(usage["prompt_tokens"], 4)
        self.assertEqual(usage["completion_tokens"], 2)
        self.assertEqual(usage["total_tokens"], 6)
        self.assertTrue(usage["estimated"])

    def test_extract_json_payload_wraps_malformed_json(self):
        with self.assertRaises(CardLLMError):
            extract_json_payload('{"strategy": {"code": "unterminated}')

    def test_reward_generator_regenerates_after_malformed_payload(self):
        class FakePromptBuilder:
            def build(self, **kwargs):
                return "system", "user"

        class FakeClient:
            def __init__(self, valid_payload):
                self.valid_payload = valid_payload
                self.calls = []

            def complete_json(self, system_prompt, user_prompt):
                self.calls.append((system_prompt, user_prompt))
                if len(self.calls) == 1:
                    return LLMCompletionResult(
                        status_code=200,
                        content='{"strategy": {"code": "unterminated}',
                        raw_json={},
                        token_usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    )
                return LLMCompletionResult(
                    status_code=200,
                    content=json.dumps(self.valid_payload),
                    raw_json={},
                    token_usage={"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                )

        base_spec = RewardSpec(
            format_version=1,
            spec_id="base",
            version=1,
            name="base",
            reward_mode="team",
            subreward_keys=[],
            allowed_context_keys=["base_reward"],
            strategy={
                "type": "python_function",
                "entrypoint": "compute_reward",
                "code": "def compute_reward(context):\n    return context.get('base_reward', 0.0), {}, {}\n",
            },
        )
        valid_payload = base_spec.to_dict()
        valid_payload["name"] = "generated"

        generator = object.__new__(CardRewardGenerator)
        generator.llm_config = SimpleNamespace(
            payload_max_retries=1,
            output_spec_id_prefix="generated",
            output_name_prefix="generated",
            provider="test",
            model="fake",
            generation_mode="iterative",
            save_raw_prompt=False,
            save_raw_response=False,
            save_parsed_spec=False,
        )
        generator.prompt_builder = FakePromptBuilder()
        generator.client = FakeClient(valid_payload)
        with tempfile.TemporaryDirectory() as tmpdir:
            generator.output_dir = tmpdir
            spec, _, _, token_usage = generator.generate(
                base_spec=base_spec,
                env_descriptor={},
                env_info={"n_agents": 3, "episode_limit": 60},
                mode="iterative",
                version=2,
            )

        self.assertEqual(generator.client.calls[0][1], "user")
        self.assertIn("Retry instruction", generator.client.calls[1][1])
        self.assertEqual(spec.name, "generated")
        self.assertEqual(spec.metadata["payload_retry_attempts"], 1)
        self.assertEqual(token_usage["total_tokens"], 4)

    def test_feature_extraction_handles_missing_fields(self):
        features = OpponentFeatureExtractor().extract_episode(synthetic_episode())
        self.assertAlmostEqual(features["focus_fire_rate"], 1.0)
        self.assertAlmostEqual(features["low_hp_chase_ratio"], 1.0)
        self.assertGreaterEqual(features["engage_distance_proxy"], 0.0)

    def test_profile_generation_and_store_roundtrip(self):
        summary = {
            "episodes": 2,
            "means": {
                "focus_fire_rate": 0.8,
                "low_hp_chase_ratio": 0.1,
                "target_switch_rate": 0.0,
            },
        }
        profile = OpponentProfileGenerator(vector_dim=8).build(summary, ["episode.json"])
        self.assertEqual(profile.opponent_style, "aggressive_focus_fire")
        self.assertEqual(len(profile.behavior_vector), 8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = OpponentProfileStore(tmpdir).save(profile)
            loaded = OpponentProfileStore.load(path)
        self.assertEqual(loaded["opponent_style"], "aggressive_focus_fire")
        self.assertEqual(style_label_for(loaded["opponent_style"]), 1)

    def test_profile_style_normalization_handles_llm_aliases(self):
        self.assertEqual(
            normalize_opponent_style("passive", [0.05] * 8),
            "unknown",
        )
        self.assertEqual(
            normalize_opponent_style("high_target_switching", [0.4, 0.3, 0.9]),
            "scattered_target_switching",
        )
        self.assertEqual(
            normalize_opponent_style("cohesive_focus", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.2]),
            "balanced",
        )
        self.assertEqual(
            normalize_opponent_style("balanced_aggressive", [0.05] * 8),
            "unknown",
        )

    def test_profile_from_payload_normalizes_style_before_storage(self):
        payload = {
            "opponent_style": "deathball",
            "behavior_vector": [0.55, 0.45, 0.3, 0.45, 0.55, 0.4, 0.82, 0.25],
            "counter_strategy": "hold formation",
            "confidence": 0.8,
            "profile_version": "magpie-profile-v1",
        }
        profile = OpponentProfileGenerator(vector_dim=8, mode="llm")._profile_from_payload(
            payload,
            feature_summary={"means": {}},
            source_episodes=[],
        )
        self.assertEqual(profile.opponent_style, "balanced")
        self.assertEqual(style_label_for("deathball"), style_label_for("balanced"))

    def test_llm_opponent_profile_refreshes_during_episode_finish(self):
        class FakeTrajectoryStore:
            def __init__(self):
                self.summaries = []

            def finish_episode(self, env_id, outcome, env_info):
                summary = {
                    "env_id": env_id,
                    "outcome": outcome,
                    "test_mode": False,
                    "opponent_features": {"focus_fire_rate": 1.0},
                }
                self.summaries.append(summary)
                return summary

            def completed_summaries(self, test_mode=None):
                if test_mode is None:
                    return list(self.summaries)
                return [
                    summary
                    for summary in self.summaries
                    if summary["test_mode"] == bool(test_mode)
                ]

        runtime = object.__new__(CardRuntime)
        runtime.enabled = True
        runtime.card_cfg = {
            "opponent": {
                "profile": {
                    "min_episodes": 1,
                    "refresh_episodes": 1,
                    "vector_dim": 8,
                }
            }
        }
        runtime.trajectory_store = FakeTrajectoryStore()
        runtime.opponent_profile = None
        runtime._last_profile_refresh_episode_count = -1
        runtime.opponent_profile_generator = SimpleNamespace(mode="llm")
        refresh_calls = []

        def fake_refresh():
            refresh_calls.append(len(runtime.trajectory_store.completed_summaries(False)))

        runtime.refresh_opponent_profile = fake_refresh

        CardRuntime.finish_episode(runtime, 0, {"battle_won": True})

        self.assertEqual(refresh_calls, [1])

    def test_llm_opponent_profile_call_record_is_persisted(self):
        class FakeClient:
            def __init__(self, config):
                self.config = config

            def complete_json(self, system_prompt, user_prompt):
                return LLMCompletionResult(
                    status_code=200,
                    content=json.dumps(
                        {
                            "opponent_style": "balanced",
                            "behavior_vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                            "counter_strategy": "hold formation",
                            "confidence": 0.75,
                            "profile_version": "magpie-profile-v1",
                        }
                    ),
                    raw_json={"choices": [{"message": {"content": "ok"}}]},
                    token_usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 8,
                        "total_tokens": 18,
                        "estimated": False,
                    },
                    latency_seconds=0.25,
                )

        runtime = object.__new__(CardRuntime)
        runtime.args = SimpleNamespace(unique_token="record-test")
        runtime.logger = SimpleNamespace(log_stat=lambda *args: None)
        runtime.card_cfg = {
            "llm": {"env_file": "", "model": ""},
            "opponent": {
                "profile": {
                    "vector_dim": 8,
                    "record_calls": True,
                }
            },
        }
        runtime.llm_config = SimpleNamespace(
            provider="fake",
            model="fake-model",
        )
        runtime.llm_usage_totals = runtime._empty_llm_usage()
        runtime.llm_usage_by_purpose = {}
        runtime._last_feedback_t_env = 0
        runtime.orchestrator = None
        runtime.opponent_profile_generator = OpponentProfileGenerator(
            vector_dim=8, mode="llm"
        )
        runtime._llm_profile_call_counter = 0

        original_client = runtime_module.OpenAICompatibleLLMClient
        runtime_module.OpenAICompatibleLLMClient = FakeClient
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                runtime.output_dir = tmpdir
                profile = CardRuntime._build_llm_opponent_profile(
                    runtime,
                    feature_summary={"episodes": 1, "means": {"focus_fire_rate": 0.5}},
                    source_episodes=["episode_000001.json"],
                )
                index_path = os.path.join(tmpdir, "llm_profile_calls.jsonl")
                self.assertTrue(os.path.exists(index_path))
                with open(index_path, encoding="utf-8") as handle:
                    record = json.loads(handle.readline())
                artifacts_dir = record["artifacts_dir"]
                self.assertEqual(record["status"], "ok")
                self.assertEqual(record["parsed_response"]["opponent_style"], "balanced")
                self.assertTrue(os.path.exists(os.path.join(artifacts_dir, "system_prompt.txt")))
                self.assertTrue(os.path.exists(os.path.join(artifacts_dir, "user_prompt.txt")))
                self.assertTrue(os.path.exists(os.path.join(artifacts_dir, "raw_response.txt")))
                self.assertTrue(os.path.exists(os.path.join(artifacts_dir, "parsed_response.json")))
                self.assertTrue(os.path.exists(os.path.join(artifacts_dir, "token_usage.json")))
        finally:
            runtime_module.OpenAICompatibleLLMClient = original_client

        self.assertEqual(profile.opponent_style, "balanced")

    def test_magpie_pe_behavior_counterexample(self):
        trajectories = [
            {
                "episode_length": 5,
                "outcome": "loss",
                "normalized_team_return": 1.0,
                "subreward_totals": {"survival": 10.0},
                "combat_summary": {"ally_alive": 0},
                "opponent_features": {"focus_fire_rate": 1.0},
            },
            {
                "episode_length": 5,
                "outcome": "win",
                "normalized_team_return": 2.0,
                "subreward_totals": {"survival": 1.0},
                "combat_summary": {"ally_alive": 1},
                "opponent_features": {"focus_fire_rate": 1.0},
            },
        ]
        result = TPEEvaluator(
            threshold=0.75, behavior_enabled=True, behavior_weight=0.5
        ).evaluate(trajectories, {"opponent_style": "aggressive_focus_fire"})
        self.assertLess(result.behavior_accuracy, 1.0)
        self.assertIn("positive_reward_with_focus_fire_victim_death", result.hacking_flags)

    def test_oa_qmix_forward_backward_shape(self):
        args = SimpleNamespace(
            n_agents=3,
            state_shape=(5,),
            opp_profile_dim=8,
            mixing_embed_dim=4,
            hypernet_layers=1,
        )
        mixer = OAQMixer(args)
        agent_qs = th.randn(2, 4, 3, requires_grad=True)
        states = th.randn(2, 4, 5)
        opp_profile = th.zeros(2, 4, 8)
        q_tot = mixer(agent_qs, states, opp_profile)
        self.assertEqual(tuple(q_tot.shape), (2, 4, 1))
        q_tot.mean().backward()
        self.assertIsNotNone(agent_qs.grad)

    def test_zero_profile_generator_returns_zero_vector(self):
        profile = OpponentProfileGenerator(vector_dim=8, mode="zero").build(
            {"means": {"focus_fire_rate": 1.0}},
            source_episodes=["episode.json"],
        )
        self.assertEqual(profile.opponent_style, "zero_profile")
        self.assertEqual(profile.behavior_vector, [0.0] * 8)
        self.assertEqual(profile.confidence, 0.0)
        self.assertEqual(profile.generation_mode, "zero")

    def test_aux_loss_uses_nonzero_style_mask(self):
        head = OpponentAuxHead(input_dim=13, hidden_dim=4, style_classes=4, n_agents=3)
        state = th.randn(5, 5)
        opp_profile = th.randn(5, 8)
        outputs = head(state, opp_profile)
        labels = {
            "style": th.ones(5, dtype=th.long),
            "focus": th.zeros(5, dtype=th.long),
            "danger": th.zeros(5),
        }
        masks = {
            "style": th.ones(5, dtype=th.bool),
            "focus": th.zeros(5, dtype=th.bool),
            "danger": th.zeros(5, dtype=th.bool),
        }
        loss, metrics = masked_aux_loss(outputs, labels, masks)
        self.assertGreater(loss.item(), 0.0)
        self.assertGreater(metrics["opponent_aux_loss"], 0.0)

    def test_aux_loss_backpropagates_to_agent_representation(self):
        head = OpponentAuxHead(input_dim=12, hidden_dim=4, style_classes=4, n_agents=3)
        agent_repr = th.randn(5, 4, requires_grad=True)
        opp_profile = th.randn(5, 8)
        outputs = head(agent_repr, opp_profile)
        labels = {
            "style": th.ones(5, dtype=th.long),
            "focus": th.zeros(5, dtype=th.long),
            "danger": th.zeros(5),
        }
        masks = {
            "style": th.ones(5, dtype=th.bool),
            "focus": th.zeros(5, dtype=th.bool),
            "danger": th.zeros(5, dtype=th.bool),
        }
        loss, _ = masked_aux_loss(outputs, labels, masks)
        loss.backward()
        self.assertIsNotNone(agent_repr.grad)
        self.assertGreater(agent_repr.grad.abs().sum().item(), 0.0)

    def test_transient_sc2_job_failure_is_retried(self):
        calls = []
        original_run_docker = run_magpie_experiments.run_docker

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "job.log")

            def fake_run_docker(main_args, dry_run, monitored, service, log_path=None):
                calls.append(1)
                if len(calls) == 1:
                    with open(log_path, "w", encoding="utf-8") as handle:
                        handle.write(
                            "RuntimeError: parallel env worker 0 step failed: "
                            "Error during join_game: Websocket timed out.\n"
                        )
                    raise subprocess.CalledProcessError(1, ["docker", "compose"])
                with open(log_path, "w", encoding="utf-8") as handle:
                    handle.write("Finished Training\n")

            run_magpie_experiments.run_docker = fake_run_docker
            try:
                run_magpie_experiments.run_job(
                    1,
                    {
                        "stage": "standard_smac_8m",
                        "method": "magpie_llm_profile_critic_only",
                        "seed": 1,
                        "stage_config": {"env_config": "sc2"},
                        "main_args": ["src/main.py"],
                    },
                    SimpleNamespace(
                        executor="docker",
                        dry_run=False,
                        monitored=False,
                        docker_service="train",
                        job_retries=1,
                        job_retry_delay=0,
                    ),
                    log_path=log_path,
                )
            finally:
                run_magpie_experiments.run_docker = original_run_docker

            self.assertEqual(len(calls), 2)
            self.assertTrue(os.path.exists(log_path + ".attempt1.failed.log"))
            with open(log_path, encoding="utf-8") as handle:
                self.assertIn("Finished Training", handle.read())


if __name__ == "__main__":
    unittest.main()
