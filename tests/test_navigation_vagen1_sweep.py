import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parse_utils = load_module(
    "parse_utils",
    REPO_ROOT / "vagen/env/utils/parse_utils.py",
)
client_module = load_module(
    "server_client",
    REPO_ROOT / "vagen/server/client.py",
)


class NavigationVagen1SweepTest(unittest.TestCase):
    def test_single_action_parser_marks_multiple_actions_as_format_failure(self):
        response = (
            "<think><observation>target ahead</observation>"
            "<reasoning>move closer</reasoning>"
            "<prediction>closer</prediction></think>"
            "<answer>moveahead, rotateleft</answer>"
        )

        parsed = parse_utils.parse_grounding_worldmodeling(
            response=response,
            action_sep=",",
            max_actions=1,
        )

        self.assertFalse(parsed["format_correct"])
        self.assertTrue(parsed["too_many_actions"])
        self.assertEqual("too_many_actions", parsed["format_error_type"])
        self.assertEqual(["moveahead", "rotateleft"], parsed["actions"])

    def test_parser_reports_malformed_tags_and_empty_answer(self):
        malformed = parse_utils.parse_grounding_worldmodeling(
            response="<think>reasoning only</think> moveahead",
            action_sep=",",
            max_actions=1,
        )
        empty = parse_utils.parse_grounding_worldmodeling(
            response=(
                "<think><observation>target</observation>"
                "<reasoning>move</reasoning><prediction>closer</prediction></think>"
                "<answer>   </answer>"
            ),
            action_sep=",",
            max_actions=1,
        )

        self.assertEqual("missing_or_malformed_tags", malformed["format_error_type"])
        self.assertEqual("empty_answer", empty["format_error_type"])

    def test_answer_only_loss_mask_wraps_only_answer_content(self):
        mask_utils = load_module(
            "loss_mask_utils",
            REPO_ROOT / "vagen/rollout/qwen_rollout/loss_mask_utils.py",
        )
        response = (
            "<think><observation>target ahead</observation>"
            "<reasoning>move closer</reasoning>"
            "<prediction>closer</prediction></think>"
            "<answer>moveahead</answer>"
        )

        masked = mask_utils.prepare_response_for_loss_mask(
            response,
            special_tokens=("<|box_start|>", "<|box_end|>"),
            mode="answer_only",
        )

        self.assertIn("<answer><|box_start|>moveahead<|box_end|></answer>", masked)
        self.assertNotIn("<|box_start|><think>", masked)

    def test_default_loss_mask_wraps_full_response(self):
        mask_utils = load_module(
            "loss_mask_utils",
            REPO_ROOT / "vagen/rollout/qwen_rollout/loss_mask_utils.py",
        )
        response = "<think>reason</think><answer>moveahead</answer>"

        masked = mask_utils.prepare_response_for_loss_mask(
            response,
            special_tokens=("<|box_start|>", "<|box_end|>"),
            mode="default",
        )

        self.assertEqual(
            "<|box_start|><think>reason</think><answer>moveahead</answer><|box_end|>",
            masked,
        )

    def test_action_distribution_metrics_summarize_recorded_actions(self):
        action_metrics = load_module(
            "action_metrics",
            REPO_ROOT / "vagen/rollout/qwen_rollout/action_metrics.py",
        )
        records = [
            {"info": {"actions": ["moveahead"], "format_correct": True, "too_many_actions": False}},
            {"info": {"actions": ["moveahead"], "format_correct": True, "too_many_actions": False}},
            {"info": {"actions": ["rotateleft"], "format_correct": True, "too_many_actions": False}},
            {"info": {"actions": [], "format_correct": False, "format_error_type": "empty_answer"}},
        ]

        metrics = action_metrics.summarize_action_distribution(records)

        self.assertAlmostEqual(2 / 3, metrics["action/top_share"])
        self.assertAlmostEqual(0.0, metrics["action/all_same_traj"])
        self.assertAlmostEqual(2 / 3, metrics["action/share/moveahead"])
        self.assertAlmostEqual(1 / 4, metrics["format/error/empty_answer"])
        self.assertGreater(metrics["action/entropy"], 0)

    def test_action_distribution_tracks_invalid_action_name_format_error(self):
        action_metrics = load_module(
            "action_metrics",
            REPO_ROOT / "vagen/rollout/qwen_rollout/action_metrics.py",
        )
        records = [
            {
                "info": {
                    "actions": ["stay"],
                    "format_correct": False,
                    "format_error_type": "invalid_action_name",
                    "metrics": {"turn_metrics": {"action_validity_error": "invalid_action_name"}},
                }
            },
            {
                "info": {
                    "actions": ["rotatelleft"],
                    "format_correct": False,
                    "format_error_type": "invalid_action_name",
                    "metrics": {"turn_metrics": {"action_validity_error": "invalid_action_name"}},
                }
            },
            {
                "info": {
                    "actions": ["moveahead"],
                    "format_correct": True,
                    "format_error_type": "ok",
                    "metrics": {"turn_metrics": {"action_validity_error": "ok"}},
                }
            },
        ]

        metrics = action_metrics.summarize_action_distribution(records)

        self.assertAlmostEqual(2 / 3, metrics["format/error/invalid_action_name"])
        self.assertAlmostEqual(2 / 3, metrics["action/error/invalid_action_name"])
        self.assertAlmostEqual(1 / 3, metrics["action/valid_vocab_rate"])
        self.assertAlmostEqual(1 / 3, metrics["action/forbidden_stay_stop_end_rate"])
        self.assertAlmostEqual(1 / 3, metrics["action/invalid_typo_rate"])

    def test_raw_sample_collector_prioritizes_invalid_responses(self):
        raw_sample_utils = load_module(
            "raw_sample_utils",
            REPO_ROOT / "vagen/rollout/qwen_rollout/raw_sample_utils.py",
        )
        log_rst = [
            {
                "env_id": "train1",
                "config_id": "cfg",
                "history": [
                    {"info": {}, "reward": 0, "done": False},
                    {
                        "info": {
                            "llm_raw_response": "<think>x</think><answer>moveahead</answer>",
                            "llm_response": "<think>x</think><answer>moveahead</answer>",
                            "action_content": "moveahead",
                            "actions": ["moveahead"],
                            "format_correct": True,
                            "format_error_type": "ok",
                            "metrics": {"turn_metrics": {"action_is_valid": True, "action_validity_error": "ok"}},
                        },
                        "reward": 0.05,
                        "done": False,
                    },
                    {
                        "info": {
                            "llm_raw_response": "move forward",
                            "llm_response": "<think></think><answer></answer>",
                            "action_content": "",
                            "actions": [],
                            "format_correct": False,
                            "format_error_type": "missing_or_malformed_tags",
                            "metrics": {
                                "turn_metrics": {
                                    "action_is_valid": False,
                                    "action_validity_error": "no_action",
                                }
                            },
                        },
                        "reward": 0,
                        "done": False,
                    },
                ],
            }
        ]

        samples = raw_sample_utils.collect_raw_response_samples(log_rst, limit=1)

        self.assertEqual(1, len(samples))
        self.assertEqual("move forward", samples[0]["raw_response"])
        self.assertEqual("missing_or_malformed_tags", samples[0]["format_error_type"])

    def test_trainer_source_logs_raw_samples_to_stdout_and_wandb(self):
        trainer_source = (REPO_ROOT / "vagen/trainer/ppo/ray_trainer.py").read_text()
        run_script = (REPO_ROOT / "scripts/examples/vagen_base/navigation/run.sh").read_text()

        self.assertIn("collect_raw_response_samples", trainer_source)
        self.assertIn("[RAW_SAMPLE]", trainer_source)
        self.assertIn("raw_samples_to_log", trainer_source)
        self.assertIn("wandb.Table", trainer_source)
        self.assertIn("trainer.raw_samples_to_log=$RAW_SAMPLES_TO_LOG", run_script)

    def test_vagen1_defaults_are_vagen_first_not_limit20(self):
        run_script = (REPO_ROOT / "scripts/examples/vagen_base/navigation_vagen1/run.sh").read_text()
        base_run_script = (REPO_ROOT / "scripts/examples/vagen_base/navigation/run.sh").read_text()

        self.assertIn("ROLLOUT_LIMIT_MM_PER_PROMPT=${ROLLOUT_LIMIT_MM_PER_PROMPT:-8}", run_script)
        self.assertIn("ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}", run_script)
        self.assertIn("ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}", run_script)
        self.assertIn("ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}", run_script)
        self.assertIn("FORMAT_REWARD=${FORMAT_REWARD:-0.1}", run_script)
        self.assertIn("rollout_manager.loss_mask_mode=$LOSS_MASK_MODE", base_run_script)

    def test_wave1_submit_script_contains_engine_sweep_variants(self):
        submit_script = (
            REPO_ROOT / "scripts/superpod/submit_navigation_vagen1_wave1_engine_smoke.sh"
        ).read_text()

        for variant in [
            "vagenrt_gpu04_limit8",
            "vagenrt_gpu06_limit8",
            "eager_gpu04_limit8",
            "eager_gpu06_limit8",
            "eager_free_gpu04_limit8",
            "eager_chunk_gpu04_limit8",
            "failed_minus_limit8_gpu06",
            "tp1_diag_gpu06_limit8",
        ]:
            self.assertIn(variant, submit_script)

        self.assertIn(
            "WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_vagen1_engine_smoke_limit8_batch8_smokedata_20260722}",
            submit_script,
        )
        self.assertIn("TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}", submit_script)
        self.assertIn("PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}", submit_script)
        self.assertIn("VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}", submit_script)
        self.assertIn(
            "ENV_CONFIG_PATH=${ENV_CONFIG_PATH:-$VAGEN_REPO/scripts/examples/vagen_base/navigation_vagen1/env_config_smoke.yaml}",
            submit_script,
        )

    def test_wave2_submit_script_uses_vagen_runtime_speed_variants(self):
        submit_script = (
            REPO_ROOT / "scripts/superpod/submit_navigation_vagen1_wave2_speed_debug5.sh"
        ).read_text()
        variant_script = (
            REPO_ROOT / "scripts/superpod/configure_navigation_vagen1_variant.sh"
        ).read_text()

        for variant in [
            "speed_b8_rmb1_w1_turn20",
            "speed_b8_rmb4_w1_turn20",
            "speed_b8_rmb8_w1_turn20",
            "speed_b16_rmb4_w2_turn20",
            "speed_b16_rmb8_w2_turn20",
            "speed_b16_rmb16_w4_turn20",
            "speed_b32_rmb8_w4_turn20",
            "speed_b32_rmb16_w4_turn20",
            "speed_b64_rmb16_w4_turn20",
            "speed_b64_rmb32_w4_turn20",
            "external_b16_rmb16_w4_turn20",
            "external_b32_rmb16_w4_turn20",
            "external_b32_rmb32_w8_turn20",
            "external2_train2x4_b16_rmb16_w4x2",
            "external2_train2x4_b32_rmb16_w4x2",
        ]:
            self.assertIn(variant, variant_script)

        for submitted_variant in [
            "speed_b8_rmb4_w1_turn20",
            "speed_b8_rmb8_w1_turn20",
            "speed_b16_rmb8_w2_turn20",
            "speed_b16_rmb16_w4_turn20",
            "speed_b32_rmb8_w4_turn20",
            "speed_b32_rmb16_w4_turn20",
            "speed_b64_rmb16_w4_turn20",
            "speed_b64_rmb32_w4_turn20",
        ]:
            self.assertIn(submitted_variant, submit_script)

        self.assertNotIn("speed_b8_rmb1_w1_turn20", submit_script)
        self.assertIn("TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-6}", submit_script)
        self.assertIn("FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-False}", submit_script)
        self.assertIn("env_config_speed.yaml", submit_script)
        self.assertIn("set_vagen_runtime 0.6 8", variant_script)
        self.assertIn("export ROLLOUT_MINI_BATCH_SIZE=\"$rollout_mini_batch_size\"", variant_script)
        self.assertIn("export SERVER_NAVIGATION_MAX_WORKERS=\"$server_workers\"", variant_script)

    def test_eager_stability_variants_are_recognized_by_variant_config(self):
        variant_script = (
            REPO_ROOT / "scripts/superpod/configure_navigation_vagen1_variant.sh"
        ).read_text()

        self.assertIn("eager_tiny)", variant_script)
        self.assertIn("eager_actionpen)", variant_script)
        self.assertIn("set_eager_runtime 0.4 8", variant_script)

    def test_external_env_submit_script_launches_server_training_pairs(self):
        submit_script = (
            REPO_ROOT / "scripts/superpod/submit_navigation_vagen1_external_env_debug.sh"
        ).read_text()

        for variant in [
            "external_b16_rmb16_w4_turn20",
            "external_b32_rmb16_w4_turn20",
        ]:
            self.assertIn(variant, submit_script)

        self.assertIn("navigation_vagen1_external_env_debug5_20260722", submit_script)
        self.assertIn("source \"$VAGEN_REPO/scripts/superpod/load_modules.sh\"", submit_script)
        self.assertIn("run_navigation_vagen1_ai2thor_server.sbatch", submit_script)
        self.assertIn("run_navigation_vagen1_4gpu_external_server.sbatch", submit_script)
        self.assertIn("--dependency=after:\"$server_job\"", submit_script)
        self.assertIn("numeric env server job id", submit_script)
        self.assertIn("numeric training job id", submit_script)
        self.assertIn("SERVER_READY_FILE=\"$ready_file\"", submit_script)
        self.assertIn("SERVER_NAVIGATION_MAX_WORKERS=\"$SERVER_NAVIGATION_MAX_WORKERS\"", submit_script)
        self.assertIn("TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-6}", submit_script)
        self.assertIn("FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-False}", submit_script)

    def test_sharded_batch_env_client_routes_each_env_to_one_server(self):
        original_client = client_module.BatchEnvClient

        class FakeClient:
            instances = []

            def __init__(self, base_url, timeout=600, max_workers=10):
                self.base_url = base_url
                self.calls = []
                self.env_configs = {}
                FakeClient.instances.append(self)

            def check_server_health(self):
                return {"status": "ok", "base_url": self.base_url}

            def wait_for_server(self, max_retries=10, retry_delay=1.0):
                return True

            def create_environments_batch(self, ids2configs):
                self.calls.append(("create", tuple(sorted(ids2configs))))
                self.env_configs.update(ids2configs)

            def reset_batch(self, ids2seeds):
                self.calls.append(("reset", tuple(sorted(ids2seeds))))
                return {env_id: ({"obs_str": self.base_url}, {"server": self.base_url}) for env_id in ids2seeds}

            def step_batch(self, ids2actions):
                self.calls.append(("step", tuple(sorted(ids2actions))))
                return {
                    env_id: ({"obs_str": self.base_url}, 0.0, False, {"metrics": {"turn_metrics": {}, "traj_metrics": {}}})
                    for env_id in ids2actions
                }

            def compute_reward_batch(self, env_ids):
                self.calls.append(("reward", tuple(sorted(env_ids))))
                return {env_id: 0.0 for env_id in env_ids}

            def get_system_prompts_batch(self, env_ids):
                self.calls.append(("system_prompt", tuple(sorted(env_ids))))
                return {env_id: self.base_url for env_id in env_ids}

            def close_batch(self, env_ids=None):
                env_ids = list(self.env_configs) if env_ids is None else list(env_ids)
                self.calls.append(("close", tuple(sorted(env_ids))))
                for env_id in env_ids:
                    self.env_configs.pop(env_id, None)

        client_module.BatchEnvClient = FakeClient
        try:
            client = client_module.ShardedBatchEnvClient(
                base_urls="http://server-a:7001, http://server-b:7002",
                timeout=7,
                max_workers=2,
            )
            ids2configs = {f"train{i}": {"env_name": "navigation"} for i in range(1, 9)}

            client.create_environments_batch(ids2configs)
            created_by_env = {}
            for instance in FakeClient.instances:
                for _, env_ids in [call for call in instance.calls if call[0] == "create"]:
                    for env_id in env_ids:
                        created_by_env[env_id] = instance.base_url
            self.assertEqual(set(ids2configs), set(created_by_env))
            self.assertEqual({"http://server-a:7001", "http://server-b:7002"}, set(created_by_env.values()))

            client.reset_batch({env_id: 42 for env_id in ids2configs})
            client.step_batch({env_id: "moveahead" for env_id in ids2configs})
            client.compute_reward_batch(list(ids2configs))
            client.get_system_prompts_batch(list(ids2configs))
            client.close_batch(list(ids2configs))

            routed_ops = {"reset", "step", "reward", "system_prompt", "close"}
            for instance in FakeClient.instances:
                for op, env_ids in instance.calls:
                    if op in routed_ops:
                        for env_id in env_ids:
                            self.assertEqual(instance.base_url, created_by_env[env_id])
        finally:
            client_module.BatchEnvClient = original_client

    def test_multi_external_submit_script_launches_two_servers_per_training_job(self):
        submit_script = (
            REPO_ROOT / "scripts/superpod/submit_navigation_vagen1_external_multi_env_debug.sh"
        ).read_text()
        variant_script = (
            REPO_ROOT / "scripts/superpod/configure_navigation_vagen1_variant.sh"
        ).read_text()

        for variant in [
            "external2_b16_rmb16_w4x2",
            "external2_b32_rmb16_w4x2",
        ]:
            self.assertIn(variant, submit_script)
            self.assertIn(variant, variant_script)

        self.assertIn("ENV_SERVERS_PER_VARIANT=${ENV_SERVERS_PER_VARIANT:-2}", submit_script)
        self.assertIn("SERVER_READY_FILES", submit_script)
        self.assertIn("after:${server_jobs_dependency}", submit_script)
        self.assertIn("ROLLOUT_BASE_URLS", (REPO_ROOT / "scripts/superpod/run_navigation_vagen1_4gpu_external_server.sbatch").read_text())
        self.assertIn("rollout_manager.base_urls=\"$ROLLOUT_BASE_URLS\"", (REPO_ROOT / "scripts/examples/vagen_base/navigation/run.sh").read_text())

    def test_two_node_external_submit_script_launches_multi_server_8gpu_training(self):
        submit_script = (
            REPO_ROOT / "scripts/superpod/submit_navigation_vagen1_external_2node4gpu_debug.sh"
        ).read_text()
        train_script = (
            REPO_ROOT / "scripts/superpod/run_navigation_vagen1_2node4gpu_external_server.sbatch"
        ).read_text()

        self.assertIn("external2_train2x4_b16_rmb16_w4x2", submit_script)
        self.assertIn("run_navigation_vagen1_ai2thor_server.sbatch", submit_script)
        self.assertIn("run_navigation_vagen1_2node4gpu_external_server.sbatch", submit_script)
        self.assertIn("ENV_SERVERS_PER_VARIANT=${ENV_SERVERS_PER_VARIANT:-2}", submit_script)
        self.assertIn("TRAIN_NNODES=${TRAIN_NNODES:-2}", submit_script)
        self.assertIn("EXPECTED_RAY_GPUS=${EXPECTED_RAY_GPUS:-8}", submit_script)
        self.assertIn("sbatch --parsable", submit_script)
        self.assertIn("numeric 2node4gpu training job id", submit_script)
        self.assertIn("trainer.nnodes=$TRAIN_NNODES", (REPO_ROOT / "scripts/examples/vagen_base/navigation/run.sh").read_text())
        self.assertIn("ray.init(address=ray_address", (REPO_ROOT / "vagen/trainer/main_ppo.py").read_text())
        self.assertIn("RAY_HEAD_ADDRESS", train_script)
        self.assertIn("TRAIN_NNODES=$TRAIN_NNODES", train_script)

    def test_speed_env_config_supports_debug5_batches(self):
        import yaml

        config_path = REPO_ROOT / "scripts/examples/vagen_base/navigation_vagen1/env_config_speed.yaml"
        config = yaml.safe_load(config_path.read_text())

        self.assertEqual({"env1", "env2"}, set(config))
        for env_spec in config.values():
            self.assertEqual(128, env_spec["train_size"])
            self.assertEqual(4, env_spec["test_size"])
            self.assertEqual(1, env_spec["env_config"]["max_actions_per_step"])

    def test_dense_light_env_config_uses_small_shaping_terms(self):
        import yaml

        config_path = REPO_ROOT / "scripts/examples/vagen_base/navigation_vagen1/env_config_dense_light.yaml"
        config = yaml.safe_load(config_path.read_text())

        self.assertEqual({"env1", "env2"}, set(config))
        for env_spec in config.values():
            env_config = env_spec["env_config"]
            self.assertEqual("anti_collapse_progress_v1", env_config["dense_reward_mode"])
            self.assertEqual(1, env_config["max_actions_per_step"])
            self.assertEqual(0.01, env_config["progress_reward"])
            self.assertEqual(-0.01, env_config["regress_penalty"])
            self.assertEqual(-0.01, env_config["repeat_action_penalty"])
            self.assertEqual(3, env_config["repeat_action_start"])
            self.assertEqual(-0.03, env_config["repeat_action_penalty_cap"])
            self.assertEqual(-0.05, env_config["invalid_action_penalty"])

    def test_dense_action_penalty_env_config_strengthens_only_action_penalties(self):
        import yaml

        config_path = REPO_ROOT / "scripts/examples/vagen_base/navigation_vagen1/env_config_dense_action_penalty.yaml"
        config = yaml.safe_load(config_path.read_text())

        self.assertEqual({"env1", "env2"}, set(config))
        for env_spec in config.values():
            env_config = env_spec["env_config"]
            self.assertEqual("anti_collapse_progress_v1", env_config["dense_reward_mode"])
            self.assertEqual(1, env_config["max_actions_per_step"])
            self.assertEqual(0.01, env_config["progress_reward"])
            self.assertEqual(-0.01, env_config["regress_penalty"])
            self.assertEqual(-0.02, env_config["repeat_action_penalty"])
            self.assertEqual(3, env_config["repeat_action_start"])
            self.assertEqual(-0.06, env_config["repeat_action_penalty_cap"])
            self.assertEqual(-0.08, env_config["invalid_action_penalty"])

    def test_dense_guard_env_config_uses_stronger_invalid_action_penalty(self):
        import yaml

        config_path = REPO_ROOT / "scripts/examples/vagen_base/navigation_vagen1/env_config_dense_guard.yaml"
        config = yaml.safe_load(config_path.read_text())

        self.assertEqual({"env1", "env2"}, set(config))
        for env_spec in config.values():
            env_config = env_spec["env_config"]
            self.assertEqual("anti_collapse_progress_v1", env_config["dense_reward_mode"])
            self.assertEqual(1, env_config["max_actions_per_step"])
            self.assertEqual(0.01, env_config["progress_reward"])
            self.assertEqual(-0.01, env_config["regress_penalty"])
            self.assertEqual(-0.02, env_config["repeat_action_penalty"])
            self.assertEqual(3, env_config["repeat_action_start"])
            self.assertEqual(-0.06, env_config["repeat_action_penalty_cap"])
            self.assertEqual(-0.12, env_config["invalid_action_penalty"])

    def test_dense_ac_env_configs_use_v2_reward_terms(self):
        import yaml

        expected = {
            "mild": (-0.01, -0.03, -0.10, -0.01, -0.005, 0.85),
            "base": (-0.02, -0.06, -0.12, -0.02, -0.01, 0.85),
            "strong": (-0.03, -0.09, -0.15, -0.03, -0.02, 0.80),
        }

        for name, values in expected.items():
            config_path = REPO_ROOT / f"scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_{name}.yaml"
            config = yaml.safe_load(config_path.read_text())
            repeat_penalty, repeat_cap, invalid_penalty, stagnation_penalty, top_penalty, top_threshold = values

            self.assertEqual({"env1", "env2"}, set(config))
            for env_spec in config.values():
                env_config = env_spec["env_config"]
                self.assertEqual("anti_collapse_progress_v2", env_config["dense_reward_mode"])
                self.assertEqual(1, env_config["max_actions_per_step"])
                self.assertEqual(0.01, env_config["progress_reward"])
                self.assertEqual(-0.01, env_config["regress_penalty"])
                self.assertEqual(repeat_penalty, env_config["repeat_action_penalty"])
                self.assertEqual(3, env_config["repeat_action_start"])
                self.assertEqual(repeat_cap, env_config["repeat_action_penalty_cap"])
                self.assertEqual(invalid_penalty, env_config["invalid_action_penalty"])
                self.assertEqual(stagnation_penalty, env_config["stagnation_repeat_penalty"])
                self.assertEqual(3, env_config["stagnation_repeat_start"])
                self.assertEqual(0.02, env_config["stagnation_delta_eps"])
                self.assertEqual(5, env_config["action_balance_min_steps"])
                self.assertEqual(top_threshold, env_config["action_top_share_penalty_threshold"])
                self.assertEqual(top_penalty, env_config["action_top_share_penalty"])
                self.assertEqual(-0.02 if name == "base" else (-0.01 if name == "mild" else -0.03), env_config["all_same_traj_penalty"])

    def test_dense_ac_submit_script_launches_three_6h_variants(self):
        submit_script = (
            REPO_ROOT / "scripts/superpod/submit_navigation_vagen1_ac_turn20_ctx8192_6h.sh"
        ).read_text()
        env_source = (REPO_ROOT / "vagen/env/navigation/env.py").read_text()

        self.assertIn("--time=06:00:00", submit_script)
        self.assertIn("SLURM_EXCLUDE_NODES=${SLURM_EXCLUDE_NODES:-dgx-26,dgx-32,dgx-35,dgx-37}", submit_script)
        self.assertIn('--exclude="$SLURM_EXCLUDE_NODES"', submit_script)
        self.assertIn("TOTAL_TRAINING_STEPS=30", submit_script)
        self.assertIn("MAX_TURNS=20", submit_script)
        self.assertIn("ROLLOUT_MAX_TRAJECTORY_LENGTH=8192", submit_script)
        self.assertIn("SERVER_NAVIGATION_MAX_WORKERS=1", submit_script)
        self.assertIn("ROLLOUT_MINI_BATCH_SIZE=2", submit_script)
        self.assertIn("LOSS_MASK_MODE=default", submit_script)
        self.assertIn("FORMAT_REWARD=0.05", submit_script)
        for name in ["mild", "base", "strong"]:
            self.assertIn(f"env_config_dense_ac_{name}.yaml", submit_script)
        self.assertIn("dense_stagnation_repeat_penalty", env_source)
        self.assertIn("dense_action_balance_penalty", env_source)

    def test_ac_base_lite_guarded_config_and_submit_script(self):
        import yaml

        config_path = REPO_ROOT / "scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_base_lite_guarded.yaml"
        config = yaml.safe_load(config_path.read_text())
        submit_script = (
            REPO_ROOT / "scripts/superpod/submit_navigation_vagen1_ac_base_lite_guarded_8gpu_60step.sh"
        ).read_text()

        self.assertEqual({"env1", "env2"}, set(config))
        for env_spec in config.values():
            env_config = env_spec["env_config"]
            self.assertEqual("anti_collapse_progress_v2", env_config["dense_reward_mode"])
            self.assertEqual(1, env_config["max_actions_per_step"])
            self.assertEqual(0.01, env_config["progress_reward"])
            self.assertEqual(-0.01, env_config["regress_penalty"])
            self.assertEqual(-0.015, env_config["repeat_action_penalty"])
            self.assertEqual(3, env_config["repeat_action_start"])
            self.assertEqual(-0.04, env_config["repeat_action_penalty_cap"])
            self.assertEqual(-0.08, env_config["invalid_action_penalty"])
            self.assertEqual(-0.015, env_config["stagnation_repeat_penalty"])
            self.assertEqual(-0.04, env_config["stagnation_repeat_penalty_cap"])
            self.assertEqual(-0.01, env_config["action_top_share_penalty"])
            self.assertEqual(-0.015, env_config["all_same_traj_penalty"])

        self.assertIn("--time=12:00:00", submit_script)
        self.assertIn("TOTAL_TRAINING_STEPS=60", submit_script)
        self.assertIn("TEST_FREQ=15", submit_script)
        self.assertIn("SAVE_FREQ=15", submit_script)
        self.assertIn("REMOVE_PREVIOUS_CKPT_IN_SAVE=True", submit_script)
        self.assertIn("SAVE_CRITIC_CKPT=False", submit_script)
        self.assertIn("SAVE_OPTIMIZER_CKPT=False", submit_script)
        self.assertIn("SERVER_NAVIGATION_MAX_WORKERS=1", submit_script)
        self.assertIn("ROLLOUT_MINI_BATCH_SIZE=2", submit_script)
        self.assertIn("LOSS_MASK_MODE=default", submit_script)
        self.assertIn("FORMAT_REWARD=0.05", submit_script)
        self.assertIn("env_config_dense_ac_base_lite_guarded.yaml", submit_script)
        self.assertIn("actor-only", submit_script)
        self.assertIn("no-optimizer", submit_script)

    def test_ac_guarded_success_and_diversity_variants(self):
        import yaml

        expected = {
            "success": {
                "repeat_action_penalty": -0.01,
                "repeat_action_penalty_cap": -0.03,
                "invalid_action_penalty": -0.08,
                "stagnation_repeat_penalty": -0.01,
                "stagnation_repeat_penalty_cap": -0.03,
                "action_top_share_penalty_threshold": 0.85,
                "action_top_share_penalty": -0.005,
                "all_same_traj_penalty": -0.01,
            },
            "diversity": {
                "repeat_action_penalty": -0.015,
                "repeat_action_penalty_cap": -0.04,
                "invalid_action_penalty": -0.10,
                "stagnation_repeat_penalty": -0.015,
                "stagnation_repeat_penalty_cap": -0.04,
                "action_top_share_penalty_threshold": 0.80,
                "action_top_share_penalty": -0.015,
                "all_same_traj_penalty": -0.02,
            },
        }

        for name, values in expected.items():
            config_path = REPO_ROOT / f"scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_{name}_guarded.yaml"
            submit_path = REPO_ROOT / f"scripts/superpod/submit_navigation_vagen1_ac_{name}_guarded_8gpu_60step.sh"
            config = yaml.safe_load(config_path.read_text())
            submit_script = submit_path.read_text()

            self.assertEqual({"env1", "env2"}, set(config))
            for env_spec in config.values():
                env_config = env_spec["env_config"]
                self.assertEqual("anti_collapse_progress_v2", env_config["dense_reward_mode"])
                self.assertEqual(1, env_config["max_actions_per_step"])
                self.assertEqual(0.01, env_config["progress_reward"])
                self.assertEqual(-0.01, env_config["regress_penalty"])
                for key, value in values.items():
                    self.assertEqual(value, env_config[key])

            self.assertIn("--time=12:00:00", submit_script)
            self.assertIn("TOTAL_TRAINING_STEPS=60", submit_script)
            self.assertIn("TEST_FREQ=15", submit_script)
            self.assertIn("SAVE_FREQ=15", submit_script)
            self.assertIn("REMOVE_PREVIOUS_CKPT_IN_SAVE=True", submit_script)
            self.assertIn("SERVER_NAVIGATION_MAX_WORKERS=1", submit_script)
            self.assertIn("ROLLOUT_MINI_BATCH_SIZE=2", submit_script)
            self.assertIn("LOSS_MASK_MODE=default", submit_script)
            self.assertIn("FORMAT_REWARD=0.05", submit_script)

    def test_498593_infer_merges_hf_model_to_node_local_storage(self):
        script = (
            REPO_ROOT / "scripts/superpod/run_navigation_498593_step30_full_infer_local_1gpu.sbatch"
        ).read_text()

        self.assertIn("MERGE_ACTOR_DIR=${MERGE_ACTOR_DIR:-$VAGEN_NODE_LOCAL_ROOT/actor_for_hf_merge}", script)
        self.assertIn('HF_DIR="$MERGE_ACTOR_DIR/huggingface"', script)
        self.assertIn("merging FSDP shards to node-local", script)
        self.assertIn('ln -s {} "$MERGE_ACTOR_DIR/"', script)
        self.assertIn('cp -a {} "$HF_DIR/"', script)
        self.assertIn('--model_path "$MODEL_PATH"', script)
        self.assertNotIn("MODEL_LINK", script)

    def test_eager_guard_variant_is_recognized(self):
        variant_script = (REPO_ROOT / "scripts/superpod/configure_navigation_vagen1_variant.sh").read_text()

        self.assertIn("eager_guard)", variant_script)
        self.assertIn("set_eager_runtime 0.4 8", variant_script)

    def test_rollout_logging_skips_non_numeric_turn_metrics(self):
        for manager_path in [
            REPO_ROOT / "vagen/rollout/qwen_rollout/rollout_manager.py",
            REPO_ROOT / "vagen/rollout/qwen_rollout/rollout_manager_service.py",
        ]:
            script = manager_path.read_text()
            self.assertIn("all(isinstance(item, (int, float, bool)) for item in v)", script)

    def test_trainer_final_validation_can_be_disabled_for_speed_debug(self):
        config = (REPO_ROOT / "vagen/trainer/config/ppo_trainer.yaml").read_text()
        trainer = (REPO_ROOT / "vagen/trainer/ppo/ray_trainer.py").read_text()
        run_script = (REPO_ROOT / "scripts/examples/vagen_base/navigation/run.sh").read_text()

        self.assertIn("final_val_after_train: True", config)
        self.assertIn("self.config.trainer.get('final_val_after_train', True)", trainer)
        self.assertIn("FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-True}", run_script)
        self.assertIn("trainer.final_val_after_train=$FINAL_VAL_AFTER_TRAIN", run_script)

    def test_trainer_saves_before_validation_on_shared_frequency_steps(self):
        trainer = (REPO_ROOT / "vagen/trainer/ppo/ray_trainer.py").read_text()
        marker_pos = trainer.index("# validate after checkpointing")
        save_pos = trainer.rindex("self._save_checkpoint()", 0, marker_pos)
        validate_pos = trainer.index("self._validate()", marker_pos)

        self.assertLess(save_pos, validate_pos)

    def test_actor_only_checkpoint_switches_are_wired(self):
        config = (REPO_ROOT / "vagen/trainer/config/ppo_trainer.yaml").read_text()
        trainer = (REPO_ROOT / "vagen/trainer/ppo/ray_trainer.py").read_text()
        run_script = (REPO_ROOT / "scripts/examples/vagen_base/navigation/run.sh").read_text()
        local4 = (REPO_ROOT / "scripts/superpod/run_navigation_vagen1_4gpu.sbatch").read_text()
        patch_script = (REPO_ROOT / "scripts/superpod/patch_verl_lightweight_checkpoint.sh").read_text()

        self.assertIn("save_critic_checkpoint: True", config)
        self.assertIn("save_critic_checkpoint', True", trainer)
        self.assertIn("Skipping critic checkpoint", trainer)
        self.assertIn("SAVE_CRITIC_CKPT=${SAVE_CRITIC_CKPT:-True}", run_script)
        self.assertIn("SAVE_OPTIMIZER_CKPT=${SAVE_OPTIMIZER_CKPT:-True}", run_script)
        self.assertIn('export VERL_SAVE_OPTIMIZER_CKPT="$SAVE_OPTIMIZER_CKPT"', run_script)
        self.assertIn("trainer.save_critic_checkpoint=$SAVE_CRITIC_CKPT", run_script)
        self.assertIn("patch_verl_lightweight_checkpoint.sh", local4)
        self.assertIn("VERL_SAVE_OPTIMIZER_CKPT", patch_script)
        self.assertIn("Skipping optimizer checkpoint", patch_script)

    def test_local_sbatch_copies_node_local_ray_logs_on_exit(self):
        local4 = (REPO_ROOT / "scripts/superpod/run_navigation_vagen1_4gpu.sbatch").read_text()

        self.assertIn("copy_vagen_node_local_logs", local4)
        self.assertIn('logs/node-local-${EXPERIMENT_NAME}-${SLURM_JOB_ID:-manual}', local4)
        self.assertIn('cp -a "$RAY_TMPDIR"', local4)
        self.assertIn("trap cleanup_navigation_vagen1_job EXIT", local4)

    def test_single_action_prompt_forbids_stop_like_completion_words(self):
        prompt = (REPO_ROOT / "vagen/env/navigation/prompt.py").read_text()

        self.assertIn("do not stop or declare completion", prompt)
        self.assertIn("finish, complete, and success are invalid answers", prompt)

    def test_ray_uses_configured_tmpdir_for_local_clusters(self):
        main_ppo = (REPO_ROOT / "vagen/trainer/main_ppo.py").read_text()
        local8 = (REPO_ROOT / "scripts/superpod/run_navigation_vagen1_8gpu_local_save5.sbatch").read_text()
        local4 = (REPO_ROOT / "scripts/superpod/run_navigation_vagen1_4gpu.sbatch").read_text()

        self.assertIn('os.environ.get("RAY_TMPDIR")', main_ppo)
        self.assertIn('ray_kwargs["_temp_dir"] = ray_tmpdir', main_ppo)
        for script in (local8, local4):
            self.assertIn("VAGEN_NODE_LOCAL_ROOT", script)
            self.assertIn("export RAY_TMPDIR=$VAGEN_NODE_LOCAL_ROOT/ray", script)
            self.assertNotIn("RAY_TMPDIR=${RAY_TMPDIR:-", script)
            self.assertIn("/tmp/${USER:-hligb}/vagen-navigation", script)

    def test_ray_trainer_logs_global_and_config_specific_metrics(self):
        trainer_source = (REPO_ROOT / "vagen/trainer/ppo/ray_trainer.py").read_text()

        self.assertIn("all_metrics_by_name", trainer_source)
        self.assertIn("metric_dict[f'{mode}/{k}'] = np.mean(v)", trainer_source)
        self.assertIn("metric_dict[f'{mode}/{k}/{config_id}'] = np.mean(v)", trainer_source)


if __name__ == "__main__":
    unittest.main()
