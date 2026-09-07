import re
import unittest
import importlib.util
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = REPO_ROOT / "vagen/env/navigation/prompt.py"
PROMPT_SPEC = importlib.util.spec_from_file_location("navigation_prompt", PROMPT_PATH)
PROMPT_MODULE = importlib.util.module_from_spec(PROMPT_SPEC)
PROMPT_SPEC.loader.exec_module(PROMPT_MODULE)
format_prompt = PROMPT_MODULE.format_prompt


class NavigationVagen1MinimalTest(unittest.TestCase):
    def test_vagen1_env_config_sets_single_action_for_all_splits(self):
        config_path = REPO_ROOT / "scripts/examples/vagen_base/navigation_vagen1/env_config.yaml"
        config = yaml.safe_load(config_path.read_text())

        self.assertEqual({"env1", "env2"}, set(config))
        for env_spec in config.values():
            env_config = env_spec["env_config"]
            self.assertEqual("navigation", env_spec["env_name"])
            self.assertEqual(1, env_config["max_actions_per_step"])
            self.assertFalse(env_config["use_state_reward"])
            self.assertEqual("grounding_worldmodeling", env_config["prompt_format"])
            self.assertEqual(0.02, env_config["format_reward"])
            self.assertEqual(-0.2, env_config["invalid_action_penalty"])
            self.assertEqual(1.5, env_config["success_threshold"])

    def test_single_action_prompt_uses_single_action_instruction_and_example(self):
        prompt = format_prompt["grounding_worldmodeling"](
            max_actions_per_step=1,
            action_sep=",",
        )

        self.assertIn("exactly one action", prompt.lower())
        self.assertNotIn("multiple actions", prompt.lower())
        answer_examples = re.findall(r"<answer>(.*?)</answer>", prompt, flags=re.DOTALL)
        self.assertTrue(answer_examples)
        for answer in answer_examples:
            self.assertNotIn(",", answer)

    def test_single_action_system_prompt_does_not_teach_action_chunks(self):
        prompt = PROMPT_MODULE.system_prompt(
            format="grounding_worldmodeling",
            max_actions_per_step=1,
            action_sep=",",
        )

        self.assertIn("exactly one action", prompt.lower())
        self.assertIn("<think><observation>", prompt)
        self.assertIn("<answer>moveahead</answer>", prompt)
        self.assertIn("do not write anything before <think>", prompt.lower())
        self.assertIn("There is no stay, stop, end, done, terminate, wait, noop action", prompt)
        self.assertIn("moveahead, moveback, moveright, moveleft, rotateright, rotateleft, lookup, lookdown", prompt)
        self.assertIn("do not stop", prompt.lower())
        self.assertNotIn("moveahead, moveahead", prompt)
        self.assertNotIn("multiple actions", prompt.lower())

    def test_original_step5_prompt_keeps_multi_action_instruction(self):
        prompt = format_prompt["grounding_worldmodeling"](
            max_actions_per_step=5,
            action_sep=",",
        )

        self.assertIn("up to 5 action", prompt.lower())
        self.assertIn("moveahead,moveahead", prompt.replace(" ", ""))

    def test_vagen1_run_script_uses_vagen1_config_and_equivalent_horizon(self):
        run_path = REPO_ROOT / "scripts/examples/vagen_base/navigation_vagen1/run.sh"
        script = run_path.read_text()
        base_run = (REPO_ROOT / "scripts/examples/vagen_base/navigation/run.sh").read_text()

        self.assertIn("navigation_vagen1", script)
        self.assertIn("MAX_TURNS=${MAX_TURNS:-20}", script)
        self.assertIn("ROLLOUT_WINDOW_SIZE=${ROLLOUT_WINDOW_SIZE:-5}", script)
        self.assertIn("UPDATE_WINDOW_SIZE=${UPDATE_WINDOW_SIZE:-5}", script)
        self.assertIn("MAX_TRAJECTORY_LENGTH=${MAX_TRAJECTORY_LENGTH:-16000}", script)
        self.assertIn("ROLLOUT_MAX_TRAJECTORY_LENGTH=${ROLLOUT_MAX_TRAJECTORY_LENGTH:-5000}", script)
        self.assertIn("ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}", script)
        self.assertIn("ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}", script)
        self.assertIn("ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}", script)
        self.assertIn("ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}", script)
        self.assertIn("ROLLOUT_LIMIT_MM_PER_PROMPT=${ROLLOUT_LIMIT_MM_PER_PROMPT:-8}", script)
        self.assertIn("ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}", script)
        self.assertIn("FORMAT_REWARD=${FORMAT_REWARD:-0.02}", script)
        self.assertIn("LOSS_MASK_MODE=${LOSS_MASK_MODE:-default}", script)
        self.assertIn("FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-True}", script)
        self.assertIn("../navigation/run.sh", script)
        self.assertIn("MAX_TURNS=${MAX_TURNS:-5}", base_run)
        self.assertIn("ROLLOUT_WINDOW_SIZE=${ROLLOUT_WINDOW_SIZE:-5}", base_run)
        self.assertIn("UPDATE_WINDOW_SIZE=${UPDATE_WINDOW_SIZE:-null}", base_run)
        self.assertIn("ROLLOUT_BASE_URL=${ROLLOUT_BASE_URL:-http://localhost:$SERVER_PORT}", base_run)
        self.assertIn("ROLLOUT_LIMIT_MM_PER_PROMPT=${ROLLOUT_LIMIT_MM_PER_PROMPT:-5}", base_run)
        self.assertIn("ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}", base_run)
        self.assertIn("ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}", base_run)
        self.assertIn("ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}", base_run)
        self.assertIn("ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}", base_run)
        self.assertIn("TRAIN_NNODES=${TRAIN_NNODES:-1}", base_run)
        self.assertIn("RAY_INIT_ADDRESS=${RAY_INIT_ADDRESS:-${RAY_ADDRESS:-}}", base_run)
        self.assertIn("MAX_TRAJECTORY_LENGTH=${MAX_TRAJECTORY_LENGTH:-5000}", base_run)
        self.assertIn("ROLLOUT_MAX_TRAJECTORY_LENGTH=${ROLLOUT_MAX_TRAJECTORY_LENGTH:-$MAX_TRAJECTORY_LENGTH}", base_run)
        self.assertIn("data.max_trajectory_length=$MAX_TRAJECTORY_LENGTH", base_run)
        self.assertIn("actor_rollout_ref.rollout.max_trajectory_length=$ROLLOUT_MAX_TRAJECTORY_LENGTH", base_run)
        self.assertIn("actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_TRAJECTORY_LENGTH", base_run)
        self.assertIn("VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}", base_run)
        self.assertIn("rollout_manager.max_turns=$MAX_TURNS", base_run)
        self.assertIn("rollout_manager.window_size=$ROLLOUT_WINDOW_SIZE", base_run)
        self.assertIn("+rollout_manager.update_window_size=$UPDATE_WINDOW_SIZE", base_run)
        self.assertIn("actor_rollout_ref.rollout.limit_mm_per_prompt=$ROLLOUT_LIMIT_MM_PER_PROMPT", base_run)
        self.assertIn("actor_rollout_ref.rollout.enable_chunked_prefill=$ROLLOUT_ENABLE_CHUNKED_PREFILL", base_run)
        self.assertIn("actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER", base_run)
        self.assertIn("actor_rollout_ref.rollout.free_cache_engine=$ROLLOUT_FREE_CACHE_ENGINE", base_run)
        self.assertIn("actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS", base_run)
        self.assertIn("trainer.nnodes=$TRAIN_NNODES", base_run)
        self.assertIn("trainer.val_before_train=$VAL_BEFORE_TRAIN", base_run)
        self.assertIn("trainer.final_val_after_train=$FINAL_VAL_AFTER_TRAIN", base_run)
        self.assertIn('rollout_manager.base_url="$ROLLOUT_BASE_URL"', base_run)

    def test_vagen1_4gpu_sbatch_requests_24h_and_targets_vagen1(self):
        sbatch_path = REPO_ROOT / "scripts/superpod/run_navigation_vagen1_4gpu.sbatch"
        script = sbatch_path.read_text()

        self.assertIn("#SBATCH --time=24:00:00", script)
        self.assertIn("navigation_vagen1", script)
        self.assertIn("SERVER_USE_STATE_REWARD=${SERVER_USE_STATE_REWARD:-False}", script)
        self.assertIn("SERVER_PREWARM_AI2THOR=${SERVER_PREWARM_AI2THOR:-0}", script)
        self.assertIn("SERVER_RENDER_PROBE_AI2THOR=${SERVER_RENDER_PROBE_AI2THOR:-1}", script)
        self.assertIn("SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-1}", script)
        self.assertIn("N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}", script)
        self.assertIn("VAGEN1_VARIANT=${VAGEN1_VARIANT:-manual}", script)
        self.assertIn("ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}", script)
        self.assertIn("VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}", script)
        self.assertIn("ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-1}", script)
        self.assertIn("ROLLOUT_LIMIT_MM_PER_PROMPT=${ROLLOUT_LIMIT_MM_PER_PROMPT:-8}", script)
        self.assertIn("ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}", script)
        self.assertIn("ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}", script)
        self.assertIn("ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}", script)
        self.assertIn("ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}", script)
        self.assertIn("FORMAT_REWARD=${FORMAT_REWARD:-0.02}", script)
        self.assertIn("LOSS_MASK_MODE=${LOSS_MASK_MODE:-default}", script)
        self.assertIn("FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-True}", script)
        self.assertIn("configure_navigation_vagen1_variant.sh", script)
        self.assertIn("VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}", script)
        self.assertIn("SAVE_FREQ=${SAVE_FREQ:-20}", script)
        self.assertIn("TEST_FREQ=${TEST_FREQ:-20}", script)
        self.assertIn("MAX_TURNS=${MAX_TURNS:-20}", script)
        self.assertIn("ROLLOUT_WINDOW_SIZE=${ROLLOUT_WINDOW_SIZE:-5}", script)
        self.assertIn("UPDATE_WINDOW_SIZE=${UPDATE_WINDOW_SIZE:-5}", script)
        self.assertIn("MAX_TRAJECTORY_LENGTH=${MAX_TRAJECTORY_LENGTH:-16000}", script)
        self.assertIn("ROLLOUT_MAX_TRAJECTORY_LENGTH=${ROLLOUT_MAX_TRAJECTORY_LENGTH:-5000}", script)

    def test_local_server_runs_direct_render_probe_before_starting(self):
        server_path = REPO_ROOT / "scripts/superpod/start_local_server.sh"
        script = server_path.read_text()

        self.assertIn("SERVER_RENDER_PROBE_AI2THOR=${SERVER_RENDER_PROBE_AI2THOR:-1}", script)
        self.assertIn("SERVER_RENDER_PROBE_TIMEOUT=${SERVER_RENDER_PROBE_TIMEOUT:-150}", script)
        self.assertIn("python -m vagen.utils.navigation_direct_render_probe", script)
        self.assertIn("render_probe_ok_gpu=", script)
        self.assertIn("render_probe_failed_gpu=", script)
        self.assertIn("ERROR: no AI2-THOR render-capable GPU remained", script)
        self.assertIn("SERVER_NAVIGATION_DEVICES=\"$good_devices\"", script)
        self.assertLess(
            script.index("python -m vagen.utils.navigation_direct_render_probe"),
            script.index("python -m vagen.server.server"),
        )
        self.assertIn(">> \"$SERVER_LOG\" 2>&1 &", script)

    def test_split_server_sbatch_uses_two_normal_nodes_and_remote_base_url(self):
        sbatch_path = REPO_ROOT / "scripts/superpod/run_navigation_vagen1_4gpu_split_server.sbatch"
        script = sbatch_path.read_text()

        self.assertIn("#SBATCH --partition=normal", script)
        self.assertIn("#SBATCH --qos=normal_qos", script)
        self.assertIn("#SBATCH --nodes=2", script)
        self.assertIn("AI2THOR_SERVER_NODE", script)
        self.assertIn("TRAIN_NODE", script)
        self.assertIn("ROLLOUT_BASE_URL=${ROLLOUT_BASE_URL:-http://$AI2THOR_SERVER_NODE:$SERVER_PORT}", script)
        self.assertIn("source scripts/superpod/start_local_server.sh", script)
        self.assertIn("bash scripts/examples/vagen_base/navigation_vagen1/run.sh", script)

    def test_external_server_pair_uses_ready_file_and_remote_base_url(self):
        server_script = (REPO_ROOT / "scripts/superpod/run_navigation_vagen1_ai2thor_server.sbatch").read_text()
        trainer_script = (REPO_ROOT / "scripts/superpod/run_navigation_vagen1_4gpu_external_server.sbatch").read_text()

        self.assertIn("#SBATCH --partition=normal", server_script)
        self.assertIn("#SBATCH --partition=normal", trainer_script)
        self.assertIn("#SBATCH --exclusive", server_script)
        self.assertIn("SERVER_READY_FILE", server_script)
        self.assertIn("SERVER_ADVERTISE_HOST", server_script)
        self.assertIn("ROLLOUT_BASE_URL=http://$SERVER_ADVERTISE_HOST:$SERVER_PORT", server_script)
        self.assertIn("AI2THOR_SERVER_NAVIGATION_MAX_WORKERS", server_script)
        self.assertIn("SERVER_SESSION_ID", trainer_script)
        self.assertIn("SERVER_READY_FILES", trainer_script)
        self.assertIn("source \"$ready_file\"", trainer_script)
        self.assertIn("AI2-THOR server is reachable", trainer_script)
        self.assertIn("configure_navigation_vagen1_variant.sh", trainer_script)
        self.assertIn("VAGEN1_VARIANT=${VAGEN1_VARIANT:-manual}", trainer_script)
        self.assertIn("ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}", trainer_script)
        self.assertIn("ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-16}", trainer_script)
        self.assertIn("FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-False}", trainer_script)
        self.assertIn("env_config_speed.yaml", trainer_script)
        self.assertIn('rollout_manager.base_url="$ROLLOUT_BASE_URL"', (REPO_ROOT / "scripts/examples/vagen_base/navigation/run.sh").read_text())

    def test_two_node_external_training_sbatch_starts_ray_cluster(self):
        sbatch_path = REPO_ROOT / "scripts/superpod/run_navigation_vagen1_2node4gpu_external_server.sbatch"
        script = sbatch_path.read_text()

        self.assertIn("#SBATCH --partition=normal", script)
        self.assertIn("#SBATCH --qos=normal_qos", script)
        self.assertIn("#SBATCH --nodes=2", script)
        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertIn("SERVER_READY_FILES", script)
        self.assertIn("ROLLOUT_BASE_URLS", script)
        self.assertIn("ray start --head", script)
        self.assertIn("ray start --address=\"$RAY_HEAD_ADDRESS\"", script)
        self.assertIn("EXPECTED_RAY_GPUS=${EXPECTED_RAY_GPUS:-8}", script)
        self.assertIn("TRAIN_NNODES=${TRAIN_NNODES:-2}", script)
        self.assertIn("RAY_INIT_ADDRESS=\"$RAY_HEAD_ADDRESS\"", script)
        self.assertIn("bash scripts/examples/vagen_base/navigation_vagen1/run.sh", script)

    def test_integrated_5node_sbatch_co_schedules_env_and_train(self):
        sbatch_path = REPO_ROOT / "scripts/superpod/run_navigation_vagen1_5node_integrated.sbatch"
        script = sbatch_path.read_text()

        self.assertIn("#SBATCH --nodes=5", script)
        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertIn("#SBATCH --time=12:00:00", script)
        self.assertIn("ENV_NNODES=${ENV_NNODES:-2}", script)
        self.assertIn("TRAIN_NNODES=${TRAIN_NNODES:-3}", script)
        self.assertIn("EXPECTED_RAY_GPUS=${EXPECTED_RAY_GPUS:-12}", script)
        self.assertIn("TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-96}", script)
        self.assertIn("PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-48}", script)
        self.assertIn("ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-48}", script)
        self.assertIn("TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-60}", script)
        self.assertIn("SAVE_FREQ=${SAVE_FREQ:-20}", script)
        self.assertIn("TEST_FREQ=${TEST_FREQ:-20}", script)
        self.assertIn("FORMAT_REWARD=${FORMAT_REWARD:-0.05}", script)
        self.assertIn("LOSS_MASK_MODE=${LOSS_MASK_MODE:-answer_only}", script)
        self.assertIn("start_env_server", script)
        self.assertIn("SERVER_READY_FILES", script)
        self.assertIn("AI2-THOR server is reachable", script)
        self.assertIn("TRAIN_NODES", script)
        self.assertIn("ray start --head", script)
        self.assertIn("cleanup_integrated_job", script)
        self.assertIn("bash scripts/examples/vagen_base/navigation_vagen1/run.sh", script)

    def test_integrated_3node_env2_train1_wrapper_uses_smaller_parameters(self):
        sbatch_path = REPO_ROOT / "scripts/superpod/run_navigation_vagen1_3node_env2_train1_integrated.sbatch"
        script = sbatch_path.read_text()

        self.assertIn("#SBATCH --nodes=3", script)
        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertIn("#SBATCH --time=02:00:00", script)
        self.assertIn("ENV_NNODES=${ENV_NNODES:-2}", script)
        self.assertIn("TRAIN_NNODES=${TRAIN_NNODES:-1}", script)
        self.assertIn("EXPECTED_RAY_GPUS=${EXPECTED_RAY_GPUS:-4}", script)
        self.assertIn("TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}", script)
        self.assertIn("PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}", script)
        self.assertIn("ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-16}", script)
        self.assertIn("TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-60}", script)
        self.assertIn("SAVE_FREQ=${SAVE_FREQ:-20}", script)
        self.assertIn("TEST_FREQ=${TEST_FREQ:-20}", script)
        self.assertIn("FORMAT_REWARD=${FORMAT_REWARD:-0.05}", script)
        self.assertIn("LOSS_MASK_MODE=${LOSS_MASK_MODE:-answer_only}", script)
        self.assertIn("navigation_vagen1_integrated_3node_env2_train1_answeronly_20260723", script)
        self.assertIn("run_navigation_vagen1_5node_integrated.sbatch", script)

    def test_integrated_2node_env1_train1_wrapper_uses_env4gpu_and_val5(self):
        sbatch_path = REPO_ROOT / "scripts/superpod/run_navigation_vagen1_2node_env1_train1_integrated.sbatch"
        script = sbatch_path.read_text()
        integrated_script = (
            REPO_ROOT / "scripts/superpod/run_navigation_vagen1_5node_integrated.sbatch"
        ).read_text()

        self.assertIn("#SBATCH --nodes=2", script)
        self.assertIn("#SBATCH --time=12:00:00", script)
        self.assertIn("ENV_NNODES=${ENV_NNODES:-1}", script)
        self.assertIn("TRAIN_NNODES=${TRAIN_NNODES:-1}", script)
        self.assertIn("ENV_GPUS_PER_NODE=${ENV_GPUS_PER_NODE:-4}", script)
        self.assertIn("TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}", script)
        self.assertIn("ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-2}", script)
        self.assertIn("TEST_FREQ=${TEST_FREQ:-5}", script)
        self.assertIn("RAW_SAMPLES_TO_LOG=${RAW_SAMPLES_TO_LOG:-8}", script)
        self.assertIn("SERVER_NAVIGATION_DEVICES=${SERVER_NAVIGATION_DEVICES:-0,1,2,3}", script)
        self.assertIn("SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-4}", script)
        self.assertIn("run_navigation_vagen1_5node_integrated.sbatch", script)
        self.assertIn("ENV_GPUS_PER_NODE", integrated_script)
        self.assertIn("--gres=gpu:$ENV_GPUS_PER_NODE", integrated_script)

    def test_local8_vagen1_wrapper_keeps_native_vagen_step1_baseline(self):
        sbatch_path = REPO_ROOT / "scripts/superpod/run_navigation_vagen1_8gpu_local_save5.sbatch"
        script = sbatch_path.read_text()

        self.assertIn("#SBATCH --gres=gpu:8", script)
        self.assertIn("#SBATCH --exclude=dgx-26,dgx-27,dgx-32,dgx-35,dgx-37,dgx-40", script)
        self.assertIn("#SBATCH --time=24:00:00", script)
        self.assertIn("N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}", script)
        self.assertIn("VAGEN1_VARIANT=${VAGEN1_VARIANT:-manual}", script)
        self.assertIn("ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-4}", script)
        self.assertIn("SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-2}", script)
        self.assertIn("TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}", script)
        self.assertIn("TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}", script)
        self.assertIn("PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}", script)
        self.assertIn("ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-1}", script)
        self.assertIn("SAVE_FREQ=${SAVE_FREQ:-5}", script)
        self.assertIn("TEST_FREQ=${TEST_FREQ:-20}", script)
        self.assertIn("ROLLOUT_MAX_TRAJECTORY_LENGTH=${ROLLOUT_MAX_TRAJECTORY_LENGTH:-6144}", script)
        self.assertIn("ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}", script)
        self.assertIn("ROLLOUT_LIMIT_MM_PER_PROMPT=${ROLLOUT_LIMIT_MM_PER_PROMPT:-6}", script)
        self.assertIn("ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}", script)
        self.assertIn("FORMAT_REWARD=${FORMAT_REWARD:-0.02}", script)
        self.assertIn("LOSS_MASK_MODE=${LOSS_MASK_MODE:-default}", script)
        self.assertIn("RAW_SAMPLES_TO_LOG=${RAW_SAMPLES_TO_LOG:-0}", script)
        self.assertIn("SERVER_RENDER_PROBE_AI2THOR=${SERVER_RENDER_PROBE_AI2THOR:-1}", script)
        self.assertIn("env_config.yaml", script)
        self.assertNotIn("env_config_speed.yaml", script)
        self.assertNotIn("answer_only", script)
        self.assertNotIn("fmt005", script)
        self.assertIn("run_navigation_vagen1_4gpu.sbatch", script)

    def test_local8_fmt01_and_dense_wrappers_reuse_local8_baseline(self):
        fmt01_script = (
            REPO_ROOT / "scripts/superpod/run_navigation_vagen1_8gpu_local_save5_fmt01.sbatch"
        ).read_text()
        dense_script = (
            REPO_ROOT / "scripts/superpod/run_navigation_vagen1_8gpu_local_save5_dense_light.sbatch"
        ).read_text()

        self.assertIn("#SBATCH --gres=gpu:8", fmt01_script)
        self.assertIn("#SBATCH --gres=gpu:8", dense_script)
        self.assertIn("FORMAT_REWARD=${FORMAT_REWARD:-0.1}", fmt01_script)
        self.assertIn("fmt01", fmt01_script)
        self.assertIn("FORMAT_REWARD=${FORMAT_REWARD:-0.05}", dense_script)
        self.assertIn("env_config_dense_light.yaml", dense_script)
        self.assertIn("dense-light-v1", dense_script)
        self.assertIn("run_navigation_vagen1_8gpu_local_save5.sbatch", fmt01_script)
        self.assertIn("run_navigation_vagen1_8gpu_local_save5.sbatch", dense_script)

    def test_turn10_4gpu_debug_wrapper_uses_collapse_probe_parameters(self):
        sbatch_path = REPO_ROOT / "scripts/superpod/run_navigation_vagen1_4gpu_turn10_debug.sbatch"
        script = sbatch_path.read_text()
        variant_script = (
            REPO_ROOT / "scripts/superpod/configure_navigation_vagen1_variant.sh"
        ).read_text()

        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertIn("#SBATCH --time=02:00:00", script)
        self.assertIn("VAGEN1_VARIANT=${VAGEN1_VARIANT:-turn10_defaultloss_fmt01}", script)
        self.assertIn("navigation_vagen1_turn10_4gpu_debug20_20260723", script)
        self.assertIn("TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-21}", script)
        self.assertIn("TEST_FREQ=${TEST_FREQ:-5}", script)
        self.assertIn("SAVE_FREQ=${SAVE_FREQ:-20}", script)
        self.assertIn("VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}", script)
        self.assertIn("FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-False}", script)
        self.assertIn("env_config_speed.yaml", script)
        self.assertIn("run_navigation_vagen1_4gpu.sbatch", script)

        self.assertIn("turn10_defaultloss_fmt01)", variant_script)
        self.assertIn("export MAX_TURNS=10", variant_script)
        self.assertIn("export TRAIN_BATCH_SIZE=16", variant_script)
        self.assertIn("export PPO_MINI_BATCH_SIZE=16", variant_script)
        self.assertIn("export ROLLOUT_MINI_BATCH_SIZE=4", variant_script)
        self.assertIn("export SERVER_NAVIGATION_MAX_WORKERS=2", variant_script)
        self.assertIn("export FORMAT_REWARD=0.1", variant_script)
        self.assertIn("export LOSS_MASK_MODE=default", variant_script)

    def test_turn10_answeronly_4gpu_debug_wrapper_lowers_env_pressure_without_singleton_batch(self):
        sbatch_path = REPO_ROOT / "scripts/superpod/run_navigation_vagen1_4gpu_turn10_answeronly_debug.sbatch"
        script = sbatch_path.read_text()
        variant_script = (
            REPO_ROOT / "scripts/superpod/configure_navigation_vagen1_variant.sh"
        ).read_text()

        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertIn("#SBATCH --time=02:00:00", script)
        self.assertIn("VAGEN1_VARIANT=${VAGEN1_VARIANT:-turn10_answeronly_fmt005_rmb2_w1}", script)
        self.assertIn("navigation_vagen1_turn10_answeronly_4gpu_debug5_20260723", script)
        self.assertIn("TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-6}", script)
        self.assertIn("TEST_FREQ=${TEST_FREQ:-5}", script)
        self.assertIn("SAVE_FREQ=${SAVE_FREQ:--1}", script)
        self.assertIn("VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}", script)
        self.assertIn("FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-False}", script)
        self.assertIn("env_config_speed.yaml", script)
        self.assertIn("run_navigation_vagen1_4gpu.sbatch", script)

        self.assertIn("turn10_answeronly_fmt005_rmb2_w1)", variant_script)
        self.assertIn("export MAX_TURNS=10", variant_script)
        self.assertIn("export TRAIN_BATCH_SIZE=16", variant_script)
        self.assertIn("export PPO_MINI_BATCH_SIZE=16", variant_script)
        self.assertIn("export ROLLOUT_MINI_BATCH_SIZE=2", variant_script)
        self.assertIn("export SERVER_NAVIGATION_MAX_WORKERS=1", variant_script)
        self.assertIn("export FORMAT_REWARD=0.05", variant_script)
        self.assertIn("export LOSS_MASK_MODE=answer_only", variant_script)

    def test_turn5_answeronly_4gpu_debug_wrapper_uses_short_horizon_probe_parameters(self):
        sbatch_path = REPO_ROOT / "scripts/superpod/run_navigation_vagen1_4gpu_turn5_answeronly_debug.sbatch"
        script = sbatch_path.read_text()
        variant_script = (
            REPO_ROOT / "scripts/superpod/configure_navigation_vagen1_variant.sh"
        ).read_text()

        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertIn("#SBATCH --time=02:00:00", script)
        self.assertIn("VAGEN1_VARIANT=${VAGEN1_VARIANT:-turn5_answeronly_fmt005_rmb2_w1}", script)
        self.assertIn("navigation_vagen1_turn5_answeronly_4gpu_debug5_20260723", script)
        self.assertIn("TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-6}", script)
        self.assertIn("TEST_FREQ=${TEST_FREQ:-5}", script)
        self.assertIn("SAVE_FREQ=${SAVE_FREQ:--1}", script)
        self.assertIn("VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}", script)
        self.assertIn("FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-False}", script)
        self.assertIn("env_config_speed.yaml", script)
        self.assertIn("run_navigation_vagen1_4gpu.sbatch", script)

        self.assertIn("turn5_answeronly_fmt005_rmb2_w1)", variant_script)
        self.assertIn("export MAX_TURNS=5", variant_script)
        self.assertIn("export TRAIN_BATCH_SIZE=16", variant_script)
        self.assertIn("export PPO_MINI_BATCH_SIZE=16", variant_script)
        self.assertIn("export ROLLOUT_MINI_BATCH_SIZE=2", variant_script)
        self.assertIn("export SERVER_NAVIGATION_MAX_WORKERS=1", variant_script)
        self.assertIn("export FORMAT_REWARD=0.05", variant_script)
        self.assertIn("export LOSS_MASK_MODE=answer_only", variant_script)

    def test_raw_samples_4gpu_debug_wrapper_enables_raw_logging(self):
        sbatch_path = REPO_ROOT / "scripts/superpod/run_navigation_vagen1_4gpu_raw_samples_debug.sbatch"
        script = sbatch_path.read_text()

        self.assertIn("#SBATCH --time=02:00:00", script)
        self.assertIn("RAW_SAMPLES_TO_LOG=${RAW_SAMPLES_TO_LOG:-8}", script)
        self.assertIn("TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-4}", script)
        self.assertIn("navigation_vagen1_raw_samples_4gpu_debug", script)
        self.assertIn("run_navigation_vagen1_4gpu.sbatch", script)

    def test_rollout_managers_use_optional_update_window(self):
        for manager_path in [
            REPO_ROOT / "vagen/rollout/qwen_rollout/rollout_manager.py",
            REPO_ROOT / "vagen/rollout/qwen_rollout/rollout_manager_service.py",
        ]:
            script = manager_path.read_text()
            self.assertIn('update_window_size = self.config.get("update_window_size", None)', script)
            self.assertIn("window_size=update_window_size", script)


if __name__ == "__main__":
    unittest.main()
