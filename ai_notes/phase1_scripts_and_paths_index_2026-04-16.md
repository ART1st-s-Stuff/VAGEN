# Phase1 scripts and paths index (2026-04-16)

- Scope:
  - Consolidate commonly used Phase1 training/eval/debug scripts and key directories.
  - Add the latest input-inspection script for collapse debugging.

- Core directories:
  - Phase1 config: `vagen/configs/navigation_sft_phase1.yaml`
  - Training entry scripts: `examples/train/navigation/`
  - Test/debug scripts: `scripts/test/`
  - Checkpoints root: `checkpoints/navigation-phase1-sft/`
  - Typical merged/eval outputs: `outputs/phase1_merged_models/`, `outputs/phase1_rollout_eval/`
  - Phase1 debug dataset example: `datasets/phase1_debug/`

- Training scripts:
  - `examples/train/navigation/train_sft_phase1.sh`
    - Main Phase1 SFT launcher (torchrun/srun/auto).
    - Uses `--config-name navigation_sft_phase1`.
  - `examples/train/navigation/srun_train_sft_phase1.sh`
    - SLURM wrapper for launching Phase1 training.

- Eval scripts:
  - `scripts/test/phase1_rollout_testset_and_eval.py`
    - Step-level rollout evaluation from EB-Nav trajectories.
  - `scripts/test/run_phase1_rollout_eval.sh`
    - Shell wrapper for rollout eval.
  - `scripts/test/run_phase1_rollout_eval_from_checkpoints.sh`
    - One-click path: merge checkpoint -> run rollout eval.
  - `scripts/test/debug_phase1_single_sample.py`
    - Single-sample generation/debug with top-k logit inspection.

- Data/model prep scripts:
  - `scripts/test/init_phase1_data.sh`
  - `scripts/test/init_phase1_model.sh`
  - `scripts/test/run_phase1_smoke_test.sh`

- New debugging script (added this session):
  - `scripts/test/inspect_phase1_training_input.py`
    - Purpose: inspect what a Phase1 sample looks like before model forward.
    - Prints:
      - raw and normalized `messages`
      - chat-template text (escaped view to make `\n` visible)
      - `input_ids` / `loss_mask` / `attention_mask` stats
      - supervised-token decoded preview (`loss_mask == 1`)
      - multimodal input keys passed into model
    - Includes repo-root `sys.path` bootstrap to avoid `ModuleNotFoundError: vagen`
      when executed outside repo root.

- Known quick checks for "newline-only collapse":
  - Verify `num_supervised_tokens` is non-trivial in sampled items.
  - Verify `supervised_prefix_visible` is not mostly escaped newlines (`\\n`).
  - Compare `chat_template_text_visible` and `loss_mask` alignment for a few random indices.
