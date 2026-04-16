# Phase1 one-click eval from checkpoints (2026-04-15)

- Added `scripts/test/run_phase1_rollout_eval_from_checkpoints.sh`.
- Purpose:
  - One-click path for users whose training output is FSDP shards under
    `checkpoints/navigation-phase1-sft/.../global_step_xxx`.
  - Auto-resolve latest experiment and latest checkpoint step.
  - Convert checkpoint to HF format via `verl/scripts/legacy_model_merger.py`.
  - Auto-detect:
    - merged full model in export dir, or
    - LoRA adapter in `export_dir/lora_adapter` with base model path.
  - Run `scripts/test/phase1_rollout_testset_and_eval.py` directly after merge.

- Key env vars:
  - `CHECKPOINT_ROOT`, `EXPERIMENT_NAME`, `STEP_DIR`
  - `BASE_MODEL_PATH`
  - `EXPORT_ROOT`, `FORCE_REMERGE`
  - `NUM_ROLLOUTS`, `INCLUDE_SOURCES`, `OUTPUT_DIR`
