# Phase1 rollout eval script (2026-04-15)

- Added `scripts/test/phase1_rollout_testset_and_eval.py`.
- Script behavior:
  - Loads EB-Nav dataset files (single/multi/both).
  - Selects the **last N successful rollouts** (`--num-rollouts`, default 100).
  - Expands rollout trajectories into step-level eval samples with GT action token.
  - Runs model generation on each step sample.
  - Truncates output at first action token and checks prefix format:
    `<think>...</think><|latent|><|action_start|>`.
  - Reports:
    - `format_accuracy`
    - `action_prior_accuracy`
    - `action_prior_accuracy_on_parsed`
  - Dumps `summary.json` and `predictions.jsonl`.

- Default paths:
  - single-step: `datasets/navigation/eb-nav_dataset_single_step.json`
  - multi-step: `datasets/navigation/eb-nav_dataset_multi_step.json`
  - image root: `datasets/navigation/images`
  - output: `outputs/phase1_rollout_eval`
