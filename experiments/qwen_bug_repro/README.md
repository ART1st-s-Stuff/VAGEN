# qwen-bug-repro — VAGEN navigation smoke + vision prefix-invariance probe

Branch: `nimloth/qwen-bug-repro` (from `upstream/vagen-legacy`).
Purpose: verify the Qwen2.5-VL visual-encoder prefix/batch non-invariance
(found in SFT2) also affects VAGEN's train/rollout path, using the default
Qwen model + VAGEN-original WM prompt + paper-style multi-action settings.

## Settings (human-specified)

- Model: `Qwen/Qwen2.5-VL-3B-Instruct` (default, fresh — no checkpoint init)
- Prompt: `worldmodeling` (VAGEN-original WM prompt; the `wm` alias is a
  nimloth addition that does not exist on `upstream/vagen-legacy`)
- `max_actions_per_step=5`, `max_turns=4`, `window_size=4`
- Advantage: `bi_level_gae`, `high_level_gamma=0.95`, `gamma=1.0`, `lam=1.0`
- Data: 3 navigation (`eval_set=base`) samples; 2 PPO steps; `val_before_train`
- Resources: 1 node, 2 GPUs — GPU0 env server, GPU1 rollout/train

## Why a side worktree + side verl

`upstream/vagen-legacy` has the `verl` submodule commented out and no `verl`
in its tree. `compute_bi_level_gae_advantage_return` (and the legacy
`compute_gae_advantage_return_with_loss_mask`) live only in the **legacy verl**
commit `2b46f25` (vendored by nimloth), not in upstream/main's verl. So the
VAGEN branch stays pure `upstream/vagen-legacy`, and verl `2b46f25` is provided
as a side checkout pointed to by `PYTHONPATH` (not committed into the branch).

The smoke runs from a side VAGEN worktree
(`/project/peilab/atst/nimloth/external/VAGEN.qwen-bug-repro`) so the main
checkout used by retry14 is untouched. The root repo is used via a side
worktree too (`/project/peilab/atst/nimloth.qwen-bug-repro`) so outputs/cache
do not touch the main root checkout; the shared `.venv` from the main checkout
is reused by absolute path.

## Minimal lazy-import patches vs upstream/vagen-legacy

`upstream/vagen-legacy` eagerly imports `from together import AsyncTogether`
(via `vagen/server/llm_as_judge.py`) at top level, which is pulled in
unconditionally by `vagen/env/__init__.py` (sokoban/frozenlake) and by
`vagen/server/server.py`. The `together` package is not in the shared `.venv`,
and the LLM-as-judge / state-reward path is unused here
(`use_state_reward=False`, `reward_model.enable=False`). Two minimal patches
(mirroring `nimloth/vagen-legacy-dev`) make those imports lazy so the
navigation path imports cleanly without `together`:

- `vagen/env/utils/state_reward_text_utils.py`: move the `llm_as_judge` imports
  from top-level into the `use_state_reward` branches that call them.
- `vagen/server/server.py`: move `from vagen.server.llm_as_judge import
  wandb_run_context` from top-level into the `use_state_reward` branch.

These do not change navigation semantics; only the unused LLM-as-judge import
timing changes.

## Files

- `dataset_train.yaml` / `dataset_val.yaml` — 3-sample navigation dataset specs.
- `smoke.slurm` — preempt 2-GPU job: create_dataset → env server (GPU0) →
  `vagen.trainer.main_ppo` bi_level_gae smoke (GPU1) → vision probe.
- `verify_vagen_vision_prefix_invariance.py` — collects a real multi-image
  trajectory from the env server, then measures:
  1. vision tower non-invariance: `get_image_features([img_k])` vs
     `get_image_features([all images])` per image (root cause);
  2. alignment: `input_ids` prefix match between full and last-step prefix;
  3. downstream: last-hidden / logits max_abs_diff between full-trajectory
     forward (train-side) and per-prefix forward (rollout-side) on the
     overlapping prefix tokens.
  Uses VAGEN's own `QwenVLRolloutManager` input construction + legacy verl
  `get_rope_index` so inputs match the real train/rollout path.

## Expected result

- Pipeline: `val_before_train` rollout completes + >=1 PPO step, no OOM.
- Vision: text control diff 0; image `vision_features_batch_max_diff > 0`
  (SFT2 observed ~0.39 on real records) and downstream hidden/logits diff
  non-zero → confirms VAGEN train/rollout path is affected by the same
  Qwen visual-encoder bug.
