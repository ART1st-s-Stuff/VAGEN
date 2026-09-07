# Dream Baseline Experiment Log

## Protocol

This file is the source of truth for VAGEN1 Step=1 training decisions.

- Read this file before changing training parameters.
- Record the hypothesis before changing scripts/configs.
- After every SuperPOD experiment, append the commit, clone path, job id, W&B URL, key parameter diff, startup status, speed, metrics, failure summary, and next decision.
- Do not use experiments that are not recorded here as evidence for the next change.

## 2026-07-22 VAGEN1 VAGEN-First Sweep

### Current State Before This Change

- Local branch before change: `hligb/vagen-step1-minimal-20260720`
- Starting commit: `548f3fe6b5cd2e96a2eafb3898ae91851f9dfb1c`
- `navigation_vagen1` already sets `max_actions_per_step: 1`, `max_turns: 20`, `window_size: 5`, and `update_window_size: 5`.
- The latest failed SuperPOD runs used `ROLLOUT_LIMIT_MM_PER_PROMPT=20` with eager/chunked/free-cache enabled.
- The observed failure mode was vLLM KV cache initialization failure or extremely slow rollout before learning could be evaluated.

### Hypothesis

We are solving two separate problems.

1. Engine stability and speed: the likely immediate blocker is not Step=1 itself, but the heavy vLLM multimodal runtime copied from Nimloth-style settings. `limit_mm_per_prompt=20` makes vLLM reserve too many image tokens during profiling.
2. CoT/action collapse: once the engine is stable, we need to prevent the model from getting high validation scores through repeated single actions, especially repeated `moveahead`.

### Change Rationale

- Revert VAGEN1 defaults toward VAGEN runtime:
  - `limit_mm_per_prompt=5`
  - chunked prefill off by default
  - eager off by default
  - free cache off by default
  - `gpu_memory_utilization=0.4`
- Keep VAGEN1 semantics:
  - `max_actions_per_step=1`
  - `max_turns=20`
  - recent-window update context of 5 turns
- Lower default VAGEN1 format reward to `0.1` so format compliance does not dominate sparse task success.
- Make multiple actions a real format failure instead of silently truncating them.
- Add action distribution metrics to W&B for both train and validation.
- Add `LOSS_MASK_MODE=answer_only` as a later anti-collapse variable, not the first engine variable.

### Planned Experiment Waves

Wave 1 engine smoke uses 2h normal jobs, 4 GPUs, 1 step, `VAL_BEFORE_TRAIN=False`, and W&B group `navigation_vagen1_engine_smoke_20260722`.

Variants:

- `vagenrt_gpu01_limit5`
- `vagenrt_gpu04_limit5`
- `vagenrt_gpu06_limit5`
- `eager_gpu04_limit5`
- `eager_free_gpu04_limit5`
- `eager_chunk_gpu04_limit5`
- `failed_minus_limit20_gpu06`
- `tp1_diag_gpu06_limit5`

Pass criteria:

- W&B run created.
- vLLM initializes without KV cache blocks failure.
- Step 1 finishes within 30 minutes.
- No AI2-THOR server crash, NCCL deadlock, or rollout timeout.

Wave 2 speed debug5 and Wave 4 anti-collapse debug20 scripts are prepared but should only be submitted after Wave 1 results are recorded here.

### Result Entries

#### Wave 1 Submission

- Submitted at: 2026-07-22 07:59:38 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit: `3ed39f864e496fa46e761300fb45c04615138688`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-sweep-3ed39f8`
- Submit log: `/project/peilab/hligb/vagen-navigation/logs/vagen1-wave1-submit-20260722T075938Z.log`
- W&B group: `navigation_vagen1_engine_smoke_20260722`
- Shared settings: 4GPU normal, 2h time limit, `TOTAL_TRAINING_STEPS=1`, `VAL_BEFORE_TRAIN=False`, `SAVE_FREQ=-1`, `TEST_FREQ=1`, `FORCE_GEN_DATA=1`

| Variant | Job ID | Status at Submit | Notes |
| --- | --- | --- | --- |
| `vagenrt_gpu01_limit5` | `483923` | `PD (Priority)` | VAGEN runtime, gpu mem 0.1 |
| `vagenrt_gpu04_limit5` | `483924` | `PD (Priority)` | VAGEN runtime, gpu mem 0.4 |
| `vagenrt_gpu06_limit5` | `483925` | `PD (Priority)` | VAGEN runtime, gpu mem 0.6 |
| `eager_gpu04_limit5` | `483926` | `PD (Priority)` | Only eager added |
| `eager_free_gpu04_limit5` | `483927` | `PD (Priority)` | Eager + free cache |
| `eager_chunk_gpu04_limit5` | `483928` | `PD (Priority)` | Eager + chunked prefill |
| `failed_minus_limit20_gpu06` | `483929` | `PD (Priority)` | Previous failed runtime except `limit_mm_per_prompt=5` |
| `tp1_diag_gpu06_limit5` | `483930` | `PD (Priority)` | TP=1 diagnostic |

Pending result fields:

- W&B URL
- vLLM initialization status
- step 1 completion time
- crash/error summary
- action distribution metrics
- next decision

### External Debug Training Walltime Change Rationale

- Prepared at: 2026-07-22 22:55 HKT
- Branch before change: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting local commit: `913956862029bcdbc5afa9dce7c6466238ceaaad`
- Change type: shorten only debug training-job walltime defaults for external-env VAGEN1 experiments.
- Parameter diff:
  - External debug train wrapper default `TRAIN_TIME`: `04:00:00 -> 02:00:00`
  - External debug env-server wrapper default `ENV_SERVER_TIME`: unchanged at `04:00:00`
- Affected future wrappers:
  - `scripts/superpod/submit_navigation_vagen1_external_env_debug.sh`
  - `scripts/superpod/submit_navigation_vagen1_external_multi_env_debug.sh`
  - `scripts/superpod/submit_navigation_vagen1_external_2node4gpu_debug.sh`
- Reason:
  - Current debug runs are for fast validation of startup, W&B, rollout connection, step 1, and debug5 behavior.
  - The env server can remain alive while training jobs are shorter; this reduces wasted train allocations when a run hangs before step 1.
  - Two hours is enough to reveal the known failure modes: queue/startup, Ray/vLLM init, env health-check, rollout timeout, or very slow step 1/debug5.
- Current Slurm jobs to update if still active:
  - `484310`: E3 `external2_b32_rmb16_w4x2` 4GPU training job.
  - `484346`: E3.5 `external2_train2x4_b16_rmb16_w4x2` 2-node training job.
- Expected observations:
  - Pending/running train jobs show `TimeLimit=02:00:00`.
  - Env jobs are not shortened.
  - Future debug submissions no longer silently request 4h train walltime.

### External Debug Training Walltime Change Result

- Applied at: 2026-07-22 22:53-22:54 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Code/doc commit pushed before Slurm update: `bb119fb`
- Local code changes:
  - `submit_navigation_vagen1_external_env_debug.sh`: default `TRAIN_TIME=02:00:00`
  - `submit_navigation_vagen1_external_multi_env_debug.sh`: default `TRAIN_TIME=02:00:00`
  - `submit_navigation_vagen1_external_2node4gpu_debug.sh`: default `TRAIN_TIME=02:00:00`
  - `ENV_SERVER_TIME` remains `04:00:00` in all three wrappers.
- Validation:
  - `bash -n` passed for the three external debug submit wrappers.
  - `python -m pytest tests/test_navigation_vagen1_minimal.py tests/test_navigation_vagen1_sweep.py -q` passed: `24 passed`.
  - `git diff --check` passed.
- SuperPOD update:
  - `484310` changed from `TimeLimit=04:00:00` to `TimeLimit=02:00:00`.
  - `484346` changed from `TimeLimit=04:00:00` to `TimeLimit=02:00:00`.
  - Confirmed by both `squeue` and `sacct` at `2026-07-22T14:53:34Z`.
- Current queue snapshot at `2026-07-22T14:54:15Z`:
  - Env jobs `484308` and `484309` were running with `TimeLimit=04:00:00`.
  - Env jobs `484344` and `484345` were pending priority with `TimeLimit=04:00:00`.
  - Train job `484310` was pending priority with `TimeLimit=02:00:00`.
  - Train job `484346` was pending dependency with `TimeLimit=02:00:00`.
- Notes:
  - `scontrol --account=peilab` is not supported on this cluster, but `scontrol update JobId=<id> Account=peilab TimeLimit=02:00:00` produced the desired time-limit change.
  - The command still printed site wrapper permission warnings, so the reliable evidence is the post-update `squeue`/`sacct` output showing `02:00:00`.
- Decision:
  - Keep debug train jobs at 2h for quick validation.
  - Keep env server jobs longer so a ready env server can survive while short training attempts fail fast or queue separately.

### E3.5 Env Reuse Change Rationale

- Prepared at: 2026-07-22 23:01 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting local commit: `9e1f953`
- Change type: resource-topology correction for the pending 2x4GPU training attempt.
- Current queue state:
  - `484308` and `484309`: existing E3 external2 env servers are running and ready.
  - `484310`: 4GPU train job using `484308/484309`, pending priority.
  - `484344` and `484345`: duplicate env server jobs for the E3.5 2x4 train, pending priority.
  - `484346`: E3.5 2x4GPU train job, pending dependency on duplicate env jobs `484344/484345`.
- Reason:
  - The 2x4GPU training test does not need a separate pair of env servers if `484308/484309` are already alive and reachable.
  - Reusing the same ready env servers avoids waiting for duplicate env jobs and avoids wasting two additional env-server allocations.
  - `484346` cannot safely be used as-is because it was submitted with dependencies and ready-file variables tied to `484344/484345`; simply clearing dependency would make it wait for the wrong ready files.
- Planned action:
  - Cancel pending duplicate env jobs `484344` and `484345`.
  - Cancel pending train job `484346`.
  - Submit a replacement 2x4GPU train job with `SERVER_READY_FILES` pointing to the existing ready files from env jobs `484308/484309`.
  - Keep the same training semantics as E3.5: `TRAIN_NNODES=2`, `N_GPUS_PER_NODE=4`, `EXPECTED_RAY_GPUS=8`, `TRAIN_BATCH_SIZE=16`, `PPO_MINI_BATCH_SIZE=16`, `ROLLOUT_MINI_BATCH_SIZE=16`, `TOTAL_TRAINING_STEPS=6`, `TEST_FREQ=5`, `SAVE_FREQ=-1`, `VAL_BEFORE_TRAIN=False`, `FINAL_VAL_AFTER_TRAIN=False`.
- Expected observations:
  - New replacement train job requests `node=2, gres/gpu=8`, `TimeLimit=02:00:00`.
  - No new env server jobs are submitted.
  - The replacement job logs `ROLLOUT_BASE_URLS` from `484308/484309` and health-checks both servers when it starts.

### E3.5 Env Reuse Result

- Applied at: 2026-07-22 23:06-23:08 HKT
- Rationale commit: `f764307`
- Action taken:
  - Canceled duplicate pending env jobs `484344` and `484345`.
  - Canceled old pending 2x4 train job `484346`, because it depended on `484344/484345` and would have used the wrong ready-file variables.
  - Submitted replacement 2x4 train job `484442` with no env dependency, using existing env ready files from `484308/484309`.
- Reused env servers:
  - `484308`: `http://10.23.0.237:7308`, ready file `navigation-ai2thor-server-navigation_vagen1_external2_b32_rmb16_w4x2_external2_52c027c_20260722T132356Z_2_server1.env`
  - `484309`: `http://10.23.1.181:7309`, ready file `navigation-ai2thor-server-navigation_vagen1_external2_b32_rmb16_w4x2_external2_52c027c_20260722T132356Z_2_server2.env`
- Replacement train job:
  - Job ID: `484442`
  - Job name: `vagen-nav-vagen1-2x4-ext`
  - Status after submit: `PENDING`, reason `Priority`
  - Dependency: `(null)`
  - Time limit: `02:00:00`
  - Resources: `node=2`, `gres/gpu=8`, `cpu=128`, `mem=768G`
  - Script: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/scripts/superpod/run_navigation_vagen1_2node4gpu_external_server.sbatch`
  - WorkDir: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
  - Intended settings: `TRAIN_NNODES=2`, `N_GPUS_PER_NODE=4`, `EXPECTED_RAY_GPUS=8`, `VAGEN1_VARIANT=external2_train2x4_b16_rmb16_w4x2`, `TOTAL_TRAINING_STEPS=6`, `TEST_FREQ=5`, `SAVE_FREQ=-1`, `VAL_BEFORE_TRAIN=False`, `FINAL_VAL_AFTER_TRAIN=False`.
- Queue after replacement:
  - `484308` and `484309` remain running env servers.
  - `484310` remains pending as the separate 4GPU train variant, also using `484308/484309`.
  - `484442` is the replacement 8GPU train variant, also using `484308/484309`.
- Risk / note:
  - If `484310` and `484442` start at the same time, they will share the same env servers and may overload AI2-THOR or make timing comparisons less clean.
  - To keep metrics clean, prefer letting only one training job actively use `484308/484309` at a time; decide based on which variant starts first and whether env servers still have enough walltime.

### E3.6 Long Env Plus 3x4 Train Change Rationale

- Prepared at: 2026-07-22 23:21 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting local commit: `694ac39`
- Change type: cancel short-window external debug jobs and resubmit a Nimloth-style long env plus multi-node train topology for overnight queueing.
- Reason:
  - Current env jobs `484308/484309` were only 4h. Slurm estimated train start times were late enough that those env servers would be near expiration before training could use them.
  - Nimloth's successful topology keeps env service separate and lets a single training job request multiple train nodes, rather than combining independent training jobs.
  - For overnight queueing, a longer env window is more important than saving a few env GPU hours.
- Planned cancellation:
  - Cancel current env jobs `484308` and `484309`.
  - Cancel current queued train jobs `484310` and `484442`.
- Planned replacement:
  - Start two external AI2-THOR env-server jobs, each 1 node x 4 GPU, 12h walltime.
  - Submit one VAGEN1 train job as a single Slurm allocation with 3 nodes x 4 GPU = 12 train GPUs, 8h walltime.
  - Keep conservative debug workload: `TRAIN_BATCH_SIZE=16`, `PPO_MINI_BATCH_SIZE=16`, `ROLLOUT_MINI_BATCH_SIZE=16`, `SERVER_NAVIGATION_MAX_WORKERS=4`, `TOTAL_TRAINING_STEPS=6`, `TEST_FREQ=5`, `SAVE_FREQ=-1`, `VAL_BEFORE_TRAIN=False`, `FINAL_VAL_AFTER_TRAIN=False`.
- Expected observations tomorrow:
  - Env jobs should remain alive long enough for the train job to start.
  - Train job should print `TRAIN_NNODES=3`, `N_GPUS_PER_NODE=4`, `EXPECTED_RAY_GPUS=12`, and Ray cluster resources with at least 12 GPUs.
  - Train job should health-check both env URLs and log `ROLLOUT_BASE_URLS`.
  - If step 1 still does not start, the next root-cause check is Ray/vLLM/FSDP startup or AI2-THOR throughput, not short env walltime.

### E3.6 Long Env Plus 3x4 Train Submission Result

- Submitted at: 2026-07-22 23:24 HKT
- Rationale commit before submission: `d0fd02e`
- Code path used on SuperPOD: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Action taken:
  - Canceled old short-window jobs `484308`, `484309`, `484310`, and `484442`.
  - Submitted two new 12h external env-server jobs.
  - Submitted one 8h single train job with 3 nodes x 4 GPU.
- New env jobs:
  - `484462`: `vagen-nav-ai2thor`, `1 node x 4 GPU`, `TimeLimit=12:00:00`, status after submit `PENDING (Priority)`.
  - `484463`: `vagen-nav-ai2thor`, `1 node x 4 GPU`, `TimeLimit=12:00:00`, status after submit `PENDING (Priority)`.
- New train job:
  - `484464`: `vagen-nav-vagen1-2x4-ext`, submitted with `--nodes=3`.
  - Resources: `node=3`, `gres/gpu=12`, `cpu=192`, `mem=1152G`.
  - Time limit: `08:00:00`.
  - Dependency: `after:484462, after:484463`.
  - Script: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/scripts/superpod/run_navigation_vagen1_2node4gpu_external_server.sbatch`.
  - WorkDir: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`.
- Ready files:
  - `/project/peilab/hligb/vagen-navigation/logs/navigation-ai2thor-server-navigation_vagen1_train3x4_b16_rmb16_w4x2_train3x4_longenv_ffaf505_20260722T152402Z_server1.env`
  - `/project/peilab/hligb/vagen-navigation/logs/navigation-ai2thor-server-navigation_vagen1_train3x4_b16_rmb16_w4x2_train3x4_longenv_ffaf505_20260722T152402Z_server2.env`
- W&B:
  - Group: `navigation_vagen1_external2_train3x4_debug5_20260722`
  - Name: `navigation_vagen1_train3x4_b16_rmb16_w4x2_external_ai2thor_train3x4_longenv_ffaf505_20260722T152402Z`
  - URL: pending until train job starts and W&B run is created.
- Training settings:
  - `VAGEN1_VARIANT=external2_train2x4_b16_rmb16_w4x2`
  - `TRAIN_NNODES=3`
  - `N_GPUS_PER_NODE=4`
  - `EXPECTED_RAY_GPUS=12`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=16`
  - `TOTAL_TRAINING_STEPS=6`
  - `TEST_FREQ=5`
  - `SAVE_FREQ=-1`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
- Notes:
  - The underlying sbatch filename still says `2node4gpu`, but submission overrides Slurm to `--nodes=3` and exports `TRAIN_NNODES=3`, `EXPECTED_RAY_GPUS=12`.
  - This is a single 12-GPU train job, not multiple independent training jobs.

## 2026-07-22 External Env Server First

### Change Rationale

- Change type: switch the next VAGEN1 stability experiments from local AI2-THOR server inside the training job to a separated env-server job plus training job.
- Reason:
  - Wave 2 showed that local `ROLLOUT_MINI_BATCH_SIZE=8/16` can finish debug5, but larger local batches such as `batch32` can still fail at `create_environments_batch` with an HTTP read timeout.
  - This suggests the immediate bottleneck is AI2-THOR environment creation/reset pressure on the same 4GPU training allocation, not `max_actions_per_step=1` itself.
  - Nimloth's working baseline separates environment serving from training: training nodes run RL/model work, while separate env node(s) run AI2-THOR HTTP servers.
  - Our current code supports a single `ROLLOUT_BASE_URL`, so the first verified external version should use one external env server. Multi-server sharding is a later code change if one server is still too slow.
- Planned code/script changes:
  - Keep VAGEN1 training logic VAGEN-first: `max_actions_per_step=1`, `max_turns=20`, `window_size=5`, `update_window_size=5`, `FORMAT_REWARD=0.1`, `LOSS_MASK_MODE=default`.
  - Update `run_navigation_vagen1_ai2thor_server.sbatch` to advertise a reachable node IP/host in its ready file.
  - Update `run_navigation_vagen1_4gpu_external_server.sbatch` to source `configure_navigation_vagen1_variant.sh`, log the external server details, and default to the stable VAGEN runtime: `ROLLOUT_GPU_MEMORY_UTILIZATION=0.6`, `ROLLOUT_LIMIT_MM_PER_PROMPT=8`, no eager/chunk/free-cache.
  - Add an external debug submit wrapper that launches one env-server job and one dependent training job per variant.
- First external debug variants:
  - `external_b16_rmb16_w4_turn20`: conservative carry-over from the healthiest local action-distribution candidate.
  - `external_b32_rmb16_w4_turn20`: tests whether external env serving can make batch32 stable.
  - `external_b32_rmb32_w8_turn20`: only run if queue pressure is acceptable; tests whether larger rollout mini-batch plus more env workers improves throughput without AI2-THOR timeout.
- Expected observations:
  - Env server writes a ready file with `ROLLOUT_BASE_URL`.
  - Training job waits for that file, reaches the external server health endpoint, creates W&B, and completes actual logged steps 1-5 with `TOTAL_TRAINING_STEPS=6`.
  - Compared with local Wave 2, `create_environments_batch` should no longer crash from training-node resource contention.
  - If step time remains slow but no longer crashes, the next direction is multi-server sharding; if it still crashes at one server, inspect server logs before increasing batch.
- Stop rules:
  - If the training job cannot reach the env server after the ready file is written, stop and fix host/IP advertisement.
  - If one external server still times out at `batch32`, keep `batch16` as the stable line and do not try `batch64` on a single server.
  - Do not start anti-collapse variants until an external debug5 run completes with usable action distribution and success metrics.

### Pending External Debug Submission

- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Code commit: `9423edaa2fbfaa92667f8ed35d3cf986e345d8e1`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-external-9423eda`
- W&B group: `navigation_vagen1_external_env_debug5_20260722`
- Shared settings: 4GPU normal env-server job plus 4GPU normal training job, `TOTAL_TRAINING_STEPS=6`, `TEST_FREQ=5`, `SAVE_FREQ=-1`, `VAL_BEFORE_TRAIN=False`, `FINAL_VAL_AFTER_TRAIN=False`, `ENV_CONFIG_PATH=navigation_vagen1/env_config_speed.yaml`, `VAL_BATCH_SIZE=1`.

| Variant | Env Job ID | Train Job ID | W&B URL | Result | Decision |
| --- | --- | --- | --- | --- | --- |
| `external_b16_rmb16_w4_turn20` | `484242` | `484243` | pending | Submitted; env job pending priority, train job pending dependency | Monitor env ready file, then W&B |
| `external_b32_rmb16_w4_turn20` | `484244` | `484245` | pending | Submitted; env job pending priority, train job pending dependency | Monitor env ready file, then W&B |

### External Debug Submit Attempt 1 Submission

- Submitted at: 2026-07-22 12:04:53 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit: `9423edaa2fbfaa92667f8ed35d3cf986e345d8e1`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-external-9423eda`
- Submit log: `/project/peilab/hligb/vagen-navigation/logs/vagen1-external-9423eda-submit-20260722T120435Z.log`
- Wrapper log: `/project/peilab/hligb/vagen-navigation/logs/vagen1-external-env-debug-submit-external_9423eda_20260722T120453Z.log`
- W&B group: `navigation_vagen1_external_env_debug5_20260722`
- Shared settings: 4GPU normal env-server job plus 4GPU normal training job, `TOTAL_TRAINING_STEPS=6`, `TEST_FREQ=5`, `SAVE_FREQ=-1`, `VAL_BEFORE_TRAIN=False`, `FINAL_VAL_AFTER_TRAIN=False`, `FORCE_GEN_DATA=1`, `ENV_CONFIG_PATH=navigation_vagen1/env_config_speed.yaml`, `VAL_BATCH_SIZE=1`, `SERVER_WAIT_SECONDS=7200`.

| Variant | Env Job ID | Train Job ID | Submit Status | Key parameters |
| --- | --- | --- | --- | --- |
| `external_b16_rmb16_w4_turn20` | `484242` | `484243` | Env `PD (Priority)`, train `PD (Dependency)` | `TRAIN_BATCH_SIZE=16`, `PPO_MINI_BATCH_SIZE=16`, `ROLLOUT_MINI_BATCH_SIZE=16`, `SERVER_NAVIGATION_MAX_WORKERS=4` |
| `external_b32_rmb16_w4_turn20` | `484244` | `484245` | Env `PD (Priority)`, train `PD (Dependency)` | `TRAIN_BATCH_SIZE=32`, `PPO_MINI_BATCH_SIZE=32`, `ROLLOUT_MINI_BATCH_SIZE=16`, `SERVER_NAVIGATION_MAX_WORKERS=4` |

Pending result fields:

- Env server ready file content
- Training job reachability check
- W&B URL
- Step 1 / step 5 status
- Average step time
- AI2-THOR timeout/crash summary
- Action distribution and success metrics

### External Debug Submit Attempt 1 Result

- Checked at: 2026-07-22 13:10 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Code commit used by submitted jobs: `9423edaa2fbfaa92667f8ed35d3cf986e345d8e1`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-external-9423eda`
- Action taken: canceled jobs `484242`, `484243`, `484244`, and `484245` after the running b16 training job crossed the 30-minute no-step gate.

| Variant | Env Job ID | Train Job ID | Result | W&B URL | Evidence | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| `external_b16_rmb16_w4_turn20` | `484242` | `484243` | Canceled after env `32m18s`, train `31m47s` | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/lmc0ud74 | Ready file was valid; training node reached `http://10.23.1.165:7242`; W&B run was created; vLLM initialized with `# cuda blocks: 93506`; no `global_steps`, mini-batch, or step timing was logged; train log mtime stayed at `2026-07-22 20:45:02 +0800`; env server health showed `active_environments=16`; env GPUs were all 100%, training GPUs were idle around 11.7-11.9 GB memory | Reject this single-server b16 setting; it is slower than local debug5 and fails the no-step gate |
| `external_b32_rmb16_w4_turn20` | `484244` | `484245` | Canceled before training started | Not created | Env ready file was valid at `http://10.23.0.237:7244`; env server was idle with `active_environments=0`; train job remained pending priority | Cancel because b16 already failed the single-server no-step gate and b32 is heavier |

Root cause update:

- The external server path itself works: ready files were produced and the training job could reach the server from the training node.
- The failure is throughput, not connectivity or vLLM initialization. With `TRAIN_BATCH_SIZE=16` and `ROLLOUT_MINI_BATCH_SIZE=16`, the env server created/held 16 active navigation environments and saturated all 4 server GPUs, while the training node waited on one long HTTP rollout request.
- A single 4-worker external server does not solve Step=1 rollout latency at b16/rmb16. It moves the bottleneck off the training node, but the wall time is still too high and now the model GPUs sit idle during env work.

Next decision:

- Do not run E2 debug20 from Attempt 1; no E1 variant passed debug5.
- Move to an E3-style split: reduce per-server active environment pressure and/or shard rollouts across multiple env servers. The next implementation should add `ROLLOUT_BASE_URLS` support with stable env routing, then test smaller per-server load before increasing global batch.

### E3 Multi-Server Sharding Change Rationale

- Change type: add multi-env-server routing for VAGEN1 service rollouts.
- Hypothesis:
  - Attempt 1 showed one external env server can be reached but is too slow when a rollout mini-batch creates 16 navigation envs on the same server.
  - If a rollout mini-batch is sharded across two env servers, `ROLLOUT_MINI_BATCH_SIZE=16` should become roughly 8 active envs per server instead of 16, while the training code still sees one logical env client.
  - This directly targets the observed bottleneck without changing VAGEN1 learning semantics, prompt, parser, reward, or vLLM runtime.
- Planned implementation:
  - Add `rollout_manager.base_urls` / `ROLLOUT_BASE_URLS` as an optional comma-separated list.
  - Add a sharded HTTP env client that routes `create/reset/step/reward/system_prompt/close` for the same `env_id` to the same server.
  - Execute per-server batch requests concurrently so two env servers actually work in parallel.
  - Keep `rollout_manager.base_url` as backward-compatible single-server fallback.
  - Add a submit wrapper that starts two env-server jobs per training job and passes both ready-file URLs into the trainer.
- First E3 debug variants:
  - `external2_b32_rmb16_w4x2`: expected per-server active env pressure is about 8 envs per rollout mini-batch.
  - `external2_b16_rmb16_w4x2`: conservative diagnostic; same global batch as failed b16 but per-server pressure should drop to about 8.
  - Do not run `external2_b64_rmb32_w4x2` until at least one 8-env-per-server variant reaches step 1; its per-server pressure would be similar to the failed single-server b16 case.
- Expected observations:
  - Multiple ready files are produced and the training job health-checks all advertised `ROLLOUT_BASE_URLS`.
  - W&B is created.
  - Step 1 completes within 30 minutes.
  - Server health shows active envs split across two servers rather than concentrated on one.
  - If both 2-server variants still fail the no-step gate, revisit AI2-THOR env creation cost directly before scaling batch.

### E3 Multi-Server Debug5 Submission

- Submitted at: 2026-07-22 13:23:56 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit: `52c027c64125e46777a5e412b6140303c642afed`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-external2-52c027c`
- Submit log: `/project/peilab/hligb/vagen-navigation/logs/vagen1-external-multi-env-debug-submit-external2_52c027c_20260722T132356Z.log`
- W&B group: `navigation_vagen1_external2_debug5_20260722`
- Shared settings: two 4GPU normal env-server jobs per 4GPU normal training job, `TOTAL_TRAINING_STEPS=6`, `TEST_FREQ=5`, `SAVE_FREQ=-1`, `VAL_BEFORE_TRAIN=False`, `FINAL_VAL_AFTER_TRAIN=False`, `FORCE_GEN_DATA=1`, `ENV_CONFIG_PATH=navigation_vagen1/env_config_speed.yaml`, `VAL_BATCH_SIZE=1`, `ENV_SERVERS_PER_VARIANT=2`.

| Variant | Env Job IDs | Train Job ID | Submit Status | Key parameters |
| --- | --- | --- | --- | --- |
| `external2_b16_rmb16_w4x2` | `484305`, `484306` | `484307` | First env server running at submit check, second env pending priority, train pending dependency | `TRAIN_BATCH_SIZE=16`, `PPO_MINI_BATCH_SIZE=16`, `ROLLOUT_MINI_BATCH_SIZE=16`, two env servers, each `SERVER_NAVIGATION_MAX_WORKERS=4` |
| `external2_b32_rmb16_w4x2` | `484308`, `484309` | `484310` | Env servers pending priority, train pending dependency | `TRAIN_BATCH_SIZE=32`, `PPO_MINI_BATCH_SIZE=32`, `ROLLOUT_MINI_BATCH_SIZE=16`, two env servers, each `SERVER_NAVIGATION_MAX_WORKERS=4` |

Pending result fields:

- Multiple ready file content and advertised `ROLLOUT_BASE_URLS`
- Training job reachability checks for all env servers
- W&B URL
- Step 1 / step 5 status
- Per-server active environment counts
- Average step time
- AI2-THOR timeout/crash summary
- Action distribution and success metrics

### E3.5 Two-Node Training Change Rationale

- Prepared at: 2026-07-22 21:50 HKT
- Branch before change: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting local commit: `0e0ee3ad84c3e859f2c7217d64efb64e0c9126cd`
- Change type: add a true two-node training job, using 2 Slurm nodes with 4 GPUs each for the VAGEN/Ray trainer, while keeping AI2-THOR env servers as separate jobs.
- Hypothesis:
  - Two independent 4GPU training jobs cannot combine into one 8GPU run. To get the effect of 8 training GPUs, one Slurm job must request `--nodes=2 --gres=gpu:4`, start a Ray head plus Ray worker, and run VAGEN with `trainer.nnodes=2` and `trainer.n_gpus_per_node=4`.
  - If the current external2 jobs are still dominated by env-server latency, this 2x4GPU training job may not make step time much faster. That result would be useful evidence: it would tell us the remaining bottleneck is AI2-THOR rollout serving rather than PPO/FSDP/vLLM training resources.
  - If training/vLLM is also a bottleneck, 2x4GPU should reduce model-side pressure and make rollout/update scheduling healthier than the 4GPU training job.
- Scope:
  - Keep VAGEN1 learning semantics unchanged: `max_actions_per_step=1`, `max_turns=20`, `window_size=5`, `update_window_size=5`, `FORMAT_REWARD=0.1`, `LOSS_MASK_MODE=default`, `SERVER_USE_STATE_REWARD=False`, no LLM judge, no WM reward.
  - Add `TRAIN_NNODES` / Ray external-address support so the same navigation run script can run either 1-node or 2-node training.
  - Add a submit wrapper that launches two env-server jobs and one dependent 2-node training job.
- First debug variant:
  - `external2_train2x4_b16_rmb16_w4x2`
  - Shared settings: 2x4GPU normal training job plus two 4GPU normal env-server jobs, `TOTAL_TRAINING_STEPS=6`, `TEST_FREQ=5`, `SAVE_FREQ=-1`, `VAL_BEFORE_TRAIN=False`, `FINAL_VAL_AFTER_TRAIN=False`, `FORCE_GEN_DATA=1`, `ENV_CONFIG_PATH=navigation_vagen1/env_config_speed.yaml`, `VAL_BATCH_SIZE=1`.
- Optional second variant:
  - `external2_train2x4_b32_rmb16_w4x2`
  - Only submit if queue pressure is acceptable or if the b16 variant starts cleanly but underuses the 8 training GPUs.
- Gates:
  - Ray cluster reports at least 8 GPUs before training starts.
  - Training log prints `trainer.nnodes=2` and `trainer.n_gpus_per_node=4`.
  - Training job health-checks all `ROLLOUT_BASE_URLS`.
  - W&B run is created and step 1 finishes within 30 minutes of training start.
  - If step 1 is still late while env servers are saturated, do not keep adding training nodes; move back to env-side throughput and sharding.

### E3 Multi-Server Debug5 Monitoring Update

- Checked at: 2026-07-22 14:03 UTC
- Action taken: canceled the old `external2_b16_rmb16_w4x2` variant jobs `484305`, `484306`, and `484307`.
- Reason:
  - Env server job `484305` had been running alone for `40:45` on `dgx-29`.
  - Its paired env server `484306` was still pending priority, and training job `484307` was still pending dependency.
  - This matched the idle-server stop rule: if one env server runs for more than 30 minutes while the paired server/train job is not ready, cancel that variant to avoid wasting GPUs.
- Evidence:
  - Ready file for `484305` existed and advertised `ROLLOUT_BASE_URL=http://10.23.0.237:7305`.
  - No paired training job had started, so no W&B run or training step could be produced from this variant.
- Remaining old E3 jobs:
  - `external2_b32_rmb16_w4x2` env jobs `484308`, `484309` and train job `484310` remained pending.
- Decision:
  - Treat `external2_b16_rmb16_w4x2` as not evaluated for speed/learning because the pair never started together.
  - Continue monitoring the remaining queued E3 b32 variant, but prioritize the new E3.5 2x4GPU training diagnostic below.

### E3.5 Two-Node Training Debug5 Submission

- Submitted at: 2026-07-22 13:57 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Code commit used by submitted jobs: `ffaf5058bd2aa19aaafab1575125973e4709021c`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Submit log: `/project/peilab/hligb/vagen-navigation/logs/vagen1-external-2node4gpu-debug-submit-train2x4_ffaf505_20260722T135749Z.log`
- W&B group: `navigation_vagen1_external2_train2x4_debug5_20260722`
- W&B workspace URL: https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/groups/navigation_vagen1_external2_train2x4_debug5_20260722/workspace
- Server-side syntax check: passed for `scripts/examples/vagen_base/navigation/run.sh`, `scripts/examples/vagen_base/navigation_vagen1/run.sh`, `scripts/superpod/run_navigation_vagen1_2node4gpu_external_server.sbatch`, `scripts/superpod/submit_navigation_vagen1_external_2node4gpu_debug.sh`, and `scripts/superpod/configure_navigation_vagen1_variant.sh`.

| Variant | Env Job IDs | Train Job ID | Submit Status | Key parameters |
| --- | --- | --- | --- | --- |
| `external2_train2x4_b16_rmb16_w4x2` | `484344`, `484345` | `484346` | Env jobs pending priority, train job pending dependency | 2 env servers, each 4GPU/4 workers; training job is 2 nodes x 4GPU with expected Ray GPUs `8`; `TRAIN_BATCH_SIZE=16`, `PPO_MINI_BATCH_SIZE=16`, `ROLLOUT_MINI_BATCH_SIZE=16`, `TOTAL_TRAINING_STEPS=6`, `TEST_FREQ=5`, `SAVE_FREQ=-1`, `VAL_BEFORE_TRAIN=False`, `FINAL_VAL_AFTER_TRAIN=False` |

Pending result fields:

- Both env ready files and advertised `ROLLOUT_BASE_URLS`
- Ray head/worker startup and `GPU >= 8` cluster-resource check
- Training log confirmation of `trainer.nnodes=2` and `trainer.n_gpus_per_node=4`
- W&B run URL
- Step 1 / step 5 status
- Average step time
- Env server active environment counts
- AI2-THOR timeout/crash summary
- Action distribution and success metrics

### External Debug Submit Attempt 0 Result

- Checked at: 2026-07-22 12:02 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit: `bfc594ec03fc3dd935d0018f055de27bb58820e7`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-external-bfc594e`
- Submit log: `/project/peilab/hligb/vagen-navigation/logs/vagen1-external-bfc594e-submit-20260722T120107Z.log`
- Result: no valid Slurm jobs were submitted. `sbatch --parsable` returned the SuperPOD warning text asking to load the Slurm module instead of a numeric job id.
- Root cause: the new external submit wrapper ran from a login-shell context where `sbatch` was not available until `scripts/superpod/load_modules.sh` loaded the Slurm module.
- Fix:
  - Source `scripts/superpod/load_modules.sh` in `submit_navigation_vagen1_external_env_debug.sh`.
  - Fail fast if either env-server or training `sbatch --parsable` output is not a numeric job id.
- Decision: recommit and resubmit the same two variants. This attempt should not be counted as a training experiment because no env or training job entered the queue.

#### Wave 1 Attempt 4 Result

- Checked at: 2026-07-22 09:07 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit: `fe666658db326b470a558592a4e8e05f4c2966bd`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-sweep-3ed39f8`
- Action taken: canceled all Attempt 4 jobs after confirming the training step could complete but validation was still using the full test split.

| Variant | Job ID | Result | W&B URL | Evidence | Decision |
| --- | --- | --- | --- | --- | --- |
| `vagenrt_gpu04_limit8` | `483995` | Canceled after 24m01s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/s0cnlc0y | Completed `Processing mini-batch 8/8`, printed `[DEBUG] step 1 rollout ends`, then entered validation; generated test parquet had 128 instances | Training step succeeds; validation split is too large for smoke |
| `vagenrt_gpu06_limit8` | `483996` | Canceled after 24m01s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/k6034bsg | Completed `Processing mini-batch 8/8`, printed `[DEBUG] step 1 rollout ends`, then entered validation; generated test parquet had 128 instances | gpu mem 0.6 is the first runtime to pass step 1 under 30m |
| `eager_gpu04_limit8` | `483997` | Canceled while pending | Not created | Canceled because Attempt 4 root cause was smoke validation data size, not eager | Retry after smoke env config |
| `eager_gpu06_limit8` | `483998` | Canceled while pending | Not created | Canceled because Attempt 4 root cause was smoke validation data size, not eager | Retry after smoke env config |
| `eager_free_gpu04_limit8` | `483999` | Canceled while pending | Not created | Canceled because Attempt 4 root cause was smoke validation data size, not free cache | Retry after smoke env config |
| `eager_chunk_gpu04_limit8` | `484000` | Canceled while pending | Not created | Canceled because Attempt 4 root cause was smoke validation data size, not chunked prefill | Retry after smoke env config |
| `failed_minus_limit8_gpu06` | `484001` | Canceled while pending | Not created | Canceled because Attempt 4 root cause was smoke validation data size | Retry after smoke env config |
| `tp1_diag_gpu06_limit8` | `484002` | Canceled while pending | Not created | Canceled because Attempt 4 root cause was smoke validation data size | Retry after smoke env config |

Root cause update:

- `TRAIN_BATCH_SIZE=8` is enough for Wave 1 to complete a training rollout step.
- The remaining smoke bottleneck is that Wave 1 still inherited the full VAGEN1 env config, so dataset generation created `test_size=64 + 64 = 128`.
- `ray_trainer.py` runs validation at `TEST_FREQ=1` and also has final validation after training, so a full test split makes a 1-step smoke look stuck even after the train step succeeds.

#### Wave 1 Attempt 5 Change Rationale

- Change type: use smoke dataset for Wave 1 only.
- Planned parameter diff:
  - Wave 1 W&B group: `navigation_vagen1_engine_smoke_limit8_batch8_20260722 -> navigation_vagen1_engine_smoke_limit8_batch8_smokedata_20260722`
  - Wave 1 `ENV_CONFIG_PATH`: full `navigation_vagen1/env_config.yaml -> navigation_vagen1/env_config_smoke.yaml`
  - Keep `TRAIN_BATCH_SIZE=8`, `PPO_MINI_BATCH_SIZE=8`, `VAL_BATCH_SIZE=1`, `TEST_FREQ=1`, `ROLLOUT_LIMIT_MM_PER_PROMPT=8`
- Reason:
  - Smoke config has `train_size=8` and `test_size=2` per env, so validation is 4 trajectories instead of 128.
  - This avoids changing trainer control flow before we know whether the normal validation path works on a small split.
- Expected result:
  - At least one runtime should finish the whole 1-step job, including validation and W&B metrics.
  - If this passes, promote gpu mem 0.6 limit8 to Wave 2 speed debug5 and start worker/batch sweeps.

#### Wave 1 Attempt 5 Submission

- Submitted at: 2026-07-22 09:09:43 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit: `6c5711ced6a9ebd69107d8122daeba0e076f3046`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-sweep-3ed39f8`
- Submit log: `/project/peilab/hligb/vagen-navigation/logs/vagen1-wave1-limit8-batch8-smokedata-submit-20260722T090943Z.log`
- W&B group: `navigation_vagen1_engine_smoke_limit8_batch8_smokedata_20260722`
- Shared settings: 4GPU normal, 2h time limit, `ENV_CONFIG_PATH=navigation_vagen1/env_config_smoke.yaml`, `TOTAL_TRAINING_STEPS=1`, `TRAIN_BATCH_SIZE=8`, `PPO_MINI_BATCH_SIZE=8`, `VAL_BATCH_SIZE=1`, `VAL_BEFORE_TRAIN=False`, `SAVE_FREQ=-1`, `TEST_FREQ=1`, `FORCE_GEN_DATA=1`, `ROLLOUT_LIMIT_MM_PER_PROMPT=8`

| Variant | Job ID | Status at Submit | Notes |
| --- | --- | --- | --- |
| `vagenrt_gpu04_limit8` | `484029` | `PD (Priority)` | VAGEN runtime, gpu mem 0.4 |
| `vagenrt_gpu06_limit8` | `484030` | `PD (Priority)` | VAGEN runtime, gpu mem 0.6 |
| `eager_gpu04_limit8` | `484031` | `PD (Priority)` | Eager, gpu mem 0.4 |
| `eager_gpu06_limit8` | `484032` | `PD (Priority)` | Eager, gpu mem 0.6 |
| `eager_free_gpu04_limit8` | `484033` | `PD (Priority)` | Eager + free cache |
| `eager_chunk_gpu04_limit8` | `484034` | `PD (Priority)` | Eager + chunked prefill |
| `failed_minus_limit8_gpu06` | `484035` | `PD (Priority)` | Previous failed runtime except `limit20 -> limit8`, smoke data |
| `tp1_diag_gpu06_limit8` | `484036` | `PD (Priority)` | TP=1 diagnostic with `limit8`, smoke data |

Pending result fields:

- W&B URL
- vLLM initialization status
- step 1 completion time
- validation completion time
- action distribution metrics
- next decision

#### Wave 1 Attempt 5 Partial Result

- Checked at: 2026-07-22 09:32 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit: `6c5711ced6a9ebd69107d8122daeba0e076f3046`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-sweep-3ed39f8`

| Variant | Job ID | Result | W&B URL | Evidence | Decision |
| --- | --- | --- | --- | --- | --- |
| `vagenrt_gpu04_limit8` | `484029` | Still running at 21m45s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/swf9afln | Smoke dataset confirmed: train 16, test 4. Reached `Processing mini-batch 8/8`, but not finished at check time | Slower than gpu0.6; keep only as comparison |
| `vagenrt_gpu06_limit8` | `484030` | Completed after 21m32s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/jwwyrhcg | Smoke dataset confirmed: train 16, test 4. Completed train rollout, optimizer step, step validation, final validation, W&B sync, and run summary | First fully passing Wave 1 runtime |

Key `484030` metrics:

- vLLM initialized successfully with `# cuda blocks: 93506`.
- Train step completed and logged `timing_s/step=741.699`, `timing_s/gen=443.575`, `timing_s/testing=277.471`.
- Train metrics: `train/success=0.125`, `train/format_correct=0.875`, `train/too_many_actions=0.0`, `train/action/top_share=0.725`, `train/action/entropy=0.418`, `train/action/all_same_traj=0.25`.
- Step-1 validation metrics: `val/success=0.25`, `val/format_correct=0.75`, `val/too_many_actions=0.0`, `val/action/top_share=0.562`, `val/action/entropy=0.532`, `val/action/all_same_traj=0.25`.
- Final validation metrics at step 2: `val/success=0.0`, `val/format_correct=0.5`, `val/too_many_actions=0.0`, `val/action/top_share=0.4`, `val/action/entropy=0.211`, `val/action/all_same_traj=0.0`.

Current conclusion:

- The original crash chain is fixed for `vagenrt_gpu06_limit8`: no KV cache failure, no 6-image prompt failure, no full-validation smoke stall.
- Step time is still too slow for formal training if we use worker=1 and frequent validation.
- Next decisions should promote `gpu_memory_utilization=0.6`, `limit_mm_per_prompt=8`, non-eager VAGEN runtime as the first stable runtime, then run speed sweeps around worker count and train batch size.

#### Wave 1 Attempt 5 Final Result

- Checked at: 2026-07-22 10:10 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit used by submitted jobs: `6c5711ced6a9ebd69107d8122daeba0e076f3046`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-sweep-3ed39f8`
- W&B group: `navigation_vagen1_engine_smoke_limit8_batch8_smokedata_20260722`
- Decision: keep `vagenrt_gpu06_limit8` as the VAGEN-first runtime seed for Wave 2; stop the eager/Nimloth-style runtime variants for now.

| Variant | Job ID | Result | W&B URL | Key evidence | Decision |
| --- | --- | --- | --- | --- | --- |
| `vagenrt_gpu04_limit8` | `484029` | Canceled at 31m05s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/swf9afln | vLLM initialized with `# cuda blocks: 35865`, reached step 1 rollout end and validation start, but validation did not finish before the 30m speed gate | Reject as slower than gpu0.6 |
| `vagenrt_gpu06_limit8` | `484030` | Completed at 21m32s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/jwwyrhcg | vLLM initialized with `# cuda blocks: 93506`; train step, step validation, final validation, W&B sync all completed | Promote to Wave 2 |
| `eager_gpu04_limit8` | `484031` | Completed at 30m39s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/xj6bni3y | Completed, but `timing_s/step=1146.140` and validation action top share was high at `0.921` | Reject as slower and more collapse-looking than VAGEN runtime |
| `eager_gpu06_limit8` | `484032` | Canceled at 29m19s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/s4r1klby | Reached step 1 rollout end and validation start, but did not complete validation by the speed gate | Stop eager direction |
| `eager_free_gpu04_limit8` | `484033` | Canceled at 7m47s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/sszvzuaq | Reached `global_steps: 1` only; canceled after eager variants were already slower than VAGEN runtime | Stop eager/free-cache direction |
| `eager_chunk_gpu04_limit8` | `484034` | Canceled while pending | Not created | Pending when VAGEN runtime already passed and eager variants were slower | Stop eager/chunk direction |
| `failed_minus_limit8_gpu06` | `484035` | Canceled while pending | Not created | This variant reintroduces eager, chunked prefill, and free cache together | Stop Nimloth-style runtime direction |
| `tp1_diag_gpu06_limit8` | `484036` | Canceled while pending | Not created | TP=1 diagnostic also uses eager runtime and is no longer needed for the first stable path | Stop TP=1 diagnostic for now |

Key `484031` comparison metrics:

- `timing_s/step=1146.140`, `timing_s/gen=686.217`, `timing_s/testing=438.987`.
- Train: `success=0.250`, `format_correct=0.750`, `too_many_actions=0.000`, `action/top_share=0.604`, `action/entropy=0.402`, `action/all_same_traj=0.125`.
- Step-1 validation: `success=0.250`, `format_correct=1.000`, `too_many_actions=0.000`, `action/top_share=0.921`, `action/entropy=0.260`, `action/all_same_traj=0.500`.

Root cause and next hypothesis:

- The first real crash chain was not caused by `max_actions_per_step=1` alone. It came from layered runtime/workload issues: Hydra duplicate `+` override, invalid `limit_mm_per_prompt=5` for 6-image prompts, full smoke batch size, then full validation split.
- After those are fixed, Step=1 can run on the VAGEN-like runtime, but speed is still dominated by rollout and validation wall time.
- Next Wave 2 should use the `vagenrt_gpu06_limit8` runtime and sweep only batch size plus AI2-THOR worker count on smoke data before attempting full-data debug20.
- Non-fatal `Traceback` blocks in completed jobs are Python multiprocessing tempdir cleanup messages after validation/W&B sync; they did not change Slurm exit code or W&B completion, but should be watched in longer runs.

#### Wave 2 Attempt 1 Change Rationale

- Change type: speed sweep after the first complete VAGEN1 engine smoke.
- Planned code/script changes:
  - Keep the runtime fixed to the winning VAGEN-like settings: `ROLLOUT_GPU_MEMORY_UTILIZATION=0.6`, `ROLLOUT_LIMIT_MM_PER_PROMPT=8`, `ROLLOUT_ENABLE_CHUNKED_PREFILL=False`, `ROLLOUT_ENFORCE_EAGER=False`, `ROLLOUT_FREE_CACHE_ENGINE=False`, `ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=4`.
  - Use a dedicated speed env config for Wave 2 first: `navigation_vagen1/env_config_speed.yaml` with `train_size=128` and `test_size=4` per split.
  - Sweep only `TRAIN_BATCH_SIZE`, `PPO_MINI_BATCH_SIZE`, `ROLLOUT_MINI_BATCH_SIZE`, and `SERVER_NAVIGATION_MAX_WORKERS`.
  - Add `FINAL_VAL_AFTER_TRAIN` with default `True`, but set it to `False` for speed debug. Step-level validation at `TEST_FREQ=5` remains enabled, so W&B still gets validation metrics.
- Reason:
  - Wave 1 showed that the VAGEN-like runtime is stable and faster than eager.
  - The remaining bottleneck is wall time in rollout/validation, not KV cache or prompt image count.
  - The trainer currently performs validation at `TEST_FREQ` and again after the last step. For a 5-step speed test with `TEST_FREQ=5`, that final validation is duplicate overhead and makes speed comparisons noisier.
  - Current VAGEN1 jobs use `ROLLOUT_MINI_BATCH_SIZE=1`, while original VAGEN base scripts use a much larger rollout mini-batch. Since `_process_in_mini_batches` resets and rolls out each mini-batch sequentially, this can make Step=1 much slower than necessary.
  - Increasing AI2-THOR workers may improve rollout throughput, but too many Unity instances can hang; therefore worker count is swept conservatively.
- Variants to submit:
  - `speed_b8_rmb1_w1_turn20`
  - `speed_b8_rmb4_w1_turn20`
  - `speed_b16_rmb4_w2_turn20`
  - `speed_b16_rmb8_w2_turn20`
  - `speed_b32_rmb8_w4_turn20`
  - `speed_b32_rmb16_w4_turn20`
  - `speed_b64_rmb16_w4_turn20`
- Expected observations:
  - W&B run creation and vLLM initialization should match Wave 1.
  - Compare `timing_s/gen`, `timing_s/testing`, and `timing_s/step`.
  - Check server init count and timeout/crash logs for worker-related AI2-THOR hangs.
  - Keep only configs that finish debug5 within the 4h limit and show no rollout timeout.

#### Wave 2 Attempt 1 Submission

- Submitted at: 2026-07-22 10:18:27 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit: `f07073af488d9a87b10e991d89b57f0f69f9c362`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-sweep-3ed39f8`
- Submit log: `/project/peilab/hligb/vagen-navigation/logs/vagen1-wave2-speed-debug5-submit-20260722T101827Z.log`
- W&B group: `navigation_vagen1_speed_debug5_vagenrt06_limit8_20260722`
- Shared settings: 4GPU normal, 4h time limit, `TOTAL_TRAINING_STEPS=5`, `SAVE_FREQ=-1`, `TEST_FREQ=5`, `VAL_BEFORE_TRAIN=False`, `FINAL_VAL_AFTER_TRAIN=False`, `FORCE_GEN_DATA=1`, `ENV_CONFIG_PATH=navigation_vagen1/env_config_speed.yaml`, `VAL_BATCH_SIZE=1`

| Variant | Job ID | Status at Submit | Key parameters |
| --- | --- | --- | --- |
| `speed_b8_rmb1_w1_turn20` | `484116` | `PD (Priority)` | `TRAIN_BATCH_SIZE=8`, `ROLLOUT_MINI_BATCH_SIZE=1`, `SERVER_NAVIGATION_MAX_WORKERS=1` |
| `speed_b8_rmb4_w1_turn20` | `484117` | `PD (Priority)` | `TRAIN_BATCH_SIZE=8`, `ROLLOUT_MINI_BATCH_SIZE=4`, `SERVER_NAVIGATION_MAX_WORKERS=1` |
| `speed_b16_rmb4_w2_turn20` | `484118` | `PD (Priority)` | `TRAIN_BATCH_SIZE=16`, `ROLLOUT_MINI_BATCH_SIZE=4`, `SERVER_NAVIGATION_MAX_WORKERS=2` |
| `speed_b16_rmb8_w2_turn20` | `484119` | `PD (Priority)` | `TRAIN_BATCH_SIZE=16`, `ROLLOUT_MINI_BATCH_SIZE=8`, `SERVER_NAVIGATION_MAX_WORKERS=2` |
| `speed_b32_rmb8_w4_turn20` | `484120` | `PD (Priority)` | `TRAIN_BATCH_SIZE=32`, `ROLLOUT_MINI_BATCH_SIZE=8`, `SERVER_NAVIGATION_MAX_WORKERS=4` |
| `speed_b32_rmb16_w4_turn20` | `484121` | `PD (Priority)` | `TRAIN_BATCH_SIZE=32`, `ROLLOUT_MINI_BATCH_SIZE=16`, `SERVER_NAVIGATION_MAX_WORKERS=4` |
| `speed_b64_rmb16_w4_turn20` | `484122` | `PD (Priority)` | `TRAIN_BATCH_SIZE=64`, `ROLLOUT_MINI_BATCH_SIZE=16`, `SERVER_NAVIGATION_MAX_WORKERS=4` |

Pending result fields:

- W&B URL
- vLLM initialization status
- step 1 completion time
- debug5 completion time
- average step time
- `timing_s/gen`, `timing_s/testing`, `timing_s/step`
- action distribution metrics
- crash/error summary
- next decision

#### Wave 2 Attempt 1 Result

- Checked at: 2026-07-22 10:39 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit used by submitted jobs: `f07073af488d9a87b10e991d89b57f0f69f9c362`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-sweep-3ed39f8`
- Action taken: canceled remaining jobs after discovering the debug5 off-by-one issue.

| Variant | Job ID | Result | W&B URL | Evidence | Decision |
| --- | --- | --- | --- | --- | --- |
| `speed_b8_rmb1_w1_turn20` | `484116` | Canceled at 18m41s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/sjcfbtem | Step 1 completed with `timing_s/gen=422.585`, `timing_s/step=443.470`; step 2 was still in rollout; no validation | Reject `ROLLOUT_MINI_BATCH_SIZE=1` as too slow |
| `speed_b8_rmb4_w1_turn20` | `484117` | Completed at 16m12s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/9wcs83fz | Completed steps 1-4, W&B synced, but no step 5 and no validation due off-by-one; step timings after warmup were `158.097`, `140.086`, `146.074` seconds | Keep as speed evidence; rerun with corrected total steps |
| `speed_b16_rmb4_w2_turn20` | `484118` | Canceled at 18m41s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/au748pvk | Completed steps 1-2 with `timing_s/step=306.008` and `315.557`; slower per step than batch8/rmb4 | Keep as partial comparison only |
| `speed_b16_rmb8_w2_turn20` | `484119` | Canceled at 1m58s | Not created before cancellation | Canceled because Attempt 1 cannot produce step5 validation | Rerun corrected |
| `speed_b32_rmb8_w4_turn20` | `484120` | Canceled while pending | Not created | Canceled because Attempt 1 cannot produce step5 validation | Rerun corrected |
| `speed_b32_rmb16_w4_turn20` | `484121` | Canceled while pending | Not created | Canceled because Attempt 1 cannot produce step5 validation | Rerun corrected |
| `speed_b64_rmb16_w4_turn20` | `484122` | Canceled while pending | Not created | Canceled because Attempt 1 cannot produce step5 validation | Rerun corrected |

Root cause:

- `RayPPOTrainer.fit()` initializes `global_steps=0`, increments once before the loop, logs steps beginning at `step:1`, then increments at the end of each training iteration.
- The stop check is `if self.global_steps >= self.total_training_steps` after that increment. Therefore `TOTAL_TRAINING_STEPS=5` completes only logged steps 1-4.
- Since validation is checked before the increment, `TEST_FREQ=5` never fires in this run.

Next change:

- Change Wave 2 debug5 script default from `TOTAL_TRAINING_STEPS=5` to `TOTAL_TRAINING_STEPS=6`.
- Keep `TEST_FREQ=5` and `FINAL_VAL_AFTER_TRAIN=False`, so the corrected run logs validation once at step 5 without duplicate final validation.
- Drop `rmb1` from the next sweep because it is already clearly too slow; add larger rollout mini-batch variants around the faster `rmb4` result.

#### Wave 2 Attempt 2 Change Rationale

- Change type: correct debug5 semantics and narrow the speed sweep.
- Planned parameter diff:
  - W&B group: `navigation_vagen1_speed_debug5_vagenrt06_limit8_20260722 -> navigation_vagen1_speed_debug5_vagenrt06_limit8_total6_20260722`
  - `TOTAL_TRAINING_STEPS: 5 -> 6`
  - Keep `TEST_FREQ=5`, `FINAL_VAL_AFTER_TRAIN=False`, `SAVE_FREQ=-1`, `ENV_CONFIG_PATH=navigation_vagen1/env_config_speed.yaml`
  - Remove `speed_b8_rmb1_w1_turn20` from the corrected sweep.
  - Add higher rollout mini-batch variants: `speed_b8_rmb8_w1_turn20`, `speed_b16_rmb16_w4_turn20`, `speed_b64_rmb32_w4_turn20`.
- Reason:
  - Attempt 1 already proves `ROLLOUT_MINI_BATCH_SIZE=1` is the slow path.
  - The corrected run must produce validation metrics at step 5 so W&B can show train/val/action curves.
  - Larger rollout mini-batches should reduce the number of sequential `rollout_manager.reset()` and `rollout_loop()` cycles per train step.
- Expected observations:
  - Step 5 validation should appear in logs and W&B.
  - Best candidate should have average post-warmup `timing_s/step` well below the old 700-1100s range.
  - Reject any worker=4 variant that hangs or shows AI2-THOR timeout despite faster early mini-batches.

#### Wave 2 Attempt 2 Submission

- Submitted at: 2026-07-22 10:39:31 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit: `388b741e1879e1ecf4c328cb7668da06b76033d7`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-sweep-3ed39f8`
- Submit log: `/project/peilab/hligb/vagen-navigation/logs/vagen1-wave2-speed-debug5-total6-submit-20260722T103931Z.log`
- W&B group: `navigation_vagen1_speed_debug5_vagenrt06_limit8_total6_20260722`
- Shared settings: 4GPU normal, 4h time limit, `TOTAL_TRAINING_STEPS=6`, `SAVE_FREQ=-1`, `TEST_FREQ=5`, `VAL_BEFORE_TRAIN=False`, `FINAL_VAL_AFTER_TRAIN=False`, `FORCE_GEN_DATA=1`, `ENV_CONFIG_PATH=navigation_vagen1/env_config_speed.yaml`, `VAL_BATCH_SIZE=1`

| Variant | Job ID | Status at Submit | Key parameters |
| --- | --- | --- | --- |
| `speed_b8_rmb4_w1_turn20` | `484136` | `PD (Priority)` | `TRAIN_BATCH_SIZE=8`, `ROLLOUT_MINI_BATCH_SIZE=4`, `SERVER_NAVIGATION_MAX_WORKERS=1` |
| `speed_b8_rmb8_w1_turn20` | `484137` | `PD (Priority)` | `TRAIN_BATCH_SIZE=8`, `ROLLOUT_MINI_BATCH_SIZE=8`, `SERVER_NAVIGATION_MAX_WORKERS=1` |
| `speed_b16_rmb8_w2_turn20` | `484138` | `PD (Priority)` | `TRAIN_BATCH_SIZE=16`, `ROLLOUT_MINI_BATCH_SIZE=8`, `SERVER_NAVIGATION_MAX_WORKERS=2` |
| `speed_b16_rmb16_w4_turn20` | `484139` | `PD (Priority)` | `TRAIN_BATCH_SIZE=16`, `ROLLOUT_MINI_BATCH_SIZE=16`, `SERVER_NAVIGATION_MAX_WORKERS=4` |
| `speed_b32_rmb8_w4_turn20` | `484140` | `PD (Priority)` | `TRAIN_BATCH_SIZE=32`, `ROLLOUT_MINI_BATCH_SIZE=8`, `SERVER_NAVIGATION_MAX_WORKERS=4` |
| `speed_b32_rmb16_w4_turn20` | `484141` | `PD (Priority)` | `TRAIN_BATCH_SIZE=32`, `ROLLOUT_MINI_BATCH_SIZE=16`, `SERVER_NAVIGATION_MAX_WORKERS=4` |
| `speed_b64_rmb16_w4_turn20` | `484142` | `PD (Priority)` | `TRAIN_BATCH_SIZE=64`, `ROLLOUT_MINI_BATCH_SIZE=16`, `SERVER_NAVIGATION_MAX_WORKERS=4` |
| `speed_b64_rmb32_w4_turn20` | `484143` | `PD (Priority)` | `TRAIN_BATCH_SIZE=64`, `ROLLOUT_MINI_BATCH_SIZE=32`, `SERVER_NAVIGATION_MAX_WORKERS=4` |

Pending result fields:

- W&B URL
- vLLM initialization status
- step 5 validation status
- average step time
- action distribution metrics
- crash/error summary
- next decision

#### Wave 2 Attempt 2 Partial Result

- Checked at: 2026-07-22 11:01 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit used by submitted jobs: `388b741e1879e1ecf4c328cb7668da06b76033d7`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-sweep-3ed39f8`
- Current status: three corrected debug5 runs completed, one batch32 env-concurrency run failed, two batch64 runs were canceled, one batch32/rmb16 run is still running.

| Variant | Job ID | Result | W&B URL | Key metrics / error | Decision |
| --- | --- | --- | --- | --- | --- |
| `speed_b8_rmb4_w1_turn20` | `484136` | Canceled at 43m19s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/7gv330pb | Logged steps 1-4 only; step 4 `timing_s/step=201.295`; too slow and unstable compared with rmb8 | Reject |
| `speed_b8_rmb8_w1_turn20` | `484137` | Completed at 22m45s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/a5yam58j | Completed step5 validation. Step 5: `train/success=0.250`, `val/success=0.125`, `train/action/top_share=0.820`, `val/action/top_share=0.925`, `timing_s/gen=107.151`, `timing_s/testing=505.492`, `timing_s/step=619.688` | Fastest complete debug5, but val action collapse risk is high |
| `speed_b16_rmb8_w2_turn20` | `484138` | Completed at 32m01s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/m75cd5wf | Completed step5 validation. Step 5: `train/success=0.000`, `val/success=0.000`, `train/action/top_share=0.472`, `val/action/top_share=0.295`, `timing_s/gen=181.691`, `timing_s/testing=517.393`, `timing_s/step=709.942` | More diverse action distribution but slower and weaker success |
| `speed_b16_rmb16_w4_turn20` | `484139` | Completed at 28m16s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/ovjt4zq9 | Completed step5 validation. Step 5: `train/success=0.188`, `val/success=0.250`, `train/action/top_share=0.435`, `val/action/top_share=0.481`, `timing_s/gen=190.018`, `timing_s/testing=485.249`, `timing_s/step=689.329` | Best stability/action-distribution candidate so far |
| `speed_b32_rmb8_w4_turn20` | `484140` | Failed after 26m15s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/ij1758ka | `requests.exceptions.ReadTimeout` in `create_environments_batch`; server returned only 7 AI2-THOR initializations before timeout | Reject batch32/rmb8/w4 |
| `speed_b32_rmb16_w4_turn20` | `484141` | Running at 20m59s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/27husy33 | Logged steps 1-2; step 2 `timing_s/step=340.989`, `train/action/top_share=0.531`; server init count 39 | Wait briefly, likely slower than b16/rmb16 |
| `speed_b64_rmb16_w4_turn20` | `484142` | Canceled at 1m02s | Not created before cancellation | Canceled after batch32/rmb8/w4 showed AI2-THOR create timeout | Stop larger batch direction |
| `speed_b64_rmb32_w4_turn20` | `484143` | Canceled at 0m32s | Not created before cancellation | Canceled after batch32/rmb8/w4 showed AI2-THOR create timeout | Stop larger batch direction |

Interim conclusion:

- The speed bottleneck was largely `ROLLOUT_MINI_BATCH_SIZE=1`; increasing it to 8 or 16 makes corrected debug5 complete.
- The fastest complete run is `speed_b8_rmb8_w1_turn20`, but its validation action top share `0.925` is too collapse-looking.
- The best current compromise is `speed_b16_rmb16_w4_turn20`: it completed in under 30 minutes and has much healthier action distribution, but it is slower than batch8/rmb8.
- Batch32 with worker=4 is risky: `speed_b32_rmb8_w4_turn20` failed from AI2-THOR environment creation timeout, so batch64 was canceled.

#### Wave 1 Attempt 3 Result

- Checked at: 2026-07-22 08:42 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit: `8032a37f596bfd13f1b16d80c458afb71d9384cc`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-sweep-3ed39f8`
- Action taken: canceled all Attempt 3 jobs after throughput evidence showed the smoke workload would not finish step 1 in a useful time.

| Variant | Job ID | Result | W&B URL | Evidence | Decision |
| --- | --- | --- | --- | --- | --- |
| `vagenrt_gpu04_limit8` | `483950` | Canceled after 20m37s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/n0oxh8bg | vLLM initialized with `# cuda blocks: 35850`, W&B created, reached `global_steps: 1`, no 6-image error; server completed only 12 AI2-THOR initializations in about 20m | Engine fix works; smoke batch is too large |
| `vagenrt_gpu06_limit8` | `483951` | Canceled after 18m36s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/rf6l1nih | vLLM initialized with `# cuda blocks: 93529`, W&B created, reached `global_steps: 1`, no 6-image error; server completed 10 initializations in about 18m | gpu mem 0.6 is engine-stable but not solving rollout throughput |
| `eager_gpu04_limit8` | `483952` | Canceled after 9m05s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/pakt3aoy | W&B created and reached `global_steps: 1`; server completed only 2 initializations before cancellation | Eager is not the first throughput fix |
| `eager_gpu06_limit8` | `483953` | Canceled while pending | Not created | Canceled because same Wave 1 workload was already too heavy | Stop batch128 smoke direction |
| `eager_free_gpu04_limit8` | `483954` | Canceled while pending | Not created | Canceled because same Wave 1 workload was already too heavy | Stop batch128 smoke direction |
| `eager_chunk_gpu04_limit8` | `483955` | Canceled while pending | Not created | Canceled because same Wave 1 workload was already too heavy | Stop batch128 smoke direction |
| `failed_minus_limit8_gpu06` | `483956` | Canceled while pending | Not created | Canceled because same Wave 1 workload was already too heavy | Stop batch128 smoke direction |
| `tp1_diag_gpu06_limit8` | `483957` | Canceled while pending | Not created | Canceled because same Wave 1 workload was already too heavy | Stop batch128 smoke direction |

Root cause update:

- Attempt 2 root cause was invalid `limit_mm_per_prompt=5`.
- Attempt 3 shows `limit8` fixes that layer: vLLM initializes, W&B starts, and the 6-image prompt no longer crashes.
- The next blocker is rollout throughput: Wave 1 used the formal default `TRAIN_BATCH_SIZE=128` with `max_turns=20` and `SERVER_NAVIGATION_MAX_WORKERS=1`.
- Step=1 has many more LLM/env interaction rounds than Step=5, so a 128-trajectory smoke is not a smoke test. It is large enough to hide whether the engine is healthy.

#### Wave 1 Attempt 4 Change Rationale

- Change type: reduce only the Wave 1 engine-smoke workload, not the formal VAGEN1 defaults.
- Planned parameter diff:
  - Wave 1 W&B group: `navigation_vagen1_engine_smoke_limit8_20260722 -> navigation_vagen1_engine_smoke_limit8_batch8_20260722`
  - Wave 1 `TRAIN_BATCH_SIZE`: inherited `128 -> 8`
  - Wave 1 `PPO_MINI_BATCH_SIZE`: inherited `32 -> 8`
  - Keep `VAL_BATCH_SIZE=1`, `SERVER_NAVIGATION_MAX_WORKERS=1`, `ROLLOUT_LIMIT_MM_PER_PROMPT=8`, `max_turns=20`, `window_size=5`
- Reason:
  - The engine smoke should answer whether a runtime variant can initialize, create W&B, execute rollout/training once, and report metrics.
  - Batch 8 should finish within the 30-minute pass criterion based on the observed 10-12 trajectory initializations in about 18-20 minutes.
  - Keeping worker count at 1 isolates the workload-size fix before changing AI2-THOR concurrency.
- Expected result:
  - At least one `limit8` runtime finishes step 1.
  - If step 1 still exceeds 30 minutes at batch 8, the next variable is `SERVER_NAVIGATION_MAX_WORKERS`, not vLLM memory.

#### Wave 1 Attempt 4 Submission

- Submitted at: 2026-07-22 08:43:30 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit: `fe666658db326b470a558592a4e8e05f4c2966bd`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-sweep-3ed39f8`
- Submit log: `/project/peilab/hligb/vagen-navigation/logs/vagen1-wave1-limit8-batch8-submit-20260722T084330Z.log`
- W&B group: `navigation_vagen1_engine_smoke_limit8_batch8_20260722`
- Shared settings: 4GPU normal, 2h time limit, `TOTAL_TRAINING_STEPS=1`, `TRAIN_BATCH_SIZE=8`, `PPO_MINI_BATCH_SIZE=8`, `VAL_BATCH_SIZE=1`, `VAL_BEFORE_TRAIN=False`, `SAVE_FREQ=-1`, `TEST_FREQ=1`, `FORCE_GEN_DATA=1`, `ROLLOUT_LIMIT_MM_PER_PROMPT=8`

| Variant | Job ID | Status at Submit | Notes |
| --- | --- | --- | --- |
| `vagenrt_gpu04_limit8` | `483995` | `PD (Priority)` | VAGEN runtime, gpu mem 0.4 |
| `vagenrt_gpu06_limit8` | `483996` | `PD (Priority)` | VAGEN runtime, gpu mem 0.6 |
| `eager_gpu04_limit8` | `483997` | `PD (Priority)` | Eager, gpu mem 0.4 |
| `eager_gpu06_limit8` | `483998` | `PD (Priority)` | Eager, gpu mem 0.6 |
| `eager_free_gpu04_limit8` | `483999` | `PD (Priority)` | Eager + free cache |
| `eager_chunk_gpu04_limit8` | `484000` | `PD (Priority)` | Eager + chunked prefill |
| `failed_minus_limit8_gpu06` | `484001` | `PD (Priority)` | Previous failed runtime except `limit20 -> limit8`, batch8 |
| `tp1_diag_gpu06_limit8` | `484002` | `PD (Priority)` | TP=1 diagnostic with `limit8`, batch8 |

Pending result fields:

- W&B URL
- vLLM initialization status
- step 1 completion time
- crash/error summary
- action distribution metrics
- next decision

#### Wave 1 Attempt 1 Result

- Checked at: 2026-07-22 08:02 UTC
- Jobs: `483923`-`483930`
- Action taken: canceled remaining jobs after first failures.
- Root cause: script/Hydra configuration error, not vLLM or AI2-THOR.
- Error:
  - `Could not append to config. An item is already at 'rollout_manager.loss_mask_mode'.`
  - We added `loss_mask_mode` to `vagen/trainer/config/ppo_trainer.yaml` but still launched with `+rollout_manager.loss_mask_mode=$LOSS_MASK_MODE`.
- Fix:
  - Change launch override to `rollout_manager.loss_mask_mode=$LOSS_MASK_MODE`.
- Decision:
  - Re-run Wave 1 after commit/push/server pull.

#### Wave 1 Attempt 2 Submission

- Submitted at: 2026-07-22 08:03:37 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit: `30a1fc8097f7696afb030a7fcb22c2e5051624bc`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-sweep-3ed39f8`
- Submit log: `/project/peilab/hligb/vagen-navigation/logs/vagen1-wave1-resubmit-20260722T080337Z.log`

| Variant | Job ID | Status at Submit |
| --- | --- | --- |
| `vagenrt_gpu01_limit5` | `483933` | `PD (Priority)` |
| `vagenrt_gpu04_limit5` | `483934` | `PD (Priority)` |
| `vagenrt_gpu06_limit5` | `483935` | `PD (Priority)` |
| `eager_gpu04_limit5` | `483936` | `PD (Priority)` |
| `eager_free_gpu04_limit5` | `483937` | `PD (Priority)` |
| `eager_chunk_gpu04_limit5` | `483938` | `PD (Priority)` |
| `failed_minus_limit20_gpu06` | `483939` | `PD (Priority)` |
| `tp1_diag_gpu06_limit5` | `483940` | `PD (Priority)` |

#### Wave 1 Attempt 2 Result

- Checked at: 2026-07-22 08:16 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Code commit used by submitted scripts: `30a1fc8097f7696afb030a7fcb22c2e5051624bc`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-sweep-3ed39f8`
- Shared settings: 4GPU normal, 2h, `TOTAL_TRAINING_STEPS=1`, `VAL_BEFORE_TRAIN=False`, `SAVE_FREQ=-1`, `TEST_FREQ=1`, `FORCE_GEN_DATA=1`, `max_turns=20`, `window_size=5`, `update_window_size=5`, `FORMAT_REWARD=0.1`, `LOSS_MASK_MODE=default`

| Variant | Job ID | Result | W&B URL | Evidence | Decision |
| --- | --- | --- | --- | --- | --- |
| `vagenrt_gpu01_limit5` | `483933` | Failed after 3m23s | Not created before failure | vLLM KV cache had `# cuda blocks: 0` with `gpu_memory_utilization=0.10`; error: `No available memory for the cache blocks` | Reject gpu mem 0.1 |
| `vagenrt_gpu04_limit5` | `483934` | Canceled after 8m29s | Not created before cancellation | Same `limit5` direction became invalid after two generation failures below; it had reached config validation but had not reached W&B or step 1 | Stop limit5 direction |
| `vagenrt_gpu06_limit5` | `483935` | Failed after 4m10s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/flvblcli | vLLM initialized and reached `global_steps: 1`, then generation failed with `limit-mm-per-prompt={"image": 5}` but prompt had 6 image items | `limit_mm_per_prompt=5` is too low for `window_size=5` |
| `eager_gpu04_limit5` | `483936` | Failed after 3m58s | https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/e0bzd9f4 | vLLM initialized and reached `global_steps: 1`, then failed with the same 6-image prompt error | Confirms root cause is limit, not eager |
| `eager_free_gpu04_limit5` | `483937` | Canceled after 12s | Not created | Canceled by stop rule after two same-root-cause `limit5` failures | Stop limit5 direction |
| `eager_chunk_gpu04_limit5` | `483938` | Canceled while pending | Not created | Canceled by stop rule | Stop limit5 direction |
| `failed_minus_limit20_gpu06` | `483939` | Canceled while pending | Not created | Canceled by stop rule | Stop limit5 direction |
| `tp1_diag_gpu06_limit5` | `483940` | Canceled while pending | Not created | Canceled by stop rule | Stop limit5 direction |

Root cause:

- `window_size=5` does not mean vLLM will see only 5 images. The prompt can include the current observation plus the 5-image history/update window, so vLLM can receive 6 image items in one prompt.
- Therefore `ROLLOUT_LIMIT_MM_PER_PROMPT=5` is an invalid lower bound for VAGEN1 with `window_size=5`.
- This is separate from the earlier `limit20` memory pressure problem: `limit5` is too small to run, while `limit20` over-reserves memory.

Next change hypothesis:

- Keep the VAGEN-first logic unchanged, but replace the invalid `limit5` sweep with `limit8`.
- Preserve `window_size=5`, `update_window_size=5`, `max_turns=20`, and the single-action parser.
- Re-run Wave 1 engine smoke with the same small 1-step jobs and compare gpu mem 0.4 vs 0.6 plus eager/no-eager.
- Expected result: vLLM should initialize and generation should no longer fail on 6-image prompts. If `limit8` is stable, move to Wave 2 speed debug5; if memory fails, test the narrower exact bound `limit6` before changing other runtime knobs.

#### Wave 1 Attempt 3 Change Rationale

- Local branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Change type: minimal runtime correction after Attempt 2 evidence.
- Files changed:
  - `scripts/superpod/configure_navigation_vagen1_variant.sh`
  - `scripts/superpod/submit_navigation_vagen1_wave1_engine_smoke.sh`
  - `scripts/superpod/run_navigation_vagen1_4gpu.sbatch`
  - `scripts/examples/vagen_base/navigation_vagen1/run.sh`
- Parameter diff:
  - VAGEN1 default `ROLLOUT_LIMIT_MM_PER_PROMPT`: `5 -> 8`
  - Wave 1 group: `navigation_vagen1_engine_smoke_20260722 -> navigation_vagen1_engine_smoke_limit8_20260722`
  - Wave 1 variants changed from `limit5` to:
    - `vagenrt_gpu04_limit8`
    - `vagenrt_gpu06_limit8`
    - `eager_gpu04_limit8`
    - `eager_gpu06_limit8`
    - `eager_free_gpu04_limit8`
    - `eager_chunk_gpu04_limit8`
    - `failed_minus_limit8_gpu06`
    - `tp1_diag_gpu06_limit8`
- Reason:
  - `limit5` is invalid for the current prompt construction because `window_size=5` can still produce 6 image items.
  - `limit8` gives headroom for the current image plus the 5-image window while staying far below the earlier over-reserving `limit20`.
- Metrics to observe:
  - W&B run creation.
  - vLLM KV cache block count.
  - No `passed 6 image items` error.
  - Step 1 completion under 30 minutes.
  - `train/action/top_share`, `train/action/entropy`, `train/format/too_many_actions`, and reward/success if step 1 completes.

#### Wave 1 Attempt 3 Submission

- Submitted at: 2026-07-22 08:20:23 UTC
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit: `8032a37f596bfd13f1b16d80c458afb71d9384cc`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-sweep-3ed39f8`
- Submit log: `/project/peilab/hligb/vagen-navigation/logs/vagen1-wave1-limit8-submit-20260722T082023Z.log`
- W&B group: `navigation_vagen1_engine_smoke_limit8_20260722`
- Shared settings: 4GPU normal, 2h time limit, `TOTAL_TRAINING_STEPS=1`, `VAL_BEFORE_TRAIN=False`, `SAVE_FREQ=-1`, `TEST_FREQ=1`, `FORCE_GEN_DATA=1`, `ROLLOUT_LIMIT_MM_PER_PROMPT=8`

| Variant | Job ID | Status at Submit | Notes |
| --- | --- | --- | --- |
| `vagenrt_gpu04_limit8` | `483950` | `PD (Priority)` | VAGEN runtime, gpu mem 0.4 |
| `vagenrt_gpu06_limit8` | `483951` | `PD (Priority)` | VAGEN runtime, gpu mem 0.6 |
| `eager_gpu04_limit8` | `483952` | `PD (Priority)` | Eager, gpu mem 0.4 |
| `eager_gpu06_limit8` | `483953` | `PD (Priority)` | Eager, gpu mem 0.6 |
| `eager_free_gpu04_limit8` | `483954` | `PD (Priority)` | Eager + free cache |
| `eager_chunk_gpu04_limit8` | `483955` | `PD (Priority)` | Eager + chunked prefill |
| `failed_minus_limit8_gpu06` | `483956` | `PD (Priority)` | Previous failed runtime except `limit20 -> limit8` |
| `tp1_diag_gpu06_limit8` | `483957` | `PD (Priority)` | TP=1 diagnostic with `limit8` |

Pending result fields:

- W&B URL
- vLLM initialization status
- step 1 completion time
- crash/error summary
- action distribution metrics
- next decision

### E3.7 Train3x4 Batch Size Correction Rationale

- Prepared at: 2026-07-22 23:36 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting local commit: `281d3d8`
- Change type: correct the 3-node x 4GPU training batch settings while keeping the new 12h env jobs alive.
- Problem:
  - E3.6 submitted one train job with `TRAIN_NNODES=3`, `N_GPUS_PER_NODE=4`, so the trainer sees `total n_gpus=12`.
  - The submitted `TRAIN_BATCH_SIZE=16` is not divisible by 12.
  - `vagen/trainer/ppo/ray_trainer.py` asserts that `real_train_batch_size` must be divisible by total GPU count, so the current pending train job would likely fail during config validation even before rollout quality can be measured.
- Planned action:
  - Keep env jobs `484462` and `484463`; do not waste the 12h env-server allocations.
  - Cancel only the pending train job `484464`.
  - Submit a replacement 3x4GPU train job with:
    - `TRAIN_BATCH_SIZE=48`
    - `PPO_MINI_BATCH_SIZE=24`
    - `ROLLOUT_MINI_BATCH_SIZE=24`
    - `TRAIN_NNODES=3`
    - `N_GPUS_PER_NODE=4`
    - `EXPECTED_RAY_GPUS=12`
    - `TOTAL_TRAINING_STEPS=6`
    - `TEST_FREQ=5`
    - `SAVE_FREQ=-1`
    - `VAL_BEFORE_TRAIN=False`
    - `FINAL_VAL_AFTER_TRAIN=False`
- Reason:
  - `48` is divisible by 12 and should avoid the trainer's batch divisibility assertion.
  - It is large enough to use a 12GPU train allocation more sensibly than batch16.
  - `ROLLOUT_MINI_BATCH_SIZE=24` should shard to roughly 12 rollout environments per external env server, which is heavier than the stable batch16 line but still less aggressive than jumping directly to batch64/96-style pressure.
- Expected observations:
  - The replacement train job starts without the batch-size assertion.
  - Logs show `TRAIN_NNODES=3`, `N_GPUS_PER_NODE=4`, `EXPECTED_RAY_GPUS=12`, and Ray resources with at least 12 GPUs.
  - Training logs show both env ready files through `ROLLOUT_BASE_URLS`.
  - If it still fails before step 1, the next check is Ray/vLLM multi-node startup or env-server throughput, not batch divisibility.

### E3.7 Train3x4 Batch Size Correction Submission Result

- Submitted at: 2026-07-22 23:37-23:38 HKT
- Local/GitHub rationale commit before Slurm action: `276b21b`
- SuperPOD code path used: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- SuperPOD code commit used by train job: `ffaf5058bd2aa19aaafab1575125973e4709021c`
- Action taken:
  - Canceled old train job `484464`.
  - Kept env jobs `484462` and `484463`.
  - Submitted replacement train job `484484`.
- Current env jobs after submission:
  - `484462`: `vagen-nav-ai2thor`, `1 node x 4 GPU`, `TimeLimit=12:00:00`, running on `dgx-54`.
  - `484463`: `vagen-nav-ai2thor`, `1 node x 4 GPU`, `TimeLimit=12:00:00`, pending priority.
- Replacement train job:
  - Job ID: `484484`
  - Job name: `vagen-nav-vagen1-2x4-ext`
  - Status after submit: `PENDING`, reason `Dependency`
  - Dependency after submit: `after:484463(unfulfilled)`; `484462` was already running, so only `484463` remained unfulfilled.
  - Resources: `node=3`, `gres/gpu=12`, `cpu=192`, `mem=1152G`
  - Time limit: `08:00:00`
  - Script: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/scripts/superpod/run_navigation_vagen1_2node4gpu_external_server.sbatch`
  - WorkDir: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Ready files:
  - `/project/peilab/hligb/vagen-navigation/logs/navigation-ai2thor-server-navigation_vagen1_train3x4_b16_rmb16_w4x2_train3x4_longenv_ffaf505_20260722T152402Z_server1.env`
  - `/project/peilab/hligb/vagen-navigation/logs/navigation-ai2thor-server-navigation_vagen1_train3x4_b16_rmb16_w4x2_train3x4_longenv_ffaf505_20260722T152402Z_server2.env`
- W&B:
  - Group: `navigation_vagen1_external2_train3x4_debug5_20260722`
  - Name: `navigation_vagen1_train3x4_b48_rmb24_external_ai2thor_train3x4_b48_rmb24_longenv_ffaf505_20260722T153758Z`
  - URL: pending until train job starts and creates the run.
- Training settings:
  - `VAGEN1_VARIANT=manual`
  - `TRAIN_NNODES=3`
  - `N_GPUS_PER_NODE=4`
  - `EXPECTED_RAY_GPUS=12`
  - `TRAIN_BATCH_SIZE=48`
  - `PPO_MINI_BATCH_SIZE=24`
  - `ROLLOUT_MINI_BATCH_SIZE=24`
  - `TOTAL_TRAINING_STEPS=6`
  - `TEST_FREQ=5`
  - `SAVE_FREQ=-1`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
- Decision:
  - This replaces the invalid `batch16` train attempt for the 12GPU topology.
  - Next check should confirm `484463` starts, then `484484` starts, reads both ready files, and reaches Ray resources with at least 12 GPUs before evaluating rollout speed.

### E3.8 Overnight Train3x4 Aggressive Batch Rationale

- Prepared at: 2026-07-22 23:43 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting local commit: `2d52b56`
- Change type: replace the pending E3.7 train job with a longer overnight run and a larger 12GPU-compatible batch.
- Reason:
  - The user wants the run to keep training overnight rather than stop after debug5.
  - `TRAIN_BATCH_SIZE=64` is invalid for 12 train GPUs because it is not divisible by 12.
  - The next clean aggressive batch size is `96`; it is divisible by 12 and should make better use of the 3-node x 4GPU training allocation.
  - Because this is no longer only a startup smoke, the run should save checkpoints. Use `SAVE_FREQ=20` and keep previous checkpoints so step20/40/60 can be inspected.
- Planned action:
  - Keep env jobs `484462` and `484463`.
  - Cancel only pending train job `484484`.
  - Submit a replacement train job with:
    - `TRAIN_NNODES=3`
    - `N_GPUS_PER_NODE=4`
    - `EXPECTED_RAY_GPUS=12`
    - `TRAIN_BATCH_SIZE=96`
    - `PPO_MINI_BATCH_SIZE=48`
    - `ROLLOUT_MINI_BATCH_SIZE=48`
    - `TOTAL_TRAINING_STEPS=60`
    - `SAVE_FREQ=20`
    - `TEST_FREQ=20`
    - `REMOVE_PREVIOUS_CKPT_IN_SAVE=False`
    - `VAL_BEFORE_TRAIN=False`
    - `FINAL_VAL_AFTER_TRAIN=False`
- Expected observations:
  - The job should not fail on the train batch divisibility assertion.
  - If it starts, W&B should show at least startup, Ray/vLLM initialization, and step20/40/60 if it survives long enough.
  - If it times out or hangs before step 1, the likely bottleneck is external AI2-THOR throughput under `ROLLOUT_MINI_BATCH_SIZE=48`, not the train batch divisibility issue.

### E3.8 Overnight Train3x4 Aggressive Batch Submission Result

- Submitted at: 2026-07-22 23:44 HKT
- Local/GitHub rationale commit before Slurm action: `b7fccc9`
- SuperPOD code path used: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- SuperPOD sync result:
  - Before pull: `ffaf5058bd2aa19aaafab1575125973e4709021c`
  - After fast-forward pull: `b7fccc9787105c566c4e30a37d20123b74799dd2`
- Action taken:
  - Canceled pending train job `484484`.
  - Kept env jobs `484462` and `484463`.
  - Submitted replacement overnight/aggressive train job `484493`.
- Current env jobs after submission:
  - `484462`: `vagen-nav-ai2thor`, `1 node x 4 GPU`, `TimeLimit=12:00:00`, running on `dgx-54`.
  - `484463`: `vagen-nav-ai2thor`, `1 node x 4 GPU`, `TimeLimit=12:00:00`, pending priority.
- Replacement train job:
  - Job ID: `484493`
  - Job name: `vagen-nav-vagen1-2x4-ext`
  - Status after submit: `PENDING`, reason `Dependency`
  - Dependency after submit: `after:484463(unfulfilled)`
  - Resources: `node=3`, `gres/gpu=12`, `cpu=192`, `mem=1152G`
  - Time limit: `08:00:00`
  - Script: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/scripts/superpod/run_navigation_vagen1_2node4gpu_external_server.sbatch`
  - WorkDir: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- W&B:
  - Group: `navigation_vagen1_external2_train3x4_overnight_20260722`
  - Name: `navigation_vagen1_train3x4_b96_rmb48_external_ai2thor_train3x4_b96_rmb48_longenv_ffaf505_20260722T154433Z`
  - URL: pending until train job starts and creates the run.
- Training settings:
  - `VAGEN1_VARIANT=manual`
  - `TRAIN_NNODES=3`
  - `N_GPUS_PER_NODE=4`
  - `EXPECTED_RAY_GPUS=12`
  - `TRAIN_BATCH_SIZE=96`
  - `PPO_MINI_BATCH_SIZE=48`
  - `ROLLOUT_MINI_BATCH_SIZE=48`
  - `TOTAL_TRAINING_STEPS=60`
  - `SAVE_FREQ=20`
  - `TEST_FREQ=20`
  - `REMOVE_PREVIOUS_CKPT_IN_SAVE=False`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
- Decision:
  - This is now the active overnight line.
  - Tomorrow, first check whether `484463` started early enough for the train job to use both ready files, then inspect whether `484493` reached Ray 12GPU initialization and step 1.

### E3.8 Overnight Train3x4 Status Check

- Checked at: 2026-07-23 11:30 HKT
- Local/GitHub commit at check: `d7979dc`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- SuperPOD clone commit at check: `d7979dc`
- Queue / accounting status:
  - `484462`: env server1, `TIMEOUT` after `12:00:28` on `dgx-54`.
  - `484463`: env server2, `RUNNING` for `03:19:30` on `dgx-29`.
  - `484493`: train job, `PENDING`, reason changed to `Priority`.
- Ready file status:
  - server1 ready file still exists and points to `http://10.23.1.181:7462`, but its Slurm job `484462` has timed out, so the ready file is stale.
  - server2 ready file exists and points to `http://10.23.0.237:7463`, and its Slurm job `484463` is still running.
- Train / W&B status:
  - No train stdout/stderr log content was present for `484493`.
  - The named train run log did not exist or was empty.
  - No W&B URL was created yet.
  - No checkpoint directories matching `navigation_vagen1_train3x4_b96_rmb48` were found.
- Interpretation:
  - The `after:` dependency only guaranteed that env jobs had started; it did not guarantee that both env servers would still be alive when the train job finally got scheduled.
  - If `484493` starts in the current state, it will likely pass the stale ready-file existence check and then fail the health check for the dead server1 URL.
- Decision:
  - Treat the active overnight topology as blocked by env lifetime / queue timing, not by VAGEN1 training code.
  - Next safe action should be to cancel `484493` and submit a fresh topology whose env servers live long enough for the queued train job, or change the train wrapper to validate server job liveness rather than trusting old ready files.

### E3.9 Integrated 5-Node Single Allocation Rationale

- Prepared at: 2026-07-23 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting local commit: `f6a5dfe`
- Change type: replace separate env-server jobs plus dependent train job with one Slurm allocation that co-schedules env and train nodes.
- Problem:
  - E3.8 showed that separate env jobs can start long before the train job is scheduled.
  - `484462` timed out after 12h while `484493` was still pending, leaving a stale ready file.
  - A dependent train job only knows the env jobs have started; it does not guarantee the env servers are still alive when training begins.
- Planned code change:
  - Add `scripts/superpod/run_navigation_vagen1_5node_integrated.sbatch`.
  - The script requests one 5-node allocation with `--gres=gpu:4` per node.
  - It starts `ENV_NNODES=2` AI2-THOR servers inside the same Slurm job.
  - It starts VAGEN1 RL on `TRAIN_NNODES=3` nodes in the same Slurm job.
  - It writes ready files only for the current `SLURM_JOB_ID` and validates both socket reachability and current-job ownership before training.
  - Cleanup kills env servers and Ray processes when the integrated job exits.
- Planned run parameters:
  - `TRAIN_BATCH_SIZE=96`
  - `PPO_MINI_BATCH_SIZE=48`
  - `ROLLOUT_MINI_BATCH_SIZE=48`
  - `TOTAL_TRAINING_STEPS=60`
  - `SAVE_FREQ=20`
  - `TEST_FREQ=20`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
  - `REMOVE_PREVIOUS_CKPT_IN_SAVE=False`
- Expected observations:
  - No stale ready file from a previous Slurm job can be accepted.
  - W&B should be created only after both env servers are live and Ray reaches 12 GPUs.
  - If the job fails, the failure should now be a real startup/throughput/training issue, not env lifetime vs queue timing.

### E3.9 Integrated 5-Node Single Allocation Submission Result

- Submitted at: 2026-07-23 11:43 HKT
- Code/doc commit used on SuperPOD: `974ccf6`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Server-side validation before submit:
  - `git pull --ff-only` updated the clone from `f6a5dfe` to `974ccf6`.
  - `bash -n scripts/superpod/run_navigation_vagen1_5node_integrated.sbatch` passed on SuperPOD.
- Cleanup before submit:
  - Canceled stale pending train job `484493`.
  - Canceled obsolete standalone env job `484463`.
- New integrated job:
  - Job ID: `484769`
  - Job name: `vagen-nav-vagen1-5n-int`
  - Status after submit: `PENDING`, reason `Priority`
  - Dependency: none
  - Time limit: `08:00:00`
  - Resources: `node=5`, `gres/gpu=20`, `cpu=320`, `mem=1920G`
  - Script: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/scripts/superpod/run_navigation_vagen1_5node_integrated.sbatch`
  - WorkDir: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- W&B:
  - Group: `navigation_vagen1_integrated_5node_overnight_20260723`
  - Name: `navigation_vagen1_integrated5n_b96_rmb48_total60_20260723T034349Z`
  - URL: pending until the integrated job starts and creates the run.
- Training topology and settings:
  - `ENV_NNODES=2`
  - `TRAIN_NNODES=3`
  - `N_GPUS_PER_NODE=4`
  - `EXPECTED_RAY_GPUS=12`
  - `TRAIN_BATCH_SIZE=96`
  - `PPO_MINI_BATCH_SIZE=48`
  - `ROLLOUT_MINI_BATCH_SIZE=48`
  - `TOTAL_TRAINING_STEPS=60`
  - `SAVE_FREQ=20`
  - `TEST_FREQ=20`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
  - `REMOVE_PREVIOUS_CKPT_IN_SAVE=False`
- Next check:
  - Confirm `484769` starts as one allocation.
  - Inspect integrated stdout for env node split, ready files owned by the current `SLURM_JOB_ID`, two successful AI2-THOR health checks, Ray reaching 12 GPUs, and W&B URL creation.

### E3.10 Integrated 3-Node Env2 Train1 Rationale

- Prepared at: 2026-07-23 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting local commit: `5ac614b`
- Change type: add a smaller single-allocation fallback while keeping the 5-node integrated job queued.
- Interpretation of the requested topology:
  - Treat "2x4GPU env, 1 train" as `ENV_NNODES=2` and `TRAIN_NNODES=1`, with each node still allocated as 4 GPUs.
  - This is a 3-node / 12GPU Slurm allocation: 8 GPUs for AI2-THOR env servers and 4 GPUs for VAGEN/Ray/vLLM training.
- Reason:
  - The 5-node / 20GPU job `484769` is stable in design but slow to schedule.
  - A 3-node single allocation should be much easier for Slurm to place while still avoiding stale ready files.
  - Keeping two env servers preserves env throughput, while reducing training to one 4GPU node avoids waiting for a 3-node training cluster.
- Planned code change:
  - Add `scripts/superpod/run_navigation_vagen1_3node_env2_train1_integrated.sbatch`.
  - The wrapper requests `--nodes=3`, `--time=04:00:00`, exports smaller defaults, then executes the generic integrated script.
- Planned run parameters:
  - `ENV_NNODES=2`
  - `TRAIN_NNODES=1`
  - `N_GPUS_PER_NODE=4`
  - `EXPECTED_RAY_GPUS=4`
  - `TRAIN_BATCH_SIZE=32`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=16`
  - `TOTAL_TRAINING_STEPS=60`
  - `SAVE_FREQ=20`
  - `TEST_FREQ=20`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
- Expected observations:
  - This job should queue faster than `484769`.
  - If it starts, it should create two current-job-owned env ready files, health-check both env servers, start a 4GPU Ray cluster, and create W&B.
  - If it is too slow after startup, the next bottleneck is likely 4GPU training/vLLM throughput rather than env server availability.
  - The 4h limit is intentionally short: this run is for parameter/topology validation, not a formal overnight run.

### E3.10 Integrated 3-Node Env2 Train1 Submission Result

- Submitted at: 2026-07-23 15:39 HKT
- Code/doc commit used on SuperPOD: `b267da5`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Server-side validation before submit:
  - `git pull --ff-only` updated the clone from `5ac614b` to `b267da5`.
  - `bash -n scripts/superpod/run_navigation_vagen1_3node_env2_train1_integrated.sbatch` passed on SuperPOD.
  - `bash -n scripts/superpod/run_navigation_vagen1_5node_integrated.sbatch` passed on SuperPOD.
- Existing 5-node job:
  - `484769` was kept queued.
  - Its latest observed start estimate moved to `2026-07-25T22:42:27 HKT`, so it is no longer a near-term validation path.
- New smaller integrated job:
  - Job ID: `484963`
  - Job name: `vagen-nav-vagen1-3n-int`
  - Status after submit: `PENDING`, reason `Priority`
  - Dependency: none
  - Time limit: `04:00:00`
  - Resources: `node=3`, `gres/gpu=12`, `cpu=192`, `mem=1152G`
  - Script: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/scripts/superpod/run_navigation_vagen1_3node_env2_train1_integrated.sbatch`
  - WorkDir: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- W&B:
  - Group: `navigation_vagen1_integrated_3node_env2_train1_20260723`
  - Name: `navigation_vagen1_integrated3n_env2_train1_b32_rmb16_20260723T073937Z`
  - URL: pending until the job starts and creates the run.
- Training topology and settings:
  - `ENV_NNODES=2`
  - `TRAIN_NNODES=1`
  - `N_GPUS_PER_NODE=4`
  - `EXPECTED_RAY_GPUS=4`
  - `TRAIN_BATCH_SIZE=32`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=16`
  - `TOTAL_TRAINING_STEPS=60`
  - `SAVE_FREQ=20`
  - `TEST_FREQ=20`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
- Next check:
  - Confirm whether `484963` receives a real start estimate before `484769`.
  - If it starts, inspect env health checks, Ray `GPU=4`, W&B creation, step 1 timing, and whether `ROLLOUT_MINI_BATCH_SIZE=16` is conservative enough.

### E3.11 Integrated 3-Node Env2 Train1 2h Rationale

- Prepared at: 2026-07-23 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting local commit: `b990a98`
- Change type: submit an additional shorter-walltime version of the E3.10 3-node integrated topology.
- Reason:
  - The current 4h E3.10 job `484963` is valid but still queued behind larger jobs.
  - A 2h request may be easier for Slurm to backfill and is enough to validate startup, env health checks, Ray/vLLM initialization, W&B creation, and whether step 1 begins.
  - This is not meant to replace the 4h run if the 4h run starts first; it is a faster scheduling probe with the same model/env parameters.
- Planned submission method:
  - Reuse `scripts/superpod/run_navigation_vagen1_3node_env2_train1_integrated.sbatch`.
  - Override only Slurm walltime at submit time with `sbatch --time=02:00:00`.
  - Set a separate W&B group/name so the 2h probe is distinguishable from the 4h probe.
- Planned run parameters:
  - `ENV_NNODES=2`
  - `TRAIN_NNODES=1`
  - `N_GPUS_PER_NODE=4`
  - `EXPECTED_RAY_GPUS=4`
  - `TRAIN_BATCH_SIZE=32`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=16`
  - `TOTAL_TRAINING_STEPS=60`
  - `SAVE_FREQ=20`
  - `TEST_FREQ=20`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
- Expected observations:
  - If the 2h job backfills earlier, it should create a W&B run and reveal whether this topology can at least enter training quickly.
  - If it starts but does not reach step 1 within the 2h window, the result still helps identify startup/env/Ray/vLLM time as the bottleneck.
  - If it remains pending while the 4h job also remains pending, the bottleneck is queue availability for 3 nodes / 12 GPUs, not walltime alone.

### E3.11 Integrated 3-Node Env2 Train1 2h Submission Result

- Submitted at: 2026-07-23 15:55 HKT
- Rationale/doc commit used on SuperPOD: `d429c9c`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Server-side validation before submit:
  - `source scripts/superpod/load_modules.sh` switched `sbatch` to `/cm/shared/apps/slurm/current/bin/sbatch`.
  - `sbatch --version` reported `slurm 23.02.6`.
  - SuperPOD clone was already at `d429c9c55b5456316130857a50b5778b42ac7802`.
  - `bash -n scripts/superpod/run_navigation_vagen1_3node_env2_train1_integrated.sbatch` passed on SuperPOD.
  - `bash -n scripts/superpod/run_navigation_vagen1_5node_integrated.sbatch` passed on SuperPOD.
- New 2h integrated job:
  - Job ID: `485011`
  - Job name: `vagen-nav-vagen1-3n-int`
  - Status after submit: `PENDING`, reason `Priority`
  - Dependency: none
  - Time limit: `02:00:00`
  - Resources: `node=3`, `gres/gpu=12`, `cpu=192`, `mem=1152G`
  - Script: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/scripts/superpod/run_navigation_vagen1_3node_env2_train1_integrated.sbatch`
  - WorkDir: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
  - Stdout: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-3node-env2-train1-integrated-485011.out`
  - Stderr: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-3node-env2-train1-integrated-485011.err`
- W&B:
  - Group: `navigation_vagen1_integrated_3node_env2_train1_2h_20260723`
  - Name: `navigation_vagen1_integrated3n_env2_train1_b32_rmb16_2h_20260723T075542Z`
  - URL: pending until the job starts and creates the run.
  - Train log: `/project/peilab/hligb/vagen-navigation/logs/navigation_vagen1_integrated3n_env2_train1_b32_rmb16_2h_20260723T075542Z.log`
- Queue context after submit:
  - `484769`: 5-node / 20GPU / 8h integrated job, still pending priority.
  - `484963`: 3-node / 12GPU / 4h integrated job, still pending priority.
  - `485011`: 3-node / 12GPU / 2h integrated job, pending priority with `START_TIME=N/A` at submission.
- Note:
  - Earlier submit attempts in this session did not create jobs because non-interactive SSH invoked the site Slurm hint wrapper before `load_modules.sh`; no duplicate Slurm job was created before `485011`.
- Next check:
  - Compare whether `485011` backfills earlier than `484963`.
  - If `485011` starts, inspect current-job env ready files, two env health checks, Ray `GPU=4`, W&B creation, and whether step 1 begins inside the 2h window.

### E3.12 4GPU Turn10 Collapse Probe Rationale

- Prepared at: 2026-07-23 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting local commit: `2cf8b1f`
- Change type: add a single-node 4GPU VAGEN1 debug wrapper for the first collapse-parameter probe.
- Reason:
  - The integrated 3-node / 12GPU jobs are still waiting on queue availability, so they are not a fast way to decide collapse parameters.
  - A single 4GPU node should queue more easily and is sufficient for checking whether the Step=1 prompt/parser/reward setup enters training and whether the policy immediately collapses to one action.
  - `max_turns=10` reduces rollout length by half versus the current VAGEN1 `max_turns=20`, which should make startup and step-time diagnosis faster.
  - This run intentionally keeps `LOSS_MASK_MODE=default`; answer-only loss is a later sweep variable after we know the short-horizon baseline can run.
- Planned code change:
  - Add variant `turn10_defaultloss_fmt01` in `scripts/superpod/configure_navigation_vagen1_variant.sh`.
  - Add wrapper `scripts/superpod/run_navigation_vagen1_4gpu_turn10_debug.sbatch`.
- Planned run parameters:
  - Topology: one normal node, 4 GPUs, local AI2-THOR server inside the training job.
  - Walltime: `02:00:00`
  - `MAX_TURNS=10`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=4`
  - `SERVER_NAVIGATION_MAX_WORKERS=2`
  - `FORMAT_REWARD=0.1`
  - `LOSS_MASK_MODE=default`
  - `TOTAL_TRAINING_STEPS=21`
  - `TEST_FREQ=5`
  - `SAVE_FREQ=20`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
  - `ENV_CONFIG_PATH=scripts/examples/vagen_base/navigation_vagen1/env_config_speed.yaml`
- Expected observations:
  - The job should schedule earlier than the 3-node integrated jobs.
  - It should create W&B, start the local AI2-THOR server, initialize vLLM, and enter step 1 without the previous long-rollout timeout.
  - Action metrics should reveal whether a default-loss, format-0.1, turn10 run already collapses.
  - If `action/top_share` quickly approaches 1 or `all_same_traj` is high, the next 4GPU probe should use `LOSS_MASK_MODE=answer_only` before adding dense reward.

### E3.12 4GPU Turn10 Collapse Probe Submission Result

- Submitted at: 2026-07-23 16:11 HKT
- Code/doc commit used on SuperPOD: `56d21fb`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Server-side validation before submit:
  - `git pull --ff-only` updated the clone from `2cf8b1f` to `56d21fb65760925e7d1be17fb4c0b1964dd31912`.
  - `sbatch --version` reported `slurm 23.02.6`.
  - `bash -n scripts/superpod/run_navigation_vagen1_4gpu_turn10_debug.sbatch` passed on SuperPOD.
  - `bash -n scripts/superpod/run_navigation_vagen1_4gpu.sbatch` passed on SuperPOD.
  - `bash -n scripts/superpod/configure_navigation_vagen1_variant.sh` passed on SuperPOD.
- New 4GPU turn10 job:
  - Job ID: `485029`
  - Job name: `vagen-nav-vagen1-t10`
  - Status after submit: `PENDING`, reason `Priority`
  - Dependency: none
  - Time limit: `02:00:00`
  - Resources: `node=1`, `gres/gpu=4`, `cpu=64`, `mem=384G`
  - Script: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/scripts/superpod/run_navigation_vagen1_4gpu_turn10_debug.sbatch`
  - WorkDir: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
  - Stdout: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-4gpu-turn10-debug-485029.out`
  - Stderr: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-4gpu-turn10-debug-485029.err`
- W&B:
  - Group: `navigation_vagen1_turn10_4gpu_debug20_20260723`
  - Name: `navigation_vagen1_turn10_defaultloss_fmt01_4gpu_debug20_20260723T081143Z`
  - URL: pending until the job starts and creates the run.
  - Train log: `/project/peilab/hligb/vagen-navigation/logs/navigation_vagen1_turn10_defaultloss_fmt01_4gpu_debug20_20260723T081143Z.log`
- Queue context after submit:
  - `484769`: 5-node / 20GPU / 8h integrated job, still pending priority.
  - `484963`: 3-node / 12GPU / 4h integrated job, still pending priority.
  - `485011`: 3-node / 12GPU / 2h integrated job, still pending priority.
  - `485029`: 1-node / 4GPU / 2h turn10 job, pending priority with `START_TIME=N/A` at submission.
- Next check:
  - Confirm whether the 1-node job gets a real start time earlier than the 3-node integrated jobs.
  - If it starts, inspect local AI2-THOR health, W&B URL, vLLM initialization, step 1 timing, and action-distribution metrics before deciding the answer-only sweep.

### E3.12 4GPU Turn10 Collapse Probe Failure Result

- Checked at: 2026-07-23 16:48 HKT
- Job ID: `485029`
- W&B URL: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/4gdazkp4`
- Slurm status:
  - `FAILED`
  - Exit code: `1:0`
  - Elapsed: `00:23:44`
  - Node: `dgx-46`
- Startup status:
  - Local navigation server started on port `5029`.
  - W&B run was created successfully.
  - Data generation succeeded.
  - vLLM initialized successfully with TP=4.
  - Training reached `global_steps: 1`.
- Failure location:
  - The first training mini-batch printed `Processing mini-batch 1/4, size: 4`.
  - The crash occurred in `rollout_manager.reset()`.
  - Traceback path:
    - `ray_trainer.py::_process_in_mini_batches`
    - `rollout_manager_service.py::reset`
    - `BatchEnvClient.create_environments_batch`
    - `requests.post(.../environments, timeout=1200)`
  - Final error: `requests.exceptions.ReadTimeout: HTTPConnectionPool(host='localhost', port=5029): Read timed out. (read timeout=1200)`.
- Server evidence:
  - Server config: `NavigationServiceConfig(max_workers=2, devices=[0, 1, 2, 3], use_state_reward=False)`.
  - AI2-THOR logged three successful controller initializations:
    - `2026-07-23 16:15:27 Initialize return`
    - `2026-07-23 16:15:29 Initialize return`
    - `2026-07-23 16:15:31 Initialize return`
  - No fourth initialization or successful `/environments` response appeared before the client timed out.
- Interpretation:
  - This crash happened before rollout generation and before any action-distribution metrics.
  - It is not evidence for or against `LOSS_MASK_MODE=default` or `answer_only`.
  - The immediate root cause is local AI2-THOR environment creation hanging when the 4GPU training job asks the same node to create a mini-batch of 4 navigation envs with `SERVER_NAVIGATION_MAX_WORKERS=2`.
  - The short `MAX_TURNS=10` did not matter yet because the job did not get past environment creation.
- Next decision:
  - Do not continue the local 4GPU turn10 line with `ROLLOUT_MINI_BATCH_SIZE=4`.
  - For a one-node local sanity run, reduce `ROLLOUT_MINI_BATCH_SIZE=1` and `SERVER_NAVIGATION_MAX_WORKERS=1`, and optionally `TOTAL_TRAINING_STEPS=6`, just to prove one step can complete.
  - For collapse experiments that need useful throughput, prefer the integrated/external-env topology so AI2-THOR env creation is not competing with vLLM/FSDP on the same 4 GPUs.

### E3.13 4GPU Turn10 Answer-Only Lower-Pressure Rationale

- Prepared at: 2026-07-23 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting local commit: `f6fa15f`
- Change type: add a smaller local 4GPU debug probe that lowers AI2-THOR env creation pressure and begins the answer-only anti-collapse test.
- Reason:
  - E3.12 reached W&B, data generation, vLLM initialization, and `global_steps: 1`, but crashed before rollout generation because `create_environments_batch` timed out while creating a mini-batch of 4 envs with 2 server workers.
  - The failure was an env creation hang, not evidence about `LOSS_MASK_MODE=default` vs `answer_only`.
  - Reducing every dimension to 1 would only test survival and would give very weak signal for speed/action-distribution behavior, so the next probe keeps a real train batch while reducing only the rollout env creation pressure.
- Planned code change:
  - Add variant `turn10_answeronly_fmt005_rmb2_w1`.
  - Add wrapper `scripts/superpod/run_navigation_vagen1_4gpu_turn10_answeronly_debug.sbatch`.
- Planned run parameters:
  - Topology: one normal node, 4 GPUs, local AI2-THOR server inside the training job.
  - Walltime: `02:00:00`
  - `MAX_TURNS=10`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `SERVER_NAVIGATION_MAX_WORKERS=1`
  - `FORMAT_REWARD=0.05`
  - `LOSS_MASK_MODE=answer_only`
  - `TOTAL_TRAINING_STEPS=6`
  - `TEST_FREQ=5`
  - `SAVE_FREQ=-1`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
- Expected observations:
  - If this still hangs in `/environments`, even 2 envs per reset are too much for local single-node co-location, and local 4GPU debug should be used only for tiny preflight.
  - If it enters rollout and finishes step 1/debug5, compare action distribution and success with the failed/default-loss line before deciding whether `answer_only` becomes the default anti-collapse setting.
  - If speed is usable but action top share is high, keep this topology and sweep `FORMAT_REWARD=0.1` vs `0.05` before adding dense reward.

### E3.13 4GPU Turn10 Answer-Only Lower-Pressure Submission Result

- Submitted at: 2026-07-23 17:02 HKT
- Code/doc commit used on SuperPOD: `d78f163`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Server-side validation before submit:
  - `git pull --ff-only` updated the clone from `f6fa15f` to `d78f163963016c50b1b9b5dad8eba09316b34eb4`.
  - `bash -n scripts/superpod/run_navigation_vagen1_4gpu_turn10_answeronly_debug.sbatch` passed on SuperPOD.
  - `bash -n scripts/superpod/run_navigation_vagen1_4gpu.sbatch` passed on SuperPOD.
  - `bash -n scripts/superpod/configure_navigation_vagen1_variant.sh` passed on SuperPOD.
- New 4GPU answer-only job:
  - Job ID: `485078`
  - Job name: `vagen-nav-v1-t10-ao`
  - Status after submit: `PENDING`, reason `Priority`
  - Dependency: none
  - Time limit: `02:00:00`
  - Resources: one normal node, `gres/gpu=4`, `cpu=64`, `mem=384G`
  - Script: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/scripts/superpod/run_navigation_vagen1_4gpu_turn10_answeronly_debug.sbatch`
  - WorkDir: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- W&B:
  - Group: `navigation_vagen1_turn10_answeronly_4gpu_debug5_20260723`
  - Name: `navigation_vagen1_turn10_answeronly_fmt005_rmb2_w1_4gpu_debug5_20260723T090226Z`
  - URL: pending until the job starts and creates the run.
  - Train log: `/project/peilab/hligb/vagen-navigation/logs/navigation_vagen1_turn10_answeronly_fmt005_rmb2_w1_4gpu_debug5_20260723T090226Z.log`
- Training settings:
  - `VAGEN1_VARIANT=turn10_answeronly_fmt005_rmb2_w1`
  - `MAX_TURNS=10`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `SERVER_NAVIGATION_MAX_WORKERS=1`
  - `FORMAT_REWARD=0.05`
  - `LOSS_MASK_MODE=answer_only`
  - `TOTAL_TRAINING_STEPS=6`
  - `TEST_FREQ=5`
  - `SAVE_FREQ=-1`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
- Next check:
  - If it starts, inspect W&B URL creation, `/environments` latency for mini-batch size 2, step 1 completion, debug5 completion, and action distribution.
  - If it fails again at env creation, stop local 4GPU nontrivial rollout tests and wait for an integrated/external-env allocation.

### E3.14 4GPU Turn5 Answer-Only Short-Horizon Probe Rationale

- Prepared at: 2026-07-23 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting local commit: `6860c75`
- Change type: add a shorter-horizon local 4GPU probe to distinguish env/runtime viability from long rollout throughput.
- Source-code check:
  - Original VAGEN navigation uses `MAX_TURNS=5` with navigation env default `max_actions_per_step=5`, so the original maximum primitive-action budget is about 25 actions.
  - VAGEN1 uses `max_actions_per_step=1`; therefore `MAX_TURNS=5` gives only 5 primitive actions and is not a fair navigation-performance baseline.
- Reason:
  - The turn10 answer-only job is useful, but if it is still slow, a turn5 probe can tell whether shortening horizon lets the local 4GPU topology enter rollout/update quickly.
  - This experiment is an engineering probe, not a success-rate probe. Low success is expected because many navigation tasks cannot be solved within 5 single actions.
  - Keeping `TRAIN_BATCH_SIZE=16` and `ROLLOUT_MINI_BATCH_SIZE=2` avoids the "all dimensions are 1" case while still lowering env creation pressure.
- Planned code change:
  - Add variant `turn5_answeronly_fmt005_rmb2_w1`.
  - Add wrapper `scripts/superpod/run_navigation_vagen1_4gpu_turn5_answeronly_debug.sbatch`.
- Planned run parameters:
  - Topology: one normal node, 4 GPUs, local AI2-THOR server inside the training job.
  - Walltime: `02:00:00`
  - `MAX_TURNS=5`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `SERVER_NAVIGATION_MAX_WORKERS=1`
  - `FORMAT_REWARD=0.05`
  - `LOSS_MASK_MODE=answer_only`
  - `TOTAL_TRAINING_STEPS=6`
  - `TEST_FREQ=5`
  - `SAVE_FREQ=-1`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
- Expected observations:
  - If turn5 enters step/update while turn10 does not, rollout horizon and AI2-THOR step throughput are the bottleneck.
  - If turn5 still hangs in `/environments`, the local 4GPU topology is failing at env creation independently of horizon length.
  - If turn5 finishes debug5, use it only for instrumentation sanity and action-metric plumbing; do not use its success rate as evidence that the final VAGEN1 task is solved.

### E3.14 4GPU Turn5 Answer-Only Short-Horizon Probe Submission Result

- Submitted at: 2026-07-23 17:19 HKT
- Code/doc commit used on SuperPOD: `f8a6519`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Server-side validation before submit:
  - `git pull --ff-only` updated the clone from `6860c75` to `f8a65191f560ce00585b4535dd5861f4937dbd07`.
  - `bash -n scripts/superpod/run_navigation_vagen1_4gpu_turn5_answeronly_debug.sbatch` passed on SuperPOD.
  - `bash -n scripts/superpod/run_navigation_vagen1_4gpu.sbatch` passed on SuperPOD.
  - `bash -n scripts/superpod/configure_navigation_vagen1_variant.sh` passed on SuperPOD.
- New 4GPU turn5 answer-only job:
  - Job ID: `485088`
  - Job name: `vagen-nav-v1-t5-ao`
  - Status after submit: `PENDING`, reason `Priority`
  - Dependency: none
  - Time limit: `02:00:00`
  - Resources: one normal node, `gres/gpu=4`, `cpu=64`, `mem=384G`
  - Script: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/scripts/superpod/run_navigation_vagen1_4gpu_turn5_answeronly_debug.sbatch`
  - WorkDir: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- W&B:
  - Group: `navigation_vagen1_turn5_answeronly_4gpu_debug5_20260723`
  - Name: `navigation_vagen1_turn5_answeronly_fmt005_rmb2_w1_4gpu_debug5_20260723T091901Z`
  - URL: pending until the job starts and creates the run.
  - Train log: `/project/peilab/hligb/vagen-navigation/logs/navigation_vagen1_turn5_answeronly_fmt005_rmb2_w1_4gpu_debug5_20260723T091901Z.log`
- Training settings:
  - `VAGEN1_VARIANT=turn5_answeronly_fmt005_rmb2_w1`
  - `MAX_TURNS=5`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `SERVER_NAVIGATION_MAX_WORKERS=1`
  - `FORMAT_REWARD=0.05`
  - `LOSS_MASK_MODE=answer_only`
  - `TOTAL_TRAINING_STEPS=6`
  - `TEST_FREQ=5`
  - `SAVE_FREQ=-1`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
- Queue context after submit:
  - `484769`: 5-node / 20GPU / 8h integrated job, pending priority, estimated start `2026-07-25T22:42:27`.
  - `484963`: 3-node / 12GPU / 4h integrated job, pending priority, estimated start `2026-07-25T10:13:23`.
  - `485011`: 3-node / 12GPU / 2h integrated job, pending priority, estimated start `2026-07-25T10:13:23`.
  - `485078`: turn10 4GPU answer-only job, pending priority, estimated start `2026-07-23T20:53:20`.
  - `485088`: turn5 4GPU answer-only job, pending priority, estimated start `N/A` immediately after submission.
- Next check:
  - Compare whether turn5 receives an earlier start estimate than turn10.
  - If both start, turn5 should be interpreted as an env/runtime sanity probe and turn10 as the more informative short-horizon training probe.

### E3.15 Integrated Answer-Only Resubmission Rationale

- Prepared at: 2026-07-23 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting local commit: `1845b0a`
- Change type: replace queued default-loss integrated runs with answer-only integrated runs.
- Reason:
  - The queued integrated jobs `484769`, `484963`, and `485011` were created before the answer-only anti-collapse direction became the active route.
  - Those old runs use `LOSS_MASK_MODE=default` and `FORMAT_REWARD=0.1`, so if they eventually start they will answer the wrong question.
  - The new target is to test env/train separation and training stability under the same anti-collapse starting point as the local probes: `LOSS_MASK_MODE=answer_only` and `FORMAT_REWARD=0.05`.
- Planned cancellation:
  - Cancel old default-loss integrated jobs `484769`, `484963`, and `485011`.
  - Keep local short-horizon answer-only jobs `485078` and `485088` unchanged.
- Planned code changes:
  - `scripts/superpod/run_navigation_vagen1_5node_integrated.sbatch`
    - time limit `08:00:00 -> 12:00:00`
    - default `FORMAT_REWARD=0.05`
    - default `LOSS_MASK_MODE=answer_only`
    - W&B group/name/tags updated to answer-only.
  - `scripts/superpod/run_navigation_vagen1_3node_env2_train1_integrated.sbatch`
    - time limit `04:00:00 -> 02:00:00`
    - default `FORMAT_REWARD=0.05`
    - default `LOSS_MASK_MODE=answer_only`
    - W&B group/name/tags updated to answer-only.
- Parameters intentionally unchanged:
  - 5-node topology remains `ENV_NNODES=2`, `TRAIN_NNODES=3`, `TRAIN_BATCH_SIZE=96`, `PPO_MINI_BATCH_SIZE=48`, `ROLLOUT_MINI_BATCH_SIZE=48`, `TOTAL_TRAINING_STEPS=60`, `MAX_TURNS=20`, `SERVER_NAVIGATION_MAX_WORKERS=4`.
  - 3-node topology remains `ENV_NNODES=2`, `TRAIN_NNODES=1`, `TRAIN_BATCH_SIZE=32`, `PPO_MINI_BATCH_SIZE=16`, `ROLLOUT_MINI_BATCH_SIZE=16`, `TOTAL_TRAINING_STEPS=60`, `MAX_TURNS=20`, `SERVER_NAVIGATION_MAX_WORKERS=4`.
- Anti-collapse stance:
  - `LOSS_MASK_MODE=answer_only` plus single-action prompt is necessary but not sufficient.
  - Keep strict single-action prompt and parser behavior where multiple actions are treated as format failure.
  - Keep lowered format reward so formatting does not dominate sparse navigation reward.
  - Use action distribution W&B metrics as the gate; if `top_share` rises toward 1 or entropy collapses, answer-only alone is not enough.
  - Do not add dense reward or WM reward until integrated answer-only first proves stable and gives interpretable action metrics.
- Expected observations:
  - 3-node/2h should be the faster queue and startup check.
  - 5-node/12h should give env servers enough time inside one allocation and enough walltime to observe step timing and checkpoint behavior if it starts.

### E3.15 Integrated Answer-Only Resubmission Result

- Submitted at: 2026-07-23 17:31 HKT
- Code/doc commit used on SuperPOD: `d39d792`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Server-side validation before submit:
  - `git pull --ff-only` updated the clone from `1845b0a` to `d39d792785d779ec349a5d2c77668dc33796c85a`.
  - `bash -n scripts/superpod/run_navigation_vagen1_3node_env2_train1_integrated.sbatch` passed on SuperPOD.
  - `bash -n scripts/superpod/run_navigation_vagen1_5node_integrated.sbatch` passed on SuperPOD.
  - `bash -n scripts/superpod/configure_navigation_vagen1_variant.sh` passed on SuperPOD.
- Canceled old default-loss integrated jobs:
  - `484769`: old 5-node / 8h / default-loss integrated job.
  - `484963`: old 3-node / 4h / default-loss integrated job.
  - `485011`: old 3-node / 2h / default-loss integrated job.
- Kept local answer-only jobs:
  - `485078`: turn10 local 4GPU answer-only job, running at the time of this resubmission.
  - `485088`: turn5 local 4GPU answer-only job, still pending.
- New 3-node answer-only integrated job:
  - Job ID: `485096`
  - Job name: `vagen-nav-vagen1-3n-int`
  - Status after submit: `PENDING`, reason `Priority`
  - Time limit: `02:00:00`
  - Resources: `3` nodes, `12` total GPUs, with `ENV_NNODES=2` and `TRAIN_NNODES=1`.
  - W&B group: `navigation_vagen1_integrated_3node_env2_train1_answeronly_20260723`
  - W&B name: `navigation_vagen1_integrated3n_env2_train1_answeronly_b32_rmb16_20260723T093142Z`
  - Train log: `/project/peilab/hligb/vagen-navigation/logs/navigation_vagen1_integrated3n_env2_train1_answeronly_b32_rmb16_20260723T093142Z.log`
  - Key settings: `FORMAT_REWARD=0.05`, `LOSS_MASK_MODE=answer_only`, `TRAIN_BATCH_SIZE=32`, `PPO_MINI_BATCH_SIZE=16`, `ROLLOUT_MINI_BATCH_SIZE=16`, `MAX_TURNS=20`, `SERVER_NAVIGATION_MAX_WORKERS=4`, `TOTAL_TRAINING_STEPS=60`, `TEST_FREQ=20`, `SAVE_FREQ=20`.
- New 5-node answer-only integrated job:
  - Job ID: `485097`
  - Job name: `vagen-nav-vagen1-5n-int`
  - Status after submit: `PENDING`, reason `Priority`
  - Time limit: `12:00:00`
  - Resources: `5` nodes, `20` total GPUs, with `ENV_NNODES=2` and `TRAIN_NNODES=3`.
  - W&B group: `navigation_vagen1_integrated_5node_answeronly_20260723`
  - W&B name: `navigation_vagen1_integrated_5node_answeronly_b96_rmb48_20260723T093142Z`
  - Train log: `/project/peilab/hligb/vagen-navigation/logs/navigation_vagen1_integrated_5node_answeronly_b96_rmb48_20260723T093142Z.log`
  - Key settings: `FORMAT_REWARD=0.05`, `LOSS_MASK_MODE=answer_only`, `TRAIN_BATCH_SIZE=96`, `PPO_MINI_BATCH_SIZE=48`, `ROLLOUT_MINI_BATCH_SIZE=48`, `MAX_TURNS=20`, `SERVER_NAVIGATION_MAX_WORKERS=4`, `TOTAL_TRAINING_STEPS=60`, `TEST_FREQ=20`, `SAVE_FREQ=20`.
- Queue snapshot immediately after submission:
  - `485078`: running on `dgx-46`, 2h local turn10 answer-only.
  - `485088`: pending, estimated start `2026-07-23T19:31:46`, 2h local turn5 answer-only.
  - `485096`: pending, start estimate `N/A`, 3-node integrated answer-only.
  - `485097`: pending, start estimate `N/A`, 5-node integrated answer-only.
- Current anti-collapse coverage:
  - Single-action prompt is active in `navigation_vagen1`.
  - Parser marks over-budget/multiple-action outputs as `format_correct=False` and `too_many_actions=True`.
  - Format reward is lowered to `0.05` for the active answer-only jobs.
  - Action distribution metrics are available through W&B keys including `action/top_share`, `action/entropy`, `action/all_same_traj`, and `format/too_many_actions`.
- Next check:
  - Inspect `485078` first because it is already running and should reveal whether local turn10 answer-only gets past vLLM init into rollout.
  - For `485096/485097`, wait for start estimates and confirm the integrated logs show answer-only, format reward 0.05, current-job env ready files, and action metrics after rollout.

### E3.13/E3.14 Local Answer-Only Debug5 Results

- Checked at: 2026-07-23 HKT
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Result summary:
  - `485078` turn10 answer-only completed successfully with exit code `0:0`.
  - `485088` turn5 answer-only completed successfully with exit code `0:0`.
  - Neither run reproduced the previous `/environments` read timeout.
  - Both runs reached logged step 5 and W&B final sync.
- `485078` turn10 details:
  - Slurm: `COMPLETED`, elapsed `00:30:18`, node `dgx-46`.
  - W&B URL: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/cnncz5li`
  - Summary path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/runs/navigation_vagen1_turn10_answeronly_fmt005_rmb2_w1_4gpu_debug5_20260723T090226Z_20260723/summary.md`
  - Final train metrics at step 5:
    - `train/success=0.188`
    - `train/action/top_share=0.656`
    - `train/action/entropy=0.697`
    - `train/action/all_same_traj=0.062`
    - `train/format/too_many_actions=0.000`
    - `train/format/correct=0.931`
    - `timing_s/step=489.467`
    - `timing_s/testing=218.195`
  - Final val metrics at step 5:
    - `val/success=0.250`
    - `val/action/top_share=0.662`
    - `val/action/entropy=0.232`
    - `val/action/all_same_traj=0.500`
    - `val/format/too_many_actions=0.000`
    - `val/format/correct=0.724`
  - Split summary:
    - `navigation_base success=0.5000`, score `5.1130`
    - `navigation_common_sense success=0.0000`, score `0.3880`
    - average success `0.2500`, score `2.7505`
- `485088` turn5 details:
  - Slurm: `COMPLETED`, elapsed `00:22:18`, node `dgx-26`.
  - W&B URL: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/fkiq8oro`
  - Summary path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/runs/navigation_vagen1_turn5_answeronly_fmt005_rmb2_w1_4gpu_debug5_20260723T091901Z_20260723/summary.md`
  - Final train metrics at step 5:
    - `train/success=0.000`
    - `train/action/top_share=0.512`
    - `train/action/entropy=0.250`
    - `train/action/all_same_traj=0.250`
    - `train/format/too_many_actions=0.000`
    - `train/format/correct=0.625`
    - `timing_s/step=269.935`
    - `timing_s/testing=123.849`
  - Final val metrics at step 5:
    - `val/success=0.125`
    - `val/action/top_share=0.600`
    - `val/action/entropy=0.063`
    - `val/action/all_same_traj=0.500`
    - `val/format/too_many_actions=0.000`
    - `val/format/correct=0.625`
  - Split summary:
    - `navigation_base success=0.0000`, score `0.0620`
    - `navigation_common_sense success=0.2500`, score `2.6750`
    - average success `0.1250`, score `1.3685`
- Interpretation:
  - Lowering local env creation pressure to `ROLLOUT_MINI_BATCH_SIZE=2` and `SERVER_NAVIGATION_MAX_WORKERS=1` fixed the immediate env creation timeout for small local debug.
  - The local topology is still too slow for formal training: turn10 step time was about 490s with validation, and turn5 was about 270s.
  - `answer_only + FORMAT_REWARD=0.05` did not immediately collapse on train action metrics: turn10 train `top_share=0.656`, `all_same_traj=0.062`, and `too_many_actions=0.000`.
  - Validation still shows low entropy and high same-trajectory fraction, especially turn5; this is expected to some extent because turn5 has only 5 single-action opportunities, but it means we still need integrated/external runs before trusting the anti-collapse result.
- Current queue after these results:
  - `485096`: 3-node integrated answer-only, pending, estimated start `2026-07-24T20:07:08`, time limit `02:00:00`.
  - `485097`: 5-node integrated answer-only, pending, estimated start `2026-07-25T14:18:38`, time limit `12:00:00`.
- Decision:
  - Keep `485096` and `485097` queued.
  - Treat local 4GPU as a successful correctness/sanity path, not a speed-sufficient training path.
  - Use the 3-node integrated job as the next meaningful speed/stability gate.

### E3.16 4GPU Local Turn20 Step60 Rationale

- Prepared at: 2026-07-23 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting local commit: `c58875a`
- Change type: submit a longer local 4GPU VAGEN1 run without changing code.
- Reason:
  - The local turn10 and turn5 answer-only debug5 jobs both completed successfully and did not reproduce the `/environments` timeout.
  - Although local 4GPU is slower than desired, the observed speed is acceptable for a first 60-step run: turn10 completed debug5 in about 30 minutes, and the final step with validation took about 489s.
  - `MAX_TURNS=20` is more meaningful for VAGEN1 navigation than turn5/turn10 because step=1 otherwise gives too few primitive actions for many tasks.
  - This run is intended to test whether the current anti-collapse settings remain stable beyond debug5 and to produce checkpoints at step 20/40/60 if training is healthy.
- Planned submission:
  - Use existing `scripts/superpod/run_navigation_vagen1_4gpu.sbatch`.
  - Request normal 4GPU local allocation with the script default `#SBATCH --time=24:00:00`.
  - Keep env creation conservative: `ROLLOUT_MINI_BATCH_SIZE=2` and `SERVER_NAVIGATION_MAX_WORKERS=1`.
- Planned run parameters:
  - `MAX_TURNS=20`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `SERVER_NAVIGATION_MAX_WORKERS=1`
  - `FORMAT_REWARD=0.05`
  - `LOSS_MASK_MODE=answer_only`
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=20`
  - `SAVE_FREQ=20`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
  - `REMOVE_PREVIOUS_CKPT_IN_SAVE=False`
  - `FORCE_GEN_DATA=1`
  - `ENV_CONFIG_PATH=scripts/examples/vagen_base/navigation_vagen1/env_config_speed.yaml`
- Expected observations:
  - Checkpoints should be saved at steps 20, 40, and 60.
  - Action distribution should remain below the collapse gate: target `train/action/top_share < 0.8`, `train/action/all_same_traj < 0.5`, and `format/too_many_actions=0`.
  - If speed is around 8-12 minutes per step, the 60-step run should fit within 24h.
  - If step time grows or action distribution collapses, stop treating local 4GPU as a candidate for formal 300-step training and wait for integrated 3-node/5-node results.

### E3.16 4GPU Local Turn20 Step60 Submission Result

- Submitted at: 2026-07-23 HKT
- Rationale commit used on SuperPOD: `18e025b`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Server-side validation before submit:
  - `git pull --ff-only` updated the clone from `dc3d4d1` to `18e025b26e8c6746cf555b0bf366cfce85e9f791`.
  - `bash -n scripts/superpod/run_navigation_vagen1_4gpu.sbatch` passed on SuperPOD.
  - `bash -n scripts/superpod/configure_navigation_vagen1_variant.sh` passed on SuperPOD.
- New local 4GPU turn20 step60 job:
  - Job ID: `485283`
  - Job name: `vagen-nav-vagen1-4g`
  - Status after submit: `PENDING`, reason `Priority`
  - Time limit: `24:00:00`
  - Resources: one normal node, `gres/gpu=4`, `cpu=64`, `mem=384G`
  - Script: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/scripts/superpod/run_navigation_vagen1_4gpu.sbatch`
  - WorkDir: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- W&B:
  - Group: `navigation_vagen1_local4gpu_turn20_answeronly_step60_20260723`
  - Name: `navigation_vagen1_local4gpu_turn20_answeronly_fmt005_rmb2_w1_step60_20260723T153812Z`
  - URL: pending until the job starts and creates the run.
  - Train log: `/project/peilab/hligb/vagen-navigation/logs/navigation_vagen1_local4gpu_turn20_answeronly_fmt005_rmb2_w1_step60_20260723T153812Z.log`
- Checkpoint path:
  - `/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_local4gpu_turn20_answeronly_fmt005_rmb2_w1_step60_20260723T153812Z`
- Key settings:
  - `VAGEN1_VARIANT=fmt005_answeronly`
  - `MAX_TURNS=20`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `SERVER_NAVIGATION_MAX_WORKERS=1`
  - `FORMAT_REWARD=0.05`
  - `LOSS_MASK_MODE=answer_only`
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=20`
  - `SAVE_FREQ=20`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
  - `REMOVE_PREVIOUS_CKPT_IN_SAVE=False`
  - `FORCE_GEN_DATA=1`
  - `ENV_CONFIG_PATH=scripts/examples/vagen_base/navigation_vagen1/env_config_speed.yaml`
- Queue snapshot after submit:
  - `485096`: 3-node integrated answer-only, pending, estimated start `2026-07-24T20:07:08`.
  - `485097`: 5-node integrated answer-only, pending, estimated start `2026-07-25T14:18:38`.
  - `485283`: local 4GPU turn20 step60, pending, start estimate `N/A` immediately after submission.
- Next check:
  - Watch for W&B URL creation.
  - If it starts before integrated jobs, use step20 metrics and checkpoint presence as the next gate.

### E3.17 VAGEN1 Format-Debug Rationale

- Prepared at: 2026-07-24 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting commit: `5af5dbb`
- Trigger:
  - The running local 4GPU turn20 job `485283` reached step 14, but train `format/correct` still fluctuates strongly.
  - W&B/logs currently expose aggregate parser metrics but not enough detail to distinguish malformed tags, empty answers, too many actions, and invalid action names.
- Current evidence:
  - `too_many_actions` is usually near zero, so the main failure is not simply multiple actions.
  - `format/correct` and `action_is_valid` differ, which means some responses can be structurally parseable while still containing invalid action names.
- Change hypothesis:
  - Single-action training needs a harder format contract in the system prompt because the previous single-action system prompt said "exactly one action" but did not include the full required `<think>...</think><answer>...</answer>` schema.
  - Parser-side failure type metrics will make future W&B curves actionable: if `missing_or_malformed_tags` dominates, improve prompt/SFT formatting; if `invalid_action_name` dominates, constrain action vocabulary or normalize aliases; if `too_many_actions` rises, tighten single-action training; if `empty_answer` rises, inspect generation/eos behavior.
- Planned code changes:
  - Add full grounding-worldmodeling single-action schema and exact one-action example to `system_prompt`.
  - Add parser `format_error_type` values: `ok`, `missing_or_malformed_tags`, `empty_answer`, `too_many_actions`.
  - Treat empty `<answer></answer>` as format failure.
  - Add env-side `action_validity_error` values: `ok`, `no_action`, `invalid_action_name`.
  - Add W&B aggregate metrics: `format/error/*` and `action/error/*`.
- Reward context:
  - Navigation env class default `format_reward` is `0.5`.
  - VAGEN1 config/run default is `0.1`.
  - Current anti-collapse jobs override it to `0.05`.
- Next experiment use:
  - Use these metrics before changing reward again.
  - If malformed tags dominate, keep `FORMAT_REWARD=0.05` but consider adding a short format SFT/no-think control.
  - If invalid action names dominate, add action alias normalization only after measuring examples.

### E3.18 VAGEN1 Raw Sample Logging Rationale

- Prepared at: 2026-07-24 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting commit: `cfcc95a`
- Trigger:
  - Aggregate metrics can identify the failure class, but they cannot show the exact model text.
  - Current running job `485283` was launched before raw sample logging existed, so it cannot answer whether malformed outputs are missing tags, writing natural-language actions, extra text after `</answer>`, or other errors.
- Change hypothesis:
  - Add a disabled-by-default raw sample logger so debug jobs can inspect actual generated responses without changing PPO behavior.
  - Prioritize invalid samples first, then include valid samples only if there is remaining budget.
- Code changes:
  - Add `vagen/rollout/qwen_rollout/raw_sample_utils.py`.
  - `recording_to_log()` now includes per-trajectory `history` so trainer-side logging can access raw turn responses.
  - Add trainer config:
    - `trainer.raw_samples_to_log`
    - `trainer.raw_samples_max_chars`
  - Add trainer logging:
    - stdout lines prefixed with `[RAW_SAMPLE]`
    - optional W&B tables: `train/raw_samples` and `val/raw_samples`
  - Add script env vars:
    - `RAW_SAMPLES_TO_LOG`
    - `RAW_SAMPLES_MAX_CHARS`
  - Add short wrapper:
    - `scripts/superpod/run_navigation_vagen1_4gpu_raw_samples_debug.sbatch`
- Default behavior:
  - `RAW_SAMPLES_TO_LOG=0`, so existing formal/training scripts do not upload raw text unless explicitly enabled.
- Recommended raw-sample debug:
  - Use `RAW_SAMPLES_TO_LOG=8`
  - `TOTAL_TRAINING_STEPS=4`
  - `TEST_FREQ=2`
  - `SAVE_FREQ=-1`
  - `MAX_TURNS=20`
  - `LOSS_MASK_MODE=answer_only`
  - `FORMAT_REWARD=0.05`
- Expected observation:
  - Logs should contain `[RAW_SAMPLE]` lines after each rollout.
  - W&B should contain `train/raw_samples` and, at eval steps, `val/raw_samples`.
  - Use these samples to decide whether to improve prompt, use no-think as a diagnostic control, normalize action aliases, or revisit format reward.

### E3.19 Integrated 2-Node Env/Train Split Rationale

- Prepared at: 2026-07-24 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting commit: `41e4d92`
- Trigger:
  - Local 4GPU job `485283` is stable enough to reach later steps but rollout generation dominates wall time.
  - We need validation every 5 training steps to see trend earlier, and we need raw samples from the updated prompt/parser code.
- Change hypothesis:
  - Keeping train batch settings close to the stable local run while moving AI2-THOR to a separate node should reduce contention between vLLM/training and environment simulation.
  - Using only two env GPUs with two server workers reduces AI2-THOR pressure while still separating env from train.
  - Frequent val every 5 steps improves observability; if validation dominates wall time too much, revert to test every 10 or 20 after debugging.
- Slurm/resource note:
  - The wrapper requests two normal nodes with `--gres=gpu:4` for compatibility with the existing homogeneous sbatch pattern.
  - The env server process is restricted to `SERVER_NAVIGATION_DEVICES=0,1` and `ENV_GPUS_PER_NODE=2`; the train node uses 4 GPUs.
- Planned script:
  - `scripts/superpod/run_navigation_vagen1_2node_env1_train1_integrated.sbatch`
- Planned settings:
  - `ENV_NNODES=1`
  - `TRAIN_NNODES=1`
  - `ENV_GPUS_PER_NODE=2`
  - `EXPECTED_RAY_GPUS=4`
  - `TOTAL_TRAINING_STEPS=60`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `SERVER_NAVIGATION_MAX_WORKERS=2`
  - `TEST_FREQ=5`
  - `SAVE_FREQ=20`
  - `MAX_TURNS=20`
  - `LOSS_MASK_MODE=answer_only`
  - `FORMAT_REWARD=0.05`
  - `RAW_SAMPLES_TO_LOG=8`
  - `RAW_SAMPLES_MAX_CHARS=2000`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
  - wall time `12:00:00`
- Expected observations:
  - W&B should show val metrics at steps 5, 10, 15, ...
  - Logs should include `[RAW_SAMPLE]` lines.
  - Compare `timing_s/gen` and `timing_s/step` against local 4GPU `485283`.
  - If step time remains dominated by generation, increasing env resources will not solve speed; if env/server timeout disappears and step time drops, continue with split topology.

### E3.19 Integrated 2-Node Env/Train Split Submission Result

- Submitted at: 2026-07-24 HKT
- Commit used on SuperPOD: `a00c773c0b0ac56e53b0a623264884857f3d795e`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Server-side validation before submit:
  - `git pull --ff-only` updated the clone from `41e4d92` to `a00c773`.
  - `bash -n scripts/superpod/run_navigation_vagen1_5node_integrated.sbatch scripts/superpod/run_navigation_vagen1_2node_env1_train1_integrated.sbatch scripts/examples/vagen_base/navigation/run.sh` passed.
- New job:
  - Job ID: `486287`
  - Job name: `vagen-nav-vagen1-2n-int`
  - Status after submit: `PENDING`, reason `Priority`
  - Start estimate snapshot: `2026-07-27T01:43:35`
  - Time limit: `12:00:00`
  - Resources: two normal nodes, homogeneous sbatch allocation `gres/gpu=4` per node; env process restricted to two devices.
- W&B:
  - Group: `navigation_vagen1_integrated_2node_env1_train1_raw_val5_20260724`
  - Name pattern: `navigation_vagen1_integrated2n_env1x2_train1x4_b16_rmb2_val5_raw_<timestamp>`
- Key settings:
  - `ENV_NNODES=1`
  - `TRAIN_NNODES=1`
  - `ENV_GPUS_PER_NODE=2`
  - `SERVER_NAVIGATION_DEVICES=0,1`
  - `SERVER_NAVIGATION_MAX_WORKERS=2`
  - `EXPECTED_RAY_GPUS=4`
  - `TOTAL_TRAINING_STEPS=60`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `TEST_FREQ=5`
  - `SAVE_FREQ=20`
  - `MAX_TURNS=20`
  - `LOSS_MASK_MODE=answer_only`
  - `FORMAT_REWARD=0.05`
  - `RAW_SAMPLES_TO_LOG=8`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
- Next check:
  - Confirm env server starts on the first node and advertises one `ROLLOUT_BASE_URL`.
  - Confirm Ray reports 4 training GPUs on the train node.
  - Confirm W&B run URL and `[RAW_SAMPLE]` log lines.
  - Compare `timing_s/step` with local job `485283`.

### E3.16 4GPU Local Turn20 Step60 Crash Result

- Checked at: 2026-07-24 HKT
- Job ID: `485283`
- Final Slurm state: `FAILED`, exit code `1:0`
- Elapsed: `05:23:56`
- Commit used by the run: `5af5dbbec910783c3d1bc123ffe8c63a267555ea`
- W&B run: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/9bhyi0u6`
- Log path:
  - `/project/peilab/hligb/vagen-navigation/logs/navigation_vagen1_local4gpu_turn20_answeronly_fmt005_rmb2_w1_step60_20260723T153812Z.log`
- Last completed training step:
  - Completed and logged `step:15`.
  - Then started the next global step rollout and crashed during `Processing mini-batch 5/8, size: 2`.
- Error:
  - `requests.exceptions.ReadTimeout: HTTPConnectionPool(host='localhost', port=5283): Read timed out. (read timeout=1200)`
  - Stack location: `QwenVLRolloutManagerService.reset -> BatchEnvClient.create_environments_batch -> POST /environments`
- Server evidence:
  - `navigation-local-server-485283.log` did not show a Python traceback.
  - The tail repeatedly shows `State reward wrapper closed` and AI2-THOR `Initialize return`, then stops before replying to the client.
- Interpretation:
  - This is not a PPO loss crash, not a checkpoint-save crash, and not an explicit CUDA OOM.
  - The root cause is still the local AI2-THOR environment creation path hanging long enough to exceed the rollout client timeout.
  - Because it happens after many successful mini-batches and without a server traceback, the likely failure is an AI2-THOR/Unity process or env creation worker stall, made worse by repeatedly creating/closing environments on the same node as vLLM/training.
- Metrics before crash:
  - Step 15 train `success=0.312`, `score=3.663`, `format/correct=0.812`, `too_many_actions=0.000`, `action/top_share=0.699`, `all_same_traj=0.312`, `timing_s/step=657.742`.
  - Base split looked high but suspiciously collapse-like: base `success=0.667`, `action/top_share=0.903`, `all_same_traj=0.667`, `moveahead share=0.803`.
- Decision:
  - Do not keep using local 4GPU turn20 as the stability path.
  - Continue with the split env/train job `486287`, because it directly tests whether removing env creation from the training node reduces this hang.
  - New runs should use raw sample logging and format failure metrics from commit `25ac376` or later.

### E3.20 Replace Env2GPU Split With Env4GPU Split Rationale

- Prepared at: 2026-07-24 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting commit: `741646f`
- Trigger:
  - User requested canceling pending `486287` and replacing it with a stronger env-node allocation.
  - `485283` failed specifically in `POST /environments` after many successful mini-batches, so the next test should target AI2-THOR server capacity/stability rather than changing PPO hyperparameters.
- Cancel result:
  - `486287` was canceled before it started.
- Change hypothesis:
  - Use a separated env node with all four GPUs exposed to AI2-THOR and `SERVER_NAVIGATION_MAX_WORKERS=4`.
  - Keep train batch and rollout mini-batch unchanged from the stable local run so we isolate the env topology variable.
- Script update:
  - `scripts/superpod/run_navigation_vagen1_2node_env1_train1_integrated.sbatch`
- Updated settings:
  - `ENV_NNODES=1`
  - `TRAIN_NNODES=1`
  - `ENV_GPUS_PER_NODE=4`
  - `SERVER_NAVIGATION_DEVICES=0,1,2,3`
  - `SERVER_NAVIGATION_MAX_WORKERS=4`
  - `EXPECTED_RAY_GPUS=4`
  - `TOTAL_TRAINING_STEPS=60`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `TEST_FREQ=5`
  - `SAVE_FREQ=20`
  - `MAX_TURNS=20`
  - `LOSS_MASK_MODE=answer_only`
  - `FORMAT_REWARD=0.05`
  - `RAW_SAMPLES_TO_LOG=8`
- Interpretation from `485283` step 15 before crash:
  - Do not change loss/reward yet: `format/correct=0.812` and `too_many_actions=0.000` indicate the anti-collapse/format settings are not obviously broken.
  - Keep watching collapse: base split had `action/top_share=0.903`, `all_same_traj=0.667`, and `moveahead share=0.803`, so raw samples and val every 5 steps are needed.
  - Do not increase batch yet: rollout generation/create-env remains the bottleneck, and bigger batches may hide or amplify env hangs.
- Expected observations:
  - If env4 split completes beyond step 15 without `/environments` timeout, local node contention was a major root cause.
  - If env4 split still times out in `/environments`, the next fix should be server-side env lifecycle hardening: environment reuse/pool, per-request timeout inside server, and worker restart after stuck AI2-THOR create.

### E3.20 Env4GPU Split Submission Result

- Submitted at: 2026-07-24 HKT
- Code/doc commit before submission: `301fd79`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Action taken:
  - Canceled pending env2/train4 integrated job `486287` before it started.
  - Pulled commit `301fd79` on SuperPOD.
  - Submitted the env4/train4 integrated replacement.
- New job:
  - Job ID: `486302`
  - Job name: `vagen-nav-vagen1-2n-int`
  - Status after submit/check: `PENDING`, reason `Priority`
  - Start estimate snapshot: `2026-07-27T01:43:35`
  - Time limit: `12:00:00`
  - Resources: two normal nodes, `gres/gpu=4` per node, total requested GPUs `8`
  - Script: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/scripts/superpod/run_navigation_vagen1_2node_env1_train1_integrated.sbatch`
  - Slurm stdout: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-2node-env1-train1-integrated-486302.out`
  - Slurm stderr: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-2node-env1-train1-integrated-486302.err`
- W&B:
  - Group: `navigation_vagen1_integrated_2node_env1x4_train1x4_raw_val5_20260724`
  - Name pattern: `navigation_vagen1_integrated2n_env1x4_train1x4_b16_rmb2_val5_raw_<timestamp>`
  - URL: pending until the job starts and creates the run.
- Key settings:
  - `ENV_NNODES=1`
  - `TRAIN_NNODES=1`
  - `ENV_GPUS_PER_NODE=4`
  - `SERVER_NAVIGATION_DEVICES=0,1,2,3`
  - `SERVER_NAVIGATION_MAX_WORKERS=4`
  - `EXPECTED_RAY_GPUS=4`
  - `TOTAL_TRAINING_STEPS=60`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `TEST_FREQ=5`
  - `SAVE_FREQ=20`
  - `MAX_TURNS=20`
  - `LOSS_MASK_MODE=answer_only`
  - `FORMAT_REWARD=0.05`
  - `RAW_SAMPLES_TO_LOG=8`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
- Decision:
  - Keep `486302` queued as the next topology gate.
  - Do not change reward/loss/batch before seeing whether the separated env4 node passes the post-step15 stability point where local `485283` failed.
  - If `486302` still hangs in `POST /environments`, move to server-side env lifecycle hardening rather than increasing rollout batch size.

### E3.21 Env4GPU Split FormatReward0.1 Control Rationale

- Prepared at: 2026-07-24 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting commit: `1a11af4`
- Trigger:
  - User requested keeping `486302` queued, canceling old integrated jobs `485096` and `485097`, and adding a `FORMAT_REWARD=0.1` control.
- Change hypothesis:
  - `486302` uses `FORMAT_REWARD=0.05`; a matched `FORMAT_REWARD=0.1` run can test whether the lower format reward is hurting early format compliance too much.
  - Keep the same env4/train4 topology and batch settings so the comparison isolates format reward rather than resource or rollout changes.
  - If `0.1` improves `format/correct` without pushing `action/top_share` or `all_same_traj` into collapse, it may be the better VAGEN1 base setting.
  - If `0.1` improves apparent success but also drives repeated `moveahead` or high same-trajectory fraction, stay with `0.05` and debug formatting through prompt/SFT/no-think diagnostics instead.
- Planned cancellation:
  - Cancel old pending multi-node integrated jobs `485096` and `485097`.
  - Keep `486302` queued.
- Planned submission:
  - Use `scripts/superpod/run_navigation_vagen1_2node_env1_train1_integrated.sbatch`.
  - Submit a matched 2-node env4/train4 job with only reward/W&B identifiers changed.
- Planned settings:
  - `ENV_NNODES=1`
  - `TRAIN_NNODES=1`
  - `ENV_GPUS_PER_NODE=4`
  - `SERVER_NAVIGATION_DEVICES=0,1,2,3`
  - `SERVER_NAVIGATION_MAX_WORKERS=4`
  - `EXPECTED_RAY_GPUS=4`
  - `TOTAL_TRAINING_STEPS=60`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `TEST_FREQ=5`
  - `SAVE_FREQ=20`
  - `MAX_TURNS=20`
  - `LOSS_MASK_MODE=answer_only`
  - `FORMAT_REWARD=0.1`
  - `RAW_SAMPLES_TO_LOG=8`
- Expected observations:
  - Compare against `486302` on format correctness, action top share, all-same trajectory fraction, success, and whether the job passes the post-step15 stability point.
  - Do not treat higher success as healthy if it is paired with `action/top_share > 0.85` or `all_same_traj > 0.5`.

### E3.21 Env4GPU Split FormatReward0.1 Control Submission Result

- Submitted at: 2026-07-24 HKT
- Rationale commit pushed before submission: `a6f67c4`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Server commit after pull: `a6f67c4`
- Action taken:
  - Kept `486302` queued as the `FORMAT_REWARD=0.05` env4/train4 run.
  - Canceled old pending integrated jobs `485096` and `485097`.
  - Submitted matched env4/train4 control with `FORMAT_REWARD=0.1`.
- New control job:
  - Job ID: `486318`
  - Job name: `vagen-nav-vagen1-2n-int`
  - Status after submit/check: `PENDING`, reason `Priority`
  - Start estimate snapshot: `2026-07-27T22:29:10`
  - Time limit: `12:00:00`
  - Resources: two normal nodes, `gres/gpu=4` per node, total requested GPUs `8`
  - Script: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/scripts/superpod/run_navigation_vagen1_2node_env1_train1_integrated.sbatch`
  - Slurm stdout: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-2node-env1-train1-integrated-486318.out`
  - Slurm stderr: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-2node-env1-train1-integrated-486318.err`
- W&B:
  - Group: `navigation_vagen1_integrated_2node_env1x4_train1x4_raw_val5_fmt01_20260724`
  - Name: `navigation_vagen1_integrated2n_env1x4_train1x4_b16_rmb2_val5_raw_fmt01_20260724T143004Z`
  - URL: pending until the job starts and creates the run.
- Key diff from `486302`:
  - `FORMAT_REWARD=0.1` instead of `0.05`.
  - W&B name/group/tags mark this as `fmt01`.
  - Topology, batch sizes, max turns, answer-only loss, raw sample logging, save/eval schedule are otherwise matched.
- Current queue after submission:
  - `486302`: `FORMAT_REWARD=0.05`, pending priority, start estimate snapshot `2026-07-27T22:29`.
  - `486318`: `FORMAT_REWARD=0.1`, pending priority, start estimate snapshot `2026-07-27T22:29`.
- Decision:
  - Treat `486302` vs `486318` as the first clean format-reward pair.
  - Primary stability gate remains whether either run gets past the post-step15 `/environments` failure point.
  - Primary collapse gate remains action distribution and raw samples, not success alone.

### E3.22 Env4GPU Split SaveFreq5 Resubmission Rationale

- Prepared at: 2026-07-24 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting commit: `2d55c16`
- Trigger:
  - User requested canceling queued env4/train4 jobs `486302` and `486318`, then resubmitting both with `SAVE_FREQ=5`.
- Change hypothesis:
  - The previous local run `485283` crashed after logged step 15, before the first planned `SAVE_FREQ=20` checkpoint.
  - For the next stability/debug pair, saving every 5 steps protects useful intermediate checkpoints at steps 5, 10, 15, and 20.
  - This does not change rollout generation, PPO loss, reward computation, or action distribution behavior; it only increases checkpoint observability and recovery.
- Planned cancellation:
  - Cancel `486302` (`FORMAT_REWARD=0.05`, `SAVE_FREQ=20`).
  - Cancel `486318` (`FORMAT_REWARD=0.1`, `SAVE_FREQ=20`).
- Planned resubmission:
  - Submit two matched env4/train4 integrated jobs using `scripts/superpod/run_navigation_vagen1_2node_env1_train1_integrated.sbatch`.
  - Use `SAVE_FREQ=5` for both.
  - Keep all other settings matched except `FORMAT_REWARD`.
- Planned pair:
  - `fmt005_save5`: `FORMAT_REWARD=0.05`, `SAVE_FREQ=5`.
  - `fmt01_save5`: `FORMAT_REWARD=0.1`, `SAVE_FREQ=5`.
- Expected observations:
  - W&B still validates every 5 steps.
  - Checkpoints should appear at steps 5/10/15 before the known risk zone.
  - If a run crashes near the old failure point, we can still inspect and reuse the closest checkpoint.

### E3.22 Env4GPU Split SaveFreq5 Resubmission Result

- Submitted at: 2026-07-24 HKT
- Rationale commit pushed before submission: `a6d4f58`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Server commit after pull: `a6d4f58`
- Action taken:
  - Canceled queued `SAVE_FREQ=20` jobs `486302` and `486318`.
  - Submitted two matched env4/train4 integrated jobs with `SAVE_FREQ=5`.
- New jobs:
  - `486324`: `FORMAT_REWARD=0.05`, `SAVE_FREQ=5`
  - `486325`: `FORMAT_REWARD=0.1`, `SAVE_FREQ=5`
- Slurm status after submit:
  - `486324`: `PENDING`, reason `Priority`, start estimate `N/A`
  - `486325`: `PENDING`, reason `Priority`, start estimate `N/A`
  - Both have `TimeLimit=12:00:00`.
- W&B:
  - `486324` group: `navigation_vagen1_integrated_2node_env1x4_train1x4_raw_val5_save5_20260724`
  - `486324` name: `navigation_vagen1_integrated2n_env1x4_train1x4_b16_rmb2_val5_save5_raw_fmt005_20260724T143519Z`
  - `486325` group: `navigation_vagen1_integrated_2node_env1x4_train1x4_raw_val5_save5_fmt01_20260724`
  - `486325` name: `navigation_vagen1_integrated2n_env1x4_train1x4_b16_rmb2_val5_save5_raw_fmt01_20260724T143519Z`
  - URLs pending until jobs start.
- Key shared settings:
  - 2 normal nodes, `gres/gpu=4` per node, total requested GPUs `8`
  - `ENV_GPUS_PER_NODE=4`
  - `SERVER_NAVIGATION_DEVICES=0,1,2,3`
  - `SERVER_NAVIGATION_MAX_WORKERS=4`
  - `EXPECTED_RAY_GPUS=4`
  - `TOTAL_TRAINING_STEPS=60`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `TEST_FREQ=5`
  - `SAVE_FREQ=5`
  - `MAX_TURNS=20`
  - `LOSS_MASK_MODE=answer_only`
  - `RAW_SAMPLES_TO_LOG=8`
- Decision:
  - Use `486324` vs `486325` as the current clean pair.
  - Earlier jobs `486302`/`486318` are superseded because their save frequency was too sparse for the known step15 crash risk.

### E3.23 DenseLightV1 Anti-Collapse/Progress Rationale

- Prepared at: 2026-07-24 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting commit: `f0adc0e`
- Trigger:
  - User requested adding action-level anti-collapse reward and progress reward, choosing reasonable parameters, and submitting a job.
- Change hypothesis:
  - The clean pair `486324`/`486325` should remain queued as reward-free VAGEN1 baselines.
  - A separate dense-light run can test whether very small shaping improves success without creating a new shortcut policy.
  - Because success reward is `+10`, shaping terms must stay tiny: one turn should not be dominated by dense reward.
- Code changes:
  - Add optional `dense_reward_mode=anti_collapse_progress_v1` to navigation env config.
  - Add distance-sign progress shaping:
    - closer than previous turn by more than `0.01m`: `+0.02`
    - farther than previous turn by more than `0.01m`: `-0.02`
  - Add repeat-action anti-collapse penalty:
    - no penalty for the first two identical consecutive valid actions.
    - from the third identical action onward: `-0.02`, capped at `-0.06`.
  - Add invalid/no-op penalty:
    - invalid action, empty/no action, or failed AI2-THOR action: `-0.05`.
  - Log dense components as numeric turn metrics:
    - `dense_reward_total`
    - `dense_progress_reward`
    - `dense_repeat_action_penalty`
    - `dense_invalid_action_penalty`
    - `dense_distance_delta`
  - Keep dense shaping off by default for existing configs.
- Safety fix:
  - Rollout log aggregation now skips non-numeric turn metrics so string diagnostic fields such as `action_validity_error` do not crash logging.
- Planned config:
  - `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_light.yaml`
- Planned submission:
  - Keep `486324` and `486325` queued.
  - Submit one env4/train4 integrated dense-light job, matched to `486324` except:
    - `ENV_CONFIG_PATH=.../env_config_dense_light.yaml`
    - `FORMAT_REWARD=0.05`
    - W&B group/name/tags mark `dense-light-v1`.
- Expected observations:
  - Success/score should improve only if action distribution remains healthy.
  - Collapse fail condition remains `action/top_share > 0.85` or `all_same_traj > 0.5`, even if success rises.
  - Compare dense component magnitudes against total score; dense should be shaping, not the main reward.

### E3.23 DenseLightV1 Anti-Collapse/Progress Submission Result

- Submitted at: 2026-07-24 HKT
- Code/doc commit used on SuperPOD: `af04c31`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Local validation:
  - `python -m pytest tests/test_navigation_vagen1_minimal.py tests/test_navigation_vagen1_sweep.py -q`: `36 passed`
  - `bash -n` passed for the integrated sbatch wrappers and navigation run scripts.
  - `git diff --check` passed with CRLF warnings only.
- Server validation:
  - `git pull --ff-only` updated the clone to `af04c31`.
  - `bash -n` passed for the integrated sbatch wrappers and navigation run scripts.
- New dense-light job:
  - Job ID: `486343`
  - Job name: `vagen-nav-vagen1-2n-int`
  - Status after submit/check: `PENDING`, reason `Priority`
  - Start estimate snapshot: `2026-07-26T12:39:30`
  - Time limit: `12:00:00`
  - Resources: two normal nodes, `gres/gpu=4` per node, total requested GPUs `8`
  - Script: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/scripts/superpod/run_navigation_vagen1_2node_env1_train1_integrated.sbatch`
  - Slurm stdout: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-2node-env1-train1-integrated-486343.out`
  - Slurm stderr: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-2node-env1-train1-integrated-486343.err`
- W&B:
  - Group: `navigation_vagen1_integrated_2node_env1x4_train1x4_dense_light_v1_20260724`
  - Name: `navigation_vagen1_integrated2n_env1x4_train1x4_b16_rmb2_val5_save5_raw_fmt005_dense_light_v1_20260724T144842Z`
  - URL: pending until the job starts.
- Key settings:
  - `ENV_CONFIG_PATH=scripts/examples/vagen_base/navigation_vagen1/env_config_dense_light.yaml`
  - `FORMAT_REWARD=0.05`
  - `SAVE_FREQ=5`
  - `TEST_FREQ=5`
  - `TOTAL_TRAINING_STEPS=60`
  - `MAX_TURNS=20`
  - `LOSS_MASK_MODE=answer_only`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `SERVER_NAVIGATION_MAX_WORKERS=4`
  - `RAW_SAMPLES_TO_LOG=8`
- Current queue after submission:
  - `486324`: no dense, `FORMAT_REWARD=0.05`, `SAVE_FREQ=5`
  - `486325`: no dense, `FORMAT_REWARD=0.1`, `SAVE_FREQ=5`
  - `486343`: dense-light-v1, `FORMAT_REWARD=0.05`, `SAVE_FREQ=5`
- Decision:
  - Keep all three queued as the current clean comparison set.
  - First judge stability: whether they pass step15 and save checkpoints at steps 5/10/15.
  - Then judge policy health: success must improve without action collapse and dense components must remain small relative to total score.

### E3.24 Local8GPU Save5 Queue Probe Rationale

- Prepared at: 2026-07-25 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting commit: `9b02d16`
- Trigger:
  - User requested adding a `486324`-matched parameter run that uses 8 GPU training with local env, mainly to compare queue time.
- Change hypothesis:
  - The current clean split-env jobs require two 4GPU nodes and are waiting a long time.
  - A single-node 8GPU local-env job may have a different scheduling profile; it may start earlier or later depending on 8GPU-node availability.
  - This is a queue/startup probe and does not replace the env-split stability line.
- Planned script:
  - `scripts/superpod/run_navigation_vagen1_8gpu_local_save5.sbatch`
- Planned settings:
  - One normal node, `gres/gpu=8`, 12h.
  - Local AI2-THOR server inside the same allocation.
  - Match `486324` policy/training parameters:
    - `FORMAT_REWARD=0.05`
    - `LOSS_MASK_MODE=answer_only`
    - `TOTAL_TRAINING_STEPS=60`
    - `TRAIN_BATCH_SIZE=16`
    - `PPO_MINI_BATCH_SIZE=16`
    - `ROLLOUT_MINI_BATCH_SIZE=2`
    - `SAVE_FREQ=5`
    - `TEST_FREQ=5`
    - `MAX_TURNS=20`
    - `RAW_SAMPLES_TO_LOG=8`
    - no dense reward, no LLM judge, `ENV_CONFIG_PATH=env_config_speed.yaml`
  - 8GPU runtime:
    - `N_GPUS_PER_NODE=8`
    - `ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=8`
    - `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - `SERVER_NAVIGATION_MAX_WORKERS=4`
- Expected observations:
  - Compare start estimate against `486324/486325/486343`.
  - If it starts earlier, use it as a fast local 8GPU probe but do not overtrust stability because previous local 4GPU crashed at `/environments`.
  - If it starts much later, keep the split-env jobs as the main queue line.

### E3.24 Local8GPU Save5 Queue Probe Submission Result

- Submitted at: 2026-07-25 HKT
- Code/doc commit used on SuperPOD: `2d2df50`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Local validation:
  - `python -m pytest tests/test_navigation_vagen1_minimal.py tests/test_navigation_vagen1_sweep.py -q`: `37 passed`
  - `bash -n` passed for the local8 wrapper, local4 wrapper, and navigation run scripts.
  - `git diff --check` passed with CRLF warnings only.
- Server validation:
  - `git pull --ff-only` updated the clone to `2d2df50`.
  - `bash -n` passed for the local8 wrapper, local4 wrapper, and navigation run scripts.
- New local8 job:
  - Job ID: `486487`
  - Job name: `vagen-nav-vagen1-8g-local`
  - Status after submit/check: `PENDING`, reason `Priority`
  - Start estimate snapshot: `2026-07-25T22:28:56`
  - Time limit: `12:00:00`
  - Scheduled node snapshot: `dgx-32`
  - Resources: one normal node, `gres/gpu=8`, `cpu=128`, `mem=768G`
  - Script: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505/scripts/superpod/run_navigation_vagen1_8gpu_local_save5.sbatch`
  - Slurm stdout: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-8gpu-local-save5-486487.out`
  - Slurm stderr: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-8gpu-local-save5-486487.err`
- W&B:
  - Group: `navigation_vagen1_local8gpu_fmt005_save5_20260725`
  - Name pattern: `navigation_vagen1_local8gpu_b16_rmb2_val5_save5_raw_fmt005_<timestamp>`
  - URL: pending until the job starts.
- Key settings:
  - local env server inside the same 8GPU allocation
  - `FORMAT_REWARD=0.05`
  - `SAVE_FREQ=5`
  - `TEST_FREQ=5`
  - `TOTAL_TRAINING_STEPS=60`
  - `MAX_TURNS=20`
  - `LOSS_MASK_MODE=answer_only`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `N_GPUS_PER_NODE=8`
  - `ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=8`
  - `SERVER_NAVIGATION_MAX_WORKERS=4`
  - `RAW_SAMPLES_TO_LOG=8`
- Queue comparison after submission:
  - `486487` local8 start estimate: `2026-07-25T22:28`
  - `486324/486325/486343` split-env start estimate: `2026-07-27T23:21`
- Decision:
  - Keep `486487` queued because it is currently estimated to start much earlier.
  - Keep split-env jobs queued as the cleaner stability path, since previous local 4GPU failed in AI2-THOR `/environments`.

### E3.24 Local8GPU Save5 Startup Update

- Checked at: 2026-07-25 01:48 HKT
- Job ID: `486487`
- Updated Slurm state: `RUNNING`
- Node: `dgx-18`
- Elapsed at check: `00:00`
- Observation:
  - The local8 job started immediately after submission, despite the first estimate showing `2026-07-25T22:28`.
  - Slurm stdout shows project storage setup and local navigation server startup.
  - Local server command started with `state_reward=False`, `max_workers=4`, and `devices=[0,1,2,3,4,5,6,7]`.
  - W&B URL is still pending; the job has not yet reached training/W&B creation.
- Decision:
  - Keep monitoring `486487` as the fastest current startup path.
  - Still keep `486324/486325/486343` queued as the cleaner split-env comparison set.

### E3.25 Local8GPU Format/Dense Pair Rationale

- Prepared at: 2026-07-25 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting commit: `47848ca`
- Trigger:
  - User observed that the local 8GPU job started quickly and requested adding local8 versions corresponding to `486325` and `486343`.
- Change hypothesis:
  - Since local8 can start much faster than the 2-node split jobs, we should mirror the current clean comparison set on one 8GPU node:
    - local8 `fmt005` no dense: already running as `486487`
    - local8 `fmt01` no dense: new wrapper
    - local8 `dense_light_v1` with `FORMAT_REWARD=0.05`: new wrapper
  - This gives faster evidence about format reward and dense shaping while retaining the split-env jobs as the cleaner stability comparison.
- Planned scripts:
  - `scripts/superpod/run_navigation_vagen1_8gpu_local_save5_fmt01.sbatch`
  - `scripts/superpod/run_navigation_vagen1_8gpu_local_save5_dense_light.sbatch`
- Shared local8 settings:
  - one normal node, `gres/gpu=8`, 12h
  - local env server inside the same allocation
  - `TOTAL_TRAINING_STEPS=60`
  - `SAVE_FREQ=5`
  - `TEST_FREQ=5`
  - `MAX_TURNS=20`
  - `LOSS_MASK_MODE=answer_only`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `RAW_SAMPLES_TO_LOG=8`
  - `N_GPUS_PER_NODE=8`
  - `ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=8`
  - `SERVER_NAVIGATION_MAX_WORKERS=4`
- Planned pair:
  - `local8_fmt01_save5`: `FORMAT_REWARD=0.1`, no dense reward.
  - `local8_dense_light_v1_save5`: `FORMAT_REWARD=0.05`, `ENV_CONFIG_PATH=env_config_dense_light.yaml`.
- Expected observations:
  - Compare queue/startup time with already running `486487`.
  - Compare policy health among local8 `fmt005`, `fmt01`, and `dense_light_v1`.
  - Do not cancel split-env jobs yet, because local env may still reproduce the previous `/environments` failure.

### E3.25 Local8GPU Format/Dense Pair Submission Result

- Submitted at: 2026-07-25 HKT
- Code/doc commit used on SuperPOD: `1b28e80`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Local validation:
  - `python -m pytest tests/test_navigation_vagen1_minimal.py tests/test_navigation_vagen1_sweep.py -q`: `38 passed`
  - `bash -n` passed for all three local8 wrappers, local4 wrapper, and navigation run scripts.
  - `git diff --check` passed with CRLF warnings only.
- Server validation:
  - `git pull --ff-only` updated the clone to `1b28e80`.
  - `bash -n` passed for all three local8 wrappers, local4 wrapper, and navigation run scripts.
- New local8 jobs:
  - `486493`: `vagen-nav-vagen1-8g-fmt01`, local8 no dense, `FORMAT_REWARD=0.1`, `SAVE_FREQ=5`
  - `486494`: `vagen-nav-vagen1-8g-dense`, local8 dense-light-v1, `FORMAT_REWARD=0.05`, `SAVE_FREQ=5`
- Slurm status after submit:
  - `486493`: `PENDING`, reason `Priority`, start estimate `N/A`
  - `486494`: `PENDING`, reason `Priority`, start estimate `N/A`
  - Existing `486487`: `RUNNING`, local8 no dense `FORMAT_REWARD=0.05`
  - Existing split-env jobs `486324/486325/486343`: still `PENDING`, start estimate snapshot `2026-07-27T23:21`
- W&B:
  - `486493` group: `navigation_vagen1_local8gpu_fmt01_save5_20260725`
  - `486493` name pattern: `navigation_vagen1_local8gpu_b16_rmb2_val5_save5_raw_fmt01_<timestamp>`
  - `486494` group: `navigation_vagen1_local8gpu_dense_light_v1_save5_20260725`
  - `486494` name pattern: `navigation_vagen1_local8gpu_b16_rmb2_val5_save5_raw_fmt005_dense_light_v1_<timestamp>`
  - URLs pending until jobs start.
- Decision:
  - The active local8 comparison set is now `486487/486493/486494`.
  - Keep the split-env jobs queued until local8 proves it can pass the previous local `/environments` stability risk.

### E3.25 Local8GPU TP8 Failure Result

- Checked at: 2026-07-25 02:00 HKT
- Jobs:
  - `486487`: local8 `FORMAT_REWARD=0.05`, no dense, `FAILED`, exit `1:0`, elapsed `00:05:53`
  - `486493`: local8 `FORMAT_REWARD=0.1`, no dense, `FAILED`, exit `1:0`, elapsed `00:02:42`
  - `486494`: local8 dense-light-v1, `FAILED`, exit `1:0`, elapsed `00:02:44`
- W&B:
  - No W&B URL was created for any of the three jobs.
  - All failed during model/vLLM initialization before training started.
- Root cause:
  - vLLM failed with `AssertionError: 3420 is not divisible by 8`.
  - The failing setting was `actor_rollout_ref.rollout.tensor_model_parallel_size=8`.
  - Qwen2.5-VL's vision module has a partition dimension `3420`, which cannot be tensor-parallel sharded by 8.
- Interpretation:
  - The single-node 8GPU allocation itself is attractive because it started immediately.
  - The local8 script should not use `ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=8`.
  - To use an 8GPU node safely, keep `N_GPUS_PER_NODE=8` for the trainer allocation but set `ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=4`, matching the previously valid TP divisor.
- Decision:
  - Mark `486487/486493/486494` as invalid TP8 probes.
  - Next local8 attempt, if submitted, must use rollout TP4.

### E3.26 Local8GPU TP4 Resubmission Rationale

- Prepared at: 2026-07-25 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting commit: `b655640`
- Trigger:
  - User requested immediately resubmitting the three local8 versions after identifying TP8 as invalid.
- Change hypothesis:
  - Single-node 8GPU scheduling is fast, but Qwen2.5-VL rollout cannot use `tensor_model_parallel_size=8` because the vision dimension `3420` is not divisible by 8.
  - Keep the 8GPU allocation and set rollout TP back to `4`, which was previously valid.
  - This tests whether local8 can become the fast-start line without changing policy/reward parameters.
- Planned script change:
  - Update `scripts/superpod/run_navigation_vagen1_8gpu_local_save5.sbatch` default `ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE` from `8` to `4`.
- Planned submissions:
  - local8 TP4 `fmt005`: no dense, `FORMAT_REWARD=0.05`
  - local8 TP4 `fmt01`: no dense, `FORMAT_REWARD=0.1`
  - local8 TP4 `dense_light_v1`: `FORMAT_REWARD=0.05`, `ENV_CONFIG_PATH=env_config_dense_light.yaml`
- Shared settings:
  - one normal node, `gres/gpu=8`, 12h
  - `N_GPUS_PER_NODE=8`
  - `ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=4`
  - `TOTAL_TRAINING_STEPS=60`
  - `SAVE_FREQ=5`
  - `TEST_FREQ=5`
  - `MAX_TURNS=20`
  - `LOSS_MASK_MODE=answer_only`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `RAW_SAMPLES_TO_LOG=8`
- Expected observations:
  - vLLM initialization should pass the previous `3420 is not divisible by 8` failure.
  - W&B URL should be created if initialization succeeds.
  - If local8 TP4 later fails in `/environments`, that is the old local-env stability issue rather than the TP8 model-sharding issue.

### E3.26 Local8GPU TP4 Resubmission Result

- Submitted at: 2026-07-25 HKT
- Code/doc commit used on SuperPOD: `281a6d4`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Local validation:
  - `python -m pytest tests/test_navigation_vagen1_minimal.py tests/test_navigation_vagen1_sweep.py -q`: `38 passed`
  - `bash -n` passed for all local8 wrappers and navigation run scripts.
  - `git diff --check` passed with CRLF warnings only.
- Server validation:
  - `git pull --ff-only` updated the clone to `281a6d4`.
  - `bash -n` passed for all local8 wrappers and navigation run scripts.
- New local8 TP4 jobs:
  - `486499`: local8 TP4 `FORMAT_REWARD=0.05`, no dense, submitted and immediately `RUNNING` on `dgx-18`
  - `486500`: local8 TP4 `FORMAT_REWARD=0.1`, no dense, `PENDING`, start estimate snapshot `2026-07-25T22:28`
  - `486501`: local8 TP4 dense-light-v1, `PENDING`, start estimate snapshot `2026-07-27T23:21`
- Queue state at submission check:
  - Existing split-env jobs `486324/486325/486343` remain pending with start estimate snapshot `2026-07-27T23:21`.
- Decision:
  - Monitor `486499` first. If TP4 passes vLLM init and creates W&B, it validates the local8 fix.
  - If `486499` fails for a non-TP reason, inspect before touching `486500/486501`.

### E3.26 Local8GPU TP4 fmt005 Failure at Step20 Validation

- Checked at: 2026-07-25 15:17 HKT
- Job:
  - `486499`: local8 TP4 `FORMAT_REWARD=0.05`, no dense, `FAILED`, exit `1:0`, elapsed `03:35:33`, node `dgx-18`
- W&B:
  - Run URL: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/vfb1ov05`
- Checkpoints:
  - Saved: `global_step_5`, `global_step_10`, `global_step_15`
  - Missing: `global_step_20`
  - Checkpoint root: `/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_local8gpu_b16_rmb2_val5_save5_raw_fmt005_20260724T180338Z`
- Root cause evidence:
  - Failure happened inside trainer validation, not inside AI2-THOR environment creation.
  - Stack: `trainer.fit()` -> `_validate()` -> `test_rollout_manager.rollout_loop()` -> `actor_rollout_wg.generate_sequences(gen_batch)`.
  - Ray reported `ActorDiedError` for a `WorkerDict` actor.
  - The dead worker printed `Fatal Python error: none_dealloc: deallocating None`.
  - The active Python stack was in vLLM generation: `vllm/core/scheduler.py` -> `llm_engine.py` -> `LLM.generate()` -> `vllm_rollout_spmd.py:generate_sequences`.
  - The same log repeatedly warned that `/project/peilab/hligb/vn-tmp/ray/486499/...` was over 95% full and that object creation would fail if spilling was required.
- Interpretation:
  - TP4 fixed the earlier TP8 initialization error; this run reached W&B and completed several save points.
  - The new failure is a validation-time vLLM/Ray worker crash around the step20 eval/save boundary.
  - This is different from the previous local 4GPU `/environments` timeout.
  - Because checkpoint saving appears to occur after validation at eval steps, step20 validation crashed before `global_step_20` was written.
- Current related jobs:
  - `486500`: local8 TP4 `FORMAT_REWARD=0.1`, no dense, still `RUNNING` at the same check.
  - `486501`: local8 TP4 dense-light-v1, still `RUNNING` at the same check.
  - Both running jobs show repeated Ray tmp directory over-95% warnings, so they may hit the same validation/vLLM stability risk.
- Decision:
  - Treat `486499` as a partially successful local8 TP4 run: vLLM init, W&B, and step5/10/15 checkpointing worked.
  - Do not interpret this as a CoT collapse result; the failure is engineering/runtime stability.
  - Next stability patch should reduce validation pressure or save before validation at eval steps, and should move Ray tmp/spill storage away from the nearly-full `/project` threshold if possible.

### E3.27 Raw Sample Collapse Diagnosis and Invalid-Action Guard

- Checked at: 2026-07-25 HKT
- Trigger:
  - User asked whether raw samples already show collapse and how to solve the step20 failure.
- Runtime diagnosis:
  - `486499` stopped during step20 validation generation with a Ray/vLLM worker crash.
  - This is still an engineering stability issue, separate from policy collapse.
  - Existing `486500/486501` are still old-code runs and may hit the same validation/Ray tmp risk.
- Raw sample diagnosis:
  - `486499` raw sample count inspected: `183`
  - Action counts among logged raw samples:
    - `stay`: `60`
    - `staythere`: `31`
    - `moveahead`: `19`
    - many other invalid names such as `stayclose`, `stayhere`, `stop`, `grasp`, `rotatel`, `rotatereight`
  - Validity:
    - `action_is_valid=False`: `156`
    - `action_is_valid=True`: `27`
    - `format_error_type=ok`: `183`
    - `action_validity_error=invalid_action_name`: `156`
  - Concrete failure example:
    - step20 repeatedly emitted `<answer>stay</answer>` from turn2 through turn9.
    - These responses were XML-format correct but invalid navigation actions.
- Root cause interpretation:
  - The current parser/metrics separate "XML format is correct" from "navigation action name is valid".
  - The model is exploiting this by producing natural-language stopping actions (`stay`, `stop`, `terminate`) that look well formatted but cannot be executed.
  - This is a collapse mode, but not the original "always moveahead" collapse. It is invalid-action/stop-word collapse.
  - `format/correct` was misleading because it stayed high even when the action vocabulary was wrong.
- Code change:
  - `vagen/env/navigation/prompt.py`
    - Single-action prompt now explicitly says there is no `stay`, `stop`, `done`, `terminate`, `wait`, or `noop` action.
    - Reward text now says format correctness requires exactly one valid action name.
  - `vagen/env/navigation/env.py`
    - If any parsed action is not in `ACTION_LOOKUP`, force `format_correct=False` and `format_error_type=invalid_action_name`.
    - In no-dense/base mode, invalid actions now receive `invalid_action_penalty` instead of only receiving zero reward.
  - `vagen/rollout/qwen_rollout/action_metrics.py`
    - W&B metrics now include `format/error/invalid_action_name`.
  - Tests updated for prompt guard and invalid-action metric tracking.
- Local validation:
  - `python -m pytest tests/test_navigation_vagen1_minimal.py tests/test_navigation_vagen1_sweep.py -q`: `39 passed`
- Decision:
  - Next runs should use this guard before interpreting anti-collapse settings.
  - Existing old-code runs can still be observed, but their `format/correct` curve is not trustworthy for action legality.

### E3.28 Tiny Progress Reward Adjustment Rationale

- Prepared at: 2026-07-25 HKT
- Trigger:
  - User agreed that a very small progress reward is acceptable.
- Change hypothesis:
  - Step=1 has much sparser success feedback than original VAGEN step=5 action chunks.
  - A tiny progress reward can help early exploration learn which single actions reduce distance, without letting dense shaping dominate the original VAGEN base objective.
  - Because raw samples show invalid stop-word collapse, the main guard remains invalid-action failure plus repeat-action penalty; progress reward is only auxiliary.
- Code/config change:
  - `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_light.yaml`
    - `progress_reward`: `0.02 -> 0.01`
    - `regress_penalty`: `-0.02 -> -0.01`
    - `repeat_action_penalty`: `-0.02 -> -0.01`
    - `repeat_action_penalty_cap`: `-0.06 -> -0.03`
    - `invalid_action_penalty`: keep `-0.05`
  - `scripts/superpod/run_navigation_vagen1_8gpu_local_save5_dense_light.sbatch`
    - default `LOSS_MASK_MODE=default`, not `answer_only`.
    - W&B tags now include `default-loss` and `tiny-progress`.
- Intended comparison:
  - `default loss + invalid-action guard + no dense`
  - `default loss + invalid-action guard + tiny progress/repeat shaping`
- Expected observations:
  - `format/error/invalid_action_name` should fall quickly after the guard.
  - `action/top_share` and `all_same_traj` should improve relative to old-code runs.
  - Success should not rise together with invalid stop-word actions.
  - If progress reward increases success but also causes a new local optimum, reduce or remove it.

### E3.29 vLLM/Ray Step20 Stability Patch Rationale

- Prepared at: 2026-07-25 HKT
- Trigger:
  - User asked how to handle the vLLM crash after accepting the tiny-progress setting.
- Root cause summary:
  - `486499` crashed during step20 validation generation, not during training update or AI2-THOR environment creation.
  - The trainer currently runs validation before checkpointing when `TEST_FREQ` and `SAVE_FREQ` coincide.
  - Therefore a validation-time vLLM/Ray worker crash prevented `global_step_20` from being saved.
  - Logs repeatedly warned that Ray session storage under `/project/peilab/hligb/vn-tmp/ray/...` was over 95% full. Ray uses filesystem usage percentage, so a large shared `/project` filesystem can trigger object-spilling risk warnings even when visible free space remains.
- Patch:
  - `vagen/trainer/ppo/ray_trainer.py`
    - Same-step checkpointing now happens before validation.
    - If validation/vLLM generation crashes, the checkpoint for that global step should already exist.
  - `vagen/trainer/main_ppo.py`
    - Local `ray.init()` now passes `_temp_dir=$RAY_TMPDIR` when `RAY_TMPDIR` is set.
  - `scripts/superpod/run_navigation_vagen1_8gpu_local_save5.sbatch`
    - Defines `VAGEN_NODE_LOCAL_ROOT=/tmp/$USER/vagen-navigation/$SLURM_JOB_ID`.
    - Moves per-job `RAY_TMPDIR`, `TMPDIR`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, and `VAGEN_AI2THOR_HOME` under node-local `/tmp`.
  - `scripts/superpod/run_navigation_vagen1_4gpu.sbatch`
    - Adds the same node-local runtime override for future 4GPU local runs.
- Non-goals:
  - This does not change reward, prompt, or policy objective.
  - This does not guarantee vLLM can never crash; it reduces Ray tmp/spill pressure and prevents eval crashes from losing same-step checkpoints.
- Expected observations:
  - New runs should not print Ray over-95%-full warnings pointing to `/project/.../vn-tmp/ray`.
  - If validation still crashes, `global_step_20` should exist when `SAVE_FREQ=20` or `SAVE_FREQ=5`.
  - If vLLM still dies with `none_dealloc`, next mitigation is to reduce validation generation pressure: lower `VAL_BATCH_SIZE`, lower val generation logging, increase `TEST_FREQ`, or reduce vLLM batched token pressure.

### E3.30 Local8 Dense Tiny-Progress 12h/24h Submission Rationale

- Prepared at: 2026-07-25 HKT
- Trigger:
  - User requested two 8GPU runs with the accepted tiny-progress settings: one 12h and one 24h, both 60 steps and `TEST_FREQ=15`.
- Shared config:
  - `N_GPUS_PER_NODE=8`
  - `ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=4`
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=15`
  - `SAVE_FREQ=5`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `ENV_CONFIG_PATH=scripts/examples/vagen_base/navigation_vagen1/env_config_dense_light.yaml`
  - `progress_reward=0.01`
  - `regress_penalty=-0.01`
  - `repeat_action_penalty=-0.01`
  - `repeat_action_start=3`
  - `repeat_action_penalty_cap=-0.03`
  - `invalid_action_penalty=-0.05`
- Runtime stability patches included:
  - Checkpoint before validation.
  - Ray local temp directory under `/tmp/$USER/vagen-navigation/$SLURM_JOB_ID`.
  - `ray.init(_temp_dir=$RAY_TMPDIR)`.
- Expected observations:
  - Step15 validation should run after step15 checkpoint is saved.
  - Step30/45/60 validation should not lose matching checkpoints if vLLM crashes.
  - `format/error/invalid_action_name` should expose stop-word collapse directly.
  - `action/top_share`, `action/all_same_traj`, and raw samples determine whether tiny-progress helps or still collapses.

### E3.30 Local8 Dense Tiny-Progress 12h/24h Submission Result

- Submitted at: 2026-07-25 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Code/doc commit on SuperPOD: `4027024`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Server validation:
  - `git pull --ff-only` updated the clone to `4027024`.
  - `bash -n` passed for:
    - `scripts/superpod/run_navigation_vagen1_8gpu_local_save5.sbatch`
    - `scripts/superpod/run_navigation_vagen1_8gpu_local_save5_dense_light.sbatch`
    - `scripts/superpod/run_navigation_vagen1_4gpu.sbatch`
- Jobs:
  - `486884`: 8GPU local dense tiny-progress, 12h, `PENDING`, reason `Priority`
  - `486885`: 8GPU local dense tiny-progress, 24h, `PENDING`, reason `Priority`
- W&B:
  - Group: `navigation_vagen1_local8gpu_tiny_progress_test15_20260725`
  - `486884` name: `navigation_vagen1_local8gpu_tiny_progress_default_test15_12h_20260725T074533Z`
  - `486885` name: `navigation_vagen1_local8gpu_tiny_progress_default_test15_24h_20260725T074533Z`
  - URLs pending until jobs start.
- Key settings:
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=15`
  - `SAVE_FREQ=5`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `ENV_CONFIG_PATH=scripts/examples/vagen_base/navigation_vagen1/env_config_dense_light.yaml`
  - `progress_reward=0.01`
  - `regress_penalty=-0.01`
  - `repeat_action_penalty=-0.01`
  - `repeat_action_start=3`
  - `repeat_action_penalty_cap=-0.03`
  - `invalid_action_penalty=-0.05`
  - `RAW_SAMPLES_TO_LOG=8`
  - `FINAL_VAL_AFTER_TRAIN=False`
  - `VAL_BEFORE_TRAIN=False`
  - `REMOVE_PREVIOUS_CKPT_IN_SAVE=False`
- Queue snapshot:
  - Both jobs pending with `START_TIME=N/A` at the first check.

### E3.31 Old-Code Local8 Runs Cancelled

- Checked/cancelled at: 2026-07-25 HKT
- Cancelled jobs:
  - `486500`: old-code local8 `FORMAT_REWARD=0.1`, no dense, W&B `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/um8ugntb`
  - `486501`: old-code local8 dense-light, W&B `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/2qxab1fy`
- Slurm result:
  - `486500`: `CANCELLED by 3733`, elapsed `03:09:22`
  - `486501`: `CANCELLED by 3733`, elapsed `03:09:22`
- Reason:
  - Both runs used old code without invalid-action guard, save-before-validation, Ray local tmp, and default-loss/tiny-progress final settings.
  - Raw samples already gave enough evidence that the old-code runs were exploiting the format/action legality gap.
- Raw sample diagnosis:
  - `486500`:
    - raw samples inspected: `152`
    - `action_is_valid=False`: `139`
    - `format_error_type=ok`: `152`
    - common invalid actions: `staythere`, `stay`, `stayclose`, `stop`
  - `486501`:
    - raw samples inspected: `160`
    - `action_is_valid=False`: `130`
    - `format_error_type=ok`: `160`
    - common invalid actions: `rotatel`, `stay`, `rotatelleft`, `stop`, `terminate`
- Checkpoints retained for failure comparison:
  - Both runs saved `global_step_5`, `global_step_10`, and `global_step_15`.
- Active next runs:
  - `486884`: new-code local8 tiny-progress 12h, pending
  - `486885`: new-code local8 tiny-progress 24h, pending
- Decision:
  - Do not use `486500/486501` for policy-quality decisions except as old-code failure evidence.
  - Use `486884/486885` as the next meaningful test of invalid-action guard + default loss + tiny progress + Ray stability patches.

### E3.32 Local8 Tiny-Progress 12h Ray Tmp Failure and Fix

- Checked at: 2026-07-25 HKT
- Job:
  - `486884`: new-code local8 tiny-progress 12h, `FAILED`, elapsed `01:40:07`, node `dgx-31`
  - W&B: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/34pbu76n`
- Failure:
  - Ray still used `/project/peilab/hligb/vn-tmp/ray/486884/session_...`.
  - Logs repeatedly printed `file_system_monitor.cc:116 ... is over 95% full`.
  - Available space under the Ray session path dropped from about `85 GB` to about `0.21 GB`.
  - This confirms the previous local tmp patch did not take effect.
- Root cause:
  - `prepare_project_storage.sh` can set `RAY_TMPDIR=/project/peilab/hligb/vn-tmp/ray/$SLURM_JOB_ID` before the local runtime override.
  - The override used `${RAY_TMPDIR:-$VAGEN_NODE_LOCAL_ROOT/ray}`, which preserves an already-set value.
  - Therefore the job kept the `/project` Ray tmp path instead of switching to node-local `/tmp`.
- Partial progress:
  - The invalid-action guard was active.
  - Raw samples now report invalid stop words as `format_error_type=invalid_action_name`, not `ok`.
  - However the policy still frequently emitted invalid stop words (`stop`, `stay`, `stayclose`), so collapse is now visible and penalized but not yet solved.
  - Checkpoints saved: `global_step_5`, `global_step_10`.
- Fix:
  - `scripts/superpod/run_navigation_vagen1_8gpu_local_save5.sbatch`
    - Force `RAY_TMPDIR=$VAGEN_NODE_LOCAL_ROOT/ray` instead of preserving an inherited value.
    - Force `TMPDIR`, `TRITON_CACHE_DIR`, and `TORCHINDUCTOR_CACHE_DIR` to node-local paths.
    - Print runtime dir values at startup.
  - `scripts/superpod/run_navigation_vagen1_4gpu.sbatch`
    - Same forced local override for future 4GPU local runs.
- Decision:
  - Symlink is not the preferred fix because `/tmp` is node-local and the compute node is not known before scheduling.
  - Force-overriding the runtime dirs inside the sbatch is safer and node-correct.
  - `486885` is still pending and should use the patched base script when it starts, because its wrapper calls the repo script at runtime.

### E3.33 Local8 Tiny-Progress 12h Resubmission Rationale

- Prepared at: 2026-07-25 HKT
- Trigger:
  - User asked to submit a fixed version of the failed `486884` 12h run.
- Relationship to `486884`:
  - Same experiment settings as `486884`.
  - Difference: submitted after commit `f842046`, which force-overrides Ray/tmp/cache runtime directories to node-local `/tmp`.
- Relationship to `486885`:
  - `486885` is the 24h version of the same fixed configuration and remains pending.
  - This new 12h run is a shorter companion probe with the same fixed code path.
- Shared settings:
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=15`
  - `SAVE_FREQ=5`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `ENV_CONFIG_PATH=scripts/examples/vagen_base/navigation_vagen1/env_config_dense_light.yaml`
  - tiny progress/repeat/invalid reward settings from E3.28.
- Expected observations:
  - Startup log must show `RAY_TMPDIR=/tmp/...`, not `/project/.../vn-tmp/ray`.
  - No repeated Ray `over 95% full` warning against `/project`.
  - If vLLM still fails after that, the root cause is not the previous Ray tmp path bug.

### E3.33 Local8 Tiny-Progress 12h Resubmission Result

- Submitted at: 2026-07-25 HKT
- Code/doc commit on SuperPOD: `bf9a06b`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Server validation:
  - `git pull --ff-only` updated the clone to `bf9a06b`.
  - `bash -n` passed for the local8 base and dense-light sbatch scripts.
- Job:
  - `487075`: fixed 12h local8 tiny-progress resubmission, `PENDING`, reason `Priority`
  - W&B name: `navigation_vagen1_local8gpu_tiny_progress_default_test15_12h_fixed_20260725T112345Z`
  - W&B group: `navigation_vagen1_local8gpu_tiny_progress_test15_20260725`
- Queue snapshot:
  - `486885`: 24h fixed-code companion, still `PENDING`, estimated start `2026-07-27T01:43:35`
  - `487075`: 12h fixed resubmission, `PENDING`, estimated start `N/A`
- Note:
  - The first log check after startup must verify the printed runtime line uses node-local `/tmp`.

### E3.34 Local8 Tiny-Progress Startup Failure Diagnosis

- Checked at: 2026-07-25 HKT
- Jobs inspected:
  - `486885`: fixed-code 24h local8 tiny-progress companion, `FAILED`, elapsed `00:24:57`, node `dgx-32`
  - `487075`: fixed 12h local8 tiny-progress resubmission, later also `FAILED`, elapsed `00:23:55`, node `dgx-32`
- W&B:
  - `486885`: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/nll3lau7`
  - `487075`: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/4540y2nt`
- Runtime-dir result:
  - The Ray tmp fix worked for both inspected runs.
  - Logs show `ray.init(_temp_dir='/tmp/hligb/vagen-navigation/<job>/ray')`.
  - This is no longer the `/project/peilab/hligb/vn-tmp/ray/...` full-filesystem failure from `486884`.
- `486885` failure:
  - The run reached W&B and `global_steps: 1`.
  - It then stopped at `Processing mini-batch 1/8, size: 2`.
  - Traceback:
    - `requests.exceptions.ReadTimeout: HTTPConnectionPool(host='localhost', port=5885): Read timed out. (read timeout=1200)`
    - Call path: rollout manager reset -> `create_environments_batch` -> local AI2-THOR HTTP server `/environments`.
  - Interpretation: local AI2-THOR environment creation did not return within 1200 seconds.
- `487075` failure:
  - W&B run created and local server started on port `5075`.
  - It reached `global_steps: 1` and `Processing mini-batch 1/8, size: 2`.
  - It then failed with the same `requests.exceptions.ReadTimeout` pattern against `localhost:5075`.
  - The server log showed initialization only, which is consistent with being blocked in env creation or first reset rather than in PPO update.
- Comparison with earlier local8 runs:
  - `486499` ran on `dgx-18` for `03:35:33`, reached step20 validation, and saved `global_step_5`, `global_step_10`, and `global_step_15`.
  - `486500` ran on `dgx-18` for `03:09:22` before manual cancellation.
  - `486501` ran on `dgx-31` for `03:09:22` before manual cancellation and reached at least `global_steps: 18`.
  - Those earlier local8 runs also used a local AI2-THOR server with `NavigationServiceConfig(max_workers=4, devices=[0,1,2,3,4,5,6,7])` and mini-batches of size 2.
  - Therefore the latest failures do not prove the local8 topology or `ROLLOUT_MINI_BATCH_SIZE=2` is intrinsically broken.
  - The strongest new signal is node-specific: both fixed-code startup failures landed on `dgx-32`, while the earlier local8 jobs that progressed used `dgx-18` or `dgx-31`.
- Current config of the failing/stuck local8 runs:
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=15`
  - `SAVE_FREQ=5`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `MAX_TURNS=20`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `SERVER_NAVIGATION_MAX_WORKERS=4`
  - local colocated AI2-THOR server and training/vLLM on the same 8GPU node
  - tiny progress/repeat/invalid reward from E3.28
- Decision:
  - Do not treat the latest failure as a vLLM or Ray tmp failure.
  - The immediate blocker is local AI2-THOR env creation/first reset.
  - Because both immediate startup failures happened on `dgx-32`, test node-specific AI2-THOR instability before concluding that the configuration itself is too aggressive.
  - A symlink is not the right fix for this failure, because the problematic directory issue is already fixed and this traceback is an HTTP env-server timeout.
  - Next probe should either exclude `dgx-32` while keeping the same local8 config, or run a tiny AI2-THOR/env-create smoke on `dgx-32` if an interactive allocation is available.
  - If excluding `dgx-32` still fails at the same point, then reduce env creation pressure before changing learning-side rewards: use fewer local server workers and/or `ROLLOUT_MINI_BATCH_SIZE=1`, or return to separated env server topology.
  - Increasing the HTTP timeout alone is not enough evidence-based mitigation, because `486885` waited 20 minutes before failing and `487075` is already sitting in the same stage.

### E3.35 Local8 Tiny-Progress Requeue Excluding dgx-32

- Submitted at: 2026-07-25 HKT
- Trigger:
  - User asked to cancel the two latest failed/stuck jobs and resubmit with unchanged parameters.
- Cancellation / old job state:
  - `486885`: already `FAILED`, elapsed `00:24:57`, node `dgx-32`.
  - `487075`: already `FAILED`, elapsed `00:23:55`, node `dgx-32`.
  - `scancel 486885 487075` was still issued as a cleanup no-op.
- Code state:
  - Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
  - SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
  - Commit on SuperPOD before submit: `6ae7292`
- Parameter policy:
  - Keep all learning/runtime parameters unchanged from E3.34.
  - Only add Slurm scheduling constraint `--exclude=dgx-32` to test whether the immediate env-create timeout is node-specific.
- Shared settings:
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=15`
  - `SAVE_FREQ=5`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `MAX_TURNS=20`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `SERVER_NAVIGATION_MAX_WORKERS=4`
  - `ENV_CONFIG_PATH=scripts/examples/vagen_base/navigation_vagen1/env_config_dense_light.yaml`
  - tiny progress/repeat/invalid reward from E3.28
  - `RAW_SAMPLES_TO_LOG=8`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
  - `REMOVE_PREVIOUS_CKPT_IN_SAVE=False`
- New jobs:
  - `487237`: 8GPU local dense tiny-progress, 12h, `PENDING (Priority)`, `--exclude=dgx-32`
    - W&B name: `navigation_vagen1_local8gpu_tiny_progress_default_test15_12h_exdgx32_20260725T140007Z`
  - `487238`: 8GPU local dense tiny-progress, 24h, `PENDING (Priority)`, `--exclude=dgx-32`
    - W&B name: `navigation_vagen1_local8gpu_tiny_progress_default_test15_24h_exdgx32_20260725T140007Z`
- W&B group:
  - `navigation_vagen1_local8gpu_tiny_progress_test15_20260725`
- Expected observations:
  - If either job starts on a node other than `dgx-32` and passes mini-batch 1/8, the E3.34 failures were likely node-specific.
  - If the same `/environments` timeout repeats off `dgx-32`, reduce env pressure next rather than changing reward: `SERVER_NAVIGATION_MAX_WORKERS=1/2` and/or `ROLLOUT_MINI_BATCH_SIZE=1`.

### E3.36 Local8 Excluding dgx-32 Results

- Checked at: 2026-07-26 HKT
- Jobs:
  - `487237`: 12h local8 tiny-progress, `FAILED`, elapsed `03:32:36`, node `dgx-31`
  - `487238`: 24h local8 tiny-progress, `FAILED`, elapsed `00:26:57`, node `dgx-37`
- W&B:
  - `487237`: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/50sh1ozl`
  - `487238`: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/xaid92yx`
- `487237` progress:
  - Excluding `dgx-32` helped this run pass the immediate step1 env-create blocker.
  - It completed many rollout/update steps and logged train metrics through `step:24`.
  - Saved checkpoints: `global_step_5`, `global_step_10`, `global_step_15`, `global_step_20`.
  - Checkpoint root: `/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_local8gpu_tiny_progress_default_test15_12h_exdgx32_20260725T140007Z`
- `487237` failure:
  - It crashed at `global_steps: 25`, `Processing mini-batch 1/8`.
  - Stack: `rollout_manager.rollout_loop()` -> `actor_rollout_wg.generate_sequences(gen_batch)` -> Ray actor died.
  - Worker printed `Fatal Python error: none_dealloc: deallocating None`.
  - Ray reported `ActorDiedError`, worker exit type `SYSTEM_ERROR`, connection error code 2.
  - Interpretation: this is the recurring vLLM/Ray worker crash during generation, not an AI2-THOR env-create timeout.
- `487237` latest metrics before crash:
  - At `step:24`: `train/success=0.438`, `train/score=4.241`, `timing_s/step=440.812`.
  - `train/action/top_share=0.882`, `train/action/entropy=0.224`, `train/action/all_same_traj=0.688`.
  - `train/format/correct=0.863`, `train/format/too_many_actions=0.057`, `train/format/error/invalid_action_name=0.078`.
  - This suggests learning is happening, but action concentration and repeated/same-trajectory behavior are still concerning.
- `487238` failure:
  - It started on `dgx-37`, created W&B, and reached `global_steps: 1`.
  - It failed at `Processing mini-batch 1/8, size: 2` with `requests.exceptions.ReadTimeout` against `localhost:5238` after 1200 seconds.
  - Server log showed startup but no healthy progression beyond initial environment creation.
  - Interpretation: `dgx-37` also shows local AI2-THOR env-create/first-reset instability.
- Decision:
  - The local8 configuration is not completely broken: `487237` on `dgx-31` ran to step24 and saved a usable step20 checkpoint.
  - Node-specific env instability is real, but not limited to `dgx-32`; `dgx-37` also failed immediately.
  - The next stability target is vLLM/Ray generation crash after step20 plus node filtering.
  - Candidate next run should keep useful learning settings but reduce vLLM generation pressure: disable CUDA graph with `ROLLOUT_ENFORCE_EAGER=True` or reduce vLLM batched-token/model-len pressure; also consider excluding `dgx-32,dgx-37`.
  - Do not increase env/server pressure until the vLLM crash is controlled.

### E3.37 Eager vLLM and Stronger Action-Penalty Rationale

- Prepared at: 2026-07-26 HKT
- Trigger:
  - User approved trying `ROLLOUT_ENFORCE_EAGER=True` and asked for a second anti-collapse run with stronger action-level penalty.
- Evidence from `487237`:
  - The run passed the local env-create phase on `dgx-31`, reached `step:24`, and saved `global_step_20`.
  - It crashed at `global_steps: 25` during vLLM generation with `Fatal Python error: none_dealloc: deallocating None` and Ray `ActorDiedError`.
  - This points to vLLM/Ray generation stability, not reward/parser/env-create.
  - Learning was visible: success reached `0.625` at step19 and `0.688` at step22, but action concentration remained high.
  - Collapse indicators near the end: `step24 train/action/top_share=0.882`, `all_same_traj=0.688`, `moveahead_share=0.701`, `format/too_many_actions=0.057`, `invalid_action_name=0.078`.
- Planned jobs:
  - `eager_tiny`: keep `487237` learning parameters unchanged and set only `ROLLOUT_ENFORCE_EAGER=True`.
  - `eager_actionpen`: keep eager on, but use a stronger action-penalty env config.
- New stronger action-penalty config:
  - File: `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_action_penalty.yaml`
  - Keep tiny progress/regress shaping unchanged:
    - `progress_reward=0.01`
    - `regress_penalty=-0.01`
  - Strengthen only action-level penalties:
    - `repeat_action_penalty: -0.01 -> -0.02`
    - `repeat_action_penalty_cap: -0.03 -> -0.06`
    - `invalid_action_penalty: -0.05 -> -0.08`
    - `repeat_action_start=3`
- Expected observations:
  - `eager_tiny` tests whether disabling CUDA graph prevents the step20+ `none_dealloc` crash.
  - `eager_actionpen` tests whether stronger action-level penalties reduce `top_share`, `all_same_traj`, invalid actions, and repeated multi-action patterns without suppressing success.
  - Both should exclude `dgx-32,dgx-37` because those nodes showed immediate local AI2-THOR env-create instability.

### E3.37 Eager vLLM and Action-Penalty Submission Result

- Submitted at: 2026-07-26 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Code commit on SuperPOD before submit: `7b9313c`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Server validation:
  - `git pull --ff-only` updated the clone to `7b9313c`.
  - `bash -n` passed for:
    - `scripts/superpod/run_navigation_vagen1_8gpu_local_save5.sbatch`
    - `scripts/superpod/run_navigation_vagen1_8gpu_local_save5_dense_light.sbatch`
- Shared settings:
  - `N_GPUS_PER_NODE=8`
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=15`
  - `SAVE_FREQ=5`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `MAX_TURNS=20`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `SERVER_NAVIGATION_MAX_WORKERS=4`
  - `ROLLOUT_ENFORCE_EAGER=True`
  - `RAW_SAMPLES_TO_LOG=8`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
  - Slurm exclude: `dgx-32,dgx-37`
- New jobs:
  - `487856`: `eager_tiny`, 12h, `PENDING (Priority)` at submit.
    - W&B name: `navigation_vagen1_local8gpu_eager_tiny_test15_12h_20260726T073021Z`
    - Env config: `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_light.yaml`
  - `487857`: `eager_actionpen`, 12h, `PENDING (Priority)` at submit.
    - W&B name: `navigation_vagen1_local8gpu_eager_actionpen_test15_12h_20260726T073021Z`
    - Env config: `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_action_penalty.yaml`
- W&B group:
  - `navigation_vagen1_local8gpu_eager_stability_20260726`
- Gates:
  - `eager_tiny` passes if it reaches step60 or at least passes the step25 region without `none_dealloc`/Ray actor death.
  - `eager_actionpen` is judged against `eager_tiny` on `train/action/top_share`, `all_same_traj`, `format/error/invalid_action_name`, `too_many_actions`, and success.

### E3.38 Eager Submission Wrapper Failure

- Checked at: 2026-07-28 HKT
- Jobs:
  - `487856`: `FAILED`, exit `2:0`, elapsed `00:00:01`, node `dgx-46`
  - `487857`: `FAILED`, exit `2:0`, elapsed `00:00:01`, node `dgx-46`
- W&B:
  - No W&B run was created for either job.
- Checkpoints:
  - No checkpoints were created.
- Failure:
  - `487856` stderr: `Unknown VAGEN1_VARIANT=eager_tiny`
  - `487857` stderr: `Unknown VAGEN1_VARIANT=eager_actionpen`
- Root cause:
  - The 8GPU wrapper passes through to `run_navigation_vagen1_4gpu.sbatch`, which sources `scripts/superpod/configure_navigation_vagen1_variant.sh` whenever `VAGEN1_VARIANT != manual`.
  - `eager_tiny` and `eager_actionpen` were used as descriptive run labels, but they were not defined as recognized variants in `configure_navigation_vagen1_variant.sh`.
  - Therefore both jobs exited before loading Python, W&B, AI2-THOR, or vLLM.
- Interpretation:
  - This is a submission/wrapper bug, not evidence about `ROLLOUT_ENFORCE_EAGER=True` or the stronger action-penalty reward.
- Next fix:
  - Either resubmit with `VAGEN1_VARIANT=manual` while keeping descriptive names/tags, or add explicit `eager_tiny` and `eager_actionpen` cases to the variant config.
  - The lower-risk immediate resubmission path is `VAGEN1_VARIANT=manual`, because all intended parameters are already passed explicitly by environment variables.

### E3.39 Eager Variant Wrapper Fix Rationale

- Prepared at: 2026-07-28 HKT
- Trigger:
  - User asked why the jobs still failed after the previous "fix" and requested a real correction plus submission.
- Root cause reminder:
  - E3.38 only recorded the failure; it did not change already-submitted Slurm job environments.
  - The real code path failed because `VAGEN1_VARIANT=eager_tiny/eager_actionpen` was not recognized by `configure_navigation_vagen1_variant.sh`.
- Code fix:
  - Add explicit variant cases:
    - `eager_tiny`: `set_eager_runtime 0.4 8`
    - `eager_actionpen`: `set_eager_runtime 0.4 8`
  - Keep reward differences controlled by `ENV_CONFIG_PATH`:
    - `eager_tiny` uses `env_config_dense_light.yaml`
    - `eager_actionpen` uses `env_config_dense_action_penalty.yaml`
- Expected effect:
  - The jobs should no longer exit in 1 second with `Unknown VAGEN1_VARIANT`.
  - Startup logs should show `actor_rollout_ref.rollout.enforce_eager=True`.

### E3.39 Eager Variant Wrapper Fix Submission Result

- Submitted at: 2026-07-28 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Fix commit used on SuperPOD: `338fe1f`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Validation:
  - Local: `python -m pytest tests/test_navigation_vagen1_sweep.py -q` passed, `24 passed`.
  - Local: `bash -n scripts/superpod/configure_navigation_vagen1_variant.sh` passed.
  - SuperPOD: `git pull --ff-only` updated the clone to `338fe1f`.
  - SuperPOD: `bash -n` passed for the variant config and local8 wrappers.
- New corrected jobs:
  - `492843`: corrected `eager_tiny`, 12h, `PENDING (Priority)` at submit.
    - W&B name: `navigation_vagen1_local8gpu_eager_tiny_fixed_test15_12h_20260728T104753Z`
    - Env config: `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_light.yaml`
  - `492844`: corrected `eager_actionpen`, 12h, `PENDING (Priority)` at submit.
    - W&B name: `navigation_vagen1_local8gpu_eager_actionpen_fixed_test15_12h_20260728T104753Z`
    - Env config: `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_action_penalty.yaml`
- Shared settings:
  - `ROLLOUT_ENFORCE_EAGER=True`
  - `VAGEN1_VARIANT=eager_tiny/eager_actionpen` now recognized by the variant config
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=15`
  - `SAVE_FREQ=5`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `SERVER_NAVIGATION_MAX_WORKERS=4`
  - Slurm exclude: `dgx-32,dgx-37`
- W&B group:
  - `navigation_vagen1_local8gpu_eager_stability_20260728`

### E3.40 Corrected Eager Jobs Result

- Checked at: 2026-07-30 HKT
- Jobs:
  - `492843`: corrected `eager_tiny`, `FAILED`, exit `1:0`, elapsed `00:24:17`, node `dgx-35`
  - `492844`: corrected `eager_actionpen`, `FAILED`, exit `1:0`, elapsed `00:23:48`, node `dgx-35`
- W&B:
  - `492843`: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/pzmgvm4b`
  - `492844`: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/lo4lixe9`
- Checkpoints:
  - Checkpoint root directories were created for both runs, but no `global_step_*` checkpoint was saved because neither run completed step 1.
- Confirmed intended runtime settings:
  - Both jobs reached Python training startup.
  - `VAGEN1_VARIANT=eager_tiny/eager_actionpen` was now accepted.
  - vLLM logs confirmed `actor_rollout_ref.rollout.enforce_eager=True`.
  - `ROLLOUT_GPU_MEMORY_UTILIZATION=0.4`, `ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192`, `ROLLOUT_MAX_MODEL_LEN=5000`, `ROLLOUT_LIMIT_MM_PER_PROMPT=8`.
- Failure:
  - Both jobs reached `global_steps: 1`, `Processing mini-batch 1/8, size: 2`.
  - Both failed while calling the local env server endpoint `/environments`.
  - Error: `requests.exceptions.ReadTimeout: HTTPConnectionPool(host='localhost', port=5843/5844): Read timed out. (read timeout=1200)`.
  - Server logs showed Flask startup with `navigation.max_workers=4` and devices `[0,1,2,3,4,5,6,7]`, but no successful environment creation progress before the client timed out.
- Interpretation:
  - E3.39 fixed the wrapper bug, but the corrected eager jobs did not reach the vLLM stability question.
  - The current blocker is local AI2-THOR environment creation/reset hanging on `dgx-35` during the first mini-batch.
  - This matches previous local-env failures on `dgx-32` and `dgx-37`, while `dgx-31` was able to run through step 24 before hitting the separate vLLM/Ray crash.
  - Therefore `ROLLOUT_ENFORCE_EAGER=True` is still untested for the step20+ `none_dealloc` failure.
- Decision:
  - Do not interpret these jobs as anti-collapse evidence, because no rollout metrics were logged.
  - Next direction should focus on env stability before reward tuning:
    - avoid local env on nodes that have shown first mini-batch `/environments` timeout: `dgx-32`, `dgx-35`, `dgx-37`;
    - or use the external env-server topology so training can fail/restart independently from long-lived env servers;
    - reduce initial env pressure if using local env again, for example `SERVER_NAVIGATION_MAX_WORKERS=1-2`, `ROLLOUT_MINI_BATCH_SIZE=1`, and possibly `MAX_TURNS=10` for a smoke gate before returning to 20.

### E3.41 Local8 Eager Worker2 Rationale

- Prepared at: 2026-07-30 HKT
- Trigger:
  - User approved rerunning the two eager experiments with `SERVER_NAVIGATION_MAX_WORKERS=2` and `ROLLOUT_MINI_BATCH_SIZE=2`.
- Hypothesis:
  - E3.40 failed before rollout metrics because local AI2-THOR `/environments` creation hung on `dgx-35` with `SERVER_NAVIGATION_MAX_WORKERS=4`.
  - Keeping `ROLLOUT_MINI_BATCH_SIZE=2` preserves the mini-batch pressure used in the prior learning run, while lowering server workers from 4 to 2 should reduce simultaneous Unity/AI2-THOR process creation pressure.
  - Excluding `dgx-32,dgx-35,dgx-37` avoids nodes already observed to fail first-mini-batch local env creation.
- Planned jobs:
  - `eager_tiny_w2`: same as corrected `eager_tiny`, but `SERVER_NAVIGATION_MAX_WORKERS=2`.
  - `eager_actionpen_w2`: same as corrected `eager_actionpen`, but `SERVER_NAVIGATION_MAX_WORKERS=2`.
- Shared settings:
  - `N_GPUS_PER_NODE=8`
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=15`
  - `SAVE_FREQ=5`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `MAX_TURNS=20`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `SERVER_NAVIGATION_MAX_WORKERS=2`
  - `ROLLOUT_ENFORCE_EAGER=True`
  - `RAW_SAMPLES_TO_LOG=8`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
  - Slurm exclude: `dgx-32,dgx-35,dgx-37`
- Expected observations:
  - Primary gate: pass `global_steps: 1` and complete at least step 5 without `/environments` timeout.
  - If both fail before step 1 again, the next local-env smoke should use `SERVER_NAVIGATION_MAX_WORKERS=1` and/or `ROLLOUT_MINI_BATCH_SIZE=1`, or return to external env servers.
  - If either reaches step20+, compare against `487237` for vLLM/Ray stability under `ROLLOUT_ENFORCE_EAGER=True`.

### E3.41 Local8 Eager Worker2 Submission Result

- Submitted at: 2026-07-30 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit used on SuperPOD: `9f76bd366f9cb4bfe33ecb5594f1f1daf48eac3b`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Validation:
  - Local `bash -n` passed for local8 wrappers and variant config.
  - Local `git diff --check` passed.
  - SuperPOD `git pull --ff-only` updated the clone to `9f76bd3`.
  - SuperPOD `bash -n` passed for local8 wrappers and variant config.
- New jobs:
  - `496841`: `eager_tiny_w2`, 12h, `PENDING (Priority)` at submit.
    - W&B name: `navigation_vagen1_local8gpu_eager_tiny_w2_rmb2_test15_12h_20260729T165020Z`
    - Env config: `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_light.yaml`
  - `496842`: `eager_actionpen_w2`, 12h, `PENDING (Priority)` at submit.
    - W&B name: `navigation_vagen1_local8gpu_eager_actionpen_w2_rmb2_test15_12h_20260729T165020Z`
    - Env config: `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_action_penalty.yaml`
- W&B group:
  - `navigation_vagen1_local8gpu_eager_worker2_20260730`
- Shared settings:
  - `ROLLOUT_ENFORCE_EAGER=True`
  - `VAGEN1_VARIANT=eager_tiny/eager_actionpen`
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=15`
  - `SAVE_FREQ=5`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `SERVER_NAVIGATION_MAX_WORKERS=2`
  - Slurm exclude: `dgx-32,dgx-35,dgx-37`
- Initial queue state:
  - Both jobs were `PENDING (Priority)`.
  - Slurm did not provide a start-time estimate at submission (`START_TIME=N/A`).

### E3.42 Local8 Eager Worker2 Result

- Checked at: 2026-07-30 HKT
- Jobs:
  - `496841`: `eager_tiny_w2`, `FAILED`, exit `1:0`, elapsed `03:44:49`, node `dgx-54`.
  - `496842`: `eager_actionpen_w2`, `FAILED`, exit `0:53`, elapsed `00:00:00`, node `dgx-26`.
- W&B:
  - `496841`: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/cesfl46y`
  - `496842`: no W&B run and no log files were created.
- Checkpoints:
  - `496841` saved:
    - `/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_local8gpu_eager_tiny_w2_rmb2_test15_12h_20260729T165020Z/global_step_5`
    - `/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_local8gpu_eager_tiny_w2_rmb2_test15_12h_20260729T165020Z/global_step_10`
    - `/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_local8gpu_eager_tiny_w2_rmb2_test15_12h_20260729T165020Z/global_step_15`
- `496841` progress:
  - Confirmed intended settings: `SERVER_NAVIGATION_MAX_WORKERS=2`, `ROLLOUT_MINI_BATCH_SIZE=2`, `ROLLOUT_ENFORCE_EAGER=True`, `LOSS_MASK_MODE=default`, `FORMAT_REWARD=0.05`.
  - It passed the first-mini-batch blocker that killed E3.40.
  - It logged steps 1-14 and saved `global_step_15`.
  - Representative metrics:
    - step1: `train/success=0.250`, `train/score=2.279`, `top_share=0.920`, `entropy=0.217`, `all_same_traj=0.438`, `format_correct=0.934`, `timing_s/step=807.557`.
    - step5: `train/success=0.438`, `train/score=4.323`, `top_share=0.923`, `entropy=0.213`, `all_same_traj=0.562`, `format_correct=0.947`, `timing_s/step=811.322`.
    - step14: `train/success=0.438`, `train/score=4.210`, `top_share=0.887`, `entropy=0.302`, `all_same_traj=0.375`, `format_correct=0.816`, `invalid_action_name=0.184`, `timing_s/step=777.811`.
- `496841` failure:
  - After the step15 checkpoint save, the job failed with `requests.exceptions.ReadTimeout: HTTPConnectionPool(host='localhost', port=5841): Read timed out. (read timeout=1200)`.
  - Stack path: `rollout_manager.reset()` -> `env_client.create_environments_batch()` -> local env server `/environments`.
  - Interpretation: lowering workers from 4 to 2 improved startup stability, but local AI2-THOR server still eventually hangs during create/reset under long-running step=1 training.
- `496842` failure:
  - Slurm state: `FAILED`, `Reason=RaisedSignal:53(Real-time_signal_19)`, `ExitCode=0:53`, `RunTime=00:00:00`.
  - No stdout/stderr/server log/W&B run was created.
  - Interpretation: this is a Slurm/node-start failure before the batch script body ran, not a reward, vLLM, or AI2-THOR training failure.
- Decision:
  - `SERVER_NAVIGATION_MAX_WORKERS=2` is better than 4 for passing step1, but still not stable enough for a 60-step local-env run.
  - `ROLLOUT_ENFORCE_EAGER=True` still has not been tested past the old step20+ vLLM crash region, because the run died from env timeout around step15.
  - The dominant blocker remains local AI2-THOR server reliability, not anti-collapse reward tuning.
  - Next technical direction should be one of:
    - local smoke with `SERVER_NAVIGATION_MAX_WORKERS=1` while keeping `ROLLOUT_MINI_BATCH_SIZE=2`, to test whether a single server worker can survive beyond step20;
    - external env server topology, where env processes are isolated and can be monitored/restarted separately;
    - add retry/recreate logic around `/environments` create/reset so one hung AI2-THOR worker does not kill the entire PPO job.

### E3.43 Worker2 Raw Sample Collapse Inspection And Worker1 Rationale

- Prepared at: 2026-07-30 HKT
- Source run:
  - `496841` / W&B `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/cesfl46y`
- Raw sample availability:
  - The stdout log contains `120` `[RAW_SAMPLE]` rows.
  - These rows prioritize format/action failures, so they are especially useful for diagnosing collapse and invalid-action behavior, not for estimating the full success distribution.
- Raw sample observations:
  - Repeated invalid `stay` appeared across many consecutive turns:
    - examples around step10-12 repeatedly output `<answer>stay</answer>` with reasoning such as staying near a deep vessel, trash can, or coffee machine.
  - Repeated invalid terminal-like actions also appeared:
    - step13 repeatedly output `<answer>end</answer>` after claiming the lamp task was complete.
    - step14 repeatedly output `<answer>stop</answer>` after seeing a digital device.
  - Typo invalid actions appeared:
    - examples include `<answer>rotatelleft</answer>` and `<answer>rotatetleft</answer>`.
  - The think/reasoning text is also repetitive within a trajectory: once it thinks an object is found, it often repeats the same observation-reasoning-prediction pattern instead of choosing a valid navigation action.
- Metric interpretation from `496841`:
  - This is not a complete single-action collapse to only `moveahead`, but it is a clear collapse-like pattern:
    - `train/action/top_share` stayed high, mostly `0.85-0.96`.
    - `train/action/entropy` stayed low, mostly `0.10-0.37`.
    - `train/action/all_same_traj` often reached `0.5-0.8`.
  - `moveahead` is still dominant in aggregate on many steps, but raw samples show a second failure mode: invalid "do nothing / finish" actions (`stay`, `stop`, `end`) and repeated nearly identical CoT.
  - `too_many_actions` is near zero, so the prompt/parser single-action constraint is working for action count. The bigger issue is invalid action vocabulary plus repeated action/reasoning.
- Decision:
  - Lowering env server workers to `1` is justified to test long-run local AI2-THOR stability without changing learning/reward knobs.
  - Keep `ROLLOUT_MINI_BATCH_SIZE=2` to avoid making the run even slower unless worker1 still hangs.
  - Keep both reward variants:
    - `eager_tiny_w1` tests whether the tiny reward can run beyond step20 when env pressure is lower.
    - `eager_actionpen_w1` retests stronger invalid/repeat penalty because the previous actionpen job never entered Python.
  - If worker1 reaches step20 but still shows high invalid `stay/stop/end`, the next anti-collapse change should explicitly constrain/penalize invalid non-navigation actions and possibly add a valid-action vocabulary reminder to the prompt.

### E3.43 Local8 Eager Worker1 Submission Result

- Submitted at: 2026-07-30 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit used on SuperPOD: `231c923643be923ed0a5116a030709cc26a798f1`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- New jobs:
  - `498186`: `eager_tiny_w1`, 12h, `PENDING (Priority)` at submit.
    - W&B name: `navigation_vagen1_local8gpu_eager_tiny_w1_rmb2_test15_12h_20260730T075522Z`
    - Env config: `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_light.yaml`
  - `498187`: `eager_actionpen_w1`, 12h, `PENDING (Priority)` at submit.
    - W&B name: `navigation_vagen1_local8gpu_eager_actionpen_w1_rmb2_test15_12h_20260730T075522Z`
    - Env config: `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_action_penalty.yaml`
- W&B group:
  - `navigation_vagen1_local8gpu_eager_worker1_20260730`
- Shared settings:
  - `SERVER_NAVIGATION_MAX_WORKERS=1`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `ROLLOUT_ENFORCE_EAGER=True`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `MAX_TURNS=20`
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=15`
  - `SAVE_FREQ=5`
  - Slurm exclude: `dgx-32,dgx-35,dgx-37`
- Initial queue state:
  - Both jobs were `PENDING (Priority)`.
  - Slurm did not provide a start-time estimate at submission (`START_TIME=N/A`).

### E3.44 Guarded Worker1 Rationale

- Prepared at: 2026-07-30 HKT
- Trigger:
  - User asked to cancel `498186` and `498187`, then rerun with the guarded prompt/metrics changes and `SERVER_NAVIGATION_MAX_WORKERS=1`.
- Cancellation:
  - `498186`: cancelled while `PENDING`, no runtime consumed.
  - `498187`: cancelled while `PENDING`, no runtime consumed.
- Root cause from raw samples:
  - The model is not only overusing `moveahead`; it also produces invalid non-navigation actions such as `stay`, `stop`, and `end`, and typo actions such as `rotatelleft`.
  - These invalid actions often come with repetitive CoT that justifies staying or stopping near a visible object.
- Code changes:
  - Strengthen single-action navigation prompt:
    - repeat the exact valid action vocabulary in the per-turn prompt;
    - explicitly forbid `stay`, `stop`, `end`, `done`, `terminate`, `wait`, and `noop`;
    - tell the model that even when close to the target or seeing the target, it must still choose one valid navigation/camera action.
  - Add action distribution metrics:
    - `action/valid_vocab_rate`
    - `action/forbidden_stay_stop_end_rate`
    - `action/invalid_typo_rate`
  - Add `env_config_dense_guard.yaml`:
    - keep progress shaping small: `progress_reward=0.01`, `regress_penalty=-0.01`;
    - keep repeat penalty moderate: `repeat_action_penalty=-0.02`, cap `-0.06`;
    - strengthen invalid action penalty to `invalid_action_penalty=-0.12`.
- Planned run:
  - Variant name: `eager_guard_w1`
  - Use `SERVER_NAVIGATION_MAX_WORKERS=1` and `ROLLOUT_MINI_BATCH_SIZE=2`.
  - Keep `LOSS_MASK_MODE=default`, `FORMAT_REWARD=0.05`, `ROLLOUT_ENFORCE_EAGER=True`, `MAX_TURNS=20`.
- Expected observations:
  - Env stability should improve relative to worker2 if single AI2-THOR worker avoids long-run create/reset hangs.
  - Collapse health should improve if `forbidden_stay_stop_end_rate` and `invalid_typo_rate` decrease while success does not collapse.
  - If `valid_vocab_rate` stays low, the next step should be stronger prompt/action-vocabulary SFT or constrained decoding rather than more dense reward.

### E3.44 Guarded Worker1 Submission Result

- Submitted at: 2026-07-30 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit used on SuperPOD: `df99a798f2e4cf67639ea69491cdc00a8bb5353b`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Validation:
  - Local `python -m pytest tests/test_navigation_vagen1_minimal.py tests/test_navigation_vagen1_sweep.py -q`: `45 passed`.
  - Local `bash -n` passed for local8 dense wrapper and variant config.
  - Local `git diff --check` passed, with CRLF warnings only.
  - SuperPOD `git pull --ff-only` updated the clone to `df99a79`.
  - SuperPOD `bash -n` passed for local8 dense wrapper and variant config.
- New job:
  - `498192`: `eager_guard_w1`, 12h, `PENDING (Priority)` at submit.
    - W&B name: `navigation_vagen1_local8gpu_eager_guard_w1_rmb2_test15_12h_20260730T080534Z`
    - Env config: `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_guard.yaml`
- W&B group:
  - `navigation_vagen1_local8gpu_eager_guard_worker1_20260730`
- Shared settings:
  - `SERVER_NAVIGATION_MAX_WORKERS=1`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `ROLLOUT_ENFORCE_EAGER=True`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `MAX_TURNS=20`
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=15`
  - `SAVE_FREQ=5`
  - `RAW_SAMPLES_TO_LOG=8`
  - Slurm exclude: `dgx-26,dgx-32,dgx-35,dgx-37`
- Initial queue state:
  - Job was `PENDING (Priority)`.
  - Slurm did not provide a start-time estimate at submission (`START_TIME=N/A`).

### E3.45 Guarded Queue-Hedge Rationale

- Prepared at: 2026-07-31 HKT
- Trigger:
  - User noted that `498192` may wait too long and asked to queue several parameter variants.
- Keep existing job:
  - Keep `498192` pending as the guarded 12h worker1/turn20 reference.
- New queue hedge:
  - Submit shorter 4h guarded debug jobs so Slurm may start one earlier and we can get stability/collapse signals without waiting for only one long allocation.
- Planned variants:
  - `guard_w1_rmb1_turn20_4h`
    - Most conservative env pressure: `SERVER_NAVIGATION_MAX_WORKERS=1`, `ROLLOUT_MINI_BATCH_SIZE=1`, `MAX_TURNS=20`.
    - Purpose: test whether one env per rollout mini-batch survives beyond the worker2 step15 timeout.
  - `guard_w1_rmb2_turn10_4h`
    - Same worker1/rmb2 as `498192`, but `MAX_TURNS=10`.
    - Purpose: reduce rollout length and test anti-collapse metrics faster.
  - `guard_w2_rmb2_turn10_4h`
    - `SERVER_NAVIGATION_MAX_WORKERS=2`, `ROLLOUT_MINI_BATCH_SIZE=2`, `MAX_TURNS=10`.
    - Purpose: test whether short horizon lets us regain speed while avoiding long-turn env hangs.
- Shared guarded settings:
  - `ENV_CONFIG_PATH=scripts/examples/vagen_base/navigation_vagen1/env_config_dense_guard.yaml`
  - `VAGEN1_VARIANT=eager_guard`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `ROLLOUT_ENFORCE_EAGER=True`
  - `TOTAL_TRAINING_STEPS=30`
  - `TEST_FREQ=10`
  - `SAVE_FREQ=5`
  - `RAW_SAMPLES_TO_LOG=8`
  - Slurm exclude: `dgx-26,dgx-32,dgx-35,dgx-37`
- Expected observations:
  - If rmb1 reaches step20+ but is slow, env pressure is still the main blocker and we should add retry/recreate before scaling speed.
  - If turn10 variants run stably and show improved `valid_vocab_rate`, we can use turn10 as a fast anti-collapse sweep before returning to turn20.
  - If worker2/turn10 still hangs, local env worker count remains fragile even under shorter horizon.

### E3.45 Guarded Queue-Hedge Submission Result

- Submitted at: 2026-07-31 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit used on SuperPOD: `7e1b3a976338666da62e9294255839274269ab9c`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Existing guarded reference:
  - `498192`: `eager_guard_w1`, 12h, now `RUNNING` on `dgx-30` at check, start `2026-07-31T01:44:30`.
- New queue-hedge jobs:
  - `498592`: `guard_w1_rmb1_turn20_4h`, `PENDING (Priority)` at submit.
    - W&B name: `navigation_vagen1_guard_w1_rmb1_turn20_4h_20260730T174706Z`
  - `498593`: `guard_w1_rmb2_turn10_4h`, `PENDING (Priority)` at submit.
    - W&B name: `navigation_vagen1_guard_w1_rmb2_turn10_4h_20260730T174706Z`
  - `498594`: `guard_w2_rmb2_turn10_4h`, `PENDING (Priority)` at submit.
    - W&B name: `navigation_vagen1_guard_w2_rmb2_turn10_4h_20260730T174706Z`
- W&B group:
  - `navigation_vagen1_local8gpu_eager_guard_queue_hedge_20260731`
- Shared settings:
  - `ENV_CONFIG_PATH=scripts/examples/vagen_base/navigation_vagen1/env_config_dense_guard.yaml`
  - `VAGEN1_VARIANT=eager_guard`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `ROLLOUT_ENFORCE_EAGER=True`
  - `TOTAL_TRAINING_STEPS=30`
  - `TEST_FREQ=10`
  - `SAVE_FREQ=5`
  - `RAW_SAMPLES_TO_LOG=8`
  - Slurm exclude: `dgx-26,dgx-32,dgx-35,dgx-37`

### E3.46 Guarded Queue-Hedge Result Snapshot

- Checked at: 2026-07-31 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- W&B group: `navigation_vagen1_local8gpu_eager_guard_queue_hedge_20260731`

| Job | Variant | State | W&B | Last step | Saved ckpts | Key result |
| --- | --- | --- | --- | --- | --- | --- |
| `498192` | `eager_guard_w1_rmb2_turn20_12h` | `FAILED`, 3:14:17 | `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/i2upwl79` | logged step 13, `global_steps: 14` | `5,10` | Failed from vLLM input length: decoder prompt length `5369` > `ROLLOUT_MAX_MODEL_LEN=5000`. |
| `498592` | `guard_w1_rmb1_turn20_4h` | `FAILED`, 2:15:18 | `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/ofzoexog` | logged step 5, `global_steps: 6` | `5` | Failed from vLLM input length: decoder prompt length `5033` > `ROLLOUT_MAX_MODEL_LEN=5000`; also too slow at about 1501s/step. |
| `498593` | `guard_w1_rmb2_turn10_4h` | `COMPLETED`, 3:46:17 | `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/oa4w1zwl` | logged step 29, completed 30-step job | `5,10,15,20,25,30` | First guarded queue-hedge run to complete normally. |
| `498594` | `guard_w2_rmb2_turn10_4h` | `COMPLETED`, 3:39:42 | `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/bmryqhem` | logged step 29, completed 30-step job | `5,10,15,20,25,30` | Completed normally; best validation success reached `0.625`, but final format/action-vocab metrics were weaker than `498593`. |

`498593` final logged train metrics:

- `train/success=0.500`
- `train/score=4.826`
- `train/action/top_share=0.842`
- `train/action/entropy=0.376`
- `train/action/all_same_traj=0.500`
- `train/action/valid_vocab_rate=0.969`
- `train/action/forbidden_stay_stop_end_rate=0.000`
- `train/action/invalid_typo_rate=0.031`
- `train/format/correct=0.969`
- `train/format/too_many_actions=0.000`
- `timing_s/step=390.996`
- Last validation snapshot: `val/success=0.500`, `val/action/top_share=0.850`, `val/action/valid_vocab_rate=1.000`, `val/format/correct=1.000`.

`498594` final logged train metrics:

- `train/success=0.312`
- `train/score=2.823`
- `train/action/top_share=0.746`
- `train/action/entropy=0.596`
- `train/action/all_same_traj=0.375`
- `train/action/valid_vocab_rate=0.863`
- `train/action/forbidden_stay_stop_end_rate=0.100`
- `train/action/invalid_typo_rate=0.038`
- `train/format/correct=0.863`
- `timing_s/step=407.041`
- Best validation snapshot: `val/success=0.625` at step 20.

Interpretation:

- The `finished` run is `498593`, and it finished normally rather than crashing.
- The two `MAX_TURNS=20` guarded runs failed for a new, clearer reason: prompt length exceeded `ROLLOUT_MAX_MODEL_LEN=5000`. This is different from the earlier local AI2-THOR `/environments` timeout failure.
- `MAX_TURNS=10` keeps the prompt within the current vLLM model-length budget and is currently the only stable local8 setting in this queue hedge.
- The guarded prompt/reward reduced invalid vocabulary substantially, but collapse pressure is still visible: `top_share` remains around `0.84-0.88` and `all_same_traj` can reach `0.5`.

Decision:

- Treat `guard_w1_rmb2_turn10_4h` as the current stable baseline candidate.
- Do not continue `MAX_TURNS=20` with `ROLLOUT_MAX_MODEL_LEN=5000`; either reduce context/prompt growth further or increase model length with a careful vLLM memory check.
- `guard_w1_rmb1_turn20_4h` is not useful: it is both too slow and still exceeds model length.
- Prefer `SERVER_NAVIGATION_MAX_WORKERS=1` and `ROLLOUT_MINI_BATCH_SIZE=2` for future local8 VAGEN1 runs. This setting completed normally in `498593`, kept final valid-action behavior cleaner than worker2, and is the current stability-first default.

### E3.47 498593 Step30 Full Infer Plan

- Trigger: user asked to run inference from the final checkpoint of `498593` on the full train and test sets, then summarize accuracy.
- Workflow requirement: keep local Git, GitHub, and SuperPOD checkout synchronized; do not copy scripts directly to the server.
- Source training job: `498593`, `navigation_vagen1_guard_w1_rmb2_turn10_4h_20260730T174706Z`.
- Checkpoint: `/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_guard_w1_rmb2_turn10_4h_20260730T174706Z/global_step_30/actor`.
- Data:
  - `/project/peilab/hligb/vagen-navigation/data/navigation_vagen1_guard_w1_rmb2_turn10_4h_20260730T174706Z/test.parquet`
  - `/project/peilab/hligb/vagen-navigation/data/navigation_vagen1_guard_w1_rmb2_turn10_4h_20260730T174706Z/train.parquet`
- New tracked script: `scripts/superpod/run_navigation_498593_step30_full_infer_local_1gpu.sbatch`.
- Planned infer settings:
  - `1` GPU, `tensor_parallel_size=1`
  - `MAX_STEPS=10`
  - `SERVER_NAVIGATION_MAX_WORKERS=1`
  - `SERVER_MAX_WORKERS=1`
  - `CHUNK_SIZE=64`
  - W&B disabled
- Submission status:
  - A direct-copy draft was removed from the SuperPOD clone after realizing it violated the local-GitHub-server sync policy.
  - Submit only after this script and log entry are committed, pushed, and pulled on SuperPOD.

### E3.47 498593 Step30 Full Infer Submission Result

- Submitted at: 2026-08-01 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit used on SuperPOD: `0a1733c7c70b2431bfb8169025695f50869ef125`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Job ID: `500297`
- Job name: `vagen-nav-498593-infer`
- Initial state: `PENDING (Priority)`
- Resource request:
  - `1` GPU
  - `16` CPUs
  - `256G` memory
  - `24:00:00` walltime
- Script: `scripts/superpod/run_navigation_498593_step30_full_infer_local_1gpu.sbatch`
- Checkpoint:
  - `/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_guard_w1_rmb2_turn10_4h_20260730T174706Z/global_step_30/actor`
- Data:
  - test split: `/project/peilab/hligb/vagen-navigation/data/navigation_vagen1_guard_w1_rmb2_turn10_4h_20260730T174706Z/test.parquet`
  - train split: `/project/peilab/hligb/vagen-navigation/data/navigation_vagen1_guard_w1_rmb2_turn10_4h_20260730T174706Z/train.parquet`
- Inference settings:
  - `MAX_STEPS=10`
  - `CHUNK_SIZE=64`
  - `SERVER_NAVIGATION_MAX_WORKERS=1`
  - `SERVER_MAX_WORKERS=1`
  - `tensor_parallel_size=1`
  - W&B disabled
- Notes:
  - First submission attempt returned an empty job id because the Slurm module was not loaded in the remote shell.
  - After explicitly loading Slurm, `sbatch --parsable` succeeded.
  - The run may need to merge the FSDP actor shards into `actor/huggingface` before vLLM inference can start.

### E3.48 Reward A+C Turn20 Ctx8192 Sweep Rationale

- Prepared at: 2026-08-01 HKT
- Trigger:
  - User accepted the A+C plan and asked to submit three 6h variants in parallel.
- Stable runtime carried forward:
  - `SERVER_NAVIGATION_MAX_WORKERS=1`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `ROLLOUT_ENFORCE_EAGER=True`
- Horizon/context change:
  - Return to `MAX_TURNS=20`, because turn10 may be too short for the intended single-action baseline.
  - Increase `ROLLOUT_MAX_TRAJECTORY_LENGTH` and vLLM `max_model_len` to `8192`, because prior turn20 guarded runs failed at decoder prompt lengths `5033` and `5369` with the old limit `5000`.
  - Keep `MAX_TRAJECTORY_LENGTH=16000` for update-side trajectory storage.
- Reward design:
  - Add `dense_reward_mode=anti_collapse_progress_v2`.
  - Keep tiny progress shaping: `progress_reward=+0.01`, `regress_penalty=-0.01`.
  - Keep invalid action penalty for forbidden/typo actions.
  - Add state-conditioned repeat penalty:
    - repeated actions are still lightly discouraged;
    - repeated actions with no distance progress get an additional stagnation penalty.
  - Add soft action-balance penalty:
    - if the current trajectory's top action share exceeds the threshold after enough steps, apply a small penalty;
    - if a trajectory stays entirely one-action for several steps, apply an additional small penalty.
- Three parallel variants:
  - `ac_mild`: lower repeat/stagnation/balance penalties to protect success.
  - `ac_base`: planned default A+C setting.
  - `ac_strong`: stronger action-level pressure to test whether top-share can be reduced without killing success.
- Planned shared run settings:
  - `8GPU local`
  - `6h`
  - `TOTAL_TRAINING_STEPS=30`
  - `TEST_FREQ=10`
  - `SAVE_FREQ=5`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
  - `RAW_SAMPLES_TO_LOG=8`
- Expected observations:
  - Primary stability gate: pass step20 without prompt-length error or local AI2-THOR timeout.
  - Primary collapse gate: reduce `action/top_share` and `action/all_same_traj` relative to `498593`, while keeping `valid_vocab_rate >= 0.95`.
  - Do not accept a run as healthier if lower top-share comes mostly from invalid/forbidden/typo actions.
  - Use train/test full inference from `498593` as the baseline reference once job `500297` finishes.

### E3.48 Reward A+C Turn20 Ctx8192 Sweep Submission Result

- Submitted at: 2026-08-01 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit used on SuperPOD: `f5eff608d7463b12a2ab3c442dc57eab43e5e4de`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Submit wrapper: `scripts/superpod/submit_navigation_vagen1_ac_turn20_ctx8192_6h.sh`
- W&B group: `navigation_vagen1_ac_turn20_ctx8192_6h_20260801`
- Shared settings:
  - `8GPU local`
  - `TimeLimit=06:00:00`
  - `TOTAL_TRAINING_STEPS=30`
  - `TEST_FREQ=10`
  - `SAVE_FREQ=5`
  - `MAX_TURNS=20`
  - `ROLLOUT_MAX_TRAJECTORY_LENGTH=8192`
  - `MAX_TRAJECTORY_LENGTH=16000`
  - `SERVER_NAVIGATION_MAX_WORKERS=1`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `ROLLOUT_ENFORCE_EAGER=True`
  - `RAW_SAMPLES_TO_LOG=8`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
- Jobs:

| Variant | Job ID | W&B name | Env config |
| --- | --- | --- | --- |
| `ac_mild` | `500316` | `navigation_vagen1_ac_ac_mild_turn20_ctx8192_6h_20260731T173941Z` | `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_mild.yaml` |
| `ac_base` | `500317` | `navigation_vagen1_ac_ac_base_turn20_ctx8192_6h_20260731T173941Z` | `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_base.yaml` |
| `ac_strong` | `500318` | `navigation_vagen1_ac_ac_strong_turn20_ctx8192_6h_20260731T173941Z` | `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_strong.yaml` |

- Initial queue state:
  - All three jobs were `PENDING (Priority)` with `TimeLimit=06:00:00`.
  - The earlier `498593` full train/test inference job `500297` was still pending, with Slurm estimated start time `2026-08-01T04:45:34`.

### E3.49 Infer And A+C Sweep Failure Result

- Checked at: 2026-08-03 HKT
- Jobs checked: `500297`, `500316`, `500317`, `500318`

`500297` / `498593` step30 full train/test inference:

- State: `FAILED`, exit `1:0`
- Runtime: `00:04:26`
- Node: `dgx-37`
- Failure point:
  - The job started and attempted to merge the FSDP actor shards into HuggingFace format under:
    - `/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_guard_w1_rmb2_turn10_4h_20260730T174706Z/global_step_30/actor/huggingface`
  - It failed before running test/train inference.
- Error:
  - `safetensors._safetensors_rust.SafetensorError: Error while serializing: I/O error: No space left on device (os error 28)`
- Storage observation:
  - `/project` was effectively full: `50T used / 6.7G available`.
  - A partial HuggingFace model directory was left behind, about `5.4G`, containing only `model-00001-of-00002.safetensors` and config/tokenizer files.
- Interpretation:
  - This is a storage failure, not an inference/model-quality failure.
  - The full train/test infer has not produced results yet.
- Next decision:
  - Do not rerun until the partial HF merge directory is removed or completed in a location with enough space.
  - Prefer merging/writing the HuggingFace model to node-local scratch or `/home` if enough space is available, then copy only final summaries/results to `/project`.

`500316` / A+C mild:

- State: `FAILED`, exit `1:0`
- Runtime: `00:24:31`
- Node: `dgx-35`
- W&B: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/i6f0cw51`
- Progress: reached `global_steps: 1`, but no completed step metrics.
- Error:
  - `requests.exceptions.ReadTimeout: HTTPConnectionPool(host='localhost', port=5316): Read timed out. (read timeout=1200)`

`500317` / A+C base:

- State: `FAILED`, exit `1:0`
- Runtime: `00:23:53`
- Node: `dgx-35`
- W&B: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/o9n9ua0v`
- Progress: reached `global_steps: 1`, but no completed step metrics.
- Error:
  - `requests.exceptions.ReadTimeout: HTTPConnectionPool(host='localhost', port=5317): Read timed out. (read timeout=1200)`

`500318` / A+C strong:

- State: `FAILED`, exit `1:0`
- Runtime: `00:23:49`
- Node: `dgx-35`
- W&B: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/475a3mr2`
- Progress: reached `global_steps: 1`, but no completed step metrics.
- Error:
  - `requests.exceptions.ReadTimeout: HTTPConnectionPool(host='localhost', port=5318): Read timed out. (read timeout=1200)`

Interpretation:

- The A+C reward variants were not evaluated. All three died before producing training metrics.
- The shared failure happened on the same node (`dgx-35`) and the same endpoint type (`/environments` through localhost), matching earlier node-local AI2-THOR create/reset fragility.
- This does not disprove `anti_collapse_progress_v2`; it only says local env creation on this node/topology was not usable for the sweep.
- The current stable evidence remains `498593` and `498594` with turn10. For turn20, the next run should either avoid known-bad local env nodes more aggressively or return to external env-server topology.

Decision:

- Do not interpret lower/higher A+C reward strength from `500316-500318`; there are no rollout metrics.
- Before resubmitting A+C, fix the storage issue for inference and avoid using `/project` as the HF merge target.
- For training, prefer excluding `dgx-35` in addition to previously problematic local-env nodes, or use external env servers so training is not killed by local `/environments` timeout.

### E3.50 Infer Storage Fix And A+C Resubmit Rationale

- Prepared at: 2026-08-03 HKT
- Trigger:
  - User asked to solve the failed `498593` full inference and rerun the A+C three-way sweep while excluding `dgx-35`.
- Root cause being addressed:
  - Inference failed because `model_merger.py` writes merged HuggingFace weights into `local_dir/huggingface`; when `local_dir` was the checkpoint actor dir under `/project`, the merge consumed additional `/project` space and failed with `No space left on device`.
  - A+C jobs failed because all three landed on `dgx-35` and local AI2-THOR `/environments` creation timed out before any meaningful rollout metrics.
- Code fix:
  - Update `run_navigation_498593_step30_full_infer_local_1gpu.sbatch` to create a node-local temporary actor directory under `$VAGEN_NODE_LOCAL_ROOT`.
  - Symlink FSDP shard files from the original actor checkpoint into the temporary actor directory.
  - Copy only small tokenizer/config files into the temporary `huggingface` directory.
  - Run `model_merger.py --local_dir "$MERGE_ACTOR_DIR"` so large merged safetensors are written to node-local storage.
  - Point vLLM inference at the node-local HuggingFace directory for the duration of the job.
  - Update `submit_navigation_vagen1_ac_turn20_ctx8192_6h.sh` with `SLURM_EXCLUDE_NODES=dgx-26,dgx-32,dgx-35,dgx-37` and pass `--exclude`.
- Planned resubmission:
  - Rerun `500297` replacement for full test/train inference.
  - Rerun the same three A+C variants (`ac_mild`, `ac_base`, `ac_strong`) with identical reward/settings except for node exclusion.
- Expected observations:
  - Inference should pass the merge step without writing large model files under `/project`.
  - A+C jobs should not run on `dgx-35`; if they still timeout, treat local-env topology as the root issue rather than the A+C reward design.

### E3.50 Infer Storage Fix And A+C Resubmit Result

- Submitted at: 2026-08-03 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit used on SuperPOD: `de4bd60a2294686ad28fda7ba54a43f279ff2348`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Validation:
  - Local tests: `python -m pytest tests/test_navigation_vagen1_minimal.py tests/test_navigation_vagen1_sweep.py -q` passed with `48 passed`.
  - Local shell syntax: `bash -n` passed for the inference and A+C submit scripts.
  - Server shell syntax: `bash -n` passed for the inference and A+C submit scripts.
- Cleanup:
  - Removed the incomplete merged HF shard files from:
    - `/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_guard_w1_rmb2_turn10_4h_20260730T174706Z/global_step_30/actor/huggingface`
  - This only removed partial generated files from the failed merge, not the FSDP actor checkpoint shards.
- New inference job:
  - Job ID: `503273`
  - Job name: `vagen-nav-498593-infer`
  - Initial state: `PENDING (Priority)`
  - Resource request: `1GPU`, `24h`
  - Fix applied: merged HF weights will be written under job-local `$VAGEN_NODE_LOCAL_ROOT/actor_for_hf_merge/huggingface`, then used directly by vLLM.
- New A+C jobs:

| Variant | Job ID | W&B name | Env config |
| --- | --- | --- | --- |
| `ac_mild` | `503274` | `navigation_vagen1_ac_ac_mild_turn20_ctx8192_6h_20260802T170850Z` | `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_mild.yaml` |
| `ac_base` | `503275` | `navigation_vagen1_ac_ac_base_turn20_ctx8192_6h_20260802T170850Z` | `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_base.yaml` |
| `ac_strong` | `503276` | `navigation_vagen1_ac_ac_strong_turn20_ctx8192_6h_20260802T170850Z` | `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_strong.yaml` |

- A+C shared settings:
  - Same as E3.48, but with Slurm exclusion:
    - `--exclude=dgx-26,dgx-32,dgx-35,dgx-37`
- Initial queue state:
  - `503273`, `503274`, `503275`, and `503276` were all `PENDING (Priority)` after submission.

### E3.51 Infer And A+C Sweep Result

- Checked at: 2026-08-04 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Local/GitHub commit at check time: `e4c5a6b`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`

`503273` / `498593` step30 full train/test inference:

- State: `COMPLETED`, exit `0:0`
- Runtime: `00:23:00`
- Node: `dgx-46`
- The node-local HuggingFace merge fix worked. The job merged FSDP shards under `/tmp/hligb/vn-fullinfer/503273/actor_for_hf_merge/huggingface` and did not write large merged weights under `/project`.
- Output summaries:
  - `/project/peilab/hligb/vagen-navigation/eval/full_498593_step30_full_20260802T170919Z_test_all/summary.json`
  - `/project/peilab/hligb/vagen-navigation/eval/full_498593_step30_full_20260802T170919Z_train_all/summary.json`
- Full test split:
  - total `8`
  - success `3/8 = 0.375`
  - base subset success `0/4 = 0.000`
  - common_sense subset success `3/4 = 0.750`
  - score mean `3.581`
  - step mean `7.625`
- Full train split:
  - total `256`
  - success `65/256 = 0.25390625`
  - base subset success `29/128 = 0.2265625`
  - common_sense subset success `36/128 = 0.28125`
  - score mean `2.191`
  - step mean `8.453`
- Interpretation:
  - The W&B online val metric from `498593` was optimistic relative to full train/test inference.
  - The checkpoint is useful for debugging/inference, but not strong enough as a final rollout-generation baseline.
  - The split gap confirms that tiny val sets can be noisy; full train/test inference is required before deciding a checkpoint is good.

`503274` / A+C mild:

- W&B: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/nv519bhs`
- State: `FAILED`, exit `1:0`
- Runtime: `03:47:45`
- Node: `dgx-24`
- Progress: reached `global_steps: 15`; last completed checkpoint tracker remained at `10`.
- Checkpoints:
  - `global_step_5`: complete, about `99G`
  - `global_step_10`: complete, about `98G`
  - no complete `global_step_15`
  - run directory total about `197G`
- Last stable metrics around step14:
  - `train/success=0.500`
  - `train/score=4.358`
  - `train/format_correct=0.919`
  - `train/action/top_share=0.854`
  - `train/action/entropy=0.321`
  - `train/action/all_same_traj=0.562`
  - `train/action/valid_vocab_rate=0.919`
  - `timing_s/step=695.499`
- Raw samples still showed repeated typo actions such as `rotatetleft`.

`503275` / A+C base:

- W&B: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/endkx7vi`
- State: `FAILED`, exit `1:0`
- Runtime: `03:37:49`
- Node: `dgx-40`
- Progress: reached `global_steps: 15`; last completed checkpoint tracker remained at `10`.
- Checkpoints:
  - `global_step_5`: complete, about `98G`
  - `global_step_10`: complete, about `99G`
  - `global_step_15`: incomplete, about `75G`, missing complete critic/HF/data files
  - run directory total about `272G`
- Last stable metrics around step14:
  - `train/success=0.438`
  - `train/score=2.922`
  - `train/format_correct=0.887`
  - `train/action/top_share=0.833`
  - `train/action/entropy=0.417`
  - `train/action/all_same_traj=0.438`
  - `train/action/valid_vocab_rate=0.898`
  - `timing_s/step=770.131`
- Raw samples still showed multi-action output and typo actions such as `rotatel`.

`503276` / A+C strong:

- W&B: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/475a3mr2`
- State: `FAILED`, exit `1:0`
- Runtime: `03:37:42`
- Node: `dgx-52`
- Progress: reached `global_steps: 15`; last completed checkpoint tracker remained at `10`.
- Checkpoints:
  - `global_step_5`: complete, about `99G`
  - `global_step_10`: complete, about `98G`
  - `global_step_15`: incomplete, about `36G`, actor-only partial files
  - run directory total about `232G`
- Last stable metrics around step14:
  - `train/success=0.375`
  - `train/score=1.438`
  - `train/format_correct=0.944`
  - `train/action/top_share=0.913`
  - `train/action/entropy=0.222`
  - `train/action/all_same_traj=0.562`
  - `train/action/valid_vocab_rate=0.944`
  - `timing_s/step=758.801`
- Raw samples still showed typo actions such as `rotatelight` and `rotatelleft`.

Failure interpretation:

- The A+C reward code did get exercised; unlike the earlier `500316-500318` attempt, these jobs passed environment startup and trained for many steps.
- All three failed around the `SAVE_FREQ=5` checkpoint at step15, not during initial rollout or model initialization.
- `sacct` does not show memory OOM: batch `MaxRSS` was about `302-303GB` under a `768G` allocation.
- The common pattern is incomplete checkpoint directories at `global_step_15` plus very large complete checkpoints at steps 5 and 10. Each complete checkpoint is around `98-99G`, and three jobs were saving large checkpoints concurrently to `/project`.
- Root-cause hypothesis:
  - The immediate failure is checkpoint-save fragility / NFS write pressure from saving full actor + critic + optimizer states every 5 steps across three parallel 8GPU jobs.
  - This is separate from the rollout/env timeout problem; the env side was stable enough to reach step15.

Collapse interpretation:

- A+C improved some indicators in the better variants, but did not solve collapse cleanly:
  - `ac_base` step13/14 had healthier top-share and all-same values than the old `498593` trend, but valid vocab/format quality was worse.
  - `ac_mild` had the best step14 success (`0.500`) but still had `top_share=0.854` and `all_same_traj=0.562`.
  - `ac_strong` over-penalized or was less useful: lower success and higher collapse indicators at step14.
- Repeated/raw samples show the main remaining pathology is not only choosing `moveahead`; it is also repeated near-identical reasoning plus invalid/typo action names (`rotatel`, `rotatetleft`, `rotatelleft`, `stay`, `stop`).

Next decision:

- Do not run three parallel 8GPU jobs with `SAVE_FREQ=5` and full optimizer checkpointing unless storage/checkpoint behavior is changed.
- For the next stability run:
  - keep `SERVER_NAVIGATION_MAX_WORKERS=1`
  - keep `ROLLOUT_MINI_BATCH_SIZE=2`
  - keep `MAX_TURNS=20`
  - keep `ROLLOUT_MAX_TRAJECTORY_LENGTH=8192`
  - set `SAVE_FREQ=15` or `SAVE_FREQ=30` for debug, or save actor-only/lightweight checkpoints if the code supports it
  - run only one A+C variant at a time, likely `ac_base` or `ac_mild`
  - keep raw-sample logging and action distribution metrics
- Reward direction:
  - `ac_base` looks like the best balance for collapse metrics.
  - `ac_strong` should not be continued without evidence because it lowers success and does not reduce collapse enough.

### E3.52 A+C Base-Lite Guarded Rationale

- Prepared at: 2026-08-04 HKT
- Trigger:
  - User approved keeping the A+C direction and asked to further reduce collapse, then submit an sbatch run.
- Goal:
  - Continue from the best partial evidence in `503274-503276` without repeating the same checkpoint failure pattern.
  - Test whether a mild/base middle reward can reduce repeated actions and typo/invalid actions while preserving success.
- Design:
  - New env config: `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_base_lite_guarded.yaml`
  - New submit wrapper: `scripts/superpod/submit_navigation_vagen1_ac_base_lite_guarded_8gpu_60step.sh`
- Training settings:
  - `8GPU local`
  - `TimeLimit=12:00:00`
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=15`
  - `SAVE_FREQ=15`
  - `REMOVE_PREVIOUS_CKPT_IN_SAVE=True`
  - `MAX_TURNS=20`
  - `ROLLOUT_MAX_TRAJECTORY_LENGTH=8192`
  - `MAX_TRAJECTORY_LENGTH=16000`
  - `SERVER_NAVIGATION_MAX_WORKERS=1`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `ROLLOUT_ENFORCE_EAGER=True`
  - `RAW_SAMPLES_TO_LOG=8`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
- Reward settings:
  - `dense_reward_mode=anti_collapse_progress_v2`
  - `progress_reward=+0.01`
  - `regress_penalty=-0.01`
  - `repeat_action_penalty=-0.015`
  - `repeat_action_start=3`
  - `repeat_action_penalty_cap=-0.04`
  - `invalid_action_penalty=-0.08`
  - `stagnation_repeat_penalty=-0.015`
  - `stagnation_repeat_start=3`
  - `stagnation_repeat_penalty_cap=-0.04`
  - `action_top_share_penalty_threshold=0.85`
  - `action_top_share_penalty=-0.01`
  - `all_same_traj_penalty_threshold=0.5`
  - `all_same_traj_penalty=-0.015`
- Why not stronger:
  - `ac_strong` lowered success and did not clearly improve collapse metrics.
  - The main observed issues were typo/invalid action names and repeated reasoning/actions; strong dense penalties risk making the policy brittle before it has learned enough navigation.
- Why checkpoint frequency changed:
  - `503274-503276` all failed around the step15 checkpoint while writing very large full checkpoints.
  - Debug run now saves every 15 steps and removes previous checkpoints to reduce NFS pressure.
- Expected observations:
  - Complete step15/30/45/60 without checkpoint-save failure.
  - Keep `train/action/top_share` below the old `498593` collapse region where possible.
  - Keep `train/action/all_same_traj < 0.5`.
  - Improve or at least preserve `format_correct` and `valid_vocab_rate`.
  - Raw samples should show fewer typo actions (`rotatel`, `rotatetleft`, `rotatelleft`) and fewer repeated identical think/action loops.

### E3.52 A+C Base-Lite Guarded Submission

- Submitted at: 2026-08-04 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit used on SuperPOD: `700e61f515183ce3cd8f61c077b5ff8cc5f25870`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Validation:
  - Local tests: `python -m pytest tests/test_navigation_vagen1_minimal.py tests/test_navigation_vagen1_sweep.py -q` passed with `49 passed`.
  - Local shell syntax: `bash -n` passed for the submit/run scripts.
  - Server shell syntax: `bash -n` passed after syncing the clone to `700e61f`.
- Job:
  - Job ID: `505814`
  - Job name: `vagen-nav-vagen1-8g-dense`
  - W&B name: `navigation_vagen1_ac_base_lite_guarded_turn20_ctx8192_60step_20260804T123819Z`
  - W&B group: `navigation_vagen1_ac_base_lite_guarded_60step_20260804`
  - Env config: `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_base_lite_guarded.yaml`
  - Submit wrapper: `scripts/superpod/submit_navigation_vagen1_ac_base_lite_guarded_8gpu_60step.sh`
  - Slurm command target: `scripts/superpod/run_navigation_vagen1_8gpu_local_save5_dense_light.sbatch`
- Queue state at submission:
  - `PENDING (Priority)`
  - `TimeLimit=12:00:00`
  - `gres:gpu:8`
  - `ExcNodeList=dgx-[26,32,35,37]`
  - Estimated start from `squeue --start`: `2026-08-06T09:39:33`
- Log paths:
  - stdout: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-8gpu-local-dense-save5-505814.out`
  - stderr: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-8gpu-local-dense-save5-505814.err`
- Monitor gates:
  - Confirm W&B URL after job starts.
  - Confirm step15 checkpoint completes and `latest_checkpointed_iteration.txt` reaches `15`.
  - If it fails again during checkpointing, stop using full debug checkpoints and implement/check actor-only or no-optimizer checkpointing before retrying.
  - If step30/45/60 complete, run full train/test inference on the best saved checkpoint before deciding whether it can generate Nimloth SFT2 rollouts.

### E3.53 Additional Guarded Sweep Rationale

- Prepared at: 2026-08-04 HKT
- Trigger:
  - User asked to queue two more experiments with different parameters/scripts while `505814` is pending.
- Goal:
  - Keep `505814` as the center point and add two interpretable neighboring variants.
  - Avoid repeating the earlier failure mode from `503274-503276`: three parallel jobs with `SAVE_FREQ=5` full checkpoints.
- Shared settings for both new variants:
  - `8GPU local`
  - `TimeLimit=12:00:00`
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=15`
  - `SAVE_FREQ=15`
  - `REMOVE_PREVIOUS_CKPT_IN_SAVE=True`
  - `MAX_TURNS=20`
  - `ROLLOUT_MAX_TRAJECTORY_LENGTH=8192`
  - `MAX_TRAJECTORY_LENGTH=16000`
  - `SERVER_NAVIGATION_MAX_WORKERS=1`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `ROLLOUT_ENFORCE_EAGER=True`
  - `RAW_SAMPLES_TO_LOG=8`
  - `VAL_BEFORE_TRAIN=False`
  - `FINAL_VAL_AFTER_TRAIN=False`
- New variant 1: `ac_success_guarded`
  - Env config: `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_success_guarded.yaml`
  - Submit script: `scripts/superpod/submit_navigation_vagen1_ac_success_guarded_8gpu_60step.sh`
  - Purpose: preserve success/format quality if `505814` is still too punitive.
  - Reward changes vs `505814`:
    - `repeat_action_penalty=-0.01`
    - `repeat_action_penalty_cap=-0.03`
    - `stagnation_repeat_penalty=-0.01`
    - `stagnation_repeat_penalty_cap=-0.03`
    - `action_top_share_penalty=-0.005`
    - `all_same_traj_penalty=-0.01`
    - `invalid_action_penalty=-0.08`
- New variant 2: `ac_diversity_guarded`
  - Env config: `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_diversity_guarded.yaml`
  - Submit script: `scripts/superpod/submit_navigation_vagen1_ac_diversity_guarded_8gpu_60step.sh`
  - Purpose: test whether slightly stronger action-distribution pressure reduces collapse without the damage seen in `ac_strong`.
  - Reward changes vs `505814`:
    - `repeat_action_penalty=-0.015`
    - `repeat_action_penalty_cap=-0.04`
    - `stagnation_repeat_penalty=-0.015`
    - `stagnation_repeat_penalty_cap=-0.04`
    - `action_top_share_penalty_threshold=0.80`
    - `action_top_share_penalty=-0.015`
    - `all_same_traj_penalty=-0.02`
    - `invalid_action_penalty=-0.10`
- Decision logic:
  - If `success_guarded` has clearly better success but worse top-share, prefer it only if full infer improves.
  - If `diversity_guarded` lowers top-share/all-same while preserving success and valid-vocab, prefer it for the formal run.
  - If both fail at checkpoint step15, stop scheduling full-checkpoint debug and implement actor-only/no-optimizer checkpointing.

### E3.53 Additional Guarded Sweep Submission

- Submitted at: 2026-08-04 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Commit used on SuperPOD: `14c5a0713f29e65fea08bf35782041cd575284ec`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Validation:
  - Local tests: `python -m pytest tests/test_navigation_vagen1_minimal.py tests/test_navigation_vagen1_sweep.py -q` passed with `50 passed`.
  - Local shell syntax: `bash -n` passed for both new submit scripts and shared run scripts.
  - Server shell syntax: `bash -n` passed after syncing the clone to `14c5a07`.
- Jobs:

| Variant | Job ID | W&B name | Env config |
| --- | --- | --- | --- |
| `ac_success_guarded` | `505830` | `navigation_vagen1_ac_success_guarded_turn20_ctx8192_60step_20260804T125642Z` | `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_success_guarded.yaml` |
| `ac_diversity_guarded` | `505831` | `navigation_vagen1_ac_diversity_guarded_turn20_ctx8192_60step_20260804T125643Z` | `scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_diversity_guarded.yaml` |

- Queue state at submission check:
  - `505814` / center `ac_base_lite_guarded`: `PENDING (Resources)`, estimated start `2026-08-05T01:00:53`, scheduled node `dgx-39`.
  - `505830` / `ac_success_guarded`: `PENDING (Priority)`, estimated start `2026-08-05T11:09:32`, scheduled node `dgx-40`.
  - `505831` / `ac_diversity_guarded`: `PENDING (Priority)`, estimated start `2026-08-05T13:00:00`, scheduled node `dgx-39`.
- Log paths:
  - `505830` stdout: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-8gpu-local-dense-save5-505830.out`
  - `505830` stderr: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-8gpu-local-dense-save5-505830.err`
  - `505831` stdout: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-8gpu-local-dense-save5-505831.out`
  - `505831` stderr: `/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-8gpu-local-dense-save5-505831.err`
- Monitoring:
  - Compare `505814`, `505830`, and `505831` as one three-point guarded sweep.
  - Primary gate remains checkpoint stability at step15.
  - Secondary gate: success, top_share, all_same_traj, valid_vocab_rate, typo/invalid action raw samples.

### E3.54 Guarded Sweep Failure Investigation

- Checked at: 2026-08-05 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Local/GitHub commit at check time: `ae71468b42fcf4b94bffad3caec3568ff0bf98c4`
- SuperPOD clone path: `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`
- Jobs checked: `505814`, `505830`, `505831`

Result summary:

| Variant | Job ID | W&B URL | Final state | Last reliable step | Checkpoint status | Main observation |
| --- | --- | --- | --- | --- | --- | --- |
| `ac_base_lite_guarded` | `505814` | `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/ymd1ezfx` | `FAILED`, exit `1:0` | step15 metrics, entered step16/eval after checkpoint | `global_step_15` complete; `latest=15`; dir about `98G` | Best reward/success balance, but process silently ended during post-step15 rollout/eval window. |
| `ac_success_guarded` | `505830` | `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/phlcywlu` | `FAILED`, exit `1:0` | step14 metrics | `global_step_15` incomplete; latest missing/empty | Strong collapse remained; failure likely during step15 save. |
| `ac_diversity_guarded` | `505831` | `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/vagen_navigation_repro/runs/2imzfp9a` | `FAILED`, exit `1:0` | step14 metrics | checkpoint dir empty/minimal; latest missing/empty | Diversity pressure lowered top-share but hurt success/format; failure likely before/during step15 save. |

Failure evidence:

- Slurm stdout/stderr did not contain a Python traceback, `RayTaskError`, `ReadTimeout`, HTTP timeout, NCCL fatal message, or explicit CUDA OOM.
- Local AI2-THOR server logs did not show a server-side traceback; tails mostly showed normal `State reward wrapper closed` and `Initialize return`.
- Stderr only showed the command trace plus Slurm/NVML cleanup messages such as `NVML: Failed to get Compute running procs(7): Insufficient Size`; this appears at job termination and is not enough to identify the Python-side root cause.
- `ray_trainer.py` saves full checkpoints in this order: actor, critic, dataloader, then `latest_checkpointed_iteration.txt`. Therefore an incomplete `global_step_15` with no latest tracker means the process died inside the checkpoint path before atomic completion.
- `505814` is different from the other two: it completed the step15 checkpoint and logged step15 metrics, then stdout ended during repeated vLLM sleep/wake generation lines. That points to a second silent Ray/vLLM worker exit or rollout/eval-side termination after checkpointing.
- Ray node-local session logs were not found under the project log directory, so the exact worker-level death reason was lost after Slurm cleaned the node-local runtime.

Metric/collapse observations:

- `505814` step15: `train/success=0.312`, `train/score=2.164`, `format_correct=0.956`, `top_share=0.876`, `entropy=0.313`, `all_same_traj=0.375`, `invalid_typo_rate=0.044`, `timing_s/step=1481.665`.
- `505830` step14: `train/success=0.312`, `train/score=2.256`, `format_correct=0.887`, `top_share=0.962`, `entropy=0.114`, `all_same_traj=0.750`, `timing_s/step=837.069`.
- `505831` step14: `train/success=0.188`, `train/score=0.451`, `format_correct=0.766`, `top_share=0.745`, `entropy=0.350`, `all_same_traj=0.250`, `timing_s/step=773.172`.
- Raw samples still show collapse-like repeated reasoning and invalid stop-style outputs, for example repeated `<answer>stop</answer>` when the target is visible. This means A+C helped distribution in some variants, but has not solved reasoning/action collapse.

Root-cause conclusion:

- The current crash is not primarily env startup and not the old `SERVER_NAVIGATION_MAX_WORKERS` timeout.
- There are likely two coupled failure modes:
  - Full checkpoint save is too heavy and fragile on `/project`, especially actor+critic+optimizer FSDP checkpoints at 8GPU scale. This directly explains `505830` and `505831`.
  - After checkpointing, Ray/vLLM can still silently terminate during rollout/eval; this explains `505814`, but exact worker logs were not preserved.

Next decision:

- Do not submit more full-checkpoint 8GPU sweeps as-is.
- Before retrying, add an EXIT trap or cleanup hook to copy Ray/vLLM node-local logs to `/project/peilab/hligb/vagen-navigation/logs/ray-${SLURM_JOB_ID}`.
- Reduce checkpoint pressure before the next long run:
  - prefer actor-only or no-optimizer checkpointing if supported or implement it;
  - otherwise use `SAVE_FREQ=30` for debug and avoid parallel full-checkpoint runs;
  - keep `REMOVE_PREVIOUS_CKPT_IN_SAVE=True`.
- For reward direction, keep `ac_base_lite_guarded` as the best current center; do not continue `ac_success_guarded` because collapse is too high, and do not continue `ac_diversity_guarded` unless its success/format can be repaired.

### E3.55 Actor-Only Checkpoint Retry Rationale

- Prepared at: 2026-08-05 HKT
- Branch: `hligb/vagen1-vagenfirst-sweep-20260722`
- Starting commit: `bb47021a40ee02c6391ca76e08cf6aadd24ca10c`
- Goal:
  - Retry the best current reward center (`ac_base_lite_guarded`, from `505814`) without repeating the full-checkpoint write failure.
  - Preserve a checkpoint that is sufficient for rollout generation and full train/test inference.
- Code changes:
  - Add `trainer.save_critic_checkpoint`; when false, `_save_checkpoint()` saves the actor path, dataloader, and latest tracker but skips critic checkpointing.
  - Add `SAVE_CRITIC_CKPT` and `SAVE_OPTIMIZER_CKPT` environment switches in the navigation run script.
  - Export `VERL_SAVE_OPTIMIZER_CKPT` so the actual SuperPOD `verl` checkpoint manager can skip `optim_world_size_*` shards.
  - Add `scripts/superpod/patch_verl_lightweight_checkpoint.sh`, an idempotent SuperPOD-side patch for `/project/peilab/hligb/vagen-navigation/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py`.
  - Copy node-local Ray/vLLM diagnostics from `$RAY_TMPDIR` to `/project/peilab/hligb/vagen-navigation/logs/node-local-${EXPERIMENT_NAME}-${SLURM_JOB_ID}` on Slurm job exit.
  - Strengthen the single-action prompt: target visible is not a stop signal, and stop-like words remain invalid answers.
- Run settings:
  - Variant: `ac_base_lite_guarded_actoronly`
  - `8GPU local`
  - `TimeLimit=12:00:00`
  - `TOTAL_TRAINING_STEPS=60`
  - `TEST_FREQ=15`
  - `SAVE_FREQ=15`
  - `SAVE_CRITIC_CKPT=False`
  - `SAVE_OPTIMIZER_CKPT=False`
  - `REMOVE_PREVIOUS_CKPT_IN_SAVE=True`
  - `MAX_TURNS=20`
  - `ROLLOUT_MAX_TRAJECTORY_LENGTH=8192`
  - `MAX_TRAJECTORY_LENGTH=16000`
  - `SERVER_NAVIGATION_MAX_WORKERS=1`
  - `ROLLOUT_MINI_BATCH_SIZE=2`
  - `LOSS_MASK_MODE=default`
  - `FORMAT_REWARD=0.05`
  - `ROLLOUT_ENFORCE_EAGER=True`
  - Reward config remains `env_config_dense_ac_base_lite_guarded.yaml`.
- Expected observations:
  - Step15 checkpoint should be much smaller than the previous full `98G` checkpoint.
  - Checkpoint should contain actor model shards and small HF config/tokenizer files, but no critic directory and no `optim_world_size_*` files.
  - Inference should remain possible because the existing inference merger uses actor model shards and extra state, not optimizer or critic.
  - If eval still fails after step15, the copied Ray/vLLM logs should provide the missing worker-level error.
- Decision:
  - Submit only this one actor-only center run first.
  - Do not run the success/diversity variants until actor-only checkpointing is verified.
