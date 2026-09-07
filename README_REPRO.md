# VAGEN Navigation Reproduction

This repository is a Navigation-only reproduction workspace for
`VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents`.

## Scope

- Upstream code: `mll-lab-nu/VAGEN`, branch `vagen-legacy`.
- Target task: Navigation only.
- Primary model: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Primary result targets from the paper:
  - VAGEN-Base Navigation average: about `0.79`.
  - VAGEN-Full Navigation average: about `0.81`, if state-reward judging runs cleanly.

Large artifacts are not stored in git. Checkpoints, rollout dumps, generated
parquet files, model/cache directories, and images should live under:

```bash
export VAGEN_ARTIFACT_ROOT=/project/peilab/hligb/vagen-navigation
```

## Paths

Local Windows checkout:

```text
D:\cityu\学校事务\Working\world model\VAGEN-navigation-repro
```

SuperPOD checkout:

```bash
export VAGEN_REPO=/home/hligb/test_lu/VAGEN-navigation-repro
export VAGEN_ARTIFACT_ROOT=/project/peilab/hligb/vagen-navigation
export VAGEN_DATA_ROOT=/project/peilab/hligb/vagen-navigation/data
export VAGEN_VERL_ROOT=/project/peilab/hligb/vagen-navigation/verl
export VAGEN_VERL_REF=vagen-legacy
```

## Setup on SuperPOD

```bash
cd "$VAGEN_REPO"
bash scripts/superpod/setup_navigation_env.sh
```

This creates or updates the `vagen_nav` conda environment by default, checks out
the `vagen-legacy` branch of the VERL fork, installs this repository editable, and installs
Navigation runtime dependencies. Use `VAGEN_CONDA_ENV=<name>` to override the
environment name.

If the environment was previously created with the wrong dependency set, rebuild
it once:

```bash
VAGEN_RECREATE_ENV=1 bash scripts/superpod/setup_navigation_env.sh
```

## Jobs

Start the environment server:

```bash
sbatch scripts/superpod/run_navigation_server.sbatch
```

Run a one-step smoke test:

```bash
sbatch scripts/superpod/run_navigation_smoke.sbatch
```

Run the released legacy Navigation VAGEN-Base job:

```bash
sbatch scripts/superpod/run_navigation_base.sbatch
```

Run the paper-aligned Navigation VAGEN-Full attempt:

```bash
sbatch scripts/superpod/run_navigation_full_paper.sbatch
```

The full job uses `bi_level_gae`, state reward, and multi-turn reward. If
legacy Navigation state reward is unstable, report VAGEN-Base as the main
reproduction and keep the full job logs as a blocker record.

## Result Log

Create a small markdown summary for each completed run:

```text
runs/navigation_YYYYMMDD/summary.md
```

Minimum contents:

- git commit hash
- SuperPOD node/GPU summary
- command or sbatch job id
- checkpoint/artifact path
- validation success rates for `navigation_base` and `navigation_common_sense`
- average and comparison to paper target

## Sync Policy

The autosync scripts only stage small files:

- source code and config
- `README*.md`
- `scripts/**/*.sh`, `scripts/**/*.ps1`, `scripts/**/*.sbatch`
- lightweight `runs/**/*.md` and `runs/**/*.json`

They intentionally exclude checkpoints, parquet data, rollout images, W&B
directories, and generated model weights.
