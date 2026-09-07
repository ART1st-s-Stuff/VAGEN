# SuperPOD VAGEN Navigation Workflow

The repository checkout and generated artifacts are intentionally separated.

```bash
export VAGEN_REPO=/home/hligb/test_lu/VAGEN-navigation-repro
export VAGEN_ARTIFACT_ROOT=/project/peilab/hligb/vagen-navigation
export VAGEN_DATA_ROOT=/project/peilab/hligb/vagen-navigation/data
export VAGEN_VERL_ROOT=/project/peilab/hligb/vagen-navigation/verl
export VAGEN_VERL_REF=vagen-legacy
export VAGEN_CONDA_ENV=vagen_nav
cd "$VAGEN_REPO"
```

## One-time setup

```bash
bash scripts/superpod/setup_navigation_env.sh
```

The setup script follows the legacy VAGEN README and pins the external VERL
checkout to `vagen-legacy`. If an earlier setup used the wrong dependency set,
rebuild the conda environment once:

```bash
VAGEN_RECREATE_ENV=1 bash scripts/superpod/setup_navigation_env.sh
```

If `sudo` is unavailable on the login node, ask for an interactive GPU node or
load the site module that provides Vulkan. The script prints the exact packages
needed.

## Server and jobs

Jobs default to `--account=peilab` and `--partition=normal`, which is the
SuperPOD GPU partition available to this user. Override the sbatch headers if
the project allocation changes.

```bash
sbatch scripts/superpod/run_navigation_server.sbatch
sbatch scripts/superpod/run_navigation_smoke.sbatch
sbatch scripts/superpod/run_navigation_base.sbatch
sbatch scripts/superpod/run_navigation_full_paper.sbatch
```

Useful checks:

```bash
squeue -u "$USER"
tail -f /project/peilab/hligb/vagen-navigation/logs/navigation-server-*.out
tail -f /project/peilab/hligb/vagen-navigation/logs/navigation-base-*.out
```

## Autosync

Install a 5-minute cron sync on SuperPOD:

```bash
bash scripts/sync/install_superpod_autosync.sh
```

The sync script commits only small source/config/summary files. It does not
commit checkpoints, parquet files, rollout images, or W&B directories.

## OpenAI key for VAGEN Full

Navigation Full uses LLM-as-Judge process rewards. On SuperPOD, put the key outside git:

```bash
mkdir -p /project/peilab/hligb/vagen-navigation/secrets
cat > /project/peilab/hligb/vagen-navigation/secrets/openai.env <<'EOF'
export OPENAI_API_KEY=sk-...
EOF
chmod 600 /project/peilab/hligb/vagen-navigation/secrets/openai.env
```

If `SERVER_USE_STATE_REWARD=True` and no key is present, `start_local_server.sh` exits before training starts.
