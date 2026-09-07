#!/usr/bin/env bash
set -euo pipefail
export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
source "$VAGEN_REPO/scripts/superpod/prepare_project_storage.sh"
source "$VAGEN_REPO/scripts/superpod/load_modules.sh" >/dev/null 2>&1 || true
cd "$VAGEN_REPO"

scripts=(
  scripts/superpod/preflight_navigation_full_together_save25.sh
  scripts/superpod/preflight_navigation_full_4gpu_together_save25.sh
  scripts/superpod/run_navigation_full_preflight_8gpu_together_save25.sbatch
  scripts/superpod/run_navigation_full_preflight_4gpu_together_save25.sbatch
  scripts/superpod/run_navigation_full_8gpu_together_save25.sbatch
  scripts/superpod/run_navigation_full_4gpu_together_save25.sbatch
  scripts/superpod/monitor_navigation_full_together_save25.sh
)
for script in "${scripts[@]}"; do
  bash -n "$script"
done

preflight8=$(sbatch --parsable scripts/superpod/run_navigation_full_preflight_8gpu_together_save25.sbatch)
preflight4=$(sbatch --parsable scripts/superpod/run_navigation_full_preflight_4gpu_together_save25.sbatch)
full8=$(sbatch --parsable --dependency=afterok:$preflight8 scripts/superpod/run_navigation_full_8gpu_together_save25.sbatch)
full4=$(sbatch --parsable --dependency=afterok:$preflight4 scripts/superpod/run_navigation_full_4gpu_together_save25.sbatch)

status_dir="$VAGEN_REPO/runs/navigation_full_together_save25_$(date -u +%Y%m%d)"
mkdir -p "$status_dir"
cat > "$status_dir/jobs.env" <<EOF_JOBS
PREFLIGHT8_JOB=$preflight8
PREFLIGHT4_JOB=$preflight4
FULL8_JOB=$full8
FULL4_JOB=$full4
SUBMITTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
FULL8_WANDB=navigation_full_8gpu_paper_repro_20260703_together_gptoss20b_save25
FULL4_WANDB=navigation_full_4gpu_paper_repro_20260703_together_gptoss20b_save25
EOF_JOBS
cat >> "$status_dir/summary.md" <<EOF_SUMMARY

## Full Together Save25 Queue $(date -u +%Y-%m-%dT%H:%M:%SZ)

| run | job id | GPUs | dependency | W&B name | ckpt/test policy |
| --- | --- | --- | --- | --- | --- |
| preflight 8GPU | $preflight8 | 8 | none | zz_delete_preflight_* | 1 train step, save_freq=-1, cleanup W&B |
| preflight 4GPU | $preflight4 | 4 | none | zz_delete_preflight_* | 1 train step, save_freq=-1, cleanup W&B |
| Full 8GPU | $full8 | 8 | afterok:$preflight8 | navigation_full_8gpu_paper_repro_20260703_together_gptoss20b_save25 | save_freq=25, test_freq=25, keep latest |
| Full 4GPU | $full4 | 4 | afterok:$preflight4 | navigation_full_4gpu_paper_repro_20260703_together_gptoss20b_save25 | save_freq=25, test_freq=25, keep latest |
EOF_SUMMARY

monitor_log="$VAGEN_LOG_ROOT/monitor-navigation-full-together-save25-${preflight8}-${preflight4}.log"
nohup bash scripts/superpod/monitor_navigation_full_together_save25.sh "$preflight8" "$full8" "$preflight4" "$full4" > "$monitor_log" 2>&1 &
echo "Submitted Full Together save25 queue: preflight8=$preflight8 preflight4=$preflight4 full8=$full8 full4=$full4"
echo "Monitor log: $monitor_log"
squeue -u "${USER:-hligb}" || true
