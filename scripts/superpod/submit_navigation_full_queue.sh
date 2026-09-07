#!/usr/bin/env bash
set -euo pipefail
export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
source "$VAGEN_REPO/scripts/superpod/prepare_project_storage.sh"
source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
module load slurm >/dev/null 2>&1 || true
cd "$VAGEN_REPO"

for script in \
  scripts/superpod/run_navigation_full_8gpu_paper.sbatch \
  scripts/superpod/run_navigation_full_4gpu_paper.sbatch; do
  bash -n "$script"
done

full8=$(sbatch --parsable scripts/superpod/run_navigation_full_8gpu_paper.sbatch)
full4=$(sbatch --parsable scripts/superpod/run_navigation_full_4gpu_paper.sbatch)
status_dir="$VAGEN_REPO/runs/navigation_full_status_$(date -u +%Y%m%d)"
mkdir -p "$status_dir"
cat > "$status_dir/full_jobs.env" <<EOF_JOBS
FULL8_JOB=$full8
FULL4_JOB=$full4
SUBMITTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF_JOBS
{
  echo ""
  echo "## Full-Only Paper Reproduction Queue $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""
  echo "| run | job id | GPUs | W&B name | checkpoint policy |"
  echo "| --- | --- | --- | --- | --- |"
  echo "| Full 8GPU | $full8 | 8 | navigation_full_8gpu_paper_repro_20260702_ckpt50 | save_freq=50, keep checkpoints |"
  echo "| Full 4GPU | $full4 | 4 | navigation_full_4gpu_paper_repro_20260702_ckpt50 | save_freq=50, keep checkpoints |"
} >> "$status_dir/summary.md"
echo "Submitted Full-only jobs: full8=$full8 full4=$full4"
squeue -u "${USER:-hligb}" || true