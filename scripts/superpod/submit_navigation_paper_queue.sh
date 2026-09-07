#!/usr/bin/env bash
set -euo pipefail
export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
source "$VAGEN_REPO/scripts/superpod/prepare_project_storage.sh"
source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
module load slurm >/dev/null 2>&1 || true
cd "$VAGEN_REPO"
for script in \
  scripts/superpod/run_navigation_base_8gpu_paper.sbatch \
  scripts/superpod/run_navigation_full_8gpu_paper.sbatch \
  scripts/superpod/run_navigation_base_4gpu_paper.sbatch \
  scripts/superpod/run_navigation_full_4gpu_paper.sbatch; do
  bash -n "$script"
done
base8=$(sbatch --parsable scripts/superpod/run_navigation_base_8gpu_paper.sbatch)
full8=$(sbatch --parsable scripts/superpod/run_navigation_full_8gpu_paper.sbatch)
base4=$(sbatch --parsable scripts/superpod/run_navigation_base_4gpu_paper.sbatch)
full4=$(sbatch --parsable scripts/superpod/run_navigation_full_4gpu_paper.sbatch)
status_dir="$VAGEN_REPO/runs/navigation_status_$(date -u +%Y%m%d)"
mkdir -p "$status_dir"
cat > "$status_dir/paper_jobs.env" <<EOF_JOBS
BASE8_JOB=$base8
FULL8_JOB=$full8
BASE4_JOB=$base4
FULL4_JOB=$full4
SUBMITTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF_JOBS
{
  echo ""
  echo "## Paper Reproduction Queue $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""
  echo "| run | job id | GPUs | W&B name |"
  echo "| --- | --- | --- | --- |"
  echo "| Base 8GPU | $base8 | 8 | navigation_base_8gpu_paper_repro_20260621 |"
  echo "| Full 8GPU | $full8 | 8 | navigation_full_8gpu_paper_repro_20260621 |"
  echo "| Base 4GPU fallback | $base4 | 4 | navigation_base_4gpu_paper_repro_20260621 |"
  echo "| Full 4GPU fallback | $full4 | 4 | navigation_full_4gpu_paper_repro_20260621 |"
} >> "$status_dir/summary.md"
monitor_log="$VAGEN_LOG_ROOT/navigation-paper-monitor-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash scripts/superpod/monitor_navigation_jobs.sh "$base8" "$full8" "$base4" "$full4" > "$monitor_log" 2>&1 &
echo "Submitted jobs: base8=$base8 full8=$full8 base4=$base4 full4=$full4"
echo "Monitor log: $monitor_log"
squeue -u "${USER:-hligb}" || true
