#!/usr/bin/env bash
set -euo pipefail
[ "$#" -eq 4 ] || { echo "usage: $0 BASE8 FULL8 BASE4 FULL4" >&2; exit 2; }
BASE8_JOB="$1"; FULL8_JOB="$2"; BASE4_JOB="$3"; FULL4_JOB="$4"
VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
source "$VAGEN_REPO/scripts/superpod/prepare_project_storage.sh"
source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
module load slurm >/dev/null 2>&1 || true
STATUS_DIR="$VAGEN_REPO/runs/navigation_status_$(date -u +%Y%m%d)"
SUMMARY="$STATUS_DIR/summary.md"
mkdir -p "$STATUS_DIR"
state_of() { squeue -h -j "$1" -o "%T" 2>/dev/null | head -n1 | tr -d ' ' || true; }
cancel_if_active() {
  local winner="$1" fallback="$2" label="$3" winner_state fallback_state
  winner_state=$(state_of "$winner"); fallback_state=$(state_of "$fallback")
  if [ "$winner_state" = "RUNNING" ] && { [ "$fallback_state" = "PENDING" ] || [ "$fallback_state" = "RUNNING" ]; }; then
    echo "monitor: $label 8GPU job $winner RUNNING; cancel fallback $fallback ($fallback_state)"
    scancel "$fallback" || true
    echo "- $(date -u +%Y-%m-%dT%H:%M:%SZ): $label 8GPU job \`$winner\` is RUNNING; canceled 4GPU fallback \`$fallback\` ($fallback_state)." >> "$SUMMARY"
  fi
}
cleanup_preflight() {
  (cd /tmp && python "$VAGEN_REPO/scripts/superpod/cleanup_wandb_runs.py" --project "${WANDB_PROJECT_PATH:-lukieluu6-city-university-of-hong-kong/vagen_navigation_repro}" --prefix zz_delete_preflight_ --states finished failed crashed killed running preempted 2>/dev/null || true)
}
while true; do
  cancel_if_active "$BASE8_JOB" "$BASE4_JOB" Base
  cancel_if_active "$FULL8_JOB" "$FULL4_JOB" Full
  cleanup_preflight
  active=0
  for job in "$BASE8_JOB" "$FULL8_JOB" "$BASE4_JOB" "$FULL4_JOB"; do [ -n "$(state_of "$job")" ] && active=$((active+1)); done
  squeue -j "$BASE8_JOB,$FULL8_JOB,$BASE4_JOB,$FULL4_JOB" || true
  [ "$active" -eq 0 ] && { echo "- $(date -u +%Y-%m-%dT%H:%M:%SZ): monitor exited; tracked jobs left squeue." >> "$SUMMARY"; break; }
  sleep 120
done
