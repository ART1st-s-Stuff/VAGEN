#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -eq 3 ]; then
  PREFLIGHT8_JOB="$1"
  FULL8_JOB="$2"
  PREFLIGHT4_JOB=""
  FULL4_JOB="$3"
elif [ "$#" -eq 4 ]; then
  PREFLIGHT8_JOB="$1"
  FULL8_JOB="$2"
  PREFLIGHT4_JOB="$3"
  FULL4_JOB="$4"
else
  echo "usage: $0 PREFLIGHT8 FULL8 [PREFLIGHT4] FULL4" >&2
  exit 2
fi
VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
VAGEN_LOG_ROOT=${VAGEN_LOG_ROOT:-$VAGEN_ARTIFACT_ROOT/logs}
mkdir -p "$VAGEN_LOG_ROOT"
source "$VAGEN_REPO/scripts/superpod/load_modules.sh" >/dev/null 2>&1 || true
STATUS_DIR="$VAGEN_REPO/runs/navigation_full_together_save25_$(date -u +%Y%m%d)"
SUMMARY="$STATUS_DIR/summary.md"
mkdir -p "$STATUS_DIR"
state_of() { [ -n "$1" ] && squeue -h -j "$1" -o %.18T 2>/dev/null | head -n1 | tr -d ' ' || true; }
line_for_jobs() {
  local jobs="$PREFLIGHT8_JOB,$FULL8_JOB,$FULL4_JOB"
  [ -n "$PREFLIGHT4_JOB" ] && jobs="$jobs,$PREFLIGHT4_JOB"
  squeue -j "$jobs" 2>/dev/null || true
}
cancel_if_active() {
  local job="$1" label="$2" state
  [ -z "$job" ] && return 0
  state=$(state_of "$job")
  if [ "$state" = "PENDING" ] || [ "$state" = "RUNNING" ]; then
    echo "canceling $label $job ($state)"
    scancel "$job" || true
    echo "- $(date -u +%Y-%m-%dT%H:%M:%SZ): canceled $label \`$job\` ($state) because Full 8GPU is RUNNING." >> "$SUMMARY"
  fi
}
cleanup_preflight() {
  (cd /tmp && python "$VAGEN_REPO/scripts/superpod/cleanup_wandb_runs.py" \
    --project "${WANDB_PROJECT_PATH:-lukieluu6-city-university-of-hong-kong/vagen_navigation_repro}" \
    --prefix zz_delete_preflight_ \
    --states finished failed crashed killed running preempted 2>/dev/null || true)
}
report_wandb() {
  /home/hligb/.conda/envs/vagen_nav/bin/python - <<'PY' 2>/dev/null || true
import wandb
PROJECT='lukieluu6-city-university-of-hong-kong/vagen_navigation_repro'
NAMES=[
 'navigation_full_8gpu_paper_repro_20260703_together_gptoss20b_save25',
 'navigation_full_4gpu_paper_repro_20260703_together_gptoss20b_save25',
]
api=wandb.Api(timeout=30)
runs=[]
for name in NAMES:
    runs.extend(list(api.runs(PROJECT, filters={'display_name': name})))
for r in runs:
    s=r.summary
    vals=[]
    for k,v in s.items():
        lk=k.lower()
        if any(x in lk for x in ['val/success','train/success','actor/ppo_kl','critic/kl','actor/grad_norm','critic/vf_loss','grounding_reward','worldmodeling_reward']):
            if isinstance(v,(int,float)):
                vals.append((k,v))
    print('WANDB', r.name, r.state, r.url)
    for k,v in sorted(vals)[:18]:
        print(' ', k, v)
PY
}

while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  pre8_state=$(state_of "$PREFLIGHT8_JOB")
  pre4_state=$(state_of "$PREFLIGHT4_JOB")
  full8_state=$(state_of "$FULL8_JOB")
  full4_state=$(state_of "$FULL4_JOB")
  echo "[$ts] pre8=$PREFLIGHT8_JOB:$pre8_state pre4=${PREFLIGHT4_JOB:-none}:$pre4_state full8=$FULL8_JOB:$full8_state full4=$FULL4_JOB:$full4_state"
  line_for_jobs
  cleanup_preflight
  if [ "$full8_state" = "RUNNING" ]; then
    cancel_if_active "$FULL4_JOB" "Full 4GPU fallback"
    cancel_if_active "$PREFLIGHT4_JOB" "Full 4GPU preflight fallback"
  fi
  if [ $(( $(date +%s) / 600 )) -ne ${LAST_WANDB_SLOT:-0} ]; then
    LAST_WANDB_SLOT=$(( $(date +%s) / 600 ))
    echo "[$ts] W&B snapshot:"
    report_wandb | tee -a "$STATUS_DIR/wandb_snapshots.log" || true
  fi
  active=0
  for job in "$PREFLIGHT8_JOB" "$PREFLIGHT4_JOB" "$FULL8_JOB" "$FULL4_JOB"; do
    [ -n "$job" ] && [ -n "$(state_of "$job")" ] && active=$((active+1))
  done
  if [ "$active" -eq 0 ]; then
    echo "- $ts: monitor exited; tracked jobs left squeue." >> "$SUMMARY"
    break
  fi
  sleep 120
done
