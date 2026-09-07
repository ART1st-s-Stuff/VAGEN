#!/usr/bin/env bash
set -euo pipefail

RUN_KIND=${1:-navigation}
COMMAND_TEXT=${VAGEN_COMMAND_TEXT:-${2:-unknown}}
REPO_DIR=${VAGEN_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
LOG_PATH=${VAGEN_RUN_LOG:-}
if [ -z "$LOG_PATH" ] && [ -n "${EXPERIMENT_NAME:-}" ]; then
  LOG_PATH="$REPO_DIR/$EXPERIMENT_NAME.log"
fi
RUN_DATE=$(date -u +"%Y%m%d")
RUN_DIR="$REPO_DIR/runs/${RUN_KIND}_${RUN_DATE}"
SUMMARY="$RUN_DIR/summary.md"

mkdir -p "$RUN_DIR"
cd "$REPO_DIR"

COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
BRANCH=$(git branch --show-current 2>/dev/null || echo unknown)
HOST=${VAGEN_RUN_HOST:-$(hostname 2>/dev/null || echo unknown)}
JOB_ID=${VAGEN_RUN_JOB_ID:-${SLURM_JOB_ID:-manual}}
GPU_INFO=${VAGEN_GPU_INFO:-}
if [ -z "$GPU_INFO" ]; then
  GPU_INFO=$(
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
      | awk '/MiB/ {print "- " $0}' || true
  )
fi
METRICS_TABLE=$(python - "$LOG_PATH" <<'PY' || true
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else None
pending = """| split | success_rate | score |
| --- | --- | --- |
| navigation_base | pending | pending |
| navigation_common_sense | pending | pending |
| average | pending | pending |"""

if not log_path or not log_path.exists():
    print(pending)
    raise SystemExit

text = log_path.read_text(encoding="utf-8", errors="replace")
text = re.sub(r"\x1b\[[0-9;]*m", "", text)
values = {}
pattern = re.compile(
    r"val/(success|score)/NavigationEnvConfig\(eval_set=([^,\)]+)[^\)]*\):([0-9.+\-eE]+)"
)
for metric, split, value in pattern.findall(text):
    values[(split, metric)] = float(value)

rows = []
successes = []
scores = []
for split in ("base", "common_sense"):
    success = values.get((split, "success"))
    score = values.get((split, "score"))
    if success is not None:
        successes.append(success)
    if score is not None:
        scores.append(score)
    rows.append(
        f"| navigation_{split} | "
        f"{success:.4f}" if success is not None else f"| navigation_{split} | pending"
    )
    rows[-1] += f" | {score:.4f} |" if score is not None else " | pending |"

avg_success = sum(successes) / len(successes) if successes else None
avg_score = sum(scores) / len(scores) if scores else None
rows.append(
    "| average | "
    + (f"{avg_success:.4f}" if avg_success is not None else "pending")
    + " | "
    + (f"{avg_score:.4f}" if avg_score is not None else "pending")
    + " |"
)
print("| split | success_rate | score |")
print("| --- | --- | --- |")
print("\n".join(rows))
PY
)

cat > "$SUMMARY" <<EOF_SUMMARY
# ${RUN_KIND} Run Summary

- Date UTC: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
- Git branch: ${BRANCH}
- Git commit: ${COMMIT}
- Host: ${HOST}
- SLURM job id: ${JOB_ID}
- Command: \`${COMMAND_TEXT}\`
- Artifact root: \`${ARTIFACT_ROOT}\`

## GPU

${GPU_INFO:-GPU info unavailable}

## Metrics

Source log: \`${LOG_PATH:-unavailable}\`

${METRICS_TABLE}

## Notes

- Paper target for Navigation VAGEN-Base average: about 0.79.
- Paper target for Navigation VAGEN-Full average: about 0.81.
EOF_SUMMARY

echo "Wrote $SUMMARY"
