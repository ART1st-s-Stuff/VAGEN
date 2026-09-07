#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${VAGEN_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SYNC_CMD="VAGEN_REPO=$REPO_DIR bash $REPO_DIR/scripts/sync/autosync.sh"
CRON_LINE="*/5 * * * * $SYNC_CMD"
TMP_CRON=$(mktemp)

crontab -l 2>/dev/null | grep -vF "$REPO_DIR/scripts/sync/autosync.sh" > "$TMP_CRON" || true
printf '%s\n' "$CRON_LINE" >> "$TMP_CRON"
crontab "$TMP_CRON"
rm -f "$TMP_CRON"

echo "Installed 5-minute autosync cron for $REPO_DIR"
