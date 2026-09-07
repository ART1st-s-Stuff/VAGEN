#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${VAGEN_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LOG_DIR="$REPO_DIR/logs"
LOG_FILE="$LOG_DIR/autosync.log"
CONFLICT_LOG="$LOG_DIR/sync-conflicts.log"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" >> "$LOG_FILE"
}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  log "not a git repository: $REPO_DIR"
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  log "origin remote is not configured"
  exit 0
fi

if ! git pull --rebase --autostash origin main >> "$LOG_FILE" 2>&1; then
  printf '[%s] pull/rebase failed; manual resolution required\n' "$(timestamp)" >> "$CONFLICT_LOG"
  exit 0
fi

git add -- README.md README_REPRO.md .gitignore 2>/dev/null || true
git add -- ':(glob)scripts/**/*.sh' ':(glob)scripts/**/*.sbatch' ':(glob)scripts/**/*.ps1' 2>/dev/null || true
git add -- ':(glob)scripts/**/*.yaml' ':(glob)scripts/**/*.yml' 2>/dev/null || true
git add -- ':(glob)runs/**/*.md' ':(glob)runs/**/*.json' 2>/dev/null || true

if git diff --cached --quiet; then
  log "no allowlisted changes"
  exit 0
fi

git commit -m "autosync: $(timestamp)" >> "$LOG_FILE" 2>&1
if ! git push origin main >> "$LOG_FILE" 2>&1; then
  printf '[%s] push failed; manual resolution required\n' "$(timestamp)" >> "$CONFLICT_LOG"
  exit 0
fi

log "sync complete"
