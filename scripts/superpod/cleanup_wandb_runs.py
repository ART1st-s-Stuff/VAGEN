#!/usr/bin/env python3
import argparse
import sys

parser = argparse.ArgumentParser(description='Delete selected W&B runs by prefix/state.')
parser.add_argument('--project', required=True)
parser.add_argument('--prefix', action='append', default=[])
parser.add_argument('--id', action='append', default=[])
parser.add_argument('--states', nargs='*', default=['failed', 'crashed', 'killed'])
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()
try:
    import wandb
except Exception as exc:
    print(f'cleanup_wandb_runs: cannot import wandb: {exc}', file=sys.stderr)
    raise SystemExit(1)
api = wandb.Api()
states = set(args.states)
matched = 0
for run in api.runs(args.project):
    by_id = run.id in args.id
    by_prefix = any(run.name.startswith(prefix) for prefix in args.prefix)
    by_state = not states or run.state in states
    if (by_id or by_prefix) and by_state:
        matched += 1
        print(f'cleanup_wandb_runs: delete {run.id} {run.name} state={run.state}')
        if not args.dry_run:
            run.delete()
print(f'cleanup_wandb_runs: matched {matched} run(s)')
