#!/usr/bin/env bash
set -euo pipefail

VERL_ROOT=${VERL_ROOT:-/project/peilab/hligb/vagen-navigation/verl}
TARGET="$VERL_ROOT/verl/utils/checkpoint/fsdp_checkpoint_manager.py"

if [ ! -f "$TARGET" ]; then
  echo "VERL checkpoint manager not found: $TARGET" >&2
  exit 1
fi

python - "$TARGET" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
marker = "VERL_SAVE_OPTIMIZER_CKPT"
if marker in text:
    print(f"lightweight checkpoint patch already present: {path}")
    raise SystemExit(0)

old = "                torch.save(optimizer_state_dict, optim_path)  # TODO: address optimizer is None\n"
new = """                save_optimizer_ckpt = os.environ.get(\"VERL_SAVE_OPTIMIZER_CKPT\", \"True\").lower() not in {\"0\", \"false\", \"no\", \"off\"}
                if save_optimizer_ckpt:
                    torch.save(optimizer_state_dict, optim_path)  # TODO: address optimizer is None
                else:
                    print(f'[rank-{self.rank}]: Skipping optimizer checkpoint because VERL_SAVE_OPTIMIZER_CKPT=False')
"""
if old not in text:
    raise SystemExit(f"expected optimizer save line not found in {path}")

path.write_text(text.replace(old, new))
print(f"patched lightweight optimizer checkpoint switch into {path}")
PY
