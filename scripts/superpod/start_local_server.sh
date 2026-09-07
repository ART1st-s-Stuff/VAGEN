#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this script from an sbatch job so its cleanup trap stays active" >&2
  exit 1
fi

SERVER_PORT=${SERVER_PORT:-5000}
SERVER_USE_STATE_REWARD=${SERVER_USE_STATE_REWARD:-True}
VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-8}
SERVER_RENDER_PROBE_AI2THOR=${SERVER_RENDER_PROBE_AI2THOR:-1}
SERVER_RENDER_PROBE_TIMEOUT=${SERVER_RENDER_PROBE_TIMEOUT:-150}
SERVER_LOG=${SERVER_LOG:-$VAGEN_ARTIFACT_ROOT/logs/navigation-local-server-${SLURM_JOB_ID:-manual}.log}

mkdir -p "$VAGEN_ARTIFACT_ROOT/logs"

OPENAI_ENV_FILE=${OPENAI_ENV_FILE:-$VAGEN_ARTIFACT_ROOT/secrets/openai.env}
TOGETHER_ENV_FILE=${TOGETHER_ENV_FILE:-$VAGEN_ARTIFACT_ROOT/secrets/together.env}
for _judge_env_file in "$OPENAI_ENV_FILE" "$TOGETHER_ENV_FILE"; do
  if [ -f "$_judge_env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$_judge_env_file"
    set +a
  fi
done

_state_reward_lc=$(printf '%s' "$SERVER_USE_STATE_REWARD" | tr '[:upper:]' '[:lower:]')
_judge_provider_lc=$(printf '%s' "${VAGEN_JUDGE_PROVIDER:-openai}" | tr '[:upper:]' '[:lower:]')
if [ "$_state_reward_lc" = "true" ]; then
  if [ "$_judge_provider_lc" = "together" ]; then
    if [ -z "${TOGETHER_API_KEY:-}" ]; then
      cat >&2 <<EOF
ERROR: SERVER_USE_STATE_REWARD=True with VAGEN_JUDGE_PROVIDER=together requires TOGETHER_API_KEY.
Create:
  $TOGETHER_ENV_FILE
with a line like:
  export TOGETHER_API_KEY=tgp_...
Refusing to start a pseudo-Full run with zero grounding/worldmodeling reward.
EOF
      exit 2
    fi
  elif [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${OPENAI_ADMIN_KEY:-}" ]; then
    cat >&2 <<EOF
ERROR: SERVER_USE_STATE_REWARD=True requires OpenAI credentials for VAGEN Full LLM-as-Judge rewards.
Set OPENAI_API_KEY in the environment, or create:
  $OPENAI_ENV_FILE
with a line like:
  export OPENAI_API_KEY=sk-...
Refusing to start a pseudo-Full run with zero grounding/worldmodeling reward.
EOF
    exit 2
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/configure_vulkan_env.sh"

REAL_HOME="${HOME:-}"
SERVER_HOME="${VAGEN_AI2THOR_HOME:-$REAL_HOME}"
if [ -n "$SERVER_HOME" ]; then
  mkdir -p "$SERVER_HOME"
fi
if [ -n "$REAL_HOME" ] && [ -n "$SERVER_HOME" ] && [ "$SERVER_HOME" != "$REAL_HOME" ] && [ -f "$REAL_HOME/.netrc" ]; then
  ln -sf "$REAL_HOME/.netrc" "$SERVER_HOME/.netrc"
fi

AI2THOR_SHARED_HOME=${VAGEN_AI2THOR_SHARED_HOME:-$VAGEN_ARTIFACT_ROOT/ai2thor-home}
AI2THOR_SHARED_RELEASES="$AI2THOR_SHARED_HOME/.ai2thor/releases"
if [ -n "$SERVER_HOME" ]; then
  mkdir -p "$SERVER_HOME/.ai2thor" "$AI2THOR_SHARED_RELEASES"
  if [ "$SERVER_HOME" != "$AI2THOR_SHARED_HOME" ] && [ ! -e "$SERVER_HOME/.ai2thor/releases" ] && [ ! -L "$SERVER_HOME/.ai2thor/releases" ]; then
    ln -s "$AI2THOR_SHARED_RELEASES" "$SERVER_HOME/.ai2thor/releases"
  fi
  rm -f "$SERVER_HOME/.ai2thor/cuda-vulkan-mapping.json" \
    "$SERVER_HOME/.ai2thor/cuda-vulkan-mapping.json.lock"
fi

prewarm_ai2thor_cache() {
  if [ "${SERVER_PREWARM_AI2THOR:-1}" != "1" ]; then
    return 0
  fi
  if [ -z "$AI2THOR_SHARED_HOME" ]; then
    return 0
  fi
  mkdir -p "$AI2THOR_SHARED_HOME/.ai2thor" "$AI2THOR_SHARED_RELEASES"
  echo "prewarming AI2-THOR cache at $AI2THOR_SHARED_HOME"
  if command -v flock >/dev/null 2>&1; then
    (
      flock -w 900 92 || { echo "could not acquire AI2-THOR cache lock" >&2; exit 1; }
      HOME="$AI2THOR_SHARED_HOME" python - <<'PYAI2THOR'
import os
from pathlib import Path
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

home = Path.home()
print(f"AI2-THOR prewarm HOME={home}")
print(f"AI2-THOR releases={home / '.ai2thor' / 'releases'}")
controller = None
try:
    controller = Controller(
        agentMode="default",
        gridSize=0.1,
        visibilityDistance=10,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=224,
        height=224,
        fieldOfView=90,
        platform=CloudRendering,
        gpu_device=int(os.environ.get("SERVER_PREWARM_GPU_DEVICE", "0")),
        server_timeout=300,
        server_start_timeout=300,
    )
    print("AI2-THOR prewarm ok")
finally:
    if controller is not None:
        controller.stop()
PYAI2THOR
    ) 92>"$VAGEN_ARTIFACT_ROOT/.ai2thor-download.lock"
  else
    HOME="$AI2THOR_SHARED_HOME" python - <<'PYAI2THOR'
import os
from pathlib import Path
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

home = Path.home()
print(f"AI2-THOR prewarm HOME={home}")
print(f"AI2-THOR releases={home / '.ai2thor' / 'releases'}")
controller = None
try:
    controller = Controller(
        agentMode="default",
        gridSize=0.1,
        visibilityDistance=10,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=224,
        height=224,
        fieldOfView=90,
        platform=CloudRendering,
        gpu_device=int(os.environ.get("SERVER_PREWARM_GPU_DEVICE", "0")),
        server_timeout=300,
        server_start_timeout=300,
    )
    print("AI2-THOR prewarm ok")
finally:
    if controller is not None:
        controller.stop()
PYAI2THOR
  fi
}
prewarm_ai2thor_cache >> "$SERVER_LOG" 2>&1

if [ -z "${SERVER_NAVIGATION_DEVICES:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a _visible_devices <<< "$CUDA_VISIBLE_DEVICES"
    _device_count=${#_visible_devices[@]}
  else
    _device_count=${N_GPUS_PER_NODE:-4}
  fi
  if [ "$_device_count" -lt 1 ]; then
    _device_count=1
  fi
  SERVER_NAVIGATION_DEVICES="0"
  for ((_idx = 1; _idx < _device_count; _idx++)); do
    SERVER_NAVIGATION_DEVICES="${SERVER_NAVIGATION_DEVICES},${_idx}"
  done
fi
SERVER_NAVIGATION_DEVICES=${SERVER_NAVIGATION_DEVICES// /}

probe_ai2thor_render_devices() {
  if [ "${SERVER_RENDER_PROBE_AI2THOR:-1}" != "1" ]; then
    echo "AI2-THOR direct render probe disabled; devices=[${SERVER_NAVIGATION_DEVICES}]"
    return 0
  fi

  local probe_root="${VAGEN_NODE_LOCAL_ROOT:-${TMPDIR:-/tmp}/vagen-navigation-${SLURM_JOB_ID:-manual}}/render-probe"
  local good_devices=""
  local failed_devices=""
  local device
  IFS=',' read -r -a _probe_devices <<< "$SERVER_NAVIGATION_DEVICES"

  for device in "${_probe_devices[@]}"; do
    device=${device// /}
    if [ -z "$device" ]; then
      continue
    fi

    local probe_home="$probe_root/gpu_${device}"
    mkdir -p "$probe_home/.ai2thor" "$AI2THOR_SHARED_RELEASES"
    if [ ! -e "$probe_home/.ai2thor/releases" ] && [ ! -L "$probe_home/.ai2thor/releases" ]; then
      ln -s "$AI2THOR_SHARED_RELEASES" "$probe_home/.ai2thor/releases"
    fi
    if [ -n "$REAL_HOME" ] && [ -f "$REAL_HOME/.netrc" ]; then
      ln -sf "$REAL_HOME/.netrc" "$probe_home/.netrc"
    fi
    rm -f "$probe_home/.ai2thor/cuda-vulkan-mapping.json" \
      "$probe_home/.ai2thor/cuda-vulkan-mapping.json.lock"

    echo "probing AI2-THOR direct render on gpu_device=${device}; HOME=${probe_home}"
    if HOME="$probe_home" AI2THOR_HOME_ROOT="$probe_home" timeout --signal=TERM "$SERVER_RENDER_PROBE_TIMEOUT" \
      python -m vagen.utils.navigation_direct_render_probe --gpu-device "$device"; then
      echo "render_probe_ok_gpu=${device}"
      if [ -z "$good_devices" ]; then
        good_devices="$device"
      else
        good_devices="${good_devices},${device}"
      fi
    else
      echo "render_probe_failed_gpu=${device}" >&2
      if [ -z "$failed_devices" ]; then
        failed_devices="$device"
      else
        failed_devices="${failed_devices},${device}"
      fi
    fi
  done

  if [ -z "$good_devices" ]; then
    echo "ERROR: no AI2-THOR render-capable GPU remained after direct render probe; failed=[${failed_devices}]" >&2
    return 4
  fi

  if [ "$good_devices" != "$SERVER_NAVIGATION_DEVICES" ]; then
    echo "AI2-THOR direct render probe removed failed devices; before=[${SERVER_NAVIGATION_DEVICES}] after=[${good_devices}]"
  else
    echo "AI2-THOR direct render probe passed all devices=[${SERVER_NAVIGATION_DEVICES}]"
  fi
  SERVER_NAVIGATION_DEVICES="$good_devices"
}
probe_ai2thor_render_devices >> "$SERVER_LOG" 2>&1
rm -f "$SERVER_HOME/.ai2thor/cuda-vulkan-mapping.json" \
  "$SERVER_HOME/.ai2thor/cuda-vulkan-mapping.json.lock"

echo "starting navigation server on port ${SERVER_PORT}; state_reward=${SERVER_USE_STATE_REWARD}; max_workers=${SERVER_NAVIGATION_MAX_WORKERS}; devices=[${SERVER_NAVIGATION_DEVICES}]"
HOME="$SERVER_HOME" python -m vagen.server.server \
  server.port="$SERVER_PORT" \
  use_state_reward="$SERVER_USE_STATE_REWARD" \
  navigation.max_workers="$SERVER_NAVIGATION_MAX_WORKERS" \
  "navigation.devices=[${SERVER_NAVIGATION_DEVICES}]" \
  >> "$SERVER_LOG" 2>&1 &

VAGEN_SERVER_PID=$!
export VAGEN_SERVER_PID

cleanup_vagen_server() {
  if kill -0 "$VAGEN_SERVER_PID" 2>/dev/null; then
    kill "$VAGEN_SERVER_PID" 2>/dev/null || true
    wait "$VAGEN_SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup_vagen_server EXIT

python - "$SERVER_PORT" "$VAGEN_SERVER_PID" "$SERVER_LOG" <<'PY'
import os
import socket
import sys
import time

port = int(sys.argv[1])
pid = int(sys.argv[2])
log_path = sys.argv[3]
deadline = time.time() + 180

while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        print(f"navigation server exited before port {port} became available", file=sys.stderr)
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
                print(handle.read()[-4000:], file=sys.stderr)
        sys.exit(1)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            print(f"navigation server is available on 127.0.0.1:{port}")
            sys.exit(0)

    time.sleep(3)

print(f"timed out waiting for navigation server on 127.0.0.1:{port}", file=sys.stderr)
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        print(handle.read()[-4000:], file=sys.stderr)
sys.exit(1)
PY
