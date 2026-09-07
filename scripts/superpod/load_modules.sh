#!/usr/bin/env bash

if [ -f /etc/profile.d/modules.sh ]; then
  # Some cluster module scripts reference unset variables.
  set +u
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh || true
  set -u
fi

if command -v module >/dev/null 2>&1; then
  module load slurm/slurm/23.02.6 || module load slurm || true
  module load Anaconda3/2023.09-0 || module load Anaconda3 || true
  module load "nvhpc-hpcx-cuda12/23.11" || true
  module load gcc/13.1.0 || true
fi

export CC=${VAGEN_CC:-gcc}
export CXX=${VAGEN_CXX:-g++}
