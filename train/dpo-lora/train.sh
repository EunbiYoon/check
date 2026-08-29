#!/usr/bin/env bash
# Thin launcher for the LoRA/DPO training orchestrator.
#
# This script only prepares the shell environment that must exist *before*
# Python starts — conda, CUDA modules, library paths — then hands off to
# `train/dpo-lora/orchestrate.py`, which does all the scheduling
# (variant -> GPU assignment, resume, parallel workers, AUX+ALL merge).
# (train/dpo-lora is a hyphenated dir, so it is run by path, not `python -m`.)
#
# All configuration is via env vars / .env or flags forwarded to the
# orchestrator, e.g.:
#   TRAIN_VARIANTS=core,aux,all,rw TRAIN_NUM_GPUS=3 ./train/dpo-lora/train.sh
#   ./train/dpo-lora/train.sh --variants filter_on,filter_off --no-merge
set -euo pipefail

# repo root = first ancestor with config.py
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$ROOT" != "/" && ! -f "$ROOT/config.py" ]]; do ROOT="$(dirname "$ROOT")"; done
[[ -f "$ROOT/config.py" ]] || { echo "ERROR: cannot locate repo root (no config.py)" >&2; exit 1; }
cd "$ROOT"

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/pkgs}"
export HF_HOME="${HF_HOME:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-126}"

module load conda/latest cuda/12.6
if [[ -n "${CUDA_HOME:-}" ]]; then
  NVHPC_ROOT="$(cd "${CUDA_HOME}/../.." && pwd)"
  export LD_LIBRARY_PATH="${NVHPC_ROOT}/math_libs/lib64:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sal

exec python "$ROOT/train/dpo-lora/orchestrate.py" "$@"
