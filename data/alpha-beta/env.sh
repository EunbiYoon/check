#!/usr/bin/env bash
# Sourced by algorithm1.sh: CUDA/Conda + .env, with command-line env winning
# over .env. Not part of Algorithm 1 -- just the runtime it needs.

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/pkgs}"
export HF_HOME="${HF_HOME:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-126}"
export PYTHONUNBUFFERED=1

module load conda/latest cuda/12.6
if [[ -n "${CUDA_HOME:-}" ]]; then
  NVHPC_ROOT="$(cd "${CUDA_HOME}/../.." && pwd)"
  export LD_LIBRARY_PATH="${NVHPC_ROOT}/math_libs/lib64:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sal

# Command-line environment values override defaults loaded from .env.
declare -A CALLER_ENV=()
for key in A1_VARIANTS A1_INPUT_DIR A1_OUTPUT_DIR A1_MODE A1_NODES \
  A1_DEDUP A1_DEDUP_CORE A1_DEDUP_AUX A1_DEDUP_ALL \
  A1_TEACHER_MODEL A1_TEACHER_MAX_NEW_TOKENS A1_TEACHER_4BIT A1_MAX_INVALID_FRACTION \
  USE_4BIT \
  A1_MAX_PAIRS A1_MAX_PAIRS_CORE A1_MAX_PAIRS_AUX A1_MAX_PAIRS_ALL A1_MAX_PAIRS_RW \
  A1_AUX_SPECIAL_PAIRS A1_ALL_SPECIAL_PAIRS A1_RW_FACTOR A1_KEEP_SHARDS \
  A1_REQUIRE_FRONTIER_MODEL TEACHER_MODEL CUDA_VISIBLE_DEVICES; do
  [[ -v "$key" ]] && CALLER_ENV["$key"]="${!key}"
done
if [[ -f .env ]]; then set -a; source .env; set +a; fi
for key in "${!CALLER_ENV[@]}"; do printf -v "$key" '%s' "${CALLER_ENV[$key]}"; export "$key"; done
