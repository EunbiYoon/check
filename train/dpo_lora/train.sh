#!/usr/bin/env bash
# LoRA/DPO training orchestrator: trains one adapter per TRAIN_VARIANTS entry,
# one worker per GPU, then (optionally) merges AUX+ALL (paper §2.5 Eq. 4).
#
# Input per variant is either a prebuilt preference-pair JSONL (DATA_DIR) or,
# when TRAIN_TRAJECTORY_DIR is set, a blind-rollout trajectory JSONL that
# `python -m train.dpo_lora --trajectories` converts to pairs on the fly
# (Algorithm 1 lines 3-9: solver pinning + Eq.2/3 filter + no-leak paraphrase).
set -euo pipefail

# Walk up from this script until the repo root (the directory with config.py).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$ROOT" != "/" && ! -f "$ROOT/config.py" ]]; do
  ROOT="$(dirname "$ROOT")"
done
if [[ ! -f "$ROOT/config.py" ]]; then
  echo "ERROR: cannot locate repo root (no config.py above ${BASH_SOURCE[0]})" >&2
  exit 1
fi
cd "$ROOT"

# Values explicitly supplied before the script override .env.
CALLER_RUN_ID_SET="${RUN_ID+x}"
CALLER_RUN_ID="${RUN_ID-}"
CALLER_TRAIN_VARIANTS_SET="${TRAIN_VARIANTS+x}"
CALLER_TRAIN_VARIANTS="${TRAIN_VARIANTS-}"
CALLER_TRAIN_NUM_GPUS_SET="${TRAIN_NUM_GPUS+x}"
CALLER_TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS-}"
CALLER_TRAIN_AUTO_MERGE_SET="${TRAIN_AUTO_MERGE+x}"
CALLER_TRAIN_AUTO_MERGE="${TRAIN_AUTO_MERGE-}"
CALLER_TRAIN_VISIBLE_DEVICES_SET="${TRAIN_VISIBLE_DEVICES+x}"
CALLER_TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES-}"
CALLER_TRAIN_TRAJECTORY_DIR_SET="${TRAIN_TRAJECTORY_DIR+x}"
CALLER_TRAIN_TRAJECTORY_DIR="${TRAIN_TRAJECTORY_DIR-}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -n "$CALLER_RUN_ID_SET" && -n "$CALLER_RUN_ID" ]]; then
  export RUN_ID="$CALLER_RUN_ID"
else
  # RUN_ID in .env is intentionally ignored: an unspecified run starts a new session.
  export RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
fi
[[ -n "$CALLER_TRAIN_VARIANTS_SET" ]] && export TRAIN_VARIANTS="$CALLER_TRAIN_VARIANTS"
[[ -n "$CALLER_TRAIN_NUM_GPUS_SET" ]] && export TRAIN_NUM_GPUS="$CALLER_TRAIN_NUM_GPUS"
[[ -n "$CALLER_TRAIN_AUTO_MERGE_SET" ]] && export TRAIN_AUTO_MERGE="$CALLER_TRAIN_AUTO_MERGE"
[[ -n "$CALLER_TRAIN_VISIBLE_DEVICES_SET" ]] && export TRAIN_VISIBLE_DEVICES="$CALLER_TRAIN_VISIBLE_DEVICES"
[[ -n "$CALLER_TRAIN_TRAJECTORY_DIR_SET" ]] && export TRAIN_TRAJECTORY_DIR="$CALLER_TRAIN_TRAJECTORY_DIR"

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/pkgs}"
export HF_HOME="${HF_HOME:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-126}"

module load conda/latest cuda/12.6
if [[ -n "${CUDA_HOME:-}" ]]; then
  NVHPC_ROOT="$(cd "${CUDA_HOME}/../.." && pwd)"
  CUDA_PATHS="${NVHPC_ROOT}/math_libs/lib64:${CUDA_HOME}/lib64"
  export LD_LIBRARY_PATH="${CUDA_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sal

nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda is unavailable'"
python -c "import bitsandbytes.cextension as e; assert e.lib is not None and getattr(e.lib, 'compiled_with_cuda', False), 'bitsandbytes CUDA is unavailable'"

DATA_DIR="${DATA_DIR:-data/paper}"
TRAIN_TRAJECTORY_DIR="${TRAIN_TRAJECTORY_DIR:-}"
TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS:-3}"
TRAIN_VARIANTS="${TRAIN_VARIANTS:-filter_on,filter_off,core,aux,all,rw}"
TRAIN_AUTO_MERGE="${TRAIN_AUTO_MERGE:-true}"
TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES:-}"
TRAIN_COUNTERFACTUAL_MODE="${TRAIN_COUNTERFACTUAL_MODE:-horizon-aware}"
TRAIN_TEACHER_MODEL="${TRAIN_TEACHER_MODEL:-${TEACHER_MODEL:-Qwen/Qwen2.5-7B-Instruct}}"
TRAIN_TEACHER_MAX_NEW_TOKENS="${TRAIN_TEACHER_MAX_NEW_TOKENS:-1024}"
# Session run outputs: history/ after the rename, runs/ before it.
if [[ -d history || ! -d runs ]]; then RUNS_ROOT="history"; else RUNS_ROOT="runs"; fi
SESSION_DIR="${RUNS_ROOT}/${RUN_ID}"
LOG_DIR="${SESSION_DIR}/train_logs"
mkdir -p "$LOG_DIR"

if ! [[ "$TRAIN_NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || ((TRAIN_NUM_GPUS > 6)); then
  echo "ERROR: TRAIN_NUM_GPUS must be an integer from 1 to 6" >&2
  exit 1
fi

VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if ((VISIBLE_GPUS < TRAIN_NUM_GPUS)); then
  echo "ERROR: TRAIN_NUM_GPUS=${TRAIN_NUM_GPUS}, but only ${VISIBLE_GPUS} GPU(s) are visible" >&2
  exit 1
fi

if [[ -n "$TRAIN_VISIBLE_DEVICES" ]] && ((TRAIN_NUM_GPUS != 1)); then
  echo "ERROR: TRAIN_VISIBLE_DEVICES model sharding requires TRAIN_NUM_GPUS=1" >&2
  exit 1
fi

train_variant() {
  local gpu="$1"
  local variant="$2"
  local pairs="$3"
  local log_file="${LOG_DIR}/${variant}.log"
  local visible_devices="${TRAIN_VISIBLE_DEVICES:-$gpu}"
  local run_info="${SESSION_DIR}/lora/${variant}/run_info.json"
  local -a resume_args=()
  local -a input_args=(--pairs "$pairs")

  if [[ -n "$TRAIN_TRAJECTORY_DIR" ]]; then
    local trajectories="${TRAIN_TRAJECTORY_DIR}/$(basename "$pairs")"
    if [[ ! -f "$trajectories" ]]; then
      echo "ERROR: missing trajectory input ${trajectories}" >&2
      return 1
    fi
    input_args=(
      --trajectories "$trajectories"
      --teacher-model "$TRAIN_TEACHER_MODEL"
      --teacher-max-new-tokens "$TRAIN_TEACHER_MAX_NEW_TOKENS"
      --counterfactual-mode "$TRAIN_COUNTERFACTUAL_MODE"
    )
  fi

  if [[ -f "$run_info" ]] && python - "$run_info" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "completed" else 1)
PY
  then
    echo "[$(date '+%H:%M:%S')] GPU ${gpu} -> ${variant}: already completed; skipping"
    return 0
  fi

  if compgen -G "${SESSION_DIR}/lora/${variant}/adapter/checkpoint-*" >/dev/null; then
    resume_args+=(--resume)
    echo "[$(date '+%H:%M:%S')] GPU ${gpu} -> ${variant}: resuming latest checkpoint"
  fi

  echo "[$(date '+%H:%M:%S')] GPU ${gpu} -> ${variant}: ${pairs}"
  CUDA_VISIBLE_DEVICES="$visible_devices" python -m train.dpo_lora \
    --tensorboard \
    "${input_args[@]}" \
    --out "$variant" \
    "${resume_args[@]}" \
    2>&1 | sed -u "s/^/[GPU ${visible_devices}][${variant}] /" | tee "$log_file"
}

declare -A PAIRS_BY_VARIANT=(
  [filter_on]="${DATA_DIR}/filter_on.jsonl"
  [filter_off]="${DATA_DIR}/filter_off.jsonl"
  [core]="${DATA_DIR}/a_beta_core.jsonl"
  [aux]="${DATA_DIR}/a_beta_aux.jsonl"
  [all]="${DATA_DIR}/a_beta_all.jsonl"
  [rw]="${DATA_DIR}/a_beta_rw.jsonl"
)

IFS=',' read -r -a REQUESTED_VARIANTS <<< "$TRAIN_VARIANTS"
VARIANTS=()
PAIR_FILES=()
for raw_variant in "${REQUESTED_VARIANTS[@]}"; do
  variant="${raw_variant//[[:space:]]/}"
  if [[ -z "$variant" || -z "${PAIRS_BY_VARIANT[$variant]+x}" ]]; then
    echo "ERROR: invalid TRAIN_VARIANTS entry: ${raw_variant}" >&2
    exit 1
  fi
  VARIANTS+=("$variant")
  PAIR_FILES+=("${PAIRS_BY_VARIANT[$variant]}")
done

if [[ -z "$TRAIN_TRAJECTORY_DIR" ]]; then
  MISSING_PAIR_FILES=()
  for pairs in "${PAIR_FILES[@]}"; do
    if [[ ! -f "$pairs" ]]; then
      MISSING_PAIR_FILES+=("$pairs")
    fi
  done
  if ((${#MISSING_PAIR_FILES[@]})); then
    printf 'ERROR: missing pair input %s\n' "${MISSING_PAIR_FILES[@]}" >&2
    exit 1
  fi
fi

worker() {
  local gpu="$1"
  local index
  for ((index = gpu; index < ${#VARIANTS[@]}; index += WORKER_COUNT)); do
    train_variant "$gpu" "${VARIANTS[index]}" "${PAIR_FILES[index]}"
  done
}

WORKER_COUNT="$TRAIN_NUM_GPUS"
if ((WORKER_COUNT > ${#VARIANTS[@]})); then
  WORKER_COUNT="${#VARIANTS[@]}"
fi

echo "Session: ${RUN_ID}; variants: ${VARIANTS[*]}; parallel workers: ${WORKER_COUNT}"
if [[ -n "$TRAIN_TRAJECTORY_DIR" ]]; then
  echo "Trajectory source: ${TRAIN_TRAJECTORY_DIR} (pairs built on the fly)"
else
  echo "Pair source: ${DATA_DIR}"
fi
PIDS=()
for ((gpu = 0; gpu < WORKER_COUNT; gpu++)); do
  worker "$gpu" &
  PIDS+=("$!")
done

cleanup() {
  local pid
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAILED=1
  fi
done
trap - INT TERM

if ((FAILED)); then
  echo "ERROR: one or more training workers failed; merge skipped" >&2
  exit 1
fi

MERGE_SCRIPT="$(find "$ROOT" -path '*/specialist-merge/merge_adapters.py' -print -quit 2>/dev/null)"
if [[ "$TRAIN_AUTO_MERGE" == "true" ]] \
  && [[ -n "$MERGE_SCRIPT" ]] \
  && [[ -f "${SESSION_DIR}/lora/aux/adapter_config.json" ]] \
  && [[ -f "${SESSION_DIR}/lora/all/adapter_config.json" ]]; then
  python "$MERGE_SCRIPT" --checkpoint-dir "${SESSION_DIR}/lora"
else
  echo "Merge skipped (TRAIN_AUTO_MERGE=${TRAIN_AUTO_MERGE}; requires completed aux and all)"
fi

echo "Training complete: ${SESSION_DIR}/lora"
echo "Worker logs: ${LOG_DIR}"
