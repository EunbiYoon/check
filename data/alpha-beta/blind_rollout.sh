#!/usr/bin/env bash
# Algorithm 1 line 2: generate variant-specific blind trajectories in parallel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/pkgs}"
export HF_HOME="${HF_HOME:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
# Blind data generation uses full BF16; bitsandbytes is reserved for training.
unset BNB_CUDA_VERSION
export SAL_DISABLE_BNB_CONFIG=1

module load conda/latest cuda/12.6
if [[ -n "${CUDA_HOME:-}" ]]; then
  NVHPC_ROOT="$(cd "${CUDA_HOME}/../.." && pwd)"
  CUDA_PATHS="${NVHPC_ROOT}/math_libs/lib64:${CUDA_HOME}/lib64"
  export LD_LIBRARY_PATH="${CUDA_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sal

# Values supplied with the command (FOO=... ./train/blind_rollout) must win
# over defaults stored in .env.
ENV_KEYS=(
  BLIND_MODEL BLIND_GPU BLIND_VARIANTS BLIND_EPISODES_PER_COMBINATION
  BLIND_SEED BLIND_MAX_NEW_TOKENS BLIND_OUTPUT_DIR TEACHER_MODEL
  BLIND_DO_SAMPLE BLIND_TEMPERATURE
  CUDA_VISIBLE_DEVICES cuda_visible
)
declare -A CALLER_ENV=()
for key in "${ENV_KEYS[@]}"; do
  if [[ -v "$key" ]]; then CALLER_ENV["$key"]="${!key}"; fi
done
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
for key in "${!CALLER_ENV[@]}"; do
  printf -v "$key" '%s' "${CALLER_ENV[$key]}"
  export "$key"
done

MODEL="${BLIND_MODEL:-${TEACHER_MODEL:-Qwen/Qwen2.5-7B-Instruct}}"
VARIANTS="${BLIND_VARIANTS:-all}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-${cuda_visible:-${BLIND_GPU:-0}}}"
EPISODES="${BLIND_EPISODES_PER_COMBINATION:-10}"
SEED_SETTING="${BLIND_SEED:-random}"
if [[ "${SEED_SETTING,,}" == random ]]; then
  SEED="$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')"
else
  SEED="$SEED_SETTING"
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "ERROR: BLIND_SEED must be an integer or 'random': $SEED_SETTING" >&2
  exit 1
fi
echo "Blind rollout seed: $SEED"
MAX_NEW_TOKENS="${BLIND_MAX_NEW_TOKENS:-384}"
DO_SAMPLE="${BLIND_DO_SAMPLE:-true}"
TEMPERATURE="${BLIND_TEMPERATURE:-0.7}"
case "${DO_SAMPLE,,}" in
  true|1|yes|on) SAMPLE_FLAG="--do-sample" ;;
  false|0|no|off) SAMPLE_FLAG="--no-do-sample" ;;
  *) echo "ERROR: BLIND_DO_SAMPLE must be true or false: $DO_SAMPLE" >&2; exit 1 ;;
esac
OUTPUT_DIR="${BLIND_OUTPUT_DIR:-data/alpha-beta/blind-rollout/result}"

# A+beta blind rollout (paper Sec. 2, Alg. 1 line 2): frontier plays player 0
# blind; the solver/filter/paraphrase stages run later at pair time.
#
# filter_on / filter_off are the Hypothesis B precursor (paper Appendix A).
# Their blind rollout is identical to A+beta's -- frontier as player 0 against
# the narrow ID matrix set -- and the on/off distinction (reasoning-action
# coupling filter) is applied only when the DPO pairs are built, not here. So
# both variants share one trajectory file; requesting both rolls once and copies.
declare -A GAMES_BY_VARIANT=(
  [core]="pd-classic,pd-tight,pd-high-temptation,stag-hunt"
  [aux]="negotiation,auction"
  [all]="pd-classic,pd-tight,pd-high-temptation,stag-hunt,bos,matching-pennies,negotiation,auction,ipd-stage"
  [rw]="pd-classic,pd-tight,pd-high-temptation,stag-hunt,bos,matching-pennies,negotiation,auction,ipd-stage"
  [filter_off]="pd-classic,pd-tight,pd-high-temptation,stag-hunt,bos"
  [filter_on]="pd-classic,pd-tight,pd-high-temptation,stag-hunt,bos"
)
declare -A OUTPUT_BY_VARIANT=(
  [core]="a_beta_core.jsonl"
  [aux]="a_beta_aux.jsonl"
  [all]="a_beta_all.jsonl"
  [rw]="a_beta_rw.jsonl"
  [filter_off]="filter_off.jsonl"
  [filter_on]="filter_on.jsonl"
)

IFS=',' read -r -a RAW_REQUESTED <<< "$VARIANTS"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

# filter_on and filter_off produce byte-identical trajectories; if both are
# asked for, roll filter_off and copy it to filter_on after the run.
REQUESTED=()
COPY_FILTER_ON=0
for raw in "${RAW_REQUESTED[@]}"; do
  v="${raw//[[:space:]]/}"
  [[ -z "$v" ]] && continue
  REQUESTED+=("$v")
done
if printf '%s\n' "${REQUESTED[@]}" | grep -qx filter_on \
  && printf '%s\n' "${REQUESTED[@]}" | grep -qx filter_off; then
  COPY_FILTER_ON=1
  TMP=()
  for v in "${REQUESTED[@]}"; do [[ "$v" == filter_on ]] || TMP+=("$v"); done
  REQUESTED=("${TMP[@]}")
fi

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/logs"
PIDS=()
for index in "${!REQUESTED[@]}"; do
  variant="${REQUESTED[index]//[[:space:]]/}"
  if [[ -z "${GAMES_BY_VARIANT[$variant]+x}" ]]; then
    echo "ERROR: unsupported BLIND_VARIANTS entry: $variant (use core,aux,all,rw,filter_on,filter_off)" >&2
    exit 1
  fi
  gpu="${GPUS[index % ${#GPUS[@]}]//[[:space:]]/}"
  output="${OUTPUT_DIR}/${OUTPUT_BY_VARIANT[$variant]}"
  log="${OUTPUT_DIR}/logs/${variant}.log"
  echo "[$variant] physical GPU $gpu -> $output"
  CUDA_VISIBLE_DEVICES="$gpu" python data/alpha-beta/blind-rollout/offline_trajectory.py \
    --model "$MODEL" \
    --games "${GAMES_BY_VARIANT[$variant]}" \
    --episodes-per-combination "$EPISODES" \
    --seed "$SEED" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    "$SAMPLE_FLAG" \
    --output "$output" \
    2>&1 | sed -u "s/^/[$variant][GPU $gpu] /" | tee "$log" &
  PIDS+=("$!")
done

FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAILED=1; fi
done
if ((FAILED)); then
  echo "ERROR: one or more blind rollout variants failed" >&2
  exit 1
fi

if ((COPY_FILTER_ON)); then
  cp "${OUTPUT_DIR}/filter_off.jsonl" "${OUTPUT_DIR}/filter_on.jsonl"
  echo "[filter_on] copied from filter_off.jsonl (coupling filter applies at pair-build time)"
fi

echo "Blind rollout complete: $OUTPUT_DIR"
