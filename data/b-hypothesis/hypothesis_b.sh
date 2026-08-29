#!/usr/bin/env bash
# Hypothesis B data construction (paper Appendix A) — the falsified
# coupling-filter DPO baseline, GPU-sharded in the style of algorithm1.sh.
#
#   step 1 (GPU)  slot-form [EV] blind rollout           offline_trajectory.py
#                 frontier as player 0 vs heuristic opponents, [Prior][Update]
#                 [EV][Decision] reasoning so the coupling filter has an [EV] slot
#   step 2 (CPU)  parse [EV] -> tag coupling -> S1..S4 candidate pairs      build_pairs.py
#                 filter_off -> --no-coupling-filter   (wider chosen-side)
#                 filter_on  -> --coupling-filter      (coupled-only; narrower)
#
# Launch modes (mirror data/alpha-beta/algorithm1.sh):
#   ./hypothesis_b.sh                 one node: blind rollout across every GPU in
#                                     CUDA_VISIBLE_DEVICES, then merge + build pairs
#   ./hypothesis_b.sh <1|2|..>        node k of an HB_NODES-node allocation: run
#                                     only this node's slice of the (game,seed)
#                                     rollout workers
#   ./hypothesis_b.sh merge           concat the blind shards -> blind/hb_blind.jsonl,
#                                     then build every HB_VARIANTS pair file
#   ./hypothesis_b.sh pairs           skip the rollout; (re)build pairs from HB_INPUT
#
# Each rollout worker is one (game, seed) pair writing a self-contained shard,
# so a modulo split by worker never breaks anything.
#
# Env overrides (command wins over .env):
#   CUDA_VISIBLE_DEVICES  GPU id list          default: every GPU reported by nvidia-smi
#   HB_NODES              1                     multi-node world size
#   HB_GAMES             pd-classic,pd-tight,pd-high-temptation,stag-hunt,bos
#   HB_SEEDS             42                     comma-separated; more seeds = more data
#   HB_EPISODES          <config BLIND_EPISODES_PER_COMBINATION>
#   HB_MODEL             <config BLIND_MODEL>
#   HB_MAX_NEW_TOKENS    512                    slot form needs the room
#   HB_TEMPERATURE       0.7
#   HB_RESULT_DIR        data/b-hypothesis/result   root: b_filter_*.jsonl; blind/: shards + hb_blind.jsonl; logs/
#   HB_INPUT             <HB_RESULT_DIR>/blind/hb_blind.jsonl
#   HB_OUTPUT_DIR        <HB_RESULT_DIR>            b_filter_on.jsonl / b_filter_off.jsonl land here
#                                                  (train with DATA_DIR=data/b-hypothesis/result)
#   HB_VARIANTS          filter_on,filter_off   comma-separated: filter_on|filter_off
#   HB_STRATEGIES        S1,S2,S3,S4            comma-separated subset
#   HB_DEDUP             context                none | full | context | position
#   HB_MAX_PAIRS         (unset)               hard cap on the pair count
#   HB_MAX_PAIRS_<VARIANT> (unset)             per-variant cap, e.g.
#                                              HB_MAX_PAIRS_FILTER_ON=400 (Table 7)
set -euo pipefail

MODE="${1:-}"                                   # "" | 1 | 2 | ... | merge | pairs
case "$MODE" in ''|merge|pairs) ;; *[!0-9]*) echo "usage: hypothesis_b.sh [<node>|merge|pairs]" >&2; exit 1 ;; esac

# repo root = first ancestor with config.py
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # data/b-hypothesis
ROOT="$SELF_DIR"
while [[ "$ROOT" != "/" && ! -f "$ROOT/config.py" ]]; do ROOT="$(dirname "$ROOT")"; done
[[ -f "$ROOT/config.py" ]] || { echo "ERROR: cannot locate repo root (no config.py)" >&2; exit 1; }
cd "$ROOT"
SELF_REL="${SELF_DIR#"$ROOT"/}"                            # e.g. data/b-hypothesis

# command-line values override .env
ENV_KEYS=(
  CUDA_VISIBLE_DEVICES HB_NODES HB_GAMES HB_SEEDS HB_EPISODES HB_MODEL
  HB_MAX_NEW_TOKENS HB_TEMPERATURE HB_INPUT HB_RESULT_DIR HB_OUTPUT_DIR HB_VARIANTS
  HB_STRATEGIES HB_DEDUP HB_MAX_PAIRS HB_MAX_PAIRS_FILTER_ON HB_MAX_PAIRS_FILTER_OFF
  BLIND_MODEL TEACHER_MODEL
)
declare -A CALLER_ENV=()
for key in "${ENV_KEYS[@]}"; do [[ -v "$key" ]] && CALLER_ENV["$key"]="${!key}"; done
if [[ -f .env ]]; then set -a; source .env; set +a; fi
for key in "${!CALLER_ENV[@]}"; do printf -v "$key" '%s' "${CALLER_ENV[$key]}"; export "$key"; done

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/pkgs}"
export HF_HOME="${HF_HOME:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

RUN_ROLLOUT=1
[[ "$MODE" == "merge" || "$MODE" == "pairs" ]] && RUN_ROLLOUT=0

if [[ "$RUN_ROLLOUT" == 1 ]]; then
  unset BNB_CUDA_VERSION
  export SAL_DISABLE_BNB_CONFIG=1
  module load conda/latest cuda/12.6 2>/dev/null || true
  if [[ -n "${CUDA_HOME:-}" ]]; then
    NVHPC_ROOT="$(cd "${CUDA_HOME}/../.." && pwd)"
    export LD_LIBRARY_PATH="${NVHPC_ROOT}/math_libs/lib64:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda activate sal 2>/dev/null || true
fi

# The rollout script is shared with the A+β pipeline; its dir has moved across
# refactors, so locate it rather than hard-coding the path.
OT="$(find data -path '*/blind-rollout/offline_trajectory.py' -print -quit 2>/dev/null || true)"
[[ -n "$OT" ]] || { echo "ERROR: cannot find blind-rollout/offline_trajectory.py under data/" >&2; exit 1; }
# All Hypothesis B artefacts (blind shards, merged pool, pair files) live under
# this pipeline's own result/ dir — never mixed into data/blind-rollout/result.
RESULT_DIR="${HB_RESULT_DIR:-${SELF_REL}/result}"   # data/b-hypothesis/result
BLIND_DIR="${RESULT_DIR}/blind"                # per-game shards + the merged pool
BLIND="${BLIND_DIR}/hb_blind.jsonl"            # merged Hypothesis B trajectory pool

# --- config -------------------------------------------------------------------
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_LIST="$CUDA_VISIBLE_DEVICES"
else
  # Do not assume a fixed 0..4 layout: GPU nodes may expose more devices or a
  # non-contiguous set.  Slurm-provided CUDA_VISIBLE_DEVICES still wins above.
  GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
    | tr -d '[:space:]' | paste -sd, -)"
  [[ -n "$GPU_LIST" ]] || {
    echo "ERROR: no GPUs discovered; set CUDA_VISIBLE_DEVICES explicitly" >&2
    exit 1
  }
fi
NODES="${HB_NODES:-1}"
GAMES_CSV="${HB_GAMES:-pd-classic,pd-tight,pd-high-temptation,stag-hunt,bos}"
SEEDS_CSV="${HB_SEEDS:-42}"
EPISODES="${HB_EPISODES:-}"
MODEL="${HB_MODEL:-${BLIND_MODEL:-${TEACHER_MODEL:-Qwen/Qwen2.5-7B-Instruct}}}"
MAX_NEW_TOKENS="${HB_MAX_NEW_TOKENS:-512}"
TEMPERATURE="${HB_TEMPERATURE:-0.7}"
INPUT="${HB_INPUT:-$BLIND}"
OUT_DIR="${HB_OUTPUT_DIR:-$RESULT_DIR}"
VARIANTS_CSV="${HB_VARIANTS:-filter_on,filter_off}"
STRATEGIES="${HB_STRATEGIES:-S1,S2,S3,S4}"
DEDUP="${HB_DEDUP:-context}"
MAX_PAIRS="${HB_MAX_PAIRS:-}"
BUILD="data/b-hypothesis/build_pairs.py"

IFS=',' read -r -a GPUS  <<< "${GPU_LIST//[[:space:]]/}"
IFS=',' read -r -a GAMES <<< "${GAMES_CSV//[[:space:]]/}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV//[[:space:]]/}"
GPUS_PER_NODE=${#GPUS[@]}
WORLD=$((NODES * GPUS_PER_NODE))

# worker list: one (game, seed) job per line -> "<game> <seed> <tag>"
JOBS=()
for seed in "${SEEDS[@]}"; do
  for game in "${GAMES[@]}"; do
    JOBS+=("${game} ${seed} hb_${game//-/}_s${seed}")
  done
done

mkdir -p "$RESULT_DIR/logs" "$BLIND_DIR" "$OUT_DIR" "$OUT_DIR/logs"

# ---------------------------------------------------------------- pair building
build_variants() {
  [[ -f "$INPUT" ]] || {
    echo "ERROR: missing trajectory input $INPUT" >&2
    echo "       run './data/b-hypothesis/hypothesis_b.sh' (rollout) or set HB_INPUT" >&2
    exit 1
  }
  local REQUESTED
  IFS=',' read -r -a REQUESTED <<< "$VARIANTS_CSV"
  echo "Hypothesis B pairs | strategies=${STRATEGIES} | dedup=${DEDUP} | ${INPUT} -> ${OUT_DIR}"
  local OK=0 FAIL=0 pids=() raw variant flag entry
  for raw in "${REQUESTED[@]}"; do
    variant="${raw//[[:space:]]/}"; [[ -z "$variant" ]] && continue
    case "$variant" in
      filter_off) flag="--no-coupling-filter" ;;
      filter_on)  flag="--coupling-filter" ;;
      *) echo "ERROR: unknown HB_VARIANTS entry: $variant (filter_off, filter_on)" >&2; exit 1 ;;
    esac
    # per-variant cap: HB_MAX_PAIRS_<VARIANT>, else HB_MAX_PAIRS, else uncapped
    local capkey="HB_MAX_PAIRS_${variant^^}" cap
    cap="${!capkey:-${MAX_PAIRS}}"
    local EXTRA=(--dedup "$DEDUP")
    [[ -n "$cap" ]] && EXTRA+=(--max-pairs "$cap")
    # "b_" prefix marks these as the Hypothesis B pair files, distinct from the
    # A+β blind-subset "filter_*.jsonl" trajectory files in data/blind-rollout/result.
    local out="${OUT_DIR}/b_${variant}.jsonl"
    (
      python "$BUILD" --input "$INPUT" --output "$out" \
        "$flag" --strategies "$STRATEGIES" "${EXTRA[@]}" \
        2>&1 | sed -u "s/^/[${variant}] /" | tee "${OUT_DIR}/logs/b_${variant}.log"
    ) &
    pids+=("$!:$variant")
  done
  for entry in "${pids[@]}"; do
    if wait "${entry%%:*}"; then OK=$((OK+1)); else
      FAIL=$((FAIL+1)); variant="${entry#*:}"
      echo "[${variant}] WARN: no pairs written (see ${OUT_DIR}/logs/b_${variant}.log)" >&2
      [[ "$variant" == filter_on ]] && \
        echo "[${variant}] filter_on needs slot-form [EV] reasoning in the trajectory pool" >&2
    fi
  done
  echo "Hypothesis B pairs: ${OK} ok, ${FAIL} failed -> ${OUT_DIR}"
  [[ "$OK" -gt 0 ]] || { echo "ERROR: no Hypothesis B variant produced pairs" >&2; exit 1; }
}

# ---------------------------------------------------------------- merge shards
merge_shards() {
  shopt -s nullglob
  local files=("$BLIND_DIR"/hb_*_s[0-9]*.jsonl)
  ((${#files[@]})) || { echo "ERROR: no Hypothesis B shards (hb_*_s<seed>.jsonl) in $BLIND_DIR" >&2; exit 1; }
  cat "${files[@]}" > "$BLIND"
  echo "merged ${#files[@]} shards -> ${BLIND} ($(wc -l < "$BLIND") trajectories)"
}

# ---------------------------------------------------------------- rollout worker
run_rollout_job() {  # <gpu> <game> <seed> <tag>
  local gpu="$1" game="$2" seed="$3" tag="$4"
  local ep_flag=()
  [[ -n "$EPISODES" ]] && ep_flag=(--episodes-per-combination "$EPISODES")
  echo "  GPU ${gpu} -> ${game} (seed ${seed}) -> ${tag}.jsonl"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python "$OT" \
    --model "$MODEL" --games "$game" --seed "$seed" \
    --max-new-tokens "$MAX_NEW_TOKENS" --do-sample --temperature "$TEMPERATURE" \
    --reasoning-format slots "${ep_flag[@]}" \
    --output "${BLIND_DIR}/${tag}.jsonl" \
    2>&1 | sed -u "s/^/[${tag}] /" | tee "${RESULT_DIR}/logs/${tag}.log"
}

# =============================================================================
if [[ "$MODE" == "pairs" ]]; then
  build_variants
  exit 0
fi

if [[ "$MODE" == "merge" ]]; then
  merge_shards
  build_variants
  exit 0
fi

# ---- rollout: single node (MODE="") or node k of HB_NODES --------------------
nvidia-smi -L 2>/dev/null || true
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "ERROR: torch.cuda is unavailable — run the rollout on a GPU node" >&2; exit 1; }

NODE="${MODE:-1}"
[[ "$NODE" -ge 1 && "$NODE" -le "$NODES" ]] || { echo "ERROR: node $NODE out of range 1..$NODES" >&2; exit 1; }
echo "Hypothesis B blind rollout (slots) | node ${NODE}/${NODES} | gpus/node=${GPUS_PER_NODE} | world=${WORLD}"
echo "  model=${MODEL} | max-new-tokens=${MAX_NEW_TOKENS} | games=${GAMES_CSV} | seeds=${SEEDS_CSV}"

# One background worker per local GPU; each worker runs its assigned
# (game, seed) jobs sequentially so a GPU is never double-booked.
PIDS=()
for g in $(seq 0 $((GPUS_PER_NODE - 1))); do
  slot=$(( (NODE - 1) * GPUS_PER_NODE + g ))
  gpu="${GPUS[g]}"
  mine=()
  for i in "${!JOBS[@]}"; do (( i % WORLD == slot )) && mine+=("${JOBS[i]}"); done
  ((${#mine[@]})) || continue
  (
    for job in "${mine[@]}"; do
      read -r game seed tag <<< "$job"
      run_rollout_job "$gpu" "$game" "$seed" "$tag"
    done
  ) &
  PIDS+=("$!")
done
((${#PIDS[@]})) || { echo "node ${NODE}: no jobs assigned (WORLD=${WORLD}, jobs=${#JOBS[@]})"; exit 0; }

FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
((FAILED)) && { echo "node ${NODE}: a rollout worker failed" >&2; exit 1; }
echo "node ${NODE}: rollout done"

if [[ "$NODES" -gt 1 ]]; then
  echo "run './data/b-hypothesis/hypothesis_b.sh merge' once all ${NODES} nodes finish"
  exit 0
fi

# single node: chain straight into merge + pair building
merge_shards
build_variants
