#!/usr/bin/env bash
# Blind rollout for a 13-node x 3-GPU allocation (39 workers).
#
#   on node k (k = 1..13):   ./data/blind-rollout/shard13.sh k
#   after all 13 finish:     ./data/blind-rollout/shard13.sh merge
#
# Why this exists: shard.sh's unit of work is a (game, seed) pair, so it tops
# out at ~13 GPUs and one worker (the light bundle) is a 1.5x straggler. Here
# each of the four repeated games (pd-classic, pd-tight, pd-high-temptation,
# stag-hunt) x {seed 42, 142, 242} is split THREE ways over the (opponent,
# episode) work list via offline_trajectory.py --num-shards/--shard-index:
#
#     workers  0..35  -> heavy: 12 (game,seed) combos x 3 episode-shards
#     workers 36..38  -> the five one-shot games at seed 42, one file each
#
# 39 workers, ~240 generation calls each (was ~1100 on the shard.sh straggler)
# => roughly 4-5x lower wall-clock. Per-unit seeds come from (opponent_index,
# episode_index) only, so the 3-way split is bit-identical to an unsplit run.
#
# Per-shard rollouts write into data/blind-rollout/result/ as
# "<tag>.p<shard>.jsonl" (+ result/logs/<tag>.p<shard>.log). `merge` reuses
# shard.sh's merge (globs "*_s<seed>*.jsonl", assembles a_beta_*/filter_* by the
# "game" field in each row), so run either script's `merge` once at the end. The
# merge deletes the per-shard files afterwards (all inside a_beta_all.jsonl); set
# BLIND_KEEP_SHARDS=1 to keep them.
#
# Overridable: BLIND_EPISODES_PER_COMBINATION (default 12), BLIND_MODEL,
# BLIND_MAX_NEW_TOKENS, BLIND_TEMPERATURE, BLIND_REASONING_FORMAT.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE"
while [[ "$ROOT" != "/" && ! -f "$ROOT/config.py" ]]; do ROOT="$(dirname "$ROOT")"; done
cd "$ROOT"
REL="${HERE#"$ROOT"/}"

MODE="${1:?usage: shard13.sh <1..13|merge>}"
D="${REL}/result"
OT="${REL}/offline_trajectory.py"
RFMT="${BLIND_REASONING_FORMAT:-slots}"

NODES=13
GPUS_PER_NODE=3
HEAVY_SPLIT=3                                 # episode-shards per (game,seed)

# ---------------------------------------------------------------- merge
if [[ "$MODE" == "merge" ]]; then
  exec "$HERE/shard.sh" merge
fi
case "$MODE" in ''|*[!0-9]*) echo "MODE must be 1..${NODES} or merge" >&2; exit 1 ;; esac
(( MODE >= 1 && MODE <= NODES )) || { echo "node out of range 1..${NODES}: $MODE" >&2; exit 1; }

# ---------------------------------------------------------------- env (matches shard.sh)
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/pkgs}"
export HF_HOME="${HF_HOME:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
unset BNB_CUDA_VERSION
export SAL_DISABLE_BNB_CONFIG=1
module load conda/latest cuda/12.6
if [[ -n "${CUDA_HOME:-}" ]]; then
  NVHPC_ROOT="$(cd "${CUDA_HOME}/../.." && pwd)"
  export LD_LIBRARY_PATH="${NVHPC_ROOT}/math_libs/lib64:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sal
if [[ -f .env ]]; then set -a; source .env; set +a; fi

E="${BLIND_EPISODES_PER_COMBINATION:-12}"
MODEL="${BLIND_MODEL:-${TEACHER_MODEL:-Qwen/Qwen2.5-7B-Instruct}}"
_DEF_TOK=384; [[ "$RFMT" == slots ]] && _DEF_TOK=512
MAX_NEW_TOKENS="${BLIND_MAX_NEW_TOKENS:-$_DEF_TOK}"
TEMPERATURE="${BLIND_TEMPERATURE:-0.7}"
mkdir -p "$D/logs"

# 12 heavy (game,seed) combos, in worker order 0..11 -> workers 0..35 after the
# 3-way episode split.
HEAVY_GAMES=(pd-classic pd-tight pd-high-temptation stag-hunt)
HEAVY_SEEDS=(42 142 242)
HEAVY_TAGS=(pdc pdt pdh stag)

# 3 one-shot workers (36..38): game list, seed 42, run whole (no episode split).
ONESHOT_GAMES=("bos,matching-pennies" "ipd-stage,negotiation" "auction")
ONESHOT_TAGS=(os_bosmp os_ipdneg os_auction)

# run_one <games> <seed> <num-shards> <shard-index> <tag> <gpu>
run_one() {
  local games="$1" seed="$2" nshards="$3" sidx="$4" tag="$5" gpu="$6"
  local out="${D}/${tag}.p${sidx}.jsonl" log="${D}/logs/${tag}.p${sidx}.log"
  echo "  GPU ${gpu} -> ${games}  seed ${seed}  shard ${sidx}/${nshards}  -> ${out}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python "$OT" \
    --model "$MODEL" --games "$games" --seed "$seed" \
    --episodes-per-combination "$E" --max-new-tokens "$MAX_NEW_TOKENS" \
    --do-sample --temperature "$TEMPERATURE" \
    --reasoning-format "$RFMT" \
    --num-shards "$nshards" --shard-index "$sidx" \
    --output "$out" \
    2>&1 | sed -u "s/^/[${tag}.p${sidx}] /" | tee "$log"
}

# dispatch one global worker id (0..38) onto local GPU $gpu
dispatch() {
  local w="$1" gpu="$2"
  if (( w < 36 )); then
    local combo=$(( w / HEAVY_SPLIT )) sidx=$(( w % HEAVY_SPLIT ))
    local gi=$(( combo / 3 )) si=$(( combo % 3 ))
    run_one "${HEAVY_GAMES[$gi]}" "${HEAVY_SEEDS[$si]}" "$HEAVY_SPLIT" "$sidx" \
            "${HEAVY_TAGS[$gi]}_s${HEAVY_SEEDS[$si]}" "$gpu"
  elif (( w < 39 )); then
    local k=$(( w - 36 ))
    run_one "${ONESHOT_GAMES[$k]}" 42 1 0 "${ONESHOT_TAGS[$k]}_s42" "$gpu"
  else
    echo "  GPU ${gpu}: no work for worker ${w}" ; : > "${D}/idle.w${w}.jsonl"
  fi
}

nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda is unavailable'"

node="$MODE"
echo "Blind rollout | node ${node}/${NODES} | E=${E} | model=${MODEL} | format=${RFMT} | world=$((NODES*GPUS_PER_NODE))"
PIDS=()
for g in $(seq 0 $((GPUS_PER_NODE - 1))); do
  w=$(( (node - 1) * GPUS_PER_NODE + g ))
  ( dispatch "$w" "$g" ) &
  PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
((FAILED)) && { echo "node ${node}: a shard failed" >&2; exit 1; }
echo "node ${node}: done — run './data/blind-rollout/shard13.sh merge' after all ${NODES} nodes finish"
