#!/usr/bin/env bash
# Blind rollout for a 13-node x 3-GPU allocation (39 workers).
#
#   on node k (k = 1..13):   ./data/blind-rollout/shard13.sh k
#   after all 13 finish:     ./data/blind-rollout/shard13.sh merge
#
# Why this exists: shard.sh's unit of work is a (game, seed) pair, so it tops
# out at ~13 GPUs and one worker (the light bundle) is a 1.5x straggler.
#
# The work splits into two pools (rounds = generation calls):
#     heavy    12 (game,seed) combos of the repeated games, 6*12*10 = 720 each
#     one-shot the 5 one-shot games at seed 42, 32 opp-slots * ONESHOT_E rounds
# At ONESHOT_E=120 that is 8640 heavy + 3840 one-shot = 12480 rounds; an even
# 39-way split is 320 rounds/worker. An earlier version split the heavy pool 3
# ways (36 workers) and left only 3 workers for the whole one-shot pool -- those
# three did ~1280 calls each while the heavy workers did ~240, so node 13 was a
# 5x straggler. Now:
#
#     workers  0..23  -> heavy: 12 (game,seed) combos x 2 episode-shards (~360)
#     workers 24..38  -> one-shot bundle, 15-way (opponent,episode) split  (~256)
#
# Per-unit seeds come from (opponent_index, episode_index) only, so any
# --num-shards value is bit-identical to an unsplit run.
#
# Per-shard rollouts write into data/blind-rollout/result/ as
# "<tag>.p<shard>.jsonl" (+ result/logs/<tag>.p<shard>.log): "<HEAVY_TAG>_s<seed>"
# for the heavy pool, "os_s42" for the one-shot pool. `merge` reuses shard.sh's
# merge (globs "*_s<seed>*.jsonl", assembles a_beta_*/filter_* by the "game"
# field in each row), so run either script's `merge` once at the end. The merge
# deletes the per-shard files afterwards (all inside a_beta_all.jsonl); set
# BLIND_KEEP_SHARDS=1 to keep them.
#
# Overridable: BLIND_EPISODES_PER_COMBINATION (repeated games),
# BLIND_ONESHOT_EPISODES (one-shot games), BLIND_MODEL,
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
HEAVY_SPLIT=2                                 # episode-shards per (game,seed) -> 12*2 = 24 workers
ONESHOT_SPLIT=15                              # (opponent,episode)-shards for the one-shot bundle
HEAVY_WORKERS=$(( 12 * HEAVY_SPLIT ))         # 24; one-shot takes workers 24..38

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
ONESHOT_E="${BLIND_ONESHOT_EPISODES:-$E}"
MODEL="${BLIND_MODEL:-${TEACHER_MODEL:-Qwen/Qwen2.5-7B-Instruct}}"
_DEF_TOK=384; [[ "$RFMT" == slots ]] && _DEF_TOK=512
MAX_NEW_TOKENS="${BLIND_MAX_NEW_TOKENS:-$_DEF_TOK}"
TEMPERATURE="${BLIND_TEMPERATURE:-0.7}"
mkdir -p "$D/logs"

# 12 heavy (game,seed) combos, in worker order 0..11 -> workers 0..23 after the
# 2-way episode split.
HEAVY_GAMES=(pd-classic pd-tight pd-high-temptation stag-hunt)
HEAVY_SEEDS=(42 142 242)
HEAVY_TAGS=(pdc pdt pdh stag)

# One-shot pool: all five games as a single bundle at seed 42, split 15 ways over
# the (opponent,episode) work list across workers 24..38.
ONESHOT_GAMES="bos,matching-pennies,ipd-stage,negotiation,auction"

# run_one <games> <seed> <num-shards> <shard-index> <tag> <gpu> <episodes>
run_one() {
  local games="$1" seed="$2" nshards="$3" sidx="$4" tag="$5" gpu="$6" episodes="$7"
  local out="${D}/${tag}.p${sidx}.jsonl" log="${D}/logs/${tag}.p${sidx}.log"
  echo "  GPU ${gpu} -> ${games}  seed ${seed}  shard ${sidx}/${nshards}  -> ${out}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python "$OT" \
    --model "$MODEL" --games "$games" --seed "$seed" \
    --episodes-per-combination "$episodes" --max-new-tokens "$MAX_NEW_TOKENS" \
    --do-sample --temperature "$TEMPERATURE" \
    --reasoning-format "$RFMT" \
    --num-shards "$nshards" --shard-index "$sidx" \
    --output "$out" \
    2>&1 | sed -u "s/^/[${tag}.p${sidx}] /" | tee "$log"
}

# dispatch one global worker id (0..38) onto local GPU $gpu
dispatch() {
  local w="$1" gpu="$2"
  if (( w < HEAVY_WORKERS )); then
    local combo=$(( w / HEAVY_SPLIT )) sidx=$(( w % HEAVY_SPLIT ))
    local gi=$(( combo / 3 )) si=$(( combo % 3 ))
    run_one "${HEAVY_GAMES[$gi]}" "${HEAVY_SEEDS[$si]}" "$HEAVY_SPLIT" "$sidx" \
            "${HEAVY_TAGS[$gi]}_s${HEAVY_SEEDS[$si]}" "$gpu" "$E"
  elif (( w < HEAVY_WORKERS + ONESHOT_SPLIT )); then
    local sidx=$(( w - HEAVY_WORKERS ))
    run_one "$ONESHOT_GAMES" 42 "$ONESHOT_SPLIT" "$sidx" "os_s42" "$gpu" "$ONESHOT_E"
  else
    echo "  GPU ${gpu}: no work for worker ${w}" ; : > "${D}/idle.w${w}.jsonl"
  fi
}

nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda is unavailable'"

node="$MODE"
echo "Blind rollout | node ${node}/${NODES} | repeated-E=${E} | one-shot-E=${ONESHOT_E} | model=${MODEL} | format=${RFMT} | world=$((NODES*GPUS_PER_NODE))"
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
