#!/usr/bin/env bash
# Algorithm 1 launcher + sharding for a 13-node x 3-GPU allocation (39 workers),
# in the style of data/blind-rollout/shard13.sh.
#
#   single machine:            ./data/alpha-beta/algorithm1.sh
#                              (one worker per visible GPU, then merge)
#   on node k (k = 1..13):     ./data/alpha-beta/algorithm1.sh k
#   after all 13 nodes finish: ./data/alpha-beta/algorithm1.sh merge
#
# Node k runs GPUS_PER_NODE workers, one pinned per local GPU; worker
# w = (k-1)*GPUS_PER_NODE + g + 1  of  WORLD = NODES*GPUS_PER_NODE. Each worker
# is `python algorithm1.py <w>` with A1_NODES=WORLD, so it line-shards the blind
# pool to its slice and writes result/shards/<variant>.n<w>.jsonl (resumable via
# the sibling .progress file). `merge` assembles all WORLD shards.
#
# Overridable: A1_ALLOC_NODES (13), A1_GPUS_PER_NODE (3), plus every A1_* in
# env.sh / .env (A1_VARIANTS, A1_INPUT_DIR, A1_MODE, A1_MAX_PAIRS_*, ...).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$ROOT" != / && ! -f "$ROOT/config.py" ]]; do ROOT="$(dirname "$ROOT")"; done
[[ -f "$ROOT/config.py" ]] || { echo "ERROR: cannot locate repository root" >&2; exit 1; }
cd "$ROOT"
source data/alpha-beta/env.sh

NODES="${A1_ALLOC_NODES:-13}"
GPUS_PER_NODE="${A1_GPUS_PER_NODE:-3}"
WORLD=$(( NODES * GPUS_PER_NODE ))
A1="python data/alpha-beta/algorithm1.py"

# spawn_workers <world> <first-worker> <count>: one process per local GPU 0..count-1
spawn_workers() {
  local world="$1" first="$2" count="$3"
  local pids=() g worker failed=0
  for (( g = 0; g < count; g++ )); do
    worker=$(( first + g ))
    (
      CUDA_VISIBLE_DEVICES="$g" A1_NODES="$world" \
        $A1 "$worker" 2>&1 | sed -u "s/^/[w${worker} g${g}] /"
    ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return "$failed"
}

MODE="${1:-run}"

if [[ "$MODE" == "merge" ]]; then
  exec env A1_NODES="$WORLD" $A1 merge
fi

if [[ "$MODE" == "run" ]]; then
  ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  [[ "$ngpu" =~ ^[0-9]+$ && "$ngpu" -ge 1 ]] || ngpu=1
  echo "[algorithm1] single machine: ${ngpu} GPU worker(s)"
  spawn_workers "$ngpu" 1 "$ngpu" || { echo "a worker failed" >&2; exit 1; }
  exec env A1_NODES="$ngpu" $A1 merge
fi

case "$MODE" in ''|*[!0-9]*) echo "usage: algorithm1.sh <1..${NODES}|merge>" >&2; exit 1 ;; esac
(( MODE >= 1 && MODE <= NODES )) || { echo "node out of range 1..${NODES}: $MODE" >&2; exit 1; }

nvidia-smi -L || true
first=$(( (MODE - 1) * GPUS_PER_NODE + 1 ))
echo "[algorithm1] node ${MODE}/${NODES}: workers ${first}..$(( first + GPUS_PER_NODE - 1 )) of ${WORLD}"
spawn_workers "$WORLD" "$first" "$GPUS_PER_NODE" || { echo "node ${MODE}: a worker failed" >&2; exit 1; }
echo "node ${MODE}: done -- run './data/alpha-beta/algorithm1.sh merge' after all ${NODES} nodes finish"
