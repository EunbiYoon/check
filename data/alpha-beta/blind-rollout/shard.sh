#!/usr/bin/env bash
# Game-sharded blind rollout for a 4-node x 3-GPU allocation.
#
#   on node k (k = 1..4):   ./data/alpha-beta/blind-rollout/shard.sh k
#   after all 4 finish:     ./data/alpha-beta/blind-rollout/shard.sh merge
#
# 12 workers = {pd-classic, pd-tight, pd-high-temptation, stag-hunt} x {seed 42,142,242};
# the five one-shot games ride along on node 2 / GPU 0. Distinct seeds diverge
# because offline_trajectory.py seeds torch's sampler with --seed.
#
# Per-shard rollouts write straight into data/alpha-beta/blind-rollout/result/
# as "<tag>_s<seed>.jsonl" (+ result/logs/<tag>.log); `merge` concatenates just
# those into result/a_beta_all.jsonl and the core/aux/filter subsets.
#
# Overridable: BLIND_EPISODES_PER_COMBINATION (default 12), BLIND_MODEL,
# BLIND_MAX_NEW_TOKENS, BLIND_TEMPERATURE.
set -euo pipefail

# repo root = first ancestor with config.py
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$ROOT" != "/" && ! -f "$ROOT/config.py" ]]; do ROOT="$(dirname "$ROOT")"; done
cd "$ROOT"

MODE="${1:?usage: shard.sh <1|2|3|4|merge>}"
D="data/alpha-beta/blind-rollout/result"
# Per-shard rollouts land directly in $D as "<tag>_s<seed>.jsonl"; the merge
# step reads exactly those and never its own "a_beta_*" / "filter_*" outputs.

# ---------------------------------------------------------------- merge
if [[ "$MODE" == "merge" ]]; then
  shopt -s nullglob
  files=("$D"/*_s[0-9]*.jsonl)
  ((${#files[@]})) || { echo "no shard files (*_s<seed>.jsonl) in $D" >&2; exit 1; }
  cat "${files[@]}" > "$D/a_beta_all.jsonl"
  cp "$D/a_beta_all.jsonl" "$D/a_beta_rw.jsonl"
  python - <<'PY'
import json
D = "data/alpha-beta/blind-rollout/result"
rows = [json.loads(l) for l in open(f"{D}/a_beta_all.jsonl", encoding="utf-8") if l.strip()]
subsets = {
    "a_beta_core.jsonl": {"pd-classic", "pd-tight", "pd-high-temptation", "stag-hunt"},
    "a_beta_aux.jsonl": {"negotiation", "auction"},
    "filter_off.jsonl": {"pd-classic", "pd-tight", "pd-high-temptation", "stag-hunt", "bos"},
}
for name, games in subsets.items():
    kept = [r for r in rows if r.get("game") in games]
    with open(f"{D}/{name}", "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(kept)} trajectories")
PY
  cp "$D/filter_off.jsonl" "$D/filter_on.jsonl"
  echo "merged ${#files[@]} shards"
  exit 0
fi

# ---------------------------------------------------------------- env (matches blind_rollout.sh)
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
MAX_NEW_TOKENS="${BLIND_MAX_NEW_TOKENS:-384}"
TEMPERATURE="${BLIND_TEMPERATURE:-0.7}"
OT="data/alpha-beta/blind-rollout/offline_trajectory.py"
mkdir -p "$D/logs"

_LIGHT="stag-hunt,bos,matching-pennies,ipd-stage,negotiation,auction"
case "$MODE" in
  1) JOBS=("0 pd-classic 42 pdc_s42"          "1 pd-tight 42 pdt_s42"           "2 pd-high-temptation 42 pdh_s42") ;;
  2) JOBS=("0 ${_LIGHT} 42 stag_light_s42"    "1 pd-classic 142 pdc_s142"       "2 pd-tight 142 pdt_s142") ;;
  3) JOBS=("0 pd-high-temptation 142 pdh_s142" "1 stag-hunt 142 stag_s142"      "2 pd-classic 242 pdc_s242") ;;
  4) JOBS=("0 pd-tight 242 pdt_s242"          "1 pd-high-temptation 242 pdh_s242" "2 stag-hunt 242 stag_s242") ;;
  *) echo "MODE must be 1, 2, 3, 4, or merge" >&2; exit 1 ;;
esac

echo "Node ${MODE}: E=${E}, model=${MODEL}"
PIDS=()
for job in "${JOBS[@]}"; do
  read -r gpu games seed tag <<< "$job"
  echo "  GPU ${gpu} -> ${games}  (seed ${seed})"
  CUDA_VISIBLE_DEVICES="$gpu" python "$OT" \
    --model "$MODEL" --games "$games" --seed "$seed" \
    --episodes-per-combination "$E" --max-new-tokens "$MAX_NEW_TOKENS" \
    --do-sample --temperature "$TEMPERATURE" \
    --output "${D}/${tag}.jsonl" \
    2>&1 | sed -u "s/^/[${tag}] /" | tee "${D}/logs/${tag}.log" &
  PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
((FAILED)) && { echo "node ${MODE}: a shard failed" >&2; exit 1; }
echo "node ${MODE}: done"
