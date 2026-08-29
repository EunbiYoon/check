#!/usr/bin/env bash
# Algorithm 1 (paper §2.6), lines 3-9 — per-round A+β pair construction.
#
#   line 4  a* <- BR0(a1_t)                        solver-pinned/best_action.py   (Eq. 1)
#   line 6  R_cf <- Eq. 3 (horizon-aware) | Eq. 2  counterfactual-horizon/        (flag h)
#   line 7  keep the flip only if R_cf > R(tau_blind)
#   line 8  rho*_t <- reasoning label:                                            (build_pairs.py)
#             repeated games  -> LLM_front(no-leak prompt, a*)  (§2.3 teacher)
#             one-shot games  -> blind reasoning verbatim, action repinned to a*
#                                (§3.1 "solver-labelled pairs" — auction/negotiation)
#   line 9  emit pair  (x_t, rho*_t + a*  >  rho_blind_t + a0_blind_t)
#
# Variants (paper §3.1 / Table 7):
#   core/aux/all : built directly from their blind-rollout trajectory files
#   rw           : NOT a rollout — the built ALL pair file with the auction/
#                  negotiation subset upsampled 4x (A1_RW_FACTOR). Requesting rw
#                  forces all to build; rw is derived after the merge/build.
#
# Input  : blind rollout trajectory JSONL (Algorithm 1 line 2 — run blind_rollout.sh first)
# Output : preference-pair JSONL consumed by train/dpo-lora
#
# Two launch modes:
#   ./algorithm1.sh                one node, one variant per GPU (CUDA_VISIBLE_DEVICES list)
#   ./algorithm1.sh <k>            node k of an A1_NODES x 3-GPU allocation (default
#                                  8 nodes = 24 workers; set A1_NODES=13 for 39).
#                                  Each of the WORLD=A1_NODES*3 GPU workers
#                                  paraphrases its 1/WORLD line-slice of every
#                                  variant's trajectory file, sequentially.
#   ./algorithm1.sh merge          after all A1_NODES nodes finish: concatenate the
#                                  slices into ${A1_OUTPUT_DIR}/a_beta_<variant>.jsonl
# Each trajectory JSONL line is one self-contained episode, so a modulo split by
# line never breaks the counterfactual-horizon DP (it needs whole episodes only).
#
# Hypothesis B (paper §3 — reasoning-action coupling ablation): the filter_on /
# filter_off variants are built here too, as a convenience so one command covers
# every DPO variant. They use a different, CPU-only pair builder
# (data/b-hypothesis/build_pairs.py) — no teacher model, no GPU — so they are
# never line-sliced; in the multi-node mode they run once, on node 1.
#
# Env overrides (command wins over .env):
#   A1_VARIANTS   core,aux,all,rw            which trajectory files to convert
#                 [+ filter_on,filter_off]   (filter_* are Hypothesis B, see above)
#   A1_INPUT_DIR  data/blind-rollout/result
#   A1_OUTPUT_DIR data/alpha-beta/result
#   A1_NODES      8                          multi-node allocation width (WORLD = A1_NODES x 3)
#   A1_MODE       horizon-aware | fixed      Algorithm 1 flag h (Eq. 3 vs Eq. 2)
#   A1_TEACHER_MODEL, A1_TEACHER_MAX_NEW_TOKENS
#   A1_MAX_INVALID_FRACTION  0.2             abort a shard only if more than this
#                                            share of teacher paraphrases fail
#                                            structural validation; the rest are
#                                            dropped with a warning
#   A1_DEDUP      context                    merge replayed decision contexts to one
#                                            pair (none|full|context|position)
#   A1_DEDUP_<VARIANT>                       per-variant override; AUX defaults
#                                            to full so distinct one-shot
#                                            completions are not collapsed
#   A1_RW_FACTOR  4                          RW upsample multiplier for the
#                                            auction/negotiation subset of ALL
#   A1_MAX_PAIRS_<VARIANT>  (unset)           per-variant pair cap, e.g.
#                                            A1_MAX_PAIRS_CORE=500 (paper Table 7).
#                                            A1_MAX_PAIRS is the fallback for all.
#   A1_KEEP_SHARDS  (unset)                  keep ${OUT_DIR}/shards/*.jsonl after
#                                            merge; default is to delete them once
#                                            concatenated into a_beta_<variant>.jsonl
#   CUDA_VISIBLE_DEVICES  GPU list, no-arg mode only (default 0)
set -euo pipefail

A1_NODE="${1:-}"    # "" (single-node) | <node k, 1..A1_NODES> | merge
case "$A1_NODE" in
  ''|merge) ;;
  *[!0-9]*|0) echo "usage: algorithm1.sh [<node 1..A1_NODES>|merge]" >&2; exit 1 ;;
esac

# repo root = first ancestor with config.py
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$ROOT" != "/" && ! -f "$ROOT/config.py" ]]; do ROOT="$(dirname "$ROOT")"; done
cd "$ROOT"

# command-line values override .env
ENV_KEYS=(
  A1_VARIANTS A1_INPUT_DIR A1_OUTPUT_DIR A1_MODE A1_DEDUP A1_MAX_PAIRS
  A1_TEACHER_MODEL A1_TEACHER_MAX_NEW_TOKENS
  TEACHER_MODEL CUDA_VISIBLE_DEVICES cuda_visible
)
declare -A CALLER_ENV=()
for key in "${ENV_KEYS[@]}"; do
  [[ -v "$key" ]] && CALLER_ENV["$key"]="${!key}"
done
if [[ -f .env ]]; then set -a; source .env; set +a; fi
for key in "${!CALLER_ENV[@]}"; do printf -v "$key" '%s' "${CALLER_ENV[$key]}"; export "$key"; done

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

VARIANTS_CSV="${A1_VARIANTS:-core,aux,all,rw}"
IN_DIR="${A1_INPUT_DIR:-data/blind-rollout/result}"
OUT_DIR="${A1_OUTPUT_DIR:-data/alpha-beta/result}"
MODE="${A1_MODE:-horizon-aware}"
TEACHER="${A1_TEACHER_MODEL:-${TEACHER_MODEL:-Qwen/Qwen2.5-7B-Instruct}}"
TEACHER_TOKENS="${A1_TEACHER_MAX_NEW_TOKENS:-1024}"
MAX_INVALID_FRACTION="${A1_MAX_INVALID_FRACTION:-0.2}"
DEDUP="${A1_DEDUP:-context}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-${cuda_visible:-0}}"

# per-variant pair cap: A1_MAX_PAIRS_<VARIANT>, else A1_MAX_PAIRS, else none
variant_max_pairs() {  # <variant> -> echoes the cap (empty when uncapped)
  local key="A1_MAX_PAIRS_${1^^}"
  if [[ -n "${!key:-}" ]]; then echo "${!key}"; else echo "${A1_MAX_PAIRS:-}"; fi
}

variant_dedup() {  # <variant> -> echoes the merge dedup mode
  local key="A1_DEDUP_${1^^}"
  if [[ -n "${!key:-}" ]]; then
    echo "${!key}"
  elif [[ "$1" == "aux" ]]; then
    echo "full"
  else
    echo "$DEDUP"
  fi
}

NODES="${A1_NODES:-8}"                    # override for a wider allocation (e.g. 13)
GPUS_PER_NODE=3
WORLD=$((NODES * GPUS_PER_NODE))          # line-slices per variant (NODES x 3)

case "$MODE" in horizon-aware|fixed) ;; *) echo "ERROR: A1_MODE must be horizon-aware or fixed" >&2; exit 1 ;; esac
if [[ -n "$A1_NODE" && "$A1_NODE" != merge ]] && (( A1_NODE < 1 || A1_NODE > NODES )); then
  echo "ERROR: node $A1_NODE out of range 1..$NODES (set A1_NODES to change the allocation width)" >&2
  exit 1
fi

declare -A FILE_BY_VARIANT=(
  [core]="a_beta_core.jsonl"
  [aux]="a_beta_aux.jsonl"
  [all]="a_beta_all.jsonl"
  [rw]="a_beta_rw.jsonl"
  # Hypothesis B (paper §3): output basenames for the coupling-filter ablation.
  [filter_on]="filter_on.jsonl"
  [filter_off]="filter_off.jsonl"
)

# Hypothesis B (paper §3) — reasoning-action coupling ablation. Both variants
# read the same blind-rollout trajectory file and differ only by the coupling
# filter flag; the pair builder is CPU-only, so these are never sharded.
declare -A HYP_B_FLAG=(
  [filter_on]="--coupling-filter"
  [filter_off]="--no-coupling-filter"
)
HYP_B_INPUT="filter_off.jsonl"          # shared trajectory input for both variants

IFS=',' read -r -a REQUESTED <<< "$VARIANTS_CSV"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
SHARD_DIR="${OUT_DIR}/shards"
mkdir -p "$OUT_DIR" "$OUT_DIR/logs" "$SHARD_DIR"

VARIANTS=()      # A+β variants: GPU teacher pipeline, line-sliced across workers
B_VARIANTS=()    # Hypothesis B variants: CPU coupling-filter pipeline, never sliced
RW_REQUESTED=0   # paper §3.1: RW = the ALL mix, auction/negotiation pairs upsampled 4x
for raw in "${REQUESTED[@]}"; do
  v="${raw//[[:space:]]/}"; [[ -z "$v" ]] && continue
  if [[ -z "${FILE_BY_VARIANT[$v]+x}" ]]; then
    echo "ERROR: unknown A1_VARIANTS entry: $v (use core,aux,all,rw,filter_on,filter_off)" >&2
    exit 1
  fi
  # RW is not a rollout of its own — it is the built ALL pair file with the
  # auction/negotiation subset upsampled 4x (paper §3.1). Force ALL to build and
  # derive RW from it after the merge/build.
  if [[ "$v" == "rw" ]]; then
    RW_REQUESTED=1
    continue
  fi
  # Hypothesis B (paper §3): different builder, shared input, never sharded.
  if [[ -n "${HYP_B_FLAG[$v]+x}" ]]; then
    if [[ "$A1_NODE" != "merge" ]]; then
      src="${IN_DIR}/${HYP_B_INPUT}"
      [[ -f "$src" ]] || { echo "ERROR: missing trajectory input $src (run blind_rollout.sh first)" >&2; exit 1; }
    fi
    B_VARIANTS+=("$v")
    continue
  fi
  # `merge` reads the per-slice outputs, not the trajectory inputs.
  if [[ "$A1_NODE" != "merge" ]]; then
    src="${IN_DIR}/${FILE_BY_VARIANT[$v]}"
    [[ -f "$src" ]] || { echo "ERROR: missing trajectory input $src (run blind_rollout.sh first)" >&2; exit 1; }
  fi
  VARIANTS+=("$v")
done

# RW is derived from the built ALL file — make sure ALL is in the build set.
if (( RW_REQUESTED )) && [[ " ${VARIANTS[*]-} " != *" all "* ]]; then
  if [[ "$A1_NODE" != "merge" ]]; then
    src="${IN_DIR}/${FILE_BY_VARIANT[all]}"
    [[ -f "$src" ]] || { echo "ERROR: rw derives from 'all'; missing $src (run blind_rollout.sh first)" >&2; exit 1; }
  fi
  VARIANTS+=("all")
fi

# paper §3.1: RW = ALL pair file + auction/negotiation subset upsampled 4x.
derive_rw() {
  local all_file="${OUT_DIR}/${FILE_BY_VARIANT[all]}"
  local rw_file="${OUT_DIR}/${FILE_BY_VARIANT[rw]}"
  [[ -s "$all_file" ]] || { echo "ERROR: [rw] needs a built $all_file" >&2; return 1; }
  echo "[rw] deriving from ALL: auction/negotiation pairs upsampled ${A1_RW_FACTOR:-4}x"
  python data/alpha-beta/noleakage-frontier/build_pairs.py --upsample-only \
    --upsample-factor "${A1_RW_FACTOR:-4}" \
    --input "$all_file" --output "$rw_file" 2>&1 | sed -u "s/^/[rw] /"
  echo "[rw] -> ${rw_file} ($(wc -l < "$rw_file") pairs)"
}

run_build_pairs() {  # <input> <output> <log-prefix> <gpu> <log-file> <dedup> [max-pairs]
  local extra=(--dedup "${6:-none}")
  [[ -n "${7:-}" ]] && extra+=(--max-pairs "$7")
  CUDA_VISIBLE_DEVICES="$4" PYTHONUNBUFFERED=1 python data/alpha-beta/noleakage-frontier/build_pairs.py \
    --input "$1" \
    --output "$2" \
    --provider teacher \
    --teacher-model "$TEACHER" \
    --teacher-max-new-tokens "$TEACHER_TOKENS" \
    --counterfactual-mode "$MODE" \
    --max-invalid-fraction "$MAX_INVALID_FRACTION" \
    --allow-empty \
    "${extra[@]}" \
    2>&1 | sed -u "s/^/$3 /" | tee "$5"
}

# Hypothesis B (paper §3) — reasoning-action coupling ablation. CPU-only and
# deterministic (no teacher model, no GPU), so it runs directly, unsharded.
run_hyp_b() {  # <variant>
  local v="$1"
  local src="${IN_DIR}/${HYP_B_INPUT}"
  local dst="${OUT_DIR}/${FILE_BY_VARIANT[$v]}"
  local key="HB_MAX_PAIRS_${v^^}" cap extra=(--dedup "${HB_DEDUP:-context}")
  cap="${!key:-${HB_MAX_PAIRS:-}}"
  [[ -n "$cap" ]] && extra+=(--max-pairs "$cap")
  echo "[$v] Hypothesis B (coupling filter): ${src} -> ${dst}"
  PYTHONUNBUFFERED=1 python data/b-hypothesis/build_pairs.py \
    --input "$src" \
    --output "$dst" \
    "${HYP_B_FLAG[$v]}" \
    "${extra[@]}" \
    2>&1 | sed -u "s/^/[$v] /" | tee "${OUT_DIR}/logs/${v}.log"
}

# ---------------------------------------------------------------- merge
if [[ "$A1_NODE" == "merge" ]]; then
  shopt -s nullglob
  for v in "${VARIANTS[@]}"; do
    slices=("$SHARD_DIR/${v}.w"[0-9]*.jsonl)   # w0..w<WORLD-1>, including 2-digit
    ((${#slices[@]})) || { echo "ERROR: no slices for '$v' in $SHARD_DIR (run nodes 1..$NODES first)" >&2; exit 1; }
    dst="${OUT_DIR}/${FILE_BY_VARIANT[$v]}"
    cat "${slices[@]}" > "$dst"
    raw="$(wc -l < "$dst")"
    # slices could not see the whole pool; de-duplicate + cap the merged file now
    cap="$(variant_max_pairs "$v")"
    dedup_mode="$(variant_dedup "$v")"
    python data/alpha-beta/noleakage-frontier/build_pairs.py --dedup-only \
      --dedup "$dedup_mode" ${cap:+--max-pairs "$cap"} \
      --input "$dst" --output "$dst" 2>&1 | sed -u "s/^/[$v] /"
    echo "[$v] merged ${#slices[@]} slices (${raw} raw) -> ${dst} ($(wc -l < "$dst") pairs)"
    # The single merged file is the deliverable; drop the per-worker slices unless
    # A1_KEEP_SHARDS is set. (Slice .input files are already removed after each
    # worker consumes them.)
    [[ -n "${A1_KEEP_SHARDS:-}" ]] || rm -f "${slices[@]}" "$SHARD_DIR/${v}.w"[0-9]*.input
  done
  [[ -n "${A1_KEEP_SHARDS:-}" ]] || rmdir "$SHARD_DIR" 2>/dev/null || true
  # RW is derived from the just-merged ALL file (paper §3.1).
  if (( RW_REQUESTED )); then derive_rw; fi
  # Hypothesis B (paper §3) variants are unsharded — node 1 wrote them straight
  # to $OUT_DIR, so there is nothing to concatenate, just report their state.
  for v in "${B_VARIANTS[@]}"; do
    out="${OUT_DIR}/${FILE_BY_VARIANT[$v]}"
    if [[ -s "$out" ]]; then
      echo "[$v] Hypothesis B already final -> $out ($(wc -l < "$out") pairs)"
    else
      echo "WARNING: [$v] missing $out — run './data/alpha-beta/algorithm1.sh 1' first" >&2
    fi
  done
  exit 0
fi

# The GPU check only matters when there are A+β variants to paraphrase; a
# Hypothesis-B-only run (paper §3) is CPU-only.
if ((${#VARIANTS[@]})); then
  nvidia-smi -L
  python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda is unavailable'"
fi

# ---------------------------------------------------------------- 3-node x 3-GPU slices
if [[ -n "$A1_NODE" ]]; then
  node="$A1_NODE"
  echo "Algorithm 1 lines 3-9 | node ${node}/${NODES} | mode=${MODE} | teacher=${TEACHER} | world=${WORLD}"
  PIDS=()
  for g in $(seq 0 $((GPUS_PER_NODE - 1))); do
    ((${#VARIANTS[@]})) || break        # nothing to slice (Hypothesis-B-only run)
    w=$(( (node - 1) * GPUS_PER_NODE + g ))
    (
      for v in "${VARIANTS[@]}"; do
        src="${IN_DIR}/${FILE_BY_VARIANT[$v]}"
        shard_src="${SHARD_DIR}/${v}.w${w}.input"
        shard_out="${SHARD_DIR}/${v}.w${w}.jsonl"
        if [[ -f "$shard_out" ]]; then
          echo "[$v][w$w GPU $g] already completed; skipping ${shard_out}"
          continue
        fi
        awk -v w="$w" -v world="$WORLD" 'NF { if (n++ % world == w) print }' "$src" > "$shard_src"
        if [[ ! -s "$shard_src" ]]; then
          : > "$shard_out"      # fewer episodes than workers; nothing for this slice
          continue
        fi
        echo "[$v][w$w GPU $g] $(wc -l < "$shard_src") episodes -> ${shard_out}"
        # a slice can't see the whole pool; build it undeduped, dedup+cap at merge
        run_build_pairs "$shard_src" "$shard_out" "[$v][w$w GPU $g]" "$g" "${OUT_DIR}/logs/${v}.w${w}.log" none ""
        rm -f "$shard_src"     # slice input consumed; only the .jsonl is kept, until merge
      done
    ) &
    PIDS+=("$!")
  done
  FAILED=0
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && { wait "$pid" || FAILED=1; }; done
  ((FAILED)) && { echo "ERROR: node ${node}: a slice failed" >&2; exit 1; }

  # Hypothesis B (paper §3): unsharded, so build it once — on node 1 only.
  if ((${#B_VARIANTS[@]})); then
    if [[ "$node" == "1" ]]; then
      for v in "${B_VARIANTS[@]}"; do run_hyp_b "$v" || { echo "ERROR: [$v] failed" >&2; exit 1; }; done
    else
      echo "node ${node}: skipping Hypothesis B variants (${B_VARIANTS[*]}) — node 1 builds those"
    fi
  fi

  echo "node ${node}: slices done — run './data/alpha-beta/algorithm1.sh merge' after all ${NODES} nodes finish"
  exit 0
fi

# ---------------------------------------------------------------- single node, one variant per GPU
echo "Algorithm 1 lines 3-9 | mode=${MODE} (flag h) | teacher=${TEACHER} | ${IN_DIR} -> ${OUT_DIR}"
PIDS=()
for i in "${!VARIANTS[@]}"; do
  v="${VARIANTS[i]}"
  gpu="${GPUS[i % ${#GPUS[@]}]//[[:space:]]/}"
  src="${IN_DIR}/${FILE_BY_VARIANT[$v]}"
  dst="${OUT_DIR}/${FILE_BY_VARIANT[$v]}"
  echo "[$v] GPU ${gpu}: ${src} -> ${dst}"
  run_build_pairs "$src" "$dst" "[$v][GPU $gpu]" "$gpu" "${OUT_DIR}/logs/${v}.log" \
    "$(variant_dedup "$v")" "$(variant_max_pairs "$v")" &
  PIDS+=("$!")
done

# Hypothesis B (paper §3): CPU-only, run alongside the GPU variants.
for v in "${B_VARIANTS[@]}"; do
  run_hyp_b "$v" &
  PIDS+=("$!")
done

FAILED=0
for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && { wait "$pid" || FAILED=1; }; done
((FAILED)) && { echo "ERROR: one or more variants failed" >&2; exit 1; }
# RW is derived from the built ALL file (paper §3.1).
if (( RW_REQUESTED )); then derive_rw; fi
echo "Algorithm 1 pairs complete: ${OUT_DIR}"
