#!/usr/bin/env python3
r"""
Algorithm 1  A+beta data construction (per episode)              paper Sec. 2.6
-----------------------------------------------------------------------------
 Input: game G, opponent policy pi_opp, frontier paraphraser LLM_front,
        horizon-aware flag h
 2:  blind trajectory tau_blind = (x, rho_blind, a_blind), LLM_front as
     player 0  --  ALREADY ROLLED OUT by data/blind-rollout/ (README step 1);
     this script reads it from A1_INPUT_DIR and starts at line 3.
 3:  for round t = 0, ..., T-1 do
 4:      a*  <-  BR_0(a1_t)                                       > Eq. 1
 5:      if a* != a0,blind_t then
 6:          R_cf  <-  horizon-aware Eq. 3 if h else Eq. 2
 7:          if R_cf > R(tau_blind) then
 8:              rho*_t  <-  LLM_front(no-leak prompt, a*)
 9:              emit pair (x_t, rho*_t (+) a* > rho_blind_t (+) a0,blind_t)
10:          end if
11:      end if
12:  end for

Each line calls the script that implements it, via build_pairs.py (``bp``), which
itself loads:
    line 4     solver-pinned/best_action.py         bp.solver.main            (Eq. 1)
    lines 6-7  counterfactual-horizon/{fixed,optimal}_continuation.py
               folded into bp._collect_candidates (matrix decode + baseline +
               is_reconstructible + Eq. 2/3 + the R_cf > R(tau_blind) test)
    line 8     noleakage-frontier/frontier_prompt.py + the LLM_front teacher;
               one-shot games take the Sec. 3.1 solver-labelled path (no teacher)
    line 9     bp._pair_from_candidate  -> the DPO pair dict

Usage (via algorithm1.sh, which sets up conda/CUDA first):
    ./algorithm1.sh            build every A1_VARIANTS mix here, then merge
    ./algorithm1.sh <N>        run only node N of an A1_NODES allocation
    ./algorithm1.sh merge      merge node shards, then compose AUX/ALL/RW
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("a1_processor", _HERE / "processor.py")
processor = importlib.util.module_from_spec(_spec)
sys.modules["a1_processor"] = processor
_spec.loader.exec_module(processor)

bp = processor.bp          # noleakage-frontier/build_pairs.py (loads solver + counterfactual + frontier)
env = processor.env


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _own(row: dict) -> list:
    return row.get("own_actions", row.get("agent_actions"))


class _LazyTeacher:
    """LLM_front (paper Sec. 2.3): loaded once, on the first repeated-game round."""

    def __init__(self) -> None:
        self._fn = None

    def __call__(self):
        if self._fn is None:
            model = processor.teacher_model()
            use_4bit = env("A1_TEACHER_4BIT", "0") == "1"
            _log(f"[teacher] loading LLM_front ({model}, "
                 f"{'4bit' if use_4bit else 'bf16'}) -- one-time, ~30-60s ...")
            self._fn = bp.teacher_paraphraser(
                model,
                max_new_tokens=int(env("A1_TEACHER_MAX_NEW_TOKENS", "384")),
                use_4bit=use_4bit,
            )
            import torch
            if torch.cuda.is_available():
                _log(f"[teacher] ready -- {torch.cuda.get_device_name(0)}, "
                     f"{torch.cuda.memory_allocated() / 1e9:.1f} GB VRAM in use")
            else:
                _log("[teacher] ready -- WARNING: no CUDA, running on CPU "
                     "(a paraphrase will take minutes, not ~20s)")
        return self._fn


def algorithm1(row, *, mode, teacher, seen, skipped, stats, report, out) -> None:
    """Lines 3-12 for one precomputed blind trajectory."""
    own, opp = _own(row), row["opponent_actions"]
    # lines 6-7: the Sec. 2.4 counterfactual horizon filter (Eq. 2/3 + the
    # R_cf > R(tau_blind) test) lives in _collect_candidates -- a round is
    # here only if a* flips it AND the certified counterfactual beats tau_blind.
    certified = {
        c["round_index"]: c
        for c in bp._collect_candidates(row, counterfactual_mode=mode)
    }
    for t in range(len(own)):                                    # 3  for round t = 0 .. T-1
        a_star = bp.solver.main(row, opp[t])                     # 4  a* <- BR_0(a1_t)   (Eq. 1)
        if a_star == own[t]:                                     # 5  a* != a0,blind_t
            continue
        if t not in certified:                                   # 6-7 R_cf > R(tau_blind)
            continue
        cand = certified[t]
        key = (cand["prompt"], str(cand["solver_action"]))
        if key not in seen:                                      # replayed position -> paraphrase once
            paraphrase = None
            if not cand.get("one_shot"):
                _log(f"[{stats['variant']}] generating episode {stats['ep']}, "
                     f"round {t + 1}/{len(own)} ...")
                paraphrase = teacher()                           # 8  rho*_t <- LLM_front
            seen[key] = bp._pair_from_candidate(
                cand, paraphrase, on_invalid="skip", skipped=skipped
            )
            stats["built"] += 1
            report()
        pair = seen[key]
        if pair is not None:                                     # 9  emit pair
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            out.flush()
            stats["emitted"] += 1


def _resume(shard: Path, progress: Path) -> tuple[int, dict]:
    """Read how far a preempted run got: (episodes done, {context: pair} cache).

    ``progress`` holds the count of fully-written episodes; the shard's pairs
    seed ``seen`` so finished decision contexts are not re-paraphrased. A shard
    that fails to parse is treated as not started (redo the whole variant)."""
    if not (shard.exists() and progress.exists()):
        return 0, {}
    try:
        done = max(0, int(progress.read_text().strip() or "0"))
        seen = {}
        for line in shard.open(encoding="utf-8"):
            if line.strip():
                p = json.loads(line)
                seen[(p["prompt"], str(p["provenance"]["solver_action"]))] = p
        return done, seen
    except (ValueError, KeyError, json.JSONDecodeError):
        return 0, {}


def _run_mix(variant: str, node: int, nodes: int) -> None:
    mode = processor.counterfactual_mode()
    expected_model = processor.teacher_model()
    require_model = env("A1_REQUIRE_FRONTIER_MODEL", "1") != "0"
    max_invalid = float(env("A1_MAX_INVALID_FRACTION", "0.2"))

    rows = processor.blind_rows(variant)
    indices = processor.node_slice(len(rows), node, nodes)
    shard = processor.shard_path(variant, node)
    progress = shard.with_suffix(".progress")

    done, seen = _resume(shard, progress)                        # survive gpu-preempt
    if done >= len(indices) and progress.exists():
        _log(f"[{variant}] node {node}/{nodes}: already complete ({done} episodes) -- skip")
        return
    if done:
        _log(f"[{variant}] node {node}/{nodes}: resume from episode {done + 1}/{len(indices)} "
             f"({len(seen)} contexts cached)")
    else:
        _log(f"[{variant}] node {node}/{nodes}: {len(indices)}/{len(rows)} episodes")

    teacher = _LazyTeacher()
    skipped: list = []
    stats = {"variant": variant, "built": 0, "emitted": len(seen), "ep": done}
    started = time.monotonic()

    def report() -> None:
        # Each paraphrase is a full teacher generation (~10-40s), so log every
        # one -- batching to every 10th makes minutes of real work look hung.
        _log(f"[{variant}] {stats['built']} paraphrased, {stats['emitted']} pairs, "
             f"{len(skipped)} dropped -- episode {stats['ep']}/{len(indices)}, "
             f"{time.monotonic() - started:.0f}s")

    with shard.open("a" if done else "w", encoding="utf-8") as out:
        for i in range(done, len(indices)):                      # 2  each precomputed tau_blind
            stats["ep"] = i + 1
            # Heartbeat: most episodes certify no solver flip and never reach the
            # teacher, so without this the run is silent while it scans them.
            if stats["ep"] == done + 1 or stats["ep"] % 10 == 0:
                _log(f"[{variant}] scanning episode {stats['ep']}/{len(indices)} -- "
                     f"{stats['built']} paraphrased, {stats['emitted']} pairs, "
                     f"{time.monotonic() - started:.0f}s")
            row = rows[indices[i]]
            if require_model and row.get("frontier_model") != expected_model:
                sys.exit(f"episode {indices[i]}: blind rollout model "
                         f"{row.get('frontier_model')!r} != paraphraser model {expected_model!r}")
            algorithm1(row, mode=mode, teacher=teacher, seen=seen, skipped=skipped,
                       stats=stats, report=report, out=out)
            progress.write_text(str(i + 1))                      # episode i is fully written

    progress.write_text(str(len(indices)))                       # variant complete
    total = stats["emitted"] + len(skipped)
    if skipped:
        reasons: dict[str, int] = {}
        for item in skipped:
            for err in item["errors"]:
                reasons[err] = reasons.get(err, 0) + 1
        summary = "; ".join(f"{n}x {m}" for m, n in sorted(reasons.items(), key=lambda kv: -kv[1]))
        fraction = len(skipped) / max(total, 1)
        if not done and fraction > max_invalid:
            sys.exit(f"[{variant}] {len(skipped)}/{total} paraphrases failed structural "
                     f"validation ({fraction:.0%} > {max_invalid:.0%}) -- {summary}")
        _log(f"[{variant}] dropped {len(skipped)} rounds this run -- {summary}")
    _log(f"[{variant}] {stats['emitted']} pairs -> {shard}")


def main(argv: list[str]) -> int:
    arg = argv[0] if argv else "run"
    if arg == "merge":
        processor.merge()
        return 0
    node = 1 if arg == "run" else int(arg)
    nodes = int(env("A1_NODES", "1"))

    for mix in processor.variants():                             # core / aux / all from A1_VARIANTS
        _run_mix(mix, node, nodes)

    if arg == "run":                                             # single machine: assemble the mixes
        processor.merge()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
