#!/usr/bin/env python3
"""Run plumbing for Algorithm 1 (paper Sec. 2.6) -- everything that is NOT the
per-episode pseudocode.

``algorithm1.py`` owns the transcription (the ``for round t`` loop, calling the
solver / counterfactual / frontier scripts directly). This module gives it:

    env(key, default)          .env-backed config lookup
    blind_rows(variant)        the precomputed blind-rollout pool (README step 1)
    node_slice(n, node, nodes) line-shard indices
    shard_path(variant, node)  where a node writes its pairs
    variants()                 the A1_VARIANTS that are per-episode mixes
    merge()                    concat shards -> global dedup -> compose AUX/ALL -> RW

The game maths lives in noleakage-frontier/build_pairs.py, exposed here as ``bp``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config.py").is_file())
_BUILD = ROOT / "data/alpha-beta/noleakage-frontier/build_pairs.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bp = _load(_BUILD, "a1_build_pairs")            # torch-free until teacher_paraphraser()

PER_EPISODE = ("core", "aux", "all")
FILES = {v: f"a_beta_{v}.jsonl" for v in (*PER_EPISODE, "rw")}


# ------------------------------------------------------------------ env config
def env(key: str, default: str = "") -> str:
    return os.environ.get(key) or default


def _indir() -> Path:
    return ROOT / env("A1_INPUT_DIR", "data/blind-rollout/result")


def _outdir() -> Path:
    return ROOT / env("A1_OUTPUT_DIR", "data/alpha-beta/result")


def _shard_dir() -> Path:
    return _outdir() / "shards"


def _requested() -> list[str]:
    want = [v.strip() for v in env("A1_VARIANTS", "core,aux,all,rw").split(",") if v.strip()]
    bad = [v for v in want if v not in FILES]
    if bad:
        sys.exit(f"unknown A1_VARIANTS: {', '.join(bad)}")
    return want


def counterfactual_mode() -> str:
    m = env("A1_MODE", "horizon-aware")
    if m not in ("horizon-aware", "fixed"):
        sys.exit("A1_MODE must be horizon-aware or fixed")
    return m


def teacher_model() -> str:
    return env("A1_TEACHER_MODEL", env("TEACHER_MODEL", "Qwen/Qwen2.5-7B-Instruct"))


def _dedup_for(variant: str) -> str:
    return env(f"A1_DEDUP_{variant.upper()}", env("A1_DEDUP", "context"))


def _cap(variant: str) -> int | None:
    value = env(f"A1_MAX_PAIRS_{variant.upper()}", env("A1_MAX_PAIRS"))
    return int(value) if value else None


def variants() -> list[str]:
    """The requested A1_VARIANTS that are per-episode mixes (core/aux/all)."""
    return [v for v in _requested() if v in PER_EPISODE]


# ------------------------------------------------------------- per-node inputs
def blind_rows(variant: str) -> list[dict]:
    """The precomputed blind-rollout pool for ``variant`` (pseudocode line 2 is a
    separate stage: data/blind-rollout/, README step 1)."""
    source = _indir() / FILES[variant]
    if not source.is_file():
        sys.exit(f"missing blind-rollout input: {source}")
    return [json.loads(x) for x in source.open(encoding="utf-8") if x.strip()]


def node_slice(n: int, node: int, nodes: int) -> list[int]:
    return list(range(n))[node - 1 :: nodes]


def shard_path(variant: str, node: int) -> Path:
    _shard_dir().mkdir(parents=True, exist_ok=True)
    return _shard_dir() / f"{variant}.n{node}.jsonl"


# ---------------------------------------------------------------- assembly
def _compose(per_episode: list[str]) -> None:
    """paper Sec. 3.1: resample AUX (= CORE + auction/negotiation) and ALL to the
    published pair counts. Skipped when the caps are unset."""
    if not {"core", "aux", "all"}.issubset(per_episode):
        return
    aux_cap, all_cap = _cap("aux"), _cap("all")
    if not aux_cap or not all_cap:
        print(
            "[compose] A1_MAX_PAIRS_AUX / A1_MAX_PAIRS_ALL unset -- keeping the raw "
            "AUX/ALL merges (no stratified resample)",
            file=sys.stderr,
        )
        return
    out = _outdir()
    fams = bp.RW_UPSAMPLE_FAMILIES
    bp.compose_stratified_mix(
        out / FILES["aux"], out / FILES["aux"],
        total_count=aux_cap, special_count=int(env("A1_AUX_SPECIAL_PAIRS", "110")),
        special_families=fams, base=out / FILES["core"],
    )
    bp.compose_stratified_mix(
        out / FILES["all"], out / FILES["all"],
        total_count=all_cap, special_count=int(env("A1_ALL_SPECIAL_PAIRS", "137")),
        special_families=fams,
    )


def _derive_rw() -> None:
    """paper Sec. 3.1 RW mix: ALL with the auction/negotiation subset upsampled 4x."""
    out = _outdir()
    bp.upsample_pairs_file(
        out / FILES["all"], out / FILES["rw"], factor=int(env("A1_RW_FACTOR", "4"))
    )
    cap = _cap("rw")
    if cap:
        bp.dedup_pairs_file(out / FILES["rw"], out / FILES["rw"], dedup="none", max_pairs=cap)


def merge() -> None:
    """Concat node shards -> global dedup -> compose AUX/ALL -> derive RW."""
    nodes = int(env("A1_NODES", "1"))
    requested = _requested()
    per_episode = [v for v in requested if v in PER_EPISODE]
    shards = _shard_dir()

    for v in per_episode:
        parts = sorted(shards.glob(f"{v}.n*.jsonl"))
        if len(parts) != nodes:
            sys.exit(f"[{v}] found {len(parts)} shard(s), expected A1_NODES={nodes}")
        n_rows = len(blind_rows(v))
        for node in range(1, nodes + 1):
            prog = shards / f"{v}.n{node}.progress"
            if not prog.is_file():
                print(f"[{v}] node {node}: no .progress marker (pre-resume run?) -- "
                      f"assuming its shard is complete", file=sys.stderr)
                continue
            want = len(list(range(n_rows))[node - 1 :: nodes])
            got = int(prog.read_text().strip() or "0")
            if got != want:
                sys.exit(f"[{v}] node {node} incomplete: {got}/{want} episodes done -- "
                         f"wait for it, or re-run `algorithm1.py {node}` to resume")
        dst = _outdir() / FILES[v]
        raw = 0
        with dst.open("w", encoding="utf-8") as out:
            for part in parts:
                for row in part.open(encoding="utf-8"):
                    if row.strip():
                        out.write(row if row.endswith("\n") else row + "\n")
                        raw += 1
        cap = None if v in ("aux", "all") else _cap(v)
        bp.dedup_pairs_file(dst, dst, dedup=_dedup_for(v), max_pairs=cap)
        print(f"[{v}] {raw} raw shard rows merged -> {dst}", file=sys.stderr)

    _compose(per_episode)
    if "rw" in requested:
        _derive_rw()

    if not env("A1_KEEP_SHARDS"):
        for part in (*shards.glob("*.jsonl"), *shards.glob("*.progress")):
            part.unlink()
        try:
            shards.rmdir()
        except OSError:
            pass


if __name__ == "__main__":                       # `python processor.py merge` for convenience
    if sys.argv[1:] == ["merge"]:
        merge()
    elif sys.argv[1:] == ["variants"]:
        print(" ".join(variants()))
    else:
        sys.exit("usage: processor.py [merge|variants]  (Algorithm 1 itself: algorithm1.py)")
