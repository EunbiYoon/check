#!/usr/bin/env python3
"""Merge per-game eval shards (eval_gpu_parallel part_* dirs) into one metrics dict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ALL_GAMES, ID_GAMES, HO_GAMES, EPISODES_PER_ENV
from eval.metric.metrics import EvalMetrics, metrics_to_dict
from eval.table.run_table import save_json


def _load_shard_rows(shard_dir: Path, variant: str) -> dict | None:
    for name in ("table1.json", f"table1_{variant}.json", "table1_base.json"):
        p = shard_dir / "tables" / name
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data.get("rows", {})
        if variant in rows:
            return rows[variant]
        if len(rows) == 1:
            return next(iter(rows.values()))
    return None


def merge_shards(shard_root: Path, variant: str) -> dict:
    per_game_cr: dict[str, list[float]] = {g: [] for g in ALL_GAMES}
    per_axis_cr: dict[str, list[float]] = {}
    per_game_fc: dict[str, list[float]] = {g: [] for g in ALL_GAMES}
    per_game_ac: dict[str, list[float]] = {g: [] for g in ALL_GAMES}
    per_game_exp: dict[str, list[float]] = {g: [] for g in ALL_GAMES}

    shards = sorted(shard_root.parent.glob(f"{shard_root.name}_part_*"))
    if not shards:
        raise FileNotFoundError(f"No shard dirs matching {shard_root.name}_part_*")

    for sd in shards:
        row = _load_shard_rows(sd, variant)
        if not row:
            print(f"WARNING: no rows for variant={variant} in {sd}", file=sys.stderr)
            continue
        for g in ALL_GAMES:
            v = row.get("per_game_cr", {}).get(g)
            if v is not None:
                per_game_cr[g].append(float(v))
            for key, dest in (
                ("per_game_fc", per_game_fc),
                ("per_game_ac", per_game_ac),
                ("per_game_exploitability", per_game_exp),
            ):
                val = row.get(key, {}).get(g)
                if val is not None:
                    dest[g].append(float(val))
        for axis, val in row.get("per_axis_cr", {}).items():
            if val is not None:
                per_axis_cr.setdefault(axis, []).append(float(val))

    def _mean(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    def _pool_rate(per_game: dict[str, list[float]], games: tuple[str, ...]) -> float:
        num = den = 0.0
        for g in games:
            vals = per_game.get(g) or []
            if not vals:
                continue
            rate = sum(vals) / len(vals)
            w = EPISODES_PER_ENV * len(vals)
            num += rate * w
            den += w
        return num / den if den else 0.0

    cr_map = {g: _mean(per_game_cr[g]) for g in ALL_GAMES}
    cr_id_vals = [cr_map[g] for g in ID_GAMES if cr_map[g] is not None]
    cr_ho_vals = [cr_map[g] for g in HO_GAMES if cr_map[g] is not None]

    fc_map = {g: _mean(per_game_fc[g]) for g in ALL_GAMES}
    ac_map = {g: _mean(per_game_ac[g]) for g in ALL_GAMES}

    m = EvalMetrics(
        fc_id=_pool_rate(per_game_fc, ID_GAMES),
        fc_ho=_pool_rate(per_game_fc, HO_GAMES),
        ac_id=_pool_rate(per_game_ac, ID_GAMES),
        cr_id=sum(cr_id_vals) / len(cr_id_vals) if cr_id_vals else 0.0,
        cr_ho=sum(cr_ho_vals) / len(cr_ho_vals) if cr_ho_vals else 0.0,
        per_game_cr=cr_map,
        per_axis_cr={k: _mean(v) for k, v in per_axis_cr.items()},
        per_game_fc=fc_map,
        per_game_ac=ac_map,
        per_game_exploitability={g: _mean(per_game_exp[g]) for g in ALL_GAMES},
    )
    return metrics_to_dict(m)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Merge eval_gpu_parallel game shards")
    p.add_argument("--shard-root", type=Path, required=True, help="Base run dir (without _part_*)")
    p.add_argument("--variant", required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output metrics json (default: <shard-root>/../20260630_180322/metrics/<variant>.json)",
    )
    args = p.parse_args(argv)

    merged = merge_shards(args.shard_root.resolve(), args.variant)
    out = args.out
    if out is None:
        out = args.shard_root.parent / "20260630_180322" / "metrics" / f"{args.variant}.json"
    save_json(out, merged)
    print(f"Merged {args.variant} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
