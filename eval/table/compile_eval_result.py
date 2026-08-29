"""Regenerate eval/runs/<run_id>/result.md from suite.json (no GPU rollout).

  python -m eval.table.compile_eval_result
  python -m eval.table.compile_eval_result --suite runs/20260621_120000/eval/tables/suite.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from config import EVAL_DIR, LORA_DIR, RUNS_DIR
from eval.table.paths import write_latest_pointer
from eval.table.run_table import save_json
from eval.table.run_eval_suite import _training_manifest, _write_table_jsons, build_result_md


def _resolve_suite_path(suite_arg: Path | None) -> Path:
    if suite_arg and suite_arg.exists():
        return suite_arg
    latest = RUNS_DIR / "latest.json"
    if latest.exists():
        data = json.loads(latest.read_text(encoding="utf-8"))
        return ROOT / data["path"] / "suite.json"
    runs = EVAL_DIR / "runs"
    if runs.is_dir():
        candidates = sorted(runs.iterdir(), reverse=True)
        for d in candidates:
            p = d / "suite.json"
            if p.exists():
                return p
    return EVAL_DIR / "runs" / "missing" / "suite.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=None)
    args = parser.parse_args()

    suite_path = _resolve_suite_path(args.suite)
    if not suite_path.exists():
        print(f"Missing {suite_path}; run eval/table/run_eval_suite.ps1 first.")
        return 1

    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    run_dir = suite_path.parent.resolve()
    rows = suite.get("rows", {})
    ok_variants = [v for v in suite.get("variants", []) if v in rows and "error" not in rows[v]]
    ckpt = Path(suite.get("hyperparams", {}).get("checkpoint_dir", LORA_DIR))
    manifest = suite.get("table5_manifest") or _training_manifest(ckpt)
    run_id = suite.get("run_id", run_dir.name)

    _write_table_jsons(run_dir, suite)
    md = build_result_md(
        suite=suite,
        rows=rows,
        ok_variants=ok_variants,
        manifest=manifest,
        run_dir=run_dir,
        run_id=run_id,
    )
    (run_dir / "result.md").write_text(md, encoding="utf-8")
    from eval.table.build_latex_md import write_latex_md

    latex_path = write_latex_md(run_dir, suite, ok_variants=ok_variants)
    save_json(run_dir / "suite.json", suite)
    write_latest_pointer(run_dir)
    print(f"Wrote {run_dir / 'result.md'}")
    print(f"Wrote {latex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
