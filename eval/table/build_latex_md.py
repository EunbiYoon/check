"""Build latex.md (Tables 1–7 as LaTeX) from eval suite.json.

  python -m eval.table.build_latex_md
  python -m eval.table.build_latex_md --suite runs/<id>/eval/tables/suite.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    ALL_GAMES,
    EVAL_DIR,
    GAME_ABBREV,
    PAPER_TABLE1,
    PAPER_TABLE3,
    PAPER_TABLE5,
    PAPER_TABLE6,
    PAPER_TABLE7,
    VARIANT_LABELS,
)
from eval.table.run_title import build_run_title, latex_run_note

_EPS = 1e-9


def _best_mask(values: list[float | None], *, higher_is_better: bool = True) -> list[bool]:
    """Return per-index mask: True where value ties for best in the column/row."""
    nums: list[tuple[int, float]] = []
    for i, v in enumerate(values):
        if v is None:
            continue
        nums.append((i, float(v)))
    mask = [False] * len(values)
    if not nums:
        return mask
    target = max(v for _, v in nums) if higher_is_better else min(v for _, v in nums)
    for i, v in nums:
        if abs(v - target) <= _EPS:
            mask[i] = True
    return mask


def _tex_num(x: float | None, *, signed: bool = False, bold: bool = False) -> str:
    s = _fmt_num(x, signed=signed)
    if bold and s != "---":
        return rf"\textbf{{{s}}}"
    return s


def _fmt_cr(x: float | None) -> str:
    if x is None:
        return "---"
    if x >= 0:
        return f"+{x:.2f}" if abs(x) < 100 else f"{x:.2f}"
    return f"{x:.2f}"


def _fmt_num(x: float | None, *, signed: bool = False) -> str:
    if x is None:
        return "---"
    if signed:
        return _fmt_cr(x)
    return f"{x:.2f}"


def _tex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for ch, esc in repl.items():
        out = out.replace(ch, esc)
    return out


def _variant_label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant)


def _table1_latex(table1: dict, variants: list[str], hp: dict) -> str:
    note = latex_run_note(hp, kind="local")
    cols = "l" + "r" * 4
    metric_keys = ("fc_id", "fc_ho", "cr_id", "cr_ho")
    bold_by_metric = {
        key: _best_mask([table1.get(v, {}).get(key) for v in variants])
        for key in metric_keys
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Headline results — {note}. Best per column in \textbf{{bold}}.}}",
        r"\label{tab:headline}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        r"Model & $f_c$(ID) & $f_c$(HO) & CR(ID) & CR(HO) \\",
        r"\midrule",
    ]
    for i, v in enumerate(variants):
        m = table1.get(v, {})
        lines.append(
            f"{_tex_escape(_variant_label(v))} & "
            f"{_tex_num(m.get('fc_id'), bold=bold_by_metric['fc_id'][i])} & "
            f"{_tex_num(m.get('fc_ho'), bold=bold_by_metric['fc_ho'][i])} & "
            f"{_tex_num(m.get('cr_id'), signed=True, bold=bold_by_metric['cr_id'][i])} & "
            f"{_tex_num(m.get('cr_ho'), signed=True, bold=bold_by_metric['cr_ho'][i])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _table1_paper_latex() -> str:
    note = latex_run_note({}, kind="paper")
    variants = [v for v in PAPER_TABLE1 if v != "haiku"]
    cols = "l" + "r" * 4
    metric_keys = ("fc_id", "fc_ho", "cr_id", "cr_ho")
    bold_by_metric = {
        key: _best_mask([PAPER_TABLE1[v][key] for v in variants])
        for key in metric_keys
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Paper reference — {note}. Best per column in \textbf{{bold}}.}}",
        r"\label{tab:headline-paper}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        r"Model & $f_c$(ID) & $f_c$(HO) & CR(ID) & CR(HO) \\",
        r"\midrule",
    ]
    for i, v in enumerate(variants):
        m = PAPER_TABLE1[v]
        lines.append(
            f"{_tex_escape(_variant_label(v))} & "
            f"{_tex_num(m['fc_id'], bold=bold_by_metric['fc_id'][i])} & "
            f"{_tex_num(m['fc_ho'], bold=bold_by_metric['fc_ho'][i])} & "
            f"{_tex_num(m['cr_id'], signed=True, bold=bold_by_metric['cr_id'][i])} & "
            f"{_tex_num(m['cr_ho'], signed=True, bold=bold_by_metric['cr_ho'][i])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _table2_latex(table2: dict, variants: list[str]) -> str:
    cols = "l" + "r" * len(variants)
    header = "Game & " + " & ".join(_tex_escape(v) for v in variants) + r" \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-game cumulative reward (mean over episodes). Best per row in \textbf{bold}.}",
        r"\label{tab:per-game-cr}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for game in ALL_GAMES:
        vals = [table2.get(v, {}).get(game) for v in variants]
        bold = _best_mask(vals)
        row = [_tex_escape(game)]
        for i, v in enumerate(variants):
            row.append(_tex_num(vals[i], bold=bold[i]))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _table3_latex(table3: dict, variants: list[str]) -> str:
    axes = sorted({k for v in table3.values() for k in v})
    if not axes:
        axes = list(PAPER_TABLE3.keys())
    cols = "l" + "r" * len(variants)
    header = "Axis & " + " & ".join(_tex_escape(v) for v in variants) + r" \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-opponent-axis cumulative reward. Best per row in \textbf{bold}.}",
        r"\label{tab:per-axis-cr}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for axis in axes:
        vals = [table3.get(v, {}).get(axis) for v in variants]
        bold = _best_mask(vals)
        row = [_tex_escape(axis)]
        for i in range(len(variants)):
            row.append(_tex_num(vals[i], signed=True, bold=bold[i]))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _table4_frontier_latex(table4: dict) -> str:
    labels = {
        "haiku": "Claude 3.5 Haiku",
        "gemma_27b": "Gemma 3 27B",
        "llama_70b": "Llama 3.1 70B",
        "llama_8b": "Llama 3.1 8B",
        "qwen_3b_base": "Qwen 2.5-3B base",
    }
    order = ("haiku", "gemma_27b", "llama_70b", "llama_8b", "qwen_3b_base")
    rows = table4.get("rows", {})
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Frontier benchmark: CR(ID), CR(HO), and mean CR vs.\ Tit-for-Tat (TFT-CR).}",
        r"\label{tab:frontier-cr}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & CR(ID) & CR(HO) & TFT-CR \\",
        r"\midrule",
    ]
    for model in order:
        row = rows.get(model, {})
        lines.append(
            " & ".join(
                [
                    _tex_escape(labels.get(model, model)),
                    _tex_num(row.get("cr_id"), signed=True),
                    _tex_num(row.get("cr_ho"), signed=True),
                    _tex_num(row.get("tft_cr"), signed=True),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _table4_latex(table4: dict, variants: list[str]) -> str:
    abbrevs = [GAME_ABBREV[g] for g in ALL_GAMES]
    cols = "l" + "r" * len(abbrevs)
    header = "Model & " + " & ".join(_tex_escape(a) for a in abbrevs) + r" \\"
    bold_by_game = {
        game: _best_mask([table4.get(v, {}).get(game) for v in variants])
        for game in ALL_GAMES
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Final-round reasoning--action coupling $f_c$ by environment. Best per column in \textbf{bold}.}",
        r"\label{tab:per-game-fc}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for vi, v in enumerate(variants):
        row = [_tex_escape(v)]
        fc_map = table4.get(v, {})
        for game in ALL_GAMES:
            row.append(_tex_num(fc_map.get(game), bold=bold_by_game[game][vi]))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _table5_coupling_latex(table5: dict) -> str:
    col_labels = table5.get("column_labels", {})
    cols = table5.get("columns", ("fc_id", "ac_id", "fc_ho"))
    rows = table5.get("rows", {})
    labels = table5.get("labels", {})
    order: list[str] = []
    for key in table5.get("sections", {}).get("trained_3b", ()):
        if key in rows:
            order.append(key)
    for key in table5.get("sections", {}).get("frontier", ()):
        if key in rows:
            order.append(key)

    trained_keys = set(table5.get("sections", {}).get("trained_3b", ()))
    bold = {c: _best_mask([rows.get(k, {}).get(c) for k in order if k in trained_keys]) for c in cols}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Reasoning-action coupling on the 12-env eval (paper\_v01 Table~5). Local eval only.}",
        r"\label{tab:coupling-summary}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & $f_c$(ID) & ac(ID) & $f_c$(HO) \\",
        r"\midrule",
    ]
    for ki, key in enumerate(order):
        row = rows.get(key, {})
        label = labels.get(key, key)
        is_trained = key in trained_keys
        cells = [
            _tex_num(row.get(c), bold=bold[c][ki] if is_trained and ki < len(bold[c]) else False)
            for c in cols
        ]
        lines.append(f"{_tex_escape(label)} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _table5_latex(manifest: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Training manifest: DPO pairs and checkpoint availability.}",
        r"\label{tab:training-manifest}",
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r"Variant & Pairs (paper) & Best step & Trained \\",
        r"\midrule",
    ]
    for name, spec in manifest.items():
        exists = "OK" if spec.get("exists") else "MISSING"
        step = spec.get("best_step")
        step_s = str(step) if step is not None else "---"
        lines.append(
            f"{_tex_escape(name)} & {spec.get('pairs', 0)} & {step_s} & {exists} \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Paper reference (eval loss):}} \\",
    ]
    paper_rows = list(PAPER_TABLE5.items())
    losses = [r["best_eval_loss"] for _, r in paper_rows if r["best_eval_loss"] is not None]
    loss_bold = _best_mask(losses, higher_is_better=False) if losses else []
    loss_idx = 0
    for v, r in paper_rows:
        step = r["best_step"] if r["best_step"] is not None else "---"
        if r["best_eval_loss"] is not None:
            loss_s = _tex_num(r["best_eval_loss"], bold=loss_bold[loss_idx])
            loss_idx += 1
        else:
            loss_s = "---"
        lines.append(f"{_tex_escape(v)} & {r['pairs']} & {step} & {loss_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _table6_latex(table6: dict) -> str:
    cols = table6.get("columns", list(GAME_ABBREV.values()))
    labels = table6.get("labels", {})
    sections = table6.get("sections", {})
    order = list(sections.get("frontier", [])) + list(sections.get("trained_3b", []))
    if not order:
        order = list(table6.get("rows", {}).keys())

    def cell(v: float | None) -> str:
        if v is None:
            return "n/a"
        return f"{v:.2f}"

    header_cols = " ".join(f"{{\\scriptsize {c}}}" for c in cols)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-(model, env) final-round coupling rate, parsable rounds only ($n=12$ episodes per cell). Local eval.}",
        r"\label{tab:coupling-by-game}",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(cols) + "}",
        r"\toprule",
        f"Model & {header_cols} \\\\",
        r"\midrule",
    ]
    for key in order:
        row = table6.get("rows", {}).get(key, {})
        label = labels.get(key, VARIANT_LABELS.get(key, key))
        vals = " & ".join(cell(row.get(c)) for c in cols)
        lines.append(f"{_tex_escape(label)} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _table8_stag_hunt_latex(table8: dict) -> str:
    rows = table8.get("rows") or table8.get("local") or table8.get("paper_reference", PAPER_TABLE6)
    variant = table8.get("variant", "base")
    ep = table8.get("episode", "?")
    caption = (
        rf"Stag-hunt anti-coordination (local eval). Qwen 2.5-3B {variant}, "
        rf"episode {ep} vs tit-for-tat; rows where blind $\neq$ oracle."
        if table8.get("rows") or table8.get("local")
        else r"Stag-hunt anti-coordination example (paper Appendix F)."
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:stag-hunt-example}",
        r"\begin{tabular}{rrllc}",
        r"\toprule",
        r"Round & TFT & Blind & Oracle & Helps \\",
        r"\midrule",
    ]
    for row in rows:
        oracle = row["oracle"]
        if row.get("helps"):
            oracle = f"{oracle} (HELPS)"
        lines.append(
            f"{row['round']} & {_tex_escape(row['tft'])} & {_tex_escape(row['blind'])} & "
            f"{_tex_escape(oracle)} & {str(row['helps']).lower()} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _table7_training_latex(table7: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Training hyperparameters (local DPO runs). Shared recipe; rows differ by pair count and published checkpoint.}",
        r"\label{tab:training-hparams}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Variant & Pairs & Ckpt step & Final train loss \\",
        r"\midrule",
    ]
    for variant in (
        "filter_on",
        "filter_off",
        "core",
        "aux",
        "all",
        "rw",
        "merge",
    ):
        row = table7.get("rows", {}).get(variant, {})
        label = row.get("label", variant)
        pairs = row.get("pairs")
        step = row.get("best_step")
        loss = row.get("final_train_loss", row.get("best_eval_loss"))
        loss_s = f"{loss:.4f}" if isinstance(loss, (int, float)) else "---"
        lines.append(
            f"{_tex_escape(label)} & "
            f"{pairs if pairs is not None else '---'} & "
            f"{step if step is not None else '---'} & "
            f"{loss_s} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _table9_companion_latex(table9: dict, variants: list[str]) -> str:
    rows = table9.get("rows", {})
    show = [v for v in variants if v in rows]
    if not show:
        show = list(rows.keys())
    bold_ac = {
        game: _best_mask([rows.get(v, {}).get("ac", {}).get(game) for v in show])
        for game in ALL_GAMES
    }
    bold_fc = {
        game: _best_mask([rows.get(v, {}).get("fc", {}).get(game) for v in show])
        for game in ALL_GAMES
    }
    bold_exp = {
        game: _best_mask(
            [rows.get(v, {}).get("exploitability", {}).get(game) for v in show],
            higher_is_better=False,
        )
        for game in ALL_GAMES
    }
    blocks: list[str] = []
    for vi, v in enumerate(show):
        m = rows.get(v, {})
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{Companion metrics per environment ({_tex_escape(_variant_label(v))}). Local eval only.}}",
            rf"\label{{tab:companion-{_tex_escape(v)}}}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Environment & ac & $f_c$ & Exploit. \\",
            r"\midrule",
        ]
        for game in ALL_GAMES:
            lines.append(
                f"{_tex_escape(game)} & "
                f"{_tex_num(m.get('ac', {}).get(game), bold=bold_ac[game][vi])} & "
                f"{_tex_num(m.get('fc', {}).get(game), bold=bold_fc[game][vi])} & "
                f"{_tex_num(m.get('exploitability', {}).get(game), bold=bold_exp[game][vi])} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_latex_md(
    suite: dict,
    *,
    ok_variants: list[str] | None = None,
    run_id: str | None = None,
) -> str:
    ok_variants = ok_variants or [
        v for v in suite.get("variants", [])
        if v in suite.get("rows", {}) and "error" not in suite.get("rows", {})[v]
    ]
    if not ok_variants:
        ok_variants = list(suite.get("table1", {}).keys())

    run_id = run_id or suite.get("run_id", "unknown")
    hp = suite.get("hyperparams", {})
    local_title = hp.get("run_title") or build_run_title(hp, kind="local")
    paper_title = hp.get("paper_title") or build_run_title(hp, kind="paper")

    sections = [
        f"# LaTeX tables — {local_title}",
        "",
        f"> Paper reference: {paper_title}",
        "",
        f"- Run ID: `{run_id}`",
        f"- Model: `{hp.get('STUDENT_MODEL', '?')}`",
        f"- Train epochs: {hp.get('train_epochs', hp.get('epochs', '?'))}",
        f"- Episodes / env: {hp.get('episodes_per_env', '?')}",
        f"- Variants: {', '.join(ok_variants)}",
        "",
        "Optional env overrides: `EVAL_TITLE`, `TRAIN_EPOCHS`, `RUN_LABEL`",
        "",
        "Preamble (add once to your `.tex` file):",
        "",
        "```latex",
        r"\usepackage{booktabs}",
        r"\usepackage{bm}",
        "```",
        "",
    ]

    table_blocks = [
        (f"Table 1 — Headline · {local_title}", _table1_latex(suite.get("table1", {}), ok_variants, hp)),
        (f"Table 1 — {paper_title}", _table1_paper_latex()),
        ("Table 2 — Per-game CR", _table2_latex(suite.get("table2", {}), ok_variants)),
        ("Table 3 — Per-axis CR", _table3_latex(suite.get("table3", {}), ok_variants)),
        ("Table 4 — Frontier CR + TFT-CR", _table4_frontier_latex(suite.get("table4_frontier", {}))),
        ("Table 5 — Reasoning-action coupling (paper_v01)", _table5_coupling_latex(suite.get("table5_coupling", {}))),
        ("Table 6 — Per-game $f_c$ (local)", _table6_latex(suite.get("table6", {}))),
        ("Table 8 — Stag-hunt example (Appendix F)", _table8_stag_hunt_latex(suite.get("table8_stag_hunt", {}))),
        ("Table 7 — Training hyperparameters (local)", _table7_training_latex(suite.get("table7_training", {}))),
        (
            "Table 9 — Companion metrics (local)",
            _table9_companion_latex(suite.get("table9_companion", {}), ok_variants),
        ),
    ]

    for title, latex in table_blocks:
        sections += [f"## {title}", "", "```latex", latex, "```", ""]

    return "\n".join(sections).rstrip() + "\n"


def write_latex_md(run_dir: Path, suite: dict, *, ok_variants: list[str] | None = None) -> Path:
    out = run_dir / "latex.md"
    out.write_text(
        build_latex_md(suite, ok_variants=ok_variants, run_id=suite.get("run_id", run_dir.name)),
        encoding="utf-8",
    )
    return out


def _resolve_suite_path(suite_arg: Path | None) -> Path:
    if suite_arg and suite_arg.exists():
        return suite_arg
    latest = config.RUNS_DIR / "latest.json"
    if latest.exists():
        data = json.loads(latest.read_text(encoding="utf-8"))
        return ROOT / data["path"] / "suite.json"
    runs = EVAL_DIR / "runs"
    if runs.is_dir():
        for d in sorted(runs.iterdir(), reverse=True):
            p = d / "suite.json"
            if p.exists():
                return p
    raise FileNotFoundError("No suite.json found; run eval first.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build latex.md from eval suite.json")
    parser.add_argument("--suite", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Override output path")
    args = parser.parse_args(argv)

    suite_path = _resolve_suite_path(args.suite)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    run_dir = suite_path.parent
    out = args.out or (run_dir / "latex.md")
    out.write_text(build_latex_md(suite, run_id=suite.get("run_id", run_dir.name)), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
