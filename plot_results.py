import argparse
import json
import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory

# On-disk cache for the expensive permutation tests (Table 2 p-values). Keyed
# on the numeric inputs, so it only recomputes when the underlying runs change.
_MEMORY = Memory(os.path.join(".cache", "joblib"), verbose=0)

MODEL_DISPLAY_NAMES = {
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gpt-5": "GPT-5.5",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
}

MODEL_COLORS = {
    "gemini-3.1-pro-preview": "#1a73e8",
    "gpt-5":                  "#10a37f",
    "claude-sonnet-4-5":      "#cc7a00",
    "claude-sonnet-4-6":      "#cc7a00",
}

PLOTTED_MODELS = set(MODEL_COLORS.keys())

# Base models drawn in the CRIT threshold sweep (plot_fp_vs_fn). Kept separate
# from MODEL_COLORS/PLOTTED_MODELS: those gate the main EM/F1 charts and their
# trial-count consistency check, which the open-weight single-seed sweeps would
# trip. Colors are MODEL_PALETTE from bias_analysis.py so a model keeps the
# same hue in the bias figures and here; the list is alphabetical by display
# name, which is the legend order every figure in the paper uses.
SWEEP_MODELS = [
    ("claude-sonnet-4-6",                              "Claude Sonnet 4.6", "#e8e844"),
    ("deepseek-ai/DeepSeek-V4-Pro",                    "DeepSeek V4 Pro",   "#4a7fb5"),
    ("gemini-3.1-pro-preview",                         "Gemini 3.1 Pro",    "#efa130"),
    ("gpt-5",                                          "GPT-5.5",           "#8cbfac"),
    ("moonshotai/Kimi-K2.6",                           "Kimi K2.6",         "#d1615d"),
    ("local/mistralai/Mistral-Small-24B-Instruct-2501", "Mistral Small 24B", "#a68a64"),
    ("local/Qwen/Qwen3-32B-AWQ",                       "Qwen3 32B",         "#7f9c6c"),
]

# Baselines for the CRIT sweep's start point. The sweep directory holds only
# threshold runs, so the no-CRIT operating point comes from the main results
# directory.
SWEEP_BASELINE_DIR = os.path.join(
    "results", "paper13_combine-10-12", "rephrased_question")

# Hatch pattern encodes the alternative method (currently the threshold agent).
CONDITION_HATCHES = {
    "Baseline":      "",
    "Baseline+RAG":  "..",
    "CRIT":     "//",
    "CRIT+RAG": "//..",
}


def latest_paper_dir(results_root="results"):
    """Return the highest-numbered `paperN[_suffix]` directory under results_root."""
    candidates = []
    for name in os.listdir(results_root):
        m = re.match(r"^paper(\d+)", name)
        if m and os.path.isdir(os.path.join(results_root, name)):
            candidates.append((int(m.group(1)), name))
    if not candidates:
        raise FileNotFoundError(f"No paperN directory found under {results_root}")
    return max(candidates)[1]


def _read_run_config(run_dir):
    """Return (use_rag, use_domain_expertise) parsed from the run's _summary.md."""
    summary_path = glob.glob(os.path.join(run_dir, "*_summary.md"))[0]
    txt = open(summary_path).read()
    rag = re.search(r"use_rag:\*\*\s+(\S+)", txt).group(1).lower() == "true"
    de = re.search(r"use_domain_expertise:\*\*\s+(\S+)", txt).group(1).lower() == "true"
    return rag, de


def _read_confidence_threshold(run_dir):
    """Return the integer confidence_threshold recorded in the run's
    _summary.md, or None if absent (e.g. baseline / reflection runs)."""
    summary_files = glob.glob(os.path.join(run_dir, "*_summary.md"))
    if not summary_files:
        return None
    txt = open(summary_files[0]).read()
    m = re.search(r"confidence_threshold:\*\*\s+(\d+)", txt)
    return int(m.group(1)) if m else None


def _resolve_model_type_dir(scan_dir, model, model_type):
    """Map a synthetic model_type like ``threshold_t3`` back to the on-disk
    directory (``threshold``) and the threshold value that scans should match.

    Returns ``(dir_path, required_threshold)`` where required_threshold is None
    for non-threshold runs (no filtering needed)."""
    m = re.match(r"^threshold_t(\d+)$", model_type)
    if m:
        return os.path.join(scan_dir, model, "threshold"), int(m.group(1))
    return os.path.join(scan_dir, model, model_type), None


def load_results(results_dir):
    """
    Scan {results_dir}/{model}/{type}/{timestamp}/*.jsonl
    and compute F1 + EM per (model, type, use_rag, use_de) cell, averaging
    across all successful runs in that cell. Raises if cells (restricted to
    PLOTTED_MODELS) have differing trial counts.
    """
    runs_by_key = {}

    for model_dir in sorted(glob.glob(os.path.join(results_dir, "*"))):
        model = os.path.basename(model_dir)
        for type_dir in sorted(glob.glob(os.path.join(model_dir, "*"))):
            base_model_type = os.path.basename(type_dir)
            for run_dir in sorted(glob.glob(os.path.join(type_dir, "*"))):
                summary_files = glob.glob(os.path.join(run_dir, "*_summary.md"))
                jsonl_files = glob.glob(os.path.join(run_dir, "*.jsonl"))
                if not summary_files or not jsonl_files:
                    print(f"  skipping incomplete run: {run_dir}")
                    continue
                use_rag, use_de = _read_run_config(run_dir)
                # Treat each --confidence_threshold value as its own
                # model_type so trial averaging / table rows separate them.
                if base_model_type == "threshold":
                    thr = _read_confidence_threshold(run_dir)
                    model_type = f"threshold_t{thr}" if thr is not None else "threshold"
                else:
                    model_type = base_model_type
                jsonl_path = jsonl_files[0]
                rows = [json.loads(line) for line in open(jsonl_path)]
                def extract(val):
                    return val[0] if isinstance(val, list) else val
                f1_scores = [v for r in rows if (v := extract(r["f1_score"])) is not None]
                em_scores = [v for r in rows if (v := extract(r["exact_match"])) is not None]
                if not f1_scores or not em_scores:
                    print(f"  skipping run with no valid scores: {run_dir}")
                    continue
                # Per-answer-type breakdown for the question-type table.
                by_type = {}
                for r in rows:
                    at = r.get("answer_type")
                    f1 = extract(r["f1_score"])
                    em = extract(r["exact_match"])
                    if at is None or f1 is None or em is None:
                        continue
                    bucket = by_type.setdefault(at, {"f1": [], "em": []})
                    bucket["f1"].append(f1)
                    bucket["em"].append(em)
                by_type_mean = {
                    at: {
                        "f1": float(np.mean(vals["f1"])) * 100,
                        "em": float(np.mean(vals["em"])) * 100,
                    }
                    for at, vals in by_type.items()
                }
                key = (model, model_type, use_rag, use_de)
                runs_by_key.setdefault(key, []).append({
                    "f1": np.mean(f1_scores) * 100,
                    "em": np.mean(em_scores) * 100,
                    "by_type": by_type_mean,
                    "timestamp": os.path.basename(run_dir),
                })

    # Threshold (CRIT) sweeps are intentionally single-seed — the threshold is
    # a post-hoc confidence filter, so seed averaging adds nothing. Exempt them
    # from the uniformity check, which exists to catch accidentally-missing
    # baseline/reflection trials.
    trial_counts = {
        k: len(v) for k, v in runs_by_key.items()
        if k[0] in PLOTTED_MODELS and not k[1].startswith("threshold")
    }
    if trial_counts and len(set(trial_counts.values())) > 1:
        detail = "\n".join(f"  {k}: {n} trials" for k, n in sorted(trial_counts.items()))
        raise ValueError(
            f"Inconsistent trial counts across experiments (PLOTTED_MODELS only):\n{detail}"
        )

    summary = {}
    for key, runs in runs_by_key.items():
        f1s = [r["f1"] for r in runs]
        ems = [r["em"] for r in runs]
        # Stack per-answer-type means across runs.
        type_runs = {}
        for r in runs:
            for at, m in r["by_type"].items():
                bucket = type_runs.setdefault(at, {"f1": [], "em": []})
                bucket["f1"].append(m["f1"])
                bucket["em"].append(m["em"])
        by_type = {}
        for at, vals in type_runs.items():
            by_type[at] = {
                "f1": float(np.mean(vals["f1"])),
                "em": float(np.mean(vals["em"])),
                "f1_std": float(np.std(vals["f1"], ddof=1)) if len(vals["f1"]) > 1 else 0.0,
                "em_std": float(np.std(vals["em"], ddof=1)) if len(vals["em"]) > 1 else 0.0,
            }
        summary[key] = {
            "f1": float(np.mean(f1s)),
            "em": float(np.mean(ems)),
            "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "em_std": float(np.std(ems, ddof=1)) if len(ems) > 1 else 0.0,
            "by_type": by_type,
            "n_trials": len(runs),
            "timestamp": max(r["timestamp"] for r in runs),
        }
    return summary


def print_table(summary):
    print(f"\n{'Model':<25} {'Type':<11} {'RAG':<5} {'DE':<5} {'F1':>7} {'EM':>7} {'N':>3}  {'LatestRun':<20}")
    print("-" * 90)
    for (model, mt, rag, de), vals in sorted(summary.items()):
        display = MODEL_DISPLAY_NAMES.get(model, model)
        print(f"{display:<25} {mt:<11} {str(rag):<5} {str(de):<5} {vals['f1']:>6.1f}% {vals['em']:>6.1f}% {vals['n_trials']:>3}  {vals['timestamp']:<20}")


def _latex_escape(s):
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("_", r"\_")
         .replace("#", r"\#")
         .replace("$", r"\$")
    )


def _build_combined_latex_table(summary, conditions, title, fig_name, show_std=True):
    present = {m for (m, _, _, _) in summary.keys() if m in PLOTTED_MODELS}
    # Explicit ordering — Gemini last so the strongest model anchors the table.
    model_order = [
        "claude-sonnet-4-5", "claude-sonnet-4-6",
        "gpt-5",
        "gemini-3.1-pro-preview",
    ]
    models = [m for m in model_order if m in present] + sorted(present - set(model_order))

    # Method label renames for the paper-style table.
    method_rename = {}
    # CRIT (threshold) now runs on every base model, so each model shows a
    # baseline row followed by its +CRIT row — no cells are omitted.
    rows = []
    for model in models:
        for label, mt, rag, de in conditions:
            method = method_rename.get(label, label)
            rows.append((model, method, summary.get((model, mt, rag, de))))

    max_f1 = max((c["f1"] for _, _, c in rows if c is not None), default=None)
    max_em = max((c["em"] for _, _, c in rows if c is not None), default=None)

    def fmt(cell, key):
        if cell is None:
            return "--"
        val = cell[key]
        if show_std and cell["n_trials"] > 1:
            s = f"{val:.1f} $\\pm$ {cell[f'{key}_std']:.1f}"
        else:
            s = f"{val:.1f}"
        max_val = max_f1 if key == "f1" else max_em
        if max_val is not None and val == max_val:
            s = f"\\textbf{{{s}}}"
        return s

    def plain(cell, key):
        if cell is None:
            return "--"
        val = cell[key]
        if show_std and cell["n_trials"] > 1:
            return f"{val:.1f} ± {cell[f'{key}_std']:.1f}"
        return f"{val:.1f}"

    def method_label(model, method):
        # Baseline rows display the model name only; the "ours" rows (any
        # Threshold variant) stand alone.
        if method.startswith("CRIT"):
            return method
        return MODEL_DISPLAY_NAMES.get(model, model)

    body_lines = []
    plain_rows = []
    for model, method, cell in rows:
        label_text = method_label(model, method)
        body_lines.append(" & ".join([
            _latex_escape(label_text),
            fmt(cell, "f1"),
            fmt(cell, "em"),
        ]) + r" \\")
        plain_rows.append([label_text, plain(cell, "f1"), plain(cell, "em")])

    trial_counts = {c["n_trials"] for _, _, c in rows if c is not None}
    if len(trial_counts) == 1 and next(iter(trial_counts)) > 1:
        n = next(iter(trial_counts))
        n_note = (
            f", mean $\\pm$ std over $N={n}$ trials"
            if show_std else f", mean over $N={n}$ trials"
        )
    else:
        n_note = ""

    label_base = "tab:" + os.path.splitext(fig_name)[0]
    label = label_base + ("" if show_std else "_nostd")

    # Plain-text rendering embedded as LaTeX comments.
    plain_header = ["Method", "F1", "EM"]
    widths = [
        max(len(plain_header[i]), *(len(r[i]) for r in plain_rows))
        for i in range(len(plain_header))
    ]
    def line(cells):
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
    sep = "-+-".join("-" * w for w in widths)
    plain_block = [
        "% Plain-text rendering of the table above:",
        "%   " + line(plain_header),
        "%   " + sep,
    ] + ["%   " + line(r) for r in plain_rows]
    plain_comment = "\n".join(plain_block) + "\n"

    latex = (
        plain_comment +
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lrr}\n"
        "\\toprule\n"
        "Method & F1 & EM \\\\\n"
        "\\midrule\n"
        + "\n".join(body_lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{_latex_escape(title)} (\\%{n_note}).}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    return latex


def write_combined_latex_table(summary, conditions, title, fig_name, out_dir):
    """Emit two `\\begin{table}` blocks in one file: mean±std and mean-only."""
    with_std = _build_combined_latex_table(summary, conditions, title, fig_name, show_std=True)
    without_std = _build_combined_latex_table(summary, conditions, title, fig_name, show_std=False)
    tex_path = os.path.join(out_dir, os.path.splitext(fig_name)[0] + ".tex")
    with open(tex_path, "w") as f:
        f.write(with_std + "\n" + without_std)
    print(f"LaTeX table saved to {tex_path}")


ANSWER_TYPE_LABELS = [
    ("boolean", "Boolean"),
    ("txn_id_list", "Txn IDs"),
    ("account_id_list", "Account IDs"),
]


def _build_question_type_latex_table(summary, rows_spec, title, fig_name, show_std=True):
    """
    Per-(model, method) breakdown of EM by answer_type.

    rows_spec: list of (model, method_label, model_type, use_rag, use_de) tuples
               selecting which cells to render and how to name the Method column.
    """
    types = ANSWER_TYPE_LABELS

    # Per-column max (after rounding to one decimal so ties on display tie too).
    col_max = {}
    for at, _ in types:
        vals = []
        for model, method, mt, rag, de in rows_spec:
            t = (summary.get((model, mt, rag, de)) or {}).get("by_type", {}).get(at)
            if t is not None:
                vals.append(round(t["f1"], 1))
        col_max[at] = max(vals) if vals else None

    body = []
    plain_rows = []
    for model, method, mt, rag, de in rows_spec:
        cell = summary.get((model, mt, rag, de))
        model_disp = MODEL_DISPLAY_NAMES.get(model, model)
        # Baseline rows show the model name; CRIT-variant rows stand alone.
        label_text = method if method.startswith("CRIT") else model_disp
        latex_cells = [_latex_escape(label_text)]
        plain_cells = [label_text]
        for at, _ in types:
            t = (cell or {}).get("by_type", {}).get(at)
            if t is None:
                latex_cells.append("--")
                plain_cells.append("--")
                continue
            if show_std and (cell or {}).get("n_trials", 0) > 1:
                latex_str = f"{t['f1']:.1f} $\\pm$ {t['f1_std']:.1f}"
                plain_str = f"{t['f1']:.1f} ± {t['f1_std']:.1f}"
            else:
                latex_str = f"{t['f1']:.1f}"
                plain_str = f"{t['f1']:.1f}"
            if col_max[at] is not None and round(t["f1"], 1) == col_max[at]:
                latex_str = f"\\textbf{{{latex_str}}}"
            latex_cells.append(latex_str)
            plain_cells.append(plain_str)
        body.append(" & ".join(latex_cells) + r" \\")
        plain_rows.append(plain_cells)

    trial_counts = {
        summary[(m, mt, rag, de)]["n_trials"]
        for m, _, mt, rag, de in rows_spec
        if (m, mt, rag, de) in summary
    }
    if len(trial_counts) == 1 and next(iter(trial_counts)) > 1:
        n = next(iter(trial_counts))
        n_note = (
            f", mean $\\pm$ std over $N={n}$ trials"
            if show_std else f", mean over $N={n}$ trials"
        )
    else:
        n_note = ""

    col_spec = "l" + ("r" * len(types))
    header_cells = ["Method"] + [lbl for _, lbl in types]
    header = " & ".join(header_cells) + r" \\"
    label = "tab:" + os.path.splitext(fig_name)[0] + ("" if show_std else "_nostd")

    # Plain-text version embedded as LaTeX comments.
    plain_header = ["Method"] + [lbl.replace("\\$ ", "$") for _, lbl in types]
    widths = [
        max(len(plain_header[i]), *(len(r[i]) for r in plain_rows))
        for i in range(len(plain_header))
    ]
    def line(cells):
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
    sep = "-+-".join("-" * w for w in widths)
    plain_block = [
        "% Plain-text rendering of the table above:",
        "%   " + line(plain_header),
        "%   " + sep,
    ] + ["%   " + line(r) for r in plain_rows]
    plain_comment = "\n".join(plain_block) + "\n"

    latex = (
        plain_comment +
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\small\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        + "\n".join(body) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{_latex_escape(title)} (F1 \\%{n_note}).}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    return latex


def write_question_type_latex_table(summary, rows_spec, title, fig_name, out_dir):
    """Emit two `\\begin{table}` blocks in one file: mean±std and mean-only."""
    with_std = _build_question_type_latex_table(summary, rows_spec, title, fig_name, show_std=True)
    without_std = _build_question_type_latex_table(summary, rows_spec, title, fig_name, show_std=False)
    tex_path = os.path.join(out_dir, os.path.splitext(fig_name)[0] + ".tex")
    os.makedirs(out_dir, exist_ok=True)
    with open(tex_path, "w") as f:
        f.write(with_std + "\n" + without_std)
    print(f"LaTeX table saved to {tex_path}")


def _collect_extras_drops(scan_dir, rows_spec):
    """For each (model, label, model_type) row, return a stats dict per row.

    Stats:
      avg_extras / avg_drops  — mean per question (averaged across trials)
      total_extras / total_drops — sum across all trial instances
      q_with_extras / q_with_drops — # questions where mean FP/FN > 0
      n_questions
    """
    import collections
    results = []
    for model, label, model_type in rows_spec:
        empty = {
            "label": label, "avg_extras": None, "avg_drops": None,
            "total_extras": 0, "total_drops": 0,
            "q_with_extras": 0, "q_with_drops": 0, "n_questions": 0,
        }
        model_dir, required_thr = _resolve_model_type_dir(scan_dir, model, model_type)
        if not os.path.isdir(model_dir):
            results.append(empty)
            continue
        per_q = collections.defaultdict(lambda: {"extras": [], "drops": []})
        for run_dir in sorted(glob.glob(os.path.join(model_dir, "*"))):
            if (os.path.exists(os.path.join(run_dir, "FAILED.txt"))
                    or os.path.exists(os.path.join(run_dir, "RATE_LIMITED.txt"))):
                continue
            try:
                use_rag, use_de = _read_run_config(run_dir)
            except Exception:
                continue
            if use_rag or use_de:
                continue
            if required_thr is not None and _read_confidence_threshold(run_dir) != required_thr:
                continue
            jsonls = glob.glob(os.path.join(run_dir, "*.jsonl"))
            if not jsonls:
                continue
            for line in open(jsonls[0]):
                r = json.loads(line)
                if r.get("answer_type") != "txn_id_list":
                    continue
                gt = _parse_id_list(r.get("answer"))
                pred = _parse_id_list(r.get("pred"))
                # All txn-list questions, positive AND negative polarity. On a
                # negative-polarity question (gt == set()) every predicted
                # item is a FP and no FN is possible.
                per_q[r["question_id"]]["extras"].append(len(pred - gt))
                per_q[r["question_id"]]["drops"].append(len(gt - pred))
        if not per_q:
            results.append(empty)
            continue
        per_q_extras = {q: float(np.mean(v["extras"])) for q, v in per_q.items()}
        per_q_drops = {q: float(np.mean(v["drops"])) for q, v in per_q.items()}
        total_extras = sum(sum(v["extras"]) for v in per_q.values())
        total_drops = sum(sum(v["drops"]) for v in per_q.values())
        # Q w/ FP|FN: average across trials of (# questions with at least one
        # error in that trial). Sums all such (question, trial) instances and
        # divides by the number of trials so the value is comparable to the
        # per-question count.
        n_trials_extras = max((len(v["extras"]) for v in per_q.values()), default=0) or 1
        n_trials_drops = max((len(v["drops"]) for v in per_q.values()), default=0) or 1
        trials_with_extras = sum(1 for v in per_q.values() for c in v["extras"] if c > 0)
        trials_with_drops = sum(1 for v in per_q.values() for c in v["drops"] if c > 0)
        results.append({
            "label": label,
            "avg_extras": float(np.mean(list(per_q_extras.values()))),
            "avg_drops": float(np.mean(list(per_q_drops.values()))),
            "total_extras": int(total_extras),
            "total_drops": int(total_drops),
            "q_with_extras": trials_with_extras / n_trials_extras,
            "q_with_drops": trials_with_drops / n_trials_drops,
            "n_questions": len(per_q),
        })
    return results


# --------------------------------------------------------------------------
# Combined Table 2 (Overall + by-type F1 + FP/FN), with CRIT recycling,
# per-model McNemar p-value on txn questions, and seed-range reporting.
# --------------------------------------------------------------------------
TABLE2_MODELS = [
    ("claude-sonnet-4-6",             "Claude Sonnet 4.6"),
    ("gpt-5",                         "GPT-5.5"),
    ("gemini-3.1-pro-preview",        "Gemini 3.1 Pro"),
    ("deepseek-ai/DeepSeek-V4-Pro",   "DeepSeek V4 Pro"),
    ("moonshotai/Kimi-K2.6",          "Kimi K2.6"),
    ("local/Qwen/Qwen3-32B-AWQ",                       "Qwen3 32B"),
    ("local/mistralai/Mistral-Small-24B-Instruct-2501", "Mistral Small 24B"),
]
# Question types CRIT never touches — CRIT recycles the baseline model's
# predictions on these, so any measured difference would be pure noise.
_NONTXN_TYPES = {"boolean", "account_id_list", "dollar_amount"}


def _load_seed_records(scan_dir, model, kind):
    """Return a list of per-seed dicts (qid -> {"at","em","f1"}) for a
    (model, kind) cell. kind is "baseline" or "crit" (threshold, t=5 only).
    Skips RAG / domain-expertise / failed / rate-limited runs."""
    subdir = "baseline" if kind == "baseline" else "threshold"
    base = os.path.join(scan_dir, model, subdir)
    seeds = []
    for run_dir in sorted(glob.glob(os.path.join(base, "*"))):
        if (os.path.exists(os.path.join(run_dir, "FAILED.txt"))
                or os.path.exists(os.path.join(run_dir, "RATE_LIMITED.txt"))):
            continue
        jsonls = glob.glob(os.path.join(run_dir, "*.jsonl"))
        if not jsonls:
            continue
        try:
            use_rag, use_de = _read_run_config(run_dir)
        except Exception:
            continue
        if use_rag or use_de:
            continue
        if kind == "crit" and _read_confidence_threshold(run_dir) != 5:
            continue

        def _ex(v):
            return v[0] if isinstance(v, list) else v

        rec = {}
        for line in open(jsonls[0]):
            r = json.loads(line)
            em, f1 = _ex(r.get("exact_match")), _ex(r.get("f1_score"))
            if em is None or f1 is None:
                continue
            rec[r["question_id"]] = {
                "at": r.get("answer_type"), "em": float(em), "f1": float(f1),
            }
        if rec:
            seeds.append(rec)
    return seeds


def _per_q_mean(seeds, field):
    """qid -> mean of `field` across the seeds that contain that qid."""
    qids = set().union(*[set(s) for s in seeds]) if seeds else set()
    return {q: float(np.mean([s[q][field] for s in seeds if q in s])) for q in qids}


def _type_f1(seeds, at):
    """Mean over seeds of (mean F1 over questions of type `at`), as a percent."""
    per_seed = [np.mean([v["f1"] for v in s.values() if v["at"] == at])
                for s in seeds if any(v["at"] == at for v in s.values())]
    return float(np.mean(per_seed)) * 100 if per_seed else float("nan")


def _overall_seed_scores(seeds, field, recycle=None):
    """Per-seed overall mean (percent). If `recycle` (qid->value) is given,
    non-txn questions take that fixed value instead of the seed's own — this
    is how CRIT reuses the baseline's non-txn predictions."""
    out = []
    for s in seeds:
        vals = [recycle[q] if (recycle is not None and v["at"] in _NONTXN_TYPES)
                else v[field] for q, v in s.items()]
        out.append(float(np.mean(vals)) * 100)
    return out


@_MEMORY.cache
def _perm_count(pools_flat, pool_lens, nbs, obs, n_perm, seed):
    """Permutation loop for `_clustered_perm_txn_p`, split out so joblib can
    cache it on the numeric inputs alone (the F1 pools), not on the seed dicts.
    Returns how many permutations reach the observed effect."""
    pools, off = [], 0
    for n in pool_lens:
        pools.append(pools_flat[off:off + n])
        off += n
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        acc = 0.0
        for pool, nb in zip(pools, nbs):
            idx = rng.permutation(len(pool))
            acc += pool[idx[nb:]].mean() - pool[idx[:nb]].mean()
        if acc / len(pools) >= obs - 1e-12:
            count += 1
    return count


def _clustered_perm_txn_p(base_seeds, crit_seeds, n_perm=20000, seed=0):
    """One-sided p-value (H1: CRIT > baseline) for CRIT vs. baseline on
    transaction-id questions,
    using a paired permutation test with question-level clustering over all
    trials (F1). The statistic is the mean over questions of (mean CRIT F1 --
    mean baseline F1). Under H0 the baseline and CRIT trials of a question are
    exchangeable, so each permutation re-pools a question's trial F1 scores and
    re-splits them into baseline/CRIT groups of the original sizes. This uses
    the continuous F1 metric and every trial as a replicate without treating
    correlated trials as independent (the question stays the resampling unit)."""
    at = {q: v["at"] for s in base_seeds for q, v in s.items()}
    byq = {}
    for s in base_seeds:
        for q, v in s.items():
            if at.get(q) == "txn_id_list":
                byq.setdefault(q, [[], []])[0].append(v["f1"])
    for s in crit_seeds:
        for q, v in s.items():
            if at.get(q) == "txn_id_list":
                byq.setdefault(q, [[], []])[1].append(v["f1"])
    # Sorted so the cache key below doesn't depend on dict insertion order.
    qs = sorted(q for q, (bt, ct) in byq.items() if bt and ct)
    if not qs:  # model not run yet (or no txn questions) — nothing to test
        return {"p": None, "obs": float("nan"), "n_txn": 0, "n_perm": 0}
    pools = [np.array(byq[q][0] + byq[q][1]) for q in qs]
    nbs = [len(byq[q][0]) for q in qs]
    obs = float(np.mean([np.mean(byq[q][1]) - np.mean(byq[q][0]) for q in qs]))

    count = _perm_count(np.concatenate(pools),
                        np.array([len(p) for p in pools]),
                        np.array(nbs), obs, n_perm, seed)
    p = (1 + count) / (n_perm + 1)
    return {"p": p, "obs": obs, "n_txn": len(qs), "n_perm": n_perm}


def write_combined_table2(scan_dir=None, out_dir=None,
                          fig_name="table2_eval_combined.tex"):
    """Rebuild the combined Table 2 (Overall F1/EM, by-type F1, FP/FN) and
    write it straight into the overleaf directory.

    CRIT rows recycle the baseline's boolean / account-id / dollar-amount
    predictions (CRIT only edits transaction lists), so Overall and those
    columns are recomputed from the merged predictions. Each CRIT row carries
    a p-value (clustered permutation test on F1 vs. its baseline, txn questions
    only, over all trials), and the Overall F1/EM cells show the per-seed
    min--max range."""
    if scan_dir is None:
        scan_dir = os.path.join("results", latest_paper_dir(), "rephrased_question")
    if out_dir is None:
        out_dir = os.path.join("overleaf", "figs")

    def _rng(vals):
        # Omitted only for single-seed cells. A cell whose seeds agree to the
        # displayed precision still prints "(x--x)": a blank there reads as a
        # missing trial, when it actually means the spread is narrower than one
        # decimal place (e.g. Mistral baseline, 45.96/45.96/46.01).
        if not vals or len(vals) < 2:
            return ""
        return f"{{\\scriptsize({min(vals):.1f}--{max(vals):.1f})}}"

    def _rng_plain(vals):
        # Same rule as _rng, but rendered for the markdown comment block.
        if not vals or len(vals) < 2:
            return ""
        return f" ({min(vals):.1f}-{max(vals):.1f})"

    # Collect every cell first so we can bold column-best across all rows.
    cells = []  # list of dicts, in table row order
    pvals = {}
    for model, disp in TABLE2_MODELS:
        base = _load_seed_records(scan_dir, model, "baseline")
        crit = _load_seed_records(scan_dir, model, "crit")
        b_em_q, b_f1_q = _per_q_mean(base, "em"), _per_q_mean(base, "f1")

        b_f1s = _overall_seed_scores(base, "f1")
        b_ems = _overall_seed_scores(base, "em")
        c_f1s = _overall_seed_scores(crit, "f1", recycle=b_f1_q)
        c_ems = _overall_seed_scores(crit, "em", recycle=b_em_q)

        b_bool, b_acct = _type_f1(base, "boolean"), _type_f1(base, "account_id_list")
        stat = _clustered_perm_txn_p(base, crit)
        pvals[model] = stat

        cells.append({
            "label": disp, "crit": False,
            "f1": float(np.mean(b_f1s)), "em": float(np.mean(b_ems)),
            "f1_rng": _rng(b_f1s), "em_rng": _rng(b_ems),
            "f1_rng_md": _rng_plain(b_f1s), "em_rng_md": _rng_plain(b_ems),
            "boolean": b_bool, "txn": _type_f1(base, "txn_id_list"), "acct": b_acct,
        })
        cells.append({
            "label": r"\quad + CRIT", "crit": True,
            "f1": float(np.mean(c_f1s)), "em": float(np.mean(c_ems)),
            "f1_rng": _rng(c_f1s), "em_rng": _rng(c_ems),
            "f1_rng_md": _rng_plain(c_f1s), "em_rng_md": _rng_plain(c_ems),
            # Recycled from baseline (CRIT does not touch these types).
            "boolean": b_bool, "txn": _type_f1(crit, "txn_id_list"), "acct": b_acct,
            "p": stat["p"],
        })

    # FP/FN via the shared collector, keyed by the same row order.
    fpfn_spec = []
    for model, disp in TABLE2_MODELS:
        fpfn_spec.append((model, disp, "baseline"))
        fpfn_spec.append((model, r"\quad + CRIT", "threshold_t5"))
    fpfn = _collect_extras_drops(scan_dir, fpfn_spec)
    for cell, fp in zip(cells, fpfn):
        if fp.get("n_questions", 0) == 0:  # model not run -> blank, don't show 0.0
            cell.update({"q_fp": None, "fp_q": None, "q_fn": None, "fn_q": None})
        else:
            cell.update({
                "q_fp": fp["q_with_extras"], "fp_q": fp["avg_extras"],
                "q_fn": fp["q_with_drops"], "fn_q": fp["avg_drops"],
            })

    # Column-best (max for accuracy, min for error columns) for bolding.
    def _best(key, mode):
        vals = [c[key] for c in cells if c.get(key) is not None and not np.isnan(c[key])]
        return (max if mode == "max" else min)(vals) if vals else None

    best = {k: _best(k, "max") for k in ("f1", "em", "boolean", "txn", "acct")}
    best.update({k: _best(k, "min") for k in ("q_fp", "fp_q", "q_fn", "fn_q")})

    def bold(val, key, fmt=".1f", suffix=""):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "--"
        s = f"{val:{fmt}}{suffix}"
        if best.get(key) is not None and abs(val - best[key]) < 1e-9:
            s = rf"\textbf{{{s}}}"
        return s

    def pfmt(p):
        if p is None:
            return "--"
        # Bold marks significance at the 0.05 level.
        if p < 0.001:
            # \textbf (not $\mathbf{}$) so the digits match the bolded numeric
            # p-values in the same column; only the < needs math mode.
            return r"\textbf{$<$0.001}"
        # Mirror of the <0.001 rule at the top of the range: a permutation
        # p-value is never exactly 1, so never print "1.00".
        if p > 0.999:
            return r"$>$0.999"
        if p >= 0.995:          # would round to 1.00 at two decimals
            return f"{p:.3f}"
        s = f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
        return rf"\textbf{{{s}}}" if p < 0.05 else s

    def mdbold(val, key, fmt=".1f"):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "--"
        s = f"{val:{fmt}}"
        if best.get(key) is not None and abs(val - best[key]) < 1e-9:
            s = f"**{s}**"
        return s

    def pfmt_md(p):
        if p is None:
            return "--"
        if p < 0.001:
            return "**<0.001**"
        if p > 0.999:
            return ">0.999"
        if p >= 0.995:          # would round to 1.00 at two decimals
            return f"{p:.3f}"
        s = f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
        return f"**{s}**" if p < 0.05 else s

    body = []
    md_rows = []
    for c in cells:
        md_rows.append([
            "+ CRIT" if c["crit"] else c["label"],
            mdbold(c["f1"], "f1") + c["f1_rng_md"],
            mdbold(c["em"], "em") + c["em_rng_md"],
            "--" if c["crit"] else mdbold(c["boolean"], "boolean"),
            mdbold(c["txn"], "txn"),
            "--" if c["crit"] else mdbold(c["acct"], "acct"),
            mdbold(c["q_fp"], "q_fp"), mdbold(c["fp_q"], "fp_q"),
            mdbold(c["q_fn"], "q_fn"), mdbold(c["fn_q"], "fn_q", ".2f"),
            pfmt_md(c.get("p")),
        ])
    for c in cells:
        txn_cell = bold(c["txn"], "txn")
        f1_cell = bold(c["f1"], "f1") + (r"\," + c["f1_rng"] if c["f1_rng"] else "")
        em_cell = bold(c["em"], "em") + (r"\," + c["em_rng"] if c["em_rng"] else "")
        cells_tex = [
            c["label"], f1_cell, em_cell,
            "--" if c["crit"] else bold(c["boolean"], "boolean"), txn_cell,
            "--" if c["crit"] else bold(c["acct"], "acct"),
            bold(c["q_fp"], "q_fp"), bold(c["fp_q"], "fp_q", ".1f"),
            bold(c["q_fn"], "q_fn"), bold(c["fn_q"], "fn_q", ".2f"),
            pfmt(c.get("p")),
        ]
        body.append(" & ".join(cells_tex) + r" \\")
        # blank rule between model pairs (after each CRIT row except the last)
    # Insert a midrule between model groups.
    grouped = []
    for i, line in enumerate(body):
        grouped.append(line)
        if i % 2 == 1 and i != len(body) - 1:
            grouped.append(r"\midrule")

    # Markdown rendering of the same table, emitted as a LaTeX comment block so
    # it can be copy-pasted straight out of the .tex file.
    md_header = ["Method", "F1", "EM", "Boolean", "Txn IDs", "Account IDs",
                 "FP Qs", "FP/Q", "FN Qs", "FN/Q", "p"]
    md_widths = [max(len(md_header[i]), *(len(r[i]) for r in md_rows))
                 for i in range(len(md_header))]

    def md_line(cells_):
        return "| " + " | ".join(c.ljust(md_widths[i])
                                 for i, c in enumerate(cells_)) + " |"

    md_sep = "| " + " | ".join("-" * w for w in md_widths) + " |"
    md_comment = "\n".join(
        ["% Markdown rendering of the table below (copy-paste ready):",
         "%   " + md_line(md_header),
         "%   " + md_sep,
         *("%   " + md_line(r) for r in md_rows)]
    ) + "\n"

    latex = (
        md_comment +
        "% GENERATED by plot_results.write_combined_table2 -- do not edit by hand.\n"
        "\\begin{table*}[t]\n\\centering\n\\small\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Overall} & \\multicolumn{3}{c}{F1 By Question Type}"
        " & \\multicolumn{2}{c}{Extras (FP)} & \\multicolumn{2}{c}{Drops (FN)} & \\\\\n"
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-10}\n"
        "Method & F1 & EM & Boolean & Txn IDs & Account IDs & Qs & FP/Q & Qs & FN/Q & $p$ \\\\\n"
        "\\midrule\n"
        + "\n".join(grouped) + "\n"
        "\\bottomrule\n\\end{tabular}%\n}\n"
        "\\caption{Accuracy overall (F1, EM) and by question type (F1 \\%, mean over "
        "$N=3$ trials), alongside error counts on transaction-list questions: the "
        "share of questions (Qs) with at least one false positive (FP) or false "
        "negative (FN) transaction and the average number of FP and FN transactions "
        "per question. CRIT edits only transaction lists, so its Boolean and "
        "Account ID scores are identical to the corresponding baseline and are "
        "shown as --.}\n"
        "\\label{tab:eval_mt_combined_nostd}\n\\end{table*}\n"
    )

    os.makedirs(out_dir, exist_ok=True)
    tex_path = os.path.join(out_dir, fig_name)
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"Combined Table 2 saved to {tex_path}")
    for model, disp in TABLE2_MODELS:
        s = pvals[model]
        if s.get("p") is None:
            print(f"  {disp}: no txn data (model not run)")
        else:
            print(f"  {disp}: clustered perm txn F1  obs Δ={s['obs']:+.3f}  "
                  f"n_txn={s['n_txn']}  n_perm={s['n_perm']}  p={s['p']:.4g}")
    return tex_path


def write_extras_drops_latex_table(scan_dir=None, rows_spec=None,
                                   fig_name="add_vs_drop.png",
                                   out_dir=None):
    """Write a LaTeX table summarizing each model's tendency to add (FP) vs.
    drop (FN) transactions, with the same plain-text comment block as the
    other tables in this script.
    """
    if scan_dir is None:
        scan_dir = os.path.join("results", "paper10_tag-leak", "rephrased_question")
    if rows_spec is None:
        rows_spec = [
            ("claude-sonnet-4-6",      "Claude Sonnet 4.6",  "baseline"),
            ("claude-sonnet-4-6",      "+ CRIT",             "threshold_t5"),
            ("gpt-5",                  "GPT-5.5",            "baseline"),
            ("gpt-5",                  "+ CRIT",             "threshold_t5"),
            ("gemini-3.1-pro-preview", "Gemini 3.1 Pro",     "baseline"),
            ("gemini-3.1-pro-preview", "+ CRIT",             "threshold_t5"),
        ]
    if out_dir is None:
        out_dir = os.path.join("paper", "figs")

    rows = _collect_extras_drops(scan_dir, rows_spec)
    n_questions = next((r["n_questions"] for r in rows if r["n_questions"]), 0)

    def fnum(val, fmt=".1f"):
        return "--" if val is None else f"{val:{fmt}}"

    # Column order per group: Qs (avg # questions per trial that had any error
    # of that type), then FP/Q or FN/Q (mean errors per question across trials).
    body = []
    plain_rows = []
    for r in rows:
        body.append(
            " & ".join([
                _latex_escape(r["label"]),
                fnum(r["q_with_extras"]),
                fnum(r["avg_extras"]),
                fnum(r["q_with_drops"]),
                fnum(r["avg_drops"], fmt=".2f"),
            ]) + r" \\"
        )
        plain_rows.append([
            r["label"],
            fnum(r["q_with_extras"]),
            fnum(r["avg_extras"]),
            fnum(r["q_with_drops"]),
            fnum(r["avg_drops"], fmt=".2f"),
        ])

    plain_header = ["Method",
                    "Qs", "FP/Q",
                    "Qs", "FN/Q"]
    widths = [
        max(len(plain_header[i]), *(len(r[i]) for r in plain_rows))
        for i in range(len(plain_header))
    ]
    def line(cells):
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
    sep = "-+-".join("-" * w for w in widths)
    plain_comment = "\n".join(
        ["% Plain-text rendering of the table above:",
         "%   " + line(plain_header),
         "%   " + sep,
         *("%   " + line(r) for r in plain_rows)]
    ) + "\n"

    caption = (
        # f"Per-model false-positive and false-negative breakdown on all "
        # f"transaction-list questions ($N$={n_questions} questions "
        # f"$\\times$ 3 trials)."
        f"Count of transaction list questions (Q) with at least one false positive (FP) or false negative (FN) transaction and the average number of FP or FN transactions per question."
    )
    label = "tab:" + os.path.splitext(fig_name)[0]

    latex = (
        plain_comment +
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        " & \\multicolumn{2}{c}{Extras (FP)} & \\multicolumn{2}{c}{Drops (FN)} \\\\\n"
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
        "Method & Qs & FP/Q & Qs & FN/Q \\\\\n"
        "\\midrule\n"
        + "\n".join(body) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )

    os.makedirs(out_dir, exist_ok=True)
    tex_path = os.path.join(out_dir, os.path.splitext(fig_name)[0] + ".tex")
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to {tex_path}")


def _save_current_fig(fig_name, subdir=None):
    """Save the current matplotlib figure to paper/figs[/subdir] and mirror it
    into overleaf/figs (flat) so the paper picks up regenerated figures without
    a manual copy step."""
    paper_dir = os.path.join("paper", "figs", subdir) if subdir else os.path.join("paper", "figs")
    os.makedirs(paper_dir, exist_ok=True)
    plt.savefig(os.path.join(paper_dir, fig_name), dpi=300, bbox_inches="tight")
    print(f"Plot saved to {os.path.join(paper_dir, fig_name)}")
    if os.path.isdir("overleaf"):
        overleaf_dir = os.path.join("overleaf", "figs")
        os.makedirs(overleaf_dir, exist_ok=True)
        plt.savefig(os.path.join(overleaf_dir, fig_name), dpi=300, bbox_inches="tight")
        print(f"Plot saved to {os.path.join(overleaf_dir, fig_name)}")


def plot_fp_vs_fn(threshold_scan_dir=None, fig_name="fp_vs_fn.png",
                  baseline_dir=None, ylim=(-0.01, 0.25)):
    """Avg FP/Q vs FN/Q on txn-list questions: the CRIT precision/recall
    tradeoff curve for each base model.

    One dashed line per model traces T=1..5 (each point labelled with its
    threshold); the model's no-CRIT baseline is a filled marker at the start of
    that trajectory. `ylim` is fixed rather than data-driven — the two
    open-weight sweeps run off the bottom of the panel, which is the point.
    """
    if threshold_scan_dir is None:
        threshold_scan_dir = os.path.join("results", "paper12_threshold_sweep", "rephrased_question")
    if baseline_dir is None:
        baseline_dir = SWEEP_BASELINE_DIR

    # One sweep per base model (SWEEP_MODELS); a model with no sweep data in
    # threshold_scan_dir is silently skipped.
    thresholds = (1, 2, 3, 4, 5)

    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    fig, ax = plt.subplots(figsize=(7.5, 5.78))

    legend_handles = []
    outside_legend = False
    all_x = []
    for model, disp, base in SWEEP_MODELS:
        stats = _collect_extras_drops(
            threshold_scan_dir,
            [(model, f"T={t}", f"threshold_t{t}") for t in thresholds],
        )
        pts = [(s["avg_extras"], s["avg_drops"], s["label"]) for s in stats
               if s["avg_extras"] is not None and s["avg_drops"] is not None]
        if not pts:
            continue  # no sweep data for this model yet
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        all_x.extend(xs)
        ax.plot(xs, ys, color=base, linewidth=2.0, linestyle="--",
                alpha=0.85, zorder=2)
        for x, y, lbl in pts:
            ax.annotate(lbl.split("=")[-1], (x, y), textcoords="offset points",
                        xytext=(0, 6), fontsize=12, color=base,
                        ha="center", va="bottom", zorder=4)

        # No-CRIT operating point for the same model. Drawn as a bare marker,
        # not joined to the sweep: the segment from baseline to T=1 is not part
        # of the tradeoff curve, and a second line per model made the panel
        # unreadable.
        bstats = _collect_extras_drops(baseline_dir, [(model, "base", "baseline")])[0]
        if bstats["avg_extras"] is not None and bstats["avg_drops"] is not None:
            bx, by = bstats["avg_extras"], bstats["avg_drops"]
            all_x.append(bx)
            ax.scatter(bx, by, color=base, s=110, edgecolor="black",
                       linewidth=0.8, zorder=5)

        legend_handles.append(
            Line2D([0], [0], color=base, marker="o", markersize=9,
                   markeredgecolor="black", linewidth=2.0, linestyle="--",
                   label=disp)
        )

    if legend_handles:
        # With seven sweeps the curves span the whole panel, so an in-axes
        # legend covers the trajectories. Park it under the axes in three
        # columns instead.
        if len(legend_handles) > 4:
            legend = ax.legend(handles=legend_handles, fontsize=13, ncol=3,
                               loc="upper center", bbox_to_anchor=(0.5, -0.10),
                               framealpha=0.9)
            # Give the legend its own strip of canvas; tight_layout would
            # instead shrink the axes to fit it and squash the tick labels.
            # The strip keeps its absolute height (1.36in) as the figure grows,
            # so extra canvas goes to the axes rather than to white space.
            fig.set_size_inches(7.5, 7.15)
            fig.subplots_adjust(left=0.13, right=0.98, top=0.967, bottom=0.19)
            outside_legend = True
        else:
            legend = ax.legend(handles=legend_handles, fontsize=15.6,
                               loc="lower left", framealpha=0.9)

    ax.set_xlabel("Avg FP per Question", fontsize=18, fontweight="bold")
    ax.set_ylabel("Avg FN per Question", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=15)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle=":", alpha=0.4)
    # x follows the data (baselines sit further right than any T); y is pinned
    # so the panel resolves the region where six of the seven models live.
    if all_x:
        x_pad = max(0.05, 0.08 * (max(all_x) - min(all_x)))
        ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
    ax.set_ylim(*ylim)
    # Ticks land on multiples of 0.02 from 0.00, so the slight negative
    # headroom in `ylim` stays unlabelled.
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.invert_xaxis()
    ax.invert_yaxis()

    if not outside_legend:
        plt.tight_layout()
    _save_current_fig(fig_name)
    plt.close(fig)


def plot_combined_chart(summary, conditions, title, fig_name, metric="em"):
    """
    Single group of bars — for each model, draw one bar per condition adjacent.
    Color = model, hatch = condition.
    """
    models_with_results = sorted(
        {m for (m, _, _, _) in summary.keys() if m in PLOTTED_MODELS}
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.8
    bars_specs = []
    for model in models_with_results:
        for label, mt, rag, de in conditions:
            val = summary.get((model, mt, rag, de), {}).get(metric, 0)
            present = (model, mt, rag, de) in summary
            bars_specs.append((model, label, val, present))

    x = np.arange(len(bars_specs))
    for xi, (model, label, val, present) in zip(x, bars_specs):
        bar = ax.bar(
            xi, val, bar_width,
            color=MODEL_COLORS.get(model, "#888888"),
            edgecolor="white",
            hatch=CONDITION_HATCHES.get(label, ""),
        )[0]
        if present:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _, _ in bars_specs], rotation=0)
    ax.set_ylim(25, 85)

    from matplotlib.patches import Patch
    model_handles = [
        Patch(facecolor=MODEL_COLORS.get(m, "#888888"),
              edgecolor="white",
              label=MODEL_DISPLAY_NAMES.get(m, m))
        for m in models_with_results
    ]
    hatch_handles = [
        Patch(facecolor="white", edgecolor="black",
              hatch=CONDITION_HATCHES.get(label, ""), label=label)
        for label, *_ in conditions
    ]
    ax.legend(handles=model_handles + hatch_handles,
              loc='upper left', framealpha=0.9)

    plt.tight_layout()
    _save_current_fig(fig_name, subdir=metric)
    plt.close(fig)

    out_dir = os.path.join("paper", "figs", metric)
    write_combined_latex_table(summary, conditions, title, fig_name, out_dir)


def _parse_id_list(value):
    """Coerce an answer/prediction into a set of IDs.

    Handles all the on-disk shapes we've seen:
      - true list ['a', 'b']
      - python-list string "['a', 'b']"
      - JSON-list string '["a", "b"]'
      - single-element trial wrappers: [json_or_python_list_string]
    Returns an empty set on failure.
    """
    import ast
    if value is None:
        return set()
    if isinstance(value, list):
        # If items are themselves stringified lists (the per-trial wrapper),
        # parse the first entry; otherwise treat the list contents as ids.
        if value and isinstance(value[0], str) and value[0].strip().startswith(("[", "(")):
            return _parse_id_list(value[0])
        return {str(x) for x in value}
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in ("none", "null", "[]"):
            return set()
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                return {str(x) for x in v}
        except Exception:
            pass
    return set()


def _parse_gt_list(answer):
    """Number of items in a list-typed ground truth answer (kept for the
    existing gt-count plot)."""
    return len(_parse_id_list(answer))


def plot_score_vs_gt_txn_count(scan_dir=None, rows_spec=None,
                                fig_name="score_vs_gt_txn_count.png"):
    """1x2 figure: F1 (left) and EM (right) vs. ground-truth-transaction count
    for the 4 (model, method) cells that appear in the main results table.

    Defaults to paper10_tag-leak/rephrased_question and the four rows used in
    the eval_mt_combined table (Claude/GPT/Gemini baseline + Gemini Threshold).
    """
    import collections

    if scan_dir is None:
        scan_dir = os.path.join("results", "paper10_tag-leak", "rephrased_question")
    if rows_spec is None:
        rows_spec = [
            ("claude-sonnet-4-6",      "Claude Sonnet 4.6",      "baseline"),
            ("gpt-5",                  "GPT-5.5",                "baseline"),
            ("gemini-3.1-pro-preview", "Gemini 3.1 Pro",         "baseline"),
            ("gemini-3.1-pro-preview", "CRIT (ours)",   "threshold_t5"),
        ]

    line_colors = {
        ("claude-sonnet-4-6",      "baseline"):     MODEL_COLORS["claude-sonnet-4-6"],
        ("gpt-5",                  "baseline"):     MODEL_COLORS["gpt-5"],
        ("gemini-3.1-pro-preview", "baseline"):     MODEL_COLORS["gemini-3.1-pro-preview"],
        ("gemini-3.1-pro-preview", "threshold_t5"): "#7d27a9",
    }

    # Collect per-question scores once per (model, model_type) so we don't
    # re-walk the filesystem for each metric.
    series = []  # list of dicts {label, color, xs, by_metric: {f1: ys, em: ys}}
    rng = np.random.default_rng(0)

    for model, label, model_type in rows_spec:
        per_qid = collections.defaultdict(lambda: {"f1": [], "em": []})
        gt_for_qid = {}
        model_dir, required_thr = _resolve_model_type_dir(scan_dir, model, model_type)
        if not os.path.isdir(model_dir):
            continue
        for run_dir in sorted(glob.glob(os.path.join(model_dir, "*"))):
            if (os.path.exists(os.path.join(run_dir, "FAILED.txt"))
                    or os.path.exists(os.path.join(run_dir, "RATE_LIMITED.txt"))):
                continue
            try:
                use_rag, use_de = _read_run_config(run_dir)
            except Exception:
                continue
            if use_rag or use_de:
                continue
            if required_thr is not None and _read_confidence_threshold(run_dir) != required_thr:
                continue
            jsonls = glob.glob(os.path.join(run_dir, "*.jsonl"))
            if not jsonls:
                continue
            for line in open(jsonls[0]):
                r = json.loads(line)
                if r.get("answer_type") != "txn_id_list":
                    continue
                f1 = r["f1_score"]; f1 = f1[0] if isinstance(f1, list) else f1
                em = r["exact_match"]; em = em[0] if isinstance(em, list) else em
                if f1 is None or em is None:
                    continue
                qid = r["question_id"]
                per_qid[qid]["f1"].append(float(f1))
                per_qid[qid]["em"].append(float(em))
                gt_for_qid[qid] = _parse_gt_list(r.get("answer"))

        if not per_qid:
            continue
        xs = [gt_for_qid[qid] for qid in per_qid]
        ys_f1 = [float(np.mean(per_qid[qid]["f1"])) * 100 for qid in per_qid]
        ys_em = [float(np.mean(per_qid[qid]["em"])) * 100 for qid in per_qid]
        series.append({
            "label": label,
            "color": line_colors.get((model, model_type), "#888888"),
            "xs": xs,
            "f1": ys_f1,
            "em": ys_em,
        })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)
    summary_lines = []

    for ax, metric, metric_label in [(axes[0], "f1", "F1"), (axes[1], "em", "EM")]:
        for s in series:
            xs = np.array(s["xs"], dtype=float)
            ys = s[metric]
            jitter = xs + rng.uniform(-0.15, 0.15, size=len(xs))
            ax.scatter(jitter, ys, color=s["color"], alpha=0.55, s=28,
                       label=s["label"])

            binned = collections.defaultdict(list)
            for x, y in zip(s["xs"], ys):
                binned[x].append(y)
            xs_line = sorted(binned)
            ys_line = [float(np.mean(binned[x])) for x in xs_line]
            ax.plot(xs_line, ys_line, color=s["color"], linewidth=2)

            if metric == "f1":
                summary_lines.append(
                    f"  {s['label']}: n_questions={len(xs)}, F1 mean={np.mean(ys):.1f}"
                )

        ax.set_xlabel("Number of ground-truth transactions", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{metric_label} (%)", fontsize=12, fontweight="bold")
        ax.set_title(f"{metric_label}", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_xlim(left=-0.5)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    axes[0].legend(loc="lower left", framealpha=0.9)
    fig.suptitle("Score vs. ground-truth transaction count (txn_id_list)",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    _save_current_fig(fig_name)
    for line in summary_lines:
        print(line)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Name under results/ to scan (matches eval.py --results_dir). "
             "Defaults to the highest-numbered paperN directory.",
    )
    parser.add_argument(
        "--question-col", type=str, default="rephrased_question",
        help="Subdirectory name under results/<results-dir>/ to scan.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir or latest_paper_dir()
    scan_dir = os.path.join("results", results_dir, args.question_col)
    summary = load_results(scan_dir)
    print(f"\n=== {scan_dir} ===")
    print_table(summary)

    eval_mt_conditions = [
        ("Baseline",       "baseline",     False, False),
        ("CRIT",  "threshold_t5", False, False),
    ]
    plot_combined_chart(
        summary,
        conditions=eval_mt_conditions,
        title="Exact Match Accuracy",
        fig_name="eval_mt_combined.png",
        metric="em",
    )

    rag_conditions = [
        ("Baseline",     "baseline", False, False),
        ("Baseline+RAG", "baseline", True,  False),
    ]
    plot_combined_chart(
        summary,
        conditions=rag_conditions,
        title="Exact Match Accuracy (RAG comparison)",
        fig_name="eval_mt_rag.png",
        metric="em",
    )

    # Per-question-type breakdown: each model's baseline followed by its
    # +CRIT (threshold t=5) row.
    qtype_rows = [
        ("claude-sonnet-4-6",      "Baseline",     "baseline",     False, False),
        ("claude-sonnet-4-6",      "CRIT (ours)",  "threshold_t5", False, False),
        ("gpt-5",                  "Baseline",     "baseline",     False, False),
        ("gpt-5",                  "CRIT (ours)",  "threshold_t5", False, False),
        ("gemini-3.1-pro-preview", "Baseline",     "baseline",     False, False),
        ("gemini-3.1-pro-preview", "CRIT (ours)",  "threshold_t5", False, False),
    ]
    write_question_type_latex_table(
        summary,
        rows_spec=qtype_rows,
        title="Accuracy by Question Type",
        fig_name="eval_mt_by_question_type.png",
        out_dir=os.path.join("paper", "figs", "em"),
    )

    plot_score_vs_gt_txn_count()
    write_extras_drops_latex_table()
    write_combined_table2(scan_dir=scan_dir)
    plot_fp_vs_fn()
