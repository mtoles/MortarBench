"""
Bias-study analysis.

Joins the eval results under `results/bias_study/.../` against
`generated_data/bias_study_foreign/test_cases/bias_analysis.jsonl` and
reports, per modification_type (company / name / savings_org), the rate at
which each model retrieved a bias-inducing transaction — i.e. included at
least one of the case's `bias_injected_transaction_ids` in its prediction.

Three tables are printed, one per modification_type:

    rows    = language (or region for savings_org)
    columns = (model_id, model_type) — Claude / GPT / Gemini baseline + CRIT
    cells   = retrieval rate in [0, 1]

Run:
    python3 bias_analysis.py
"""

from __future__ import annotations

import ast
import glob
import json
import os
from collections import defaultdict


# Two-dataset layout (see generate_test_cases.py --bias-study):
#   bias_study_names   — foreign-origin question (company + person names).
#   bias_study_savings — savings-club question (positive cultural funds +
#                        negative money-transfer systems).
BIAS_LABELS_PATH_NAMES = (
    "generated_data/bias_study_names/test_cases/bias_analysis.jsonl"
)
BIAS_LABELS_PATH_SAVINGS = (
    "generated_data/bias_study_savings/test_cases/bias_analysis.jsonl"
)
BIAS_LABELS_PATH_MT = (
    "generated_data/bias_study_money_transfer/test_cases/bias_analysis.jsonl"
)
# Result roots. A run lives at
#   <root>/<model_id>/<model_type>/<timestamp>/<file>_results.jsonl
# where `model_id` may itself contain slashes — open-source slugs nest one or
# two extra directories deep (`deepseek-ai/DeepSeek-V4-Pro`,
# `local/Qwen/Qwen3-32B-AWQ`). `discover_runs` therefore searches each root
# recursively and reconstructs model_id from whatever sits above
# <model_type>/<timestamp>/, rather than assuming a fixed depth.
#
# Names dataset accepts two roots: the current `results/bias_study_names/` and
# the legacy single-bucket `results/bias_study/` predating the names/savings
# split. Records whose test_case_id isn't in the names labels (i.e. were
# savings cases in the legacy bucket) are silently skipped by
# `retrieval_per_run`, so mixing them is safe.
RESULTS_GLOB_NAMES = [
    "results/bias_study_names/rephrased_question",
    "results/bias_study/rephrased_question",
]
RESULTS_GLOB_SAVINGS = "results/bias_study_savings/rephrased_question"
RESULTS_GLOB_MT = "results/bias_study_money_transfer/rephrased_question"

# Back-compat shim — the old single-dataset path. Kept as the names-dataset
# alias so callers that read these still get the foreign-origin labels.
BIAS_LABELS_PATH = BIAS_LABELS_PATH_NAMES
RESULTS_GLOB = RESULTS_GLOB_NAMES

# The compilable-paper source tree. Tables and figures are emitted directly
# here (see main()) so recompiling the paper picks up a rerun of this script
# without anything being copied by hand.
LATEX_DIR = "overleaf"
LATEX_FIGS_DIR = os.path.join(LATEX_DIR, "figs")

# Short display names, keyed by the `--model_id` passed to eval.py. Open-source
# slugs are routed by prefix (see OSS_PROVIDERS in llm.py): bare `org/model` →
# Baseten, `accounts/...` → Fireworks, `local/...` → self-hosted vLLM. The same
# model can therefore appear under two slugs (e.g. Kimi via Baseten and via
# Fireworks); both map to one display name so they collapse into one column.
MODEL_DISPLAY = {
    "claude-sonnet-4-5": "Claude 4.5",
    "claude-sonnet-4-6": "Claude 4.6",
    "gpt-5": "GPT-5.5",
    "gemini-3.1-pro-preview": "Gemini 3.1",
    "deepseek-ai/DeepSeek-V4-Pro": "DeepSeek V4",
    "accounts/fireworks/models/deepseek-v4-pro": "DeepSeek V4",
    "moonshotai/Kimi-K2.6": "Kimi K2.6",
    "accounts/fireworks/models/kimi-k2p6": "Kimi K2.6",
    "local/Qwen/Qwen3-32B-AWQ": "Qwen3 32B",
    "local/mistralai/Mistral-Small-24B-Instruct-2501": "Mistral 24B",
    "local/stelterlab/Mistral-Small-24B-Instruct-2501-AWQ": "Mistral 24B",
}

# Left-to-right ordering of model columns/bars: alphabetical (case-insensitive)
# by display name, so every figure's legend and every table's column order
# agree without anyone having to remember a bespoke ranking. `<model> + CRIT`
# columns are slotted next to their own baseline by `discover_runs`.
MODEL_ORDER = sorted(
    ["Claude 4.5", "Claude 4.6", "CRIT (ours)", "DeepSeek V4", "Gemini 3.1",
     "GPT-5.5", "Kimi K2.6", "Mistral 24B", "Qwen3 32B"],
    key=str.lower,
)

# Per-model colors. The first four come from the ULAD/bank-statement figure
# palette; the open-weight models extend it with distinguishable hues that
# stay legible at the small bar widths a 7-model group produces. CRIT (ours)
# keeps the distinctive lavender accent.
MODEL_PALETTE = {
    "Claude 4.6":  "#e8e844",   # soft yellow (answer panel)
    "Claude 4.5":  "#e8e844",
    "GPT-5.5":     "#8cbfac",   # sage/teal green (header box)
    "Gemini 3.1":  "#efa130",   # warm orange (clipboards)
    "DeepSeek V4": "#4a7fb5",   # steel blue
    "Kimi K2.6":   "#d1615d",   # brick red
    "Qwen3 32B":   "#7f9c6c",   # olive green
    "Mistral 24B": "#a68a64",   # tan / khaki
    "CRIT (ours)": "#b56fbf",   # lavender (person panel)
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_CRIT_SUFFIX = " + CRIT"

# Base models whose `+ CRIT` column is drawn in the paper's bias figures. CRIT
# is run on every model, but plotting all seven variants doubles the legend to
# 14 entries and packs each language/category group with bars too thin to read.
# The figures therefore show every baseline plus Gemini's CRIT variant as the
# representative; the complete set stays in the appendix tables, which are built
# from the unfiltered run list.
PLOT_CRIT_MODELS = ("Gemini 3.1",)


def _figure_runs(runs):
    """Filter a run list for plotting: drop `<model> + CRIT` columns whose base
    model is not in PLOT_CRIT_MODELS. Baselines are always kept."""
    kept = []
    for run in runs:
        label = run[0]
        if (label.endswith(_CRIT_SUFFIX)
                and label[:-len(_CRIT_SUFFIX)] not in PLOT_CRIT_MODELS):
            continue
        kept.append(run)
    return kept


def _model_color(label, index):
    """Bar color for a model column. A `<model> + CRIT` column takes its base
    model's hue — with CRIT run on every model, a single shared lavender would
    render seven columns indistinguishable. Method is encoded by hatch instead
    (see `_model_hatch`)."""
    if label in MODEL_PALETTE:
        return MODEL_PALETTE[label]
    if label.endswith(_CRIT_SUFFIX):
        base = label[:-len(_CRIT_SUFFIX)]
        if base in MODEL_PALETTE:
            return MODEL_PALETTE[base]
        return MODEL_PALETTE["CRIT (ours)"]
    return f"C{index}"


def _model_hatch(label):
    """Hatch pattern for a model column: CRIT variants are hatched, baselines
    plain, so a (color, hatch) pair identifies (model, method)."""
    return "//" if label.endswith(_CRIT_SUFFIX) else ""


def _legend_ncol(model_labels):
    """Legend columns for a bottom-anchored figure legend. One row reads best,
    but the entries run past the figure width beyond five columns, so wrap onto
    further rows instead. Capped at four because the `<model> + CRIT` labels are
    roughly twice the width of a bare model name."""
    n = max(len(model_labels), 1)
    if n <= 5:
        return n
    return min(4, (n + 1) // 2)


def _fit_bottom_legend(fig, handles, labels, model_labels, fontsize,
                       plot_h, min_width, pad=0.35):
    """Attach a bottom-centred figure legend and resize `fig` around it.

    Adding a CRIT column per model doubles the legend's entry count, which both
    widens it past the axes and grows it from two rows to four. At a fixed
    figure size the extra rows land on top of the x tick labels, so measure the
    legend and give it its own strip instead:

      width  -> at least the legend's, so the plot spans the same canvas rather
                than floating inside a legend-width image (`bbox_inches="tight"`
                sizes the PNG to whichever is wider). Free in the paper, where
                the PNG is scaled to \\linewidth either way.
      height -> `plot_h` for the axes plus the legend's measured height, so rows
                are added to the figure rather than taken out of the plot.
    """
    legend = fig.legend(handles, labels, loc="lower center",
                        bbox_to_anchor=(0.5, 0.0),
                        ncol=_legend_ncol(model_labels),
                        fontsize=fontsize, frameon=False)
    fig.canvas.draw()  # legend extents are only known once it has been laid out
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    strip_h = bbox.height + pad
    fig.set_size_inches(max(min_width, bbox.width + 0.3), plot_h + strip_h)
    fig.tight_layout(rect=[0, strip_h / (plot_h + strip_h), 1, 1])
    return legend


def _parse_id_list(value):
    """Coerce the eval row's `pred` field (commonly a list-of-one JSON string)
    into a set of plaid IDs."""
    if value is None:
        return set()
    if isinstance(value, list):
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


def load_bias_labels(path: str = BIAS_LABELS_PATH):
    labels = {}
    for line in open(path):
        r = json.loads(line)
        labels[int(r["test_case_id"])] = r
    return labels


_MODEL_TYPES = ("baseline", "threshold", "reflection")


def discover_runs(roots=RESULTS_GLOB):
    """Return list of (column_label, model_id, model_type, path).

    `roots` is a results root (or list of roots) laid out as
    `<root>/<model_id>/<model_type>/<timestamp>/<file>_results.jsonl`. Each
    root is searched recursively because `model_id` spans a variable number of
    directories: one for a flat slug (`gpt-5`), two for a Baseten-style
    `org/model` (`deepseek-ai/DeepSeek-V4-Pro`), three for a self-hosted vLLM
    slug (`local/Qwen/Qwen3-32B-AWQ`). model_type/timestamp are always the two
    directories directly above the file, so model_id is whatever precedes them.

    The latest timestamped directory wins when multiple exist per
    (model_id, model_type). Two slugs that share a display name (the same
    model served by two providers) also share a column; the lexically latest
    timestamp across both wins.
    """
    root_list = [roots] if isinstance(roots, str) else list(roots)
    by_key = {}
    for root in root_list:
        root = os.path.normpath(root)
        for path in glob.glob(os.path.join(root, "**", "*_results.jsonl"),
                              recursive=True):
            rel_parts = os.path.relpath(path, root).split(os.sep)
            if len(rel_parts) < 4:
                continue  # not a <model>/<type>/<timestamp>/<file> layout
            model_type = rel_parts[-3]
            timestamp = rel_parts[-2]
            if model_type not in _MODEL_TYPES:
                continue
            model_id = "/".join(rel_parts[:-3])
            key = (MODEL_DISPLAY.get(model_id, model_id), model_type)
            prev = by_key.get(key)
            if prev is None or prev[1] < timestamp:
                by_key[key] = (path, timestamp, model_id)

    # CRIT is a wrapper around a base model, so every threshold column names
    # the model it wraps ("Gemini 3.1 + CRIT"). An unqualified "CRIT (ours)"
    # column reads as a standalone system next to a column of baselines, which
    # overstates coverage whenever only some models have threshold runs — and
    # identical labels would silently overwrite each other in the per-column
    # grids downstream.
    runs = []
    for (disp, model_type), (path, _ts, model_id) in by_key.items():
        if model_type == "threshold":
            column = f"{disp} + CRIT"
        elif model_type == "baseline":
            column = disp
        else:
            column = f"{disp} {model_type}"
        runs.append((column, model_id, model_type, path))

    # A "<model> + CRIT" column sorts immediately after its own baseline
    # rather than at the end of the table, so each pair reads together (the
    # same convention as Table 2 in the paper).
    order = {name: i for i, name in enumerate(MODEL_ORDER)}

    def _sort_key(r):
        label = r[0]
        base = label[:-len(" + CRIT")] if label.endswith(" + CRIT") else label
        return (order.get(base, 99), label.endswith(" + CRIT"), label)

    runs.sort(key=_sort_key)
    return runs


def _trap_descriptions(label):
    """
    Return the set of injected-transaction description strings to expect in
    the bank-statement payload for a given bias-study label. Built from
    stable label fields (entity_name / platform + direction) so it survives
    regenerations of the dataset that reshuffle plaid IDs.
    """
    descs = set()
    entity = label.get("entity_name")
    if entity:
        # Used by both the names study (foreign-origin entity_name like
        # "Gold Star Trading") and the savings study (fund_name like
        # "Chit FUND"). Single description per case.
        descs.add(f"ACH CREDIT - {entity}")
    platform = label.get("platform")
    if platform:
        # Money-transfer study. Direction determines `ACH Credit` /
        # `ACH Debit` prefix (current spec); both transactions in the
        # injected pair share the description so one entry covers both.
        # Legacy descriptions (`Credit - …` / `Debit - …`) without the ACH
        # prefix are still listed so plots regenerate against older eval
        # results from before the prefix change.
        direction = label.get("direction", "")
        if direction == "credit":
            descs.add(f"ACH Credit - {platform}")
            descs.add(f"Credit - {platform}")
        elif direction == "debit":
            descs.add(f"ACH Debit - {platform}")
            descs.add(f"Debit - {platform}")
        else:
            descs.add(f"ACH Credit - {platform}")
            descs.add(f"ACH Debit - {platform}")
            descs.add(f"Credit - {platform}")
            descs.add(f"Debit - {platform}")
    return descs


def _extract_trap_ids_from_result(result_row, trap_descriptions):
    """
    Walk the eval result's embedded bank_statement payload for transactions
    whose Description matches one of the expected trap strings; return the
    set of their TransactionIDs. Falls back to {} if the payload is missing
    or unparseable.
    """
    if not trap_descriptions:
        return set()
    bs = result_row.get("bank_statement")
    if isinstance(bs, str):
        try:
            bs = json.loads(bs)
        except (TypeError, json.JSONDecodeError):
            return set()
    if not isinstance(bs, dict):
        return set()
    out = set()
    # Post-rebuild "Transactions" array (eval.py reads this; reliable for
    # baseline + threshold runs alike).
    for t in bs.get("Transactions", []) or []:
        if t.get("Description") in trap_descriptions:
            tid = t.get("TransactionID")
            if tid:
                out.add(tid)
    # Fallback: pre-rebuild `override_accounts` shape (the dict eval.py
    # passes through unchanged if it ever skips the rebuild step).
    if not out:
        for acc in bs.get("override_accounts", []) or []:
            for t in acc.get("transactions", []) or []:
                if t.get("description") in trap_descriptions:
                    tid = t.get("transaction_id")
                    if tid:
                        out.add(tid)
    return out


def retrieval_per_run(labels, results_path):
    """
    For each result row, decide whether the model retrieved the bias-study
    trap transaction. Trap IDs are derived per-result from the embedded
    bank_statement using stable description patterns rebuilt from the label
    (entity_name / platform + direction). This survives regenerations of
    the labels file that reseed bank statements and reshuffle plaid IDs.

    Returns {test_case_id: bool_retrieved_trap}.
    """
    out = {}
    for line in open(results_path):
        r = json.loads(line)
        qid = r.get("question_id")
        if qid is None:
            continue
        try:
            tc_id = int(qid)
        except (TypeError, ValueError):
            continue
        label = labels.get(tc_id)
        if label is None:
            continue
        trap_ids = _extract_trap_ids_from_result(r, _trap_descriptions(label))
        if not trap_ids:
            # Fallback to the label's stored IDs (works when the result was
            # produced under the same regen as the current labels file).
            trap_ids = set(label.get("bias_injected_transaction_ids") or [])
        pred = _parse_id_list(r.get("pred"))
        out[tc_id] = bool(trap_ids & pred)
    return out


# ---------------------------------------------------------------------------
# Aggregation + printing
# ---------------------------------------------------------------------------


def build_table(labels, runs, modification_type):
    by_lang = defaultdict(list)
    for tc_id, label in labels.items():
        if label["modification_type"] == modification_type:
            by_lang[label["language"]].append(tc_id)
    languages = sorted(by_lang)

    columns = [r[0] for r in runs]
    rate_grid = {l: {} for l in languages}
    count_grid = {l: {} for l in languages}

    for run in runs:
        col, path = run[0], run[3]
        retrieved = retrieval_per_run(labels, path)
        for lang in languages:
            ids = by_lang[lang]
            scored = [retrieved[i] for i in ids if i in retrieved]
            total = len(scored)
            hits = sum(1 for x in scored if x)
            rate_grid[lang][col] = (hits / total) if total else None
            count_grid[lang][col] = (hits, total)

    return languages, columns, rate_grid, count_grid


def print_table(title, languages, columns, rate_grid, count_grid):
    print()
    print(f"=== {title} ===")
    if not columns:
        print("  (no runs found)")
        return
    if not languages:
        print("  (no cases for this modification_type)")
        return

    lang_w = max(len("Language"), *(len(l) for l in languages))
    col_ws = [max(len(c), 14) for c in columns]

    header = "Language".ljust(lang_w) + " | " + " | ".join(c.center(w) for c, w in zip(columns, col_ws))
    sep = "-" * lang_w + "-+-" + "-+-".join("-" * w for w in col_ws)
    print(header)
    print(sep)
    for lang in languages:
        cells = []
        for col, w in zip(columns, col_ws):
            rate = rate_grid[lang][col]
            hits, total = count_grid[lang][col]
            if rate is None or total == 0:
                cells.append("--".center(w))
            else:
                cells.append(f"{rate*100:5.1f}% ({hits}/{total})".center(w))
        print(lang.ljust(lang_w) + " | " + " | ".join(cells))

    # Aggregate row.
    cells = []
    for col, w in zip(columns, col_ws):
        hits = sum(count_grid[l][col][0] for l in languages)
        total = sum(count_grid[l][col][1] for l in languages)
        if total == 0:
            cells.append("--".center(w))
        else:
            cells.append(f"{hits/total*100:5.1f}% ({hits}/{total})".center(w))
    print(sep)
    print("ALL".ljust(lang_w) + " | " + " | ".join(cells))


# ---------------------------------------------------------------------------
# LaTeX tables — paper-ready output of the same retrieval-rate data the
# text tables above print. Two tables:
#
#   Table 1 (`tab:bias_foreign`) — rows = 4 models. Columns: a standalone
#   English baseline (Company / Personal) plus two multicolumn groups
#   (Company name / Personal name) over the 8 non-English language variants
#   (5 native + 3 Latin transliterations marked with †).
#
#   Table 2 (`tab:bias_savings`) — rows = 4 models, columns = the 16 fund
#   names from BIAS_STUDY_SAVINGS_CLUB_FUNDS.
#
# Cell = % of cases for that (model, column) where the model retrieved the
# bias-injected trap transaction; "-" when no overlap between eval results
# and labels.
# ---------------------------------------------------------------------------

_LANG_ABBR = {
    "english":        ("EN", False),
    "mandarin":       ("ZH", False),
    "mandarin_latin": ("ZH", True),
    "hindi":          ("HI", False),
    "hindi_latin":    ("HI", True),
    "spanish":        ("ES", False),
    "arabic":         ("AR", False),
    "arabic_latin":   ("AR", True),
    "french":         ("FR", False),
}
# Latin transliteration sits immediately next to its native-script parent.
# Order: Romance/Latin-script languages first, then non-Latin (each followed
# by its Latin transliteration).
_NONEN_LANG_ORDER = [
    "spanish",
    "french",
    "arabic", "arabic_latin",
    "hindi", "hindi_latin",
    "mandarin", "mandarin_latin",
]
# Generic / culture-neutral baselines first (MutAid, P2P, Microfin), then a
# divider, then culturally-specific names. The plot draws a dotted vertical
# line between the two groups.
_FUND_SHORT = {
    "MUTUAL AID FUND":   "MutAid",
    "ROSCA FUND":        "ROSCA",
    "PEER LENDING FUND": "Peer",
    "Chit FUND":         "Chit",
    "Tontines FUND":     "Tontines",
    "Hui FUND":          "Hui",
    "Tanomoshi FUND":    "Tanomoshi",
    "Kye FUND":          "Kye",
    "Susu FUND":         "Susu",
    "Pardner FUND":      "Pardner",
    "Ajo FUND":          "Ajo",
    "Stokvels FUND":     "Stokvels",
    "Arisan FUND":       "Arisan",
    "Paluwagan FUND":    "Paluwagan",
    "Cundina FUND":      "Cundina",
}
_FUND_GENERIC_COUNT = 3  # MutAid, ROSCA, Peer

# Negative cases for the savings-club question — money-transfer / remittance
# systems that are NOT savings clubs. English-branded controls (WUnion,
# Venmo, Zelle) come first as the "no-bias-expected" baseline; foreign-named
# remittance systems follow on the right. Ground truth = [] for every case
# here, so a non-zero retrieval rate is the model wrongly classifying
# transfer as savings (bias signal).
_MONEY_TRANSFER_SHORT = {
    "WESTERN UNION": "WUnion",
    "VENMO":         "Venmo",
    "ZELLE":         "Zelle",
    "PAYPAL":        "PayPal",
    "HAWALA":        "Hawala",
    "HUNDI":         "Hundi",
    "HUIKUAN":       "Huikuan",
    "PADALA":        "Padala",
    # FEIQIAN intentionally omitted — its test cases still exist in
    # bias_study_savings/ from the prior generation but are excluded from
    # both the LaTeX table and the plot.
}
_MT_GENERIC_COUNT = 4  # WUnion, Venmo, Zelle, PayPal (English-branded controls)

# Primary cultural / regional association for each fund variant. The first
# three (MutAid, ROSCA, Peer) are intentionally culture-neutral baselines.
_FUND_REGION = {
    "MUTUAL AID FUND":   "Non-specific",
    "ROSCA FUND":        "Non-specific",
    "PEER LENDING FUND": "Non-specific",
    "Chit FUND":         "India",
    "Tontines FUND":     "West Africa",
    "Hui FUND":          "China",
    "Tanomoshi FUND":    "Japan",
    "Kye FUND":          "Korea",
    "Susu FUND":         "Ghana",
    "Pardner FUND":      "Jamaica",
    "Ajo FUND":          "Nigeria",
    "Stokvels FUND":     "South Africa",
    "Arisan FUND":       "Indonesia",
    "Paluwagan FUND":    "Philippines",
    "Cundina FUND":      "Mexico",
    # HAWALA excluded — see _FUND_SHORT comment above.
}


def _latex_escape(s):
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )


def _lang_header(lang):
    abbr, transliterated = _LANG_ABBR[lang]
    return abbr + (r"$^*$" if transliterated else "")


def _retrieval_rate(retrieved_map, ids):
    scored = [retrieved_map[i] for i in ids if i in retrieved_map]
    if not scored:
        return None
    return 100.0 * sum(1 for x in scored if x) / len(scored)


def build_latex_tables(labels, runs):
    """Return (foreign_table, savings_table) — two LaTeX strings."""
    by_mod_lang = {}
    by_fund = {}
    en_company_ids, en_person_ids = [], []
    for tc_id, lab in labels.items():
        mod = lab["modification_type"]
        lang = lab["language"]
        name = lab["entity_name"]
        if mod in ("company", "name"):
            by_mod_lang.setdefault((mod, lang), []).append(tc_id)
            if lang == "english" and mod == "company":
                en_company_ids.append(tc_id)
            elif lang == "english" and mod == "name":
                en_person_ids.append(tc_id)
        elif mod == "savings_org":
            by_fund.setdefault(name, []).append(tc_id)

    model_retrieved = [(col, retrieval_per_run(labels, path))
                       for col, _, _, path in runs]

    def fmt(rate):
        return "-" if rate is None else f"{rate:.0f}"

    # ── Table 1 — foreign-origin bias ──────────────────────────────────────
    # Each group: en (slightly separated) + 8 non-English language variants +
    # a Δ column, mirroring the gap columns of the money-transfer table.
    # Δ pools the five non-transliterated non-English languages and subtracts
    # the English rate; transliterated variants are excluded so Δ measures the
    # effect of the name's own language, not of the script conversion.
    _DELTA_LANGS = ["spanish", "french", "arabic", "hindi", "mandarin"]
    n_per_group = 2 + len(_NONEN_LANG_ORDER)
    # `c@{\hspace{8pt}}` adds a small gap between the English baseline column
    # and the non-English variants, and again before the Δ column.
    group_spec = ("c@{\\hspace{8pt}}" + "c" * len(_NONEN_LANG_ORDER)
                  + "@{\\hspace{8pt}}c")
    tabular_spec = "l" + group_spec + group_spec

    def delta(retrieved, mod, en_ids):
        non_ids = [i for l in _DELTA_LANGS
                   for i in by_mod_lang.get((mod, l), [])]
        r_non = _retrieval_rate(retrieved, non_ids)
        r_en = _retrieval_rate(retrieved, en_ids)
        if r_non is None or r_en is None:
            return "-"
        return f"{r_non - r_en:+.0f}"

    header_cells = (
        r"\multirow{2}{*}{Model} & "
        r"\multicolumn{" + str(n_per_group) + r"}{c}{Company name} & "
        r"\multicolumn{" + str(n_per_group) + r"}{c}{Personal name} \\"
    )
    cmid = (
        r"\cmidrule(lr){2-" + str(1 + n_per_group) + "} "
        r"\cmidrule(lr){" + str(2 + n_per_group) + "-"
                          + str(1 + 2 * n_per_group) + "}"
    )
    one_group = (" & EN & " + " & ".join(
        _lang_header(l) for l in _NONEN_LANG_ORDER) + r" & $\Delta$")
    subhdr = one_group + one_group + r" \\"

    body_rows = []
    for col, retrieved in model_retrieved:
        cells = [_latex_escape(col),
                 fmt(_retrieval_rate(retrieved, en_company_ids))]
        for lang in _NONEN_LANG_ORDER:
            cells.append(fmt(_retrieval_rate(
                retrieved, by_mod_lang.get(("company", lang), []))))
        cells.append(delta(retrieved, "company", en_company_ids))
        cells.append(fmt(_retrieval_rate(retrieved, en_person_ids)))
        for lang in _NONEN_LANG_ORDER:
            cells.append(fmt(_retrieval_rate(
                retrieved, by_mod_lang.get(("name", lang), []))))
        cells.append(delta(retrieved, "name", en_person_ids))
        body_rows.append(" & ".join(cells) + r" \\")

    # 19 columns do not fit a single text column, so this is a full-width
    # float; \resizebox absorbs whatever the natural width overshoots by.
    foreign_tbl = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{" + tabular_spec + "}\n"
        "\\toprule\n"
        + header_cells + "\n"
        + cmid + "\n"
        + subhdr + "\n"
        "\\midrule\n"
        + "\n".join(body_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\caption{Foreign-origin retrieval rate as a function of the language of the sender's name. "
        "* indicates that the name has been transliterated into Latin script. "
        "$\\Delta$ = non-English $-$ EN, pooled over the five non-transliterated non-English languages.}\n"
        "\\label{tab:bias_foreign}\n"
        "\\end{table*}\n"
    )

    # ── Table 2 — savings-club bias ────────────────────────────────────────
    # Partition by polarity: positives (cultural savings clubs, ground truth
    # = [trap_id], cell = recall) and negatives (money-transfer systems,
    # ground truth = [], cell = false-positive rate / bias signal).
    pos_ids = {f: ids for f, ids in by_fund.items() if f in _FUND_SHORT}
    neg_ids = {f: ids for f, ids in by_fund.items() if f in _MONEY_TRANSFER_SHORT}
    pos_order = [k for k in _FUND_SHORT if k in pos_ids]
    neg_order = [k for k in _MONEY_TRANSFER_SHORT if k in neg_ids]

    if not pos_order and not neg_order:
        return foreign_tbl, ""

    n_pos, n_neg = len(pos_order), len(neg_order)
    hdr_parts = []
    if n_pos:
        hdr_parts.append(r"\multicolumn{" + str(n_pos)
                         + r"}{c}{Positives — true savings clubs (recall)}")
    if n_neg:
        hdr_parts.append(r"\multicolumn{" + str(n_neg)
                         + r"}{c}{Negatives — money transfer (false-positive rate)}")
    header_row = r"\multirow{2}{*}{Model} & " + " & ".join(hdr_parts) + r" \\"

    cmid_parts = []
    col_idx = 2
    if n_pos:
        cmid_parts.append(r"\cmidrule(lr){" + str(col_idx) + "-"
                          + str(col_idx + n_pos - 1) + "}")
        col_idx += n_pos
    if n_neg:
        cmid_parts.append(r"\cmidrule(lr){" + str(col_idx) + "-"
                          + str(col_idx + n_neg - 1) + "}")
    cmid_row = " ".join(cmid_parts)

    subhdr_cells = ([_FUND_SHORT[f] for f in pos_order]
                    + [_MONEY_TRANSFER_SHORT[f] for f in neg_order])
    subhdr = " & " + " & ".join(subhdr_cells) + r" \\"

    fund_rows = []
    for col, retrieved in model_retrieved:
        cells = [_latex_escape(col)]
        for fund in pos_order:
            cells.append(fmt(_retrieval_rate(retrieved, pos_ids[fund])))
        for fund in neg_order:
            cells.append(fmt(_retrieval_rate(retrieved, neg_ids[fund])))
        fund_rows.append(" & ".join(cells) + r" \\")

    # Column spec: positives, optional gap, negatives.
    col_spec = "l"
    if n_pos:
        col_spec += "c" * n_pos
    if n_pos and n_neg:
        col_spec += "@{\\hspace{8pt}}"
    if n_neg:
        col_spec += "c" * n_neg

    savings_tbl = (
        "\\begin{table*}[H]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{" + col_spec + "}\n"
        "\\toprule\n"
        + header_row + "\n"
        + cmid_row + "\n"
        + subhdr + "\n"
        "\\midrule\n"
        + "\n".join(fund_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Savings-club trap-transaction retrieval rate (\\%). "
        "Positives are genuine rotating savings / credit associations whose "
        "injected ACH credit is tagged as a savings-club deposit "
        "(higher recall is better). Negatives are informal value-transfer / "
        "remittance systems with the same ACH-credit framing but tagged "
        "as ordinary deposits (lower is better; non-zero is name-driven "
        "bias).}\n"
        "\\label{tab:bias_savings}\n"
        "\\end{table*}\n"
    )

    return foreign_tbl, savings_tbl


def build_bias_plot(labels, runs,
                    output_path_foreign="bias_foreign_plot.png",
                    output_path_savings_positives="bias_savings_positives.png",
                    output_path_savings_negatives="bias_savings_negatives.png"):
    """Render per-(model, language/fund) retrieval rates as separate PNGs.

    `output_path_foreign` — 2 panels (company-name + personal-name foreign-
    origin bias). 9 lang groups (EN first) × N models per panel.
    `output_path_savings_positives` — true savings clubs, plotted as recall.
    `output_path_savings_negatives` — money-transfer systems, plotted as FPR
    with a per-model Δ column = avg(foreign FPR) − avg(generic FPR).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # ── Build per-panel data ────────────────────────────────────────────────
    by_mod_lang = {}
    by_fund = {}
    en_company_ids, en_person_ids = [], []
    for tc_id, lab in labels.items():
        mod = lab["modification_type"]
        lang = lab["language"]
        name = lab["entity_name"]
        if mod in ("company", "name"):
            by_mod_lang.setdefault((mod, lang), []).append(tc_id)
            if lang == "english" and mod == "company":
                en_company_ids.append(tc_id)
            elif lang == "english" and mod == "name":
                en_person_ids.append(tc_id)
        elif mod == "savings_org":
            by_fund.setdefault(name, []).append(tc_id)

    model_retrieved = [(col, retrieval_per_run(labels, path))
                       for col, _, _, path in runs]
    model_labels = [m[0] for m in model_retrieved]

    def _rate_or_zero(retrieved, ids):
        r = (lambda rm, i: (None if not [rm[x] for x in i if x in rm]
                            else 100.0 * sum(1 for x in i if x in rm and rm[x])
                                       / sum(1 for x in i if x in rm)))(retrieved, ids)
        return 0.0 if r is None else r

    foreign_lang_keys = ["english"] + _NONEN_LANG_ORDER
    foreign_lang_disp = ["EN"] + [
        (_LANG_ABBR[l][0] + ("*" if _LANG_ABBR[l][1] else ""))
        for l in _NONEN_LANG_ORDER
    ]

    company_grid = np.array([
        [_rate_or_zero(rm,
                       en_company_ids if l == "english"
                       else by_mod_lang.get(("company", l), []))
         for l in foreign_lang_keys]
        for _, rm in model_retrieved
    ])
    person_grid = np.array([
        [_rate_or_zero(rm,
                       en_person_ids if l == "english"
                       else by_mod_lang.get(("name", l), []))
         for l in foreign_lang_keys]
        for _, rm in model_retrieved
    ])

    # Per-model bias delta = avg(ES, FR, AR, HI, ZH non-transliterated) − EN.
    # Quantifies how much the language of the entity name shifts the model's
    # foreign-classification rate above its English baseline.
    delta_langs = ["spanish", "french", "arabic", "hindi", "mandarin"]
    en_lang_idx = foreign_lang_keys.index("english")
    delta_idx = [foreign_lang_keys.index(l) for l in delta_langs]
    company_delta = (company_grid[:, delta_idx].mean(axis=1)
                     - company_grid[:, en_lang_idx])
    person_delta = (person_grid[:, delta_idx].mean(axis=1)
                    - person_grid[:, en_lang_idx])
    company_grid = np.hstack([company_grid, company_delta.reshape(-1, 1)])
    person_grid = np.hstack([person_grid, person_delta.reshape(-1, 1)])
    foreign_lang_disp = foreign_lang_disp + ["Δ"]

    # Savings funds, partitioned by polarity.
    pos_order = [k for k in _FUND_SHORT if k in by_fund]
    neg_order = [k for k in _MONEY_TRANSFER_SHORT if k in by_fund]
    pos_disp = [_FUND_SHORT[f] for f in pos_order]
    neg_disp = [_MONEY_TRANSFER_SHORT[f] for f in neg_order]

    def _fund_grid(order):
        if not order:
            return np.empty((len(model_retrieved), 0))
        return np.array([
            [_rate_or_zero(rm, by_fund[f]) for f in order]
            for _, rm in model_retrieved
        ])

    pos_grid = _fund_grid(pos_order)
    neg_grid = _fund_grid(neg_order)

    # ── Plot ────────────────────────────────────────────────────────────────
    colors = [_model_color(m, i) for i, m in enumerate(model_labels)]

    bar_w = 0.78 / max(len(model_labels), 1)

    def _draw(ax, grid, x_labels, title, divider_after=None, tick_rotation=0,
              ymin=0):
        n_groups = grid.shape[1]
        x = np.arange(n_groups)
        stub_h = 1.2
        for i, m in enumerate(model_labels):
            offset = (i - (len(model_labels) - 1) / 2) * bar_w
            xs = x + offset
            # Stubs only for near-zero bars so the slot stays visible.
            # Non-zero bars (positive or negative) draw their own visible bar.
            near_zero = [abs(v) <= 0.01 for v in grid[i]]
            stub_xs = [xv for xv, nz in zip(xs, near_zero) if nz]
            if stub_xs:
                ax.bar(stub_xs, [stub_h] * len(stub_xs), bar_w,
                       color=colors[i], edgecolor="white", linewidth=0.4,
                       alpha=0.35, zorder=2)
            ax.bar(xs, grid[i], bar_w,
                   color=colors[i], label=m, edgecolor="white", linewidth=0.4,
                   hatch=_model_hatch(m), zorder=3)
            # Annotate exact-zero bars with an explicit "0" above the stub.
            for xi, v, nz in zip(xs, grid[i], near_zero):
                if nz:
                    ax.text(xi, stub_h + 1.5, "0", ha="center", va="bottom",
                            fontsize=14, color=colors[i], zorder=4)
        ax.set_xticks(x)
        if tick_rotation:
            ax.set_xticklabels(x_labels, fontsize=15,
                               rotation=tick_rotation, ha="right")
        else:
            ax.set_xticklabels(x_labels, fontsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_ylim(ymin, 108)
        if ymin < 0:
            ax.axhline(0, color="#444", linewidth=0.7, zorder=1)
        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        if divider_after is not None:
            positions = (divider_after if isinstance(divider_after, (list, tuple))
                         else [divider_after])
            for p in positions:
                ax.axvline(p + 0.5, color="#888", linestyle="--",
                           alpha=0.4, linewidth=0.8)

    # ── Helper: print a plot's grid as a text table ────────────────────────
    def _print_plot_grid(title, col_labels, grid):
        print(f"\n=== plot values: {title} ===")
        if grid.shape[1] == 0:
            print("  (no data)")
            return
        col_w = max(8, *(len(c) for c in col_labels))
        m_w = max(len("Model"), *(len(m) for m in model_labels))
        hdr = "Model".ljust(m_w) + " | " + " | ".join(
            c.center(col_w) for c in col_labels)
        sep = "-" * m_w + "-+-" + "-+-".join("-" * col_w for _ in col_labels)
        print(hdr)
        print(sep)
        for m, row in zip(model_labels, grid):
            cells = [f"{v:6.1f}".center(col_w) for v in row]
            print(m.ljust(m_w) + " | " + " | ".join(cells))

    # ── Figure 1: foreign-origin (two panels) ──────────────────────────────
    has_foreign = bool(en_company_ids or en_person_ids
                       or any(by_mod_lang.values()))
    if output_path_foreign and has_foreign:
        _print_plot_grid("Personal Name (foreign classification rate %)",
                         foreign_lang_disp, person_grid)
        _print_plot_grid("Company Name (foreign classification rate %)",
                         foreign_lang_disp, company_grid)
        fig1, axes1 = plt.subplots(2, 1, figsize=(14, 7))
        delta_col_idx = len(foreign_lang_disp) - 1
        _draw(axes1[0], person_grid, foreign_lang_disp,
              "Language of Personal Name",
              divider_after=[0, delta_col_idx - 1])
        _draw(axes1[1], company_grid, foreign_lang_disp,
              "Language of Company Name",
              divider_after=[0, delta_col_idx - 1])
        fig1.supylabel("Foreign Classification Rate", fontsize=15, fontweight="bold")
        handles, labels_ = axes1[0].get_legend_handles_labels()
        _fit_bottom_legend(fig1, handles, labels_, model_labels,
                           fontsize=15, plot_h=5.95, min_width=14)
        plt.savefig(output_path_foreign, dpi=300, bbox_inches="tight")
        plt.close(fig1)
        print(f"Wrote foreign-origin bias plot to {output_path_foreign}")

    # ── Figure 2a: positives — true savings clubs (recall) ────────────────
    if output_path_savings_positives and pos_disp:
        _print_plot_grid("Savings-Positives (recall %)", pos_disp, pos_grid)
        fig_pos, ax_pos = plt.subplots(1, 1, figsize=(14, 4.5))
        _draw(ax_pos, pos_grid, pos_disp,
              "Positives - True Savings Clubs",
              divider_after=_FUND_GENERIC_COUNT - 1
              if len(pos_disp) > _FUND_GENERIC_COUNT else None,
              tick_rotation=45)
        ax_pos.set_ylabel("Recall (%)", fontsize=15, fontweight="bold")
        handles, labels_ = ax_pos.get_legend_handles_labels()
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        fig_pos.legend(handles, labels_, loc="lower center",
                       bbox_to_anchor=(0.5, 0.0),
                       ncol=_legend_ncol(model_labels), fontsize=15, frameon=False)
        plt.savefig(output_path_savings_positives, dpi=300, bbox_inches="tight")
        plt.close(fig_pos)
        print(f"Wrote savings-positives plot to {output_path_savings_positives}")

    # ── Figure 2b: negatives — money transfer (FPR + Δ) ───────────────────
    if output_path_savings_negatives and neg_disp:
        # Per-model Δ column = avg(foreign FPR) − avg(generic FPR). Negative
        # funds are ordered [generic ..., foreign ...] (see
        # _MONEY_TRANSFER_SHORT), so split at _MT_GENERIC_COUNT.
        g = _MT_GENERIC_COUNT
        n_neg = neg_grid.shape[1]
        has_delta = n_neg > g
        if has_delta:
            gen_means = neg_grid[:, :g].mean(axis=1)
            for_means = neg_grid[:, g:].mean(axis=1)
            delta_col = (for_means - gen_means).reshape(-1, 1)
        else:
            delta_col = None

        # Print combined values (FPR columns + Δ if present).
        printed_disp = neg_disp + (["Δ"] if has_delta else [])
        printed_grid = (np.hstack([neg_grid, delta_col])
                        if has_delta else neg_grid)
        _print_plot_grid("Savings-Negatives (FPR %; Δ = foreign − generic)",
                         printed_disp, printed_grid)

        # Single subplot: FPR columns followed by Δ column, both regions set
        # off by dotted dividers (one between generic and foreign, one
        # between foreign and Δ).
        if has_delta:
            grid = np.hstack([neg_grid, delta_col])
            labels = neg_disp + ["Δ"]
            dividers = [g - 1, n_neg - 1]
        else:
            grid = neg_grid
            labels = neg_disp
            dividers = [g - 1] if n_neg > g else None

        fig_neg, ax_neg = plt.subplots(1, 1, figsize=(14, 4))
        _draw(ax_neg, grid, labels, "", divider_after=dividers)
        ax_neg.set_ylabel("False-Positive Rate (%)",
                          fontsize=30, fontweight="bold")
        handles, labels_ = ax_neg.get_legend_handles_labels()
        plt.tight_layout(rect=[0, 0.10, 1, 1])
        fig_neg.legend(handles, labels_, loc="lower center",
                       bbox_to_anchor=(0.5, 0.0),
                       ncol=_legend_ncol(model_labels), fontsize=15, frameon=False)
        plt.savefig(output_path_savings_negatives, dpi=300, bbox_inches="tight")
        plt.close(fig_neg)
        print(f"Wrote savings-negatives plot to {output_path_savings_negatives}")


def build_money_transfer_plot(labels, runs,
                              output_path_platforms="bias_money_transfer_platforms.png",
                              output_path_questions="bias_money_transfer_questions.png",
                              output_path_categories="bias_money_transfer_categories.png"):
    """
    Two PNGs for the money-transfer-platform bias study with three platform
    categories: generic (US-branded), foreign (modern foreign platforms),
    and traditional (informal value-transfer systems).

    `output_path_platforms` — single panel; per-(model, platform) FPR
    averaged across all active questions. Bars are ordered
    [generic] | [foreign] | [traditional] | Δ₁ | Δ₂, with dotted dividers.
        Δ₁ = avg(foreign FPR) − avg(generic FPR)
        Δ₂ = avg(traditional FPR) − avg(foreign FPR)

    `output_path_questions` — two-panel figure; per-(model, question)
    deltas. Top panel = Δ₁ per question. Bottom panel = Δ₂ per question.
    Positive bars mean the more-foreign category was flagged more.

    `output_path_categories` — single panel, three bars per model: average
    FPR across all platforms in each category (US P2P / Non-US P2P / IVTSs).
    The simplest one-glance summary of the three-category gap.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from data_mutator import DataMutator
    generic_order = list(DataMutator.BIAS_STUDY_MT_PLATFORMS_GENERIC)
    foreign_order = list(DataMutator.BIAS_STUDY_MT_PLATFORMS_FOREIGN)
    traditional_order = list(DataMutator.BIAS_STUDY_MT_PLATFORMS_TRADITIONAL)
    question_order = [q["key"] for q in DataMutator.BIAS_STUDY_MT_QUESTIONS]

    by_platform = {}
    by_q_origin = {}
    for tc_id, lab in labels.items():
        by_platform.setdefault(
            (lab["platform_origin"], lab["platform"]), []).append(tc_id)
        by_q_origin.setdefault(
            (lab["question_key"], lab["platform_origin"]), []).append(tc_id)

    model_retrieved = [(col, retrieval_per_run(labels, path))
                       for col, _, _, path in runs]
    model_labels = [m[0] for m in model_retrieved]

    def _rate_or_zero(retrieved, ids):
        scored = [retrieved[i] for i in ids if i in retrieved]
        if not scored:
            return 0.0
        return 100.0 * sum(1 for x in scored if x) / len(scored)

    colors = [_model_color(m, i) for i, m in enumerate(model_labels)]
    bar_w = 0.78 / max(len(model_labels), 1)

    # ── Per-platform grids ────────────────────────────────────────────────
    def _platform_grid(order, origin):
        if not order:
            return np.empty((len(model_labels), 0))
        return np.array([
            [_rate_or_zero(rm, by_platform.get((origin, p), []))
             for p in order]
            for _, rm in model_retrieved
        ])

    gen_grid = _platform_grid(generic_order, "generic")
    for_grid = _platform_grid(foreign_order, "foreign")
    trad_grid = _platform_grid(traditional_order, "traditional")

    # Δ columns. Each is shape (n_models, 1). Avg-of-platforms makes the
    # delta robust to per-platform sparsity.
    def _avg_or_nan(grid):
        return grid.mean(axis=1) if grid.size else np.zeros(len(model_labels))

    gen_avg = _avg_or_nan(gen_grid)
    for_avg = _avg_or_nan(for_grid)
    trad_avg = _avg_or_nan(trad_grid)
    delta1 = (for_avg - gen_avg).reshape(-1, 1)
    delta2 = (trad_avg - for_avg).reshape(-1, 1)

    plat_grid = np.hstack([gen_grid, for_grid, trad_grid, delta1, delta2])
    plat_labels = (generic_order + foreign_order + traditional_order
                   + ["Δ₁ (F−G)", "Δ₂ (T−F)"])

    # ── Per-question deltas ───────────────────────────────────────────────
    q_keys = [q for q in question_order
              if any((q, o) in by_q_origin for o in _MT_ORIGINS)]

    def _per_q_rate(rm, q, origin):
        return _rate_or_zero(rm, by_q_origin.get((q, origin), []))

    delta1_grid = np.array([
        [(_per_q_rate(rm, q, "foreign") - _per_q_rate(rm, q, "generic"))
         for q in q_keys]
        for _, rm in model_retrieved
    ]) if q_keys else np.empty((len(model_labels), 0))
    delta2_grid = np.array([
        [(_per_q_rate(rm, q, "traditional") - _per_q_rate(rm, q, "foreign"))
         for q in q_keys]
        for _, rm in model_retrieved
    ]) if q_keys else np.empty((len(model_labels), 0))

    if delta1_grid.size:
        delta1_grid = np.hstack([delta1_grid, delta1_grid.mean(axis=1).reshape(-1, 1)])
        delta2_grid = np.hstack([delta2_grid, delta2_grid.mean(axis=1).reshape(-1, 1)])
        q_labels = q_keys + ["ALL"]
    else:
        q_labels = []

    def _draw_bars(ax, grid, x_labels, ylabel, divider_after=None, rot=30,
                   show_zero_axis=False, ymin_lim=None):
        n = grid.shape[1]
        x = np.arange(n)
        # Annotate exactly-zero cells with a small "0" so they're not
        # invisible — gives the reader the same scan-rhythm as nonzero bars.
        zero_offset_fraction = 0.018  # of full y-axis range, applied after ylim is set
        for i, m in enumerate(model_labels):
            offset = (i - (len(model_labels) - 1) / 2) * bar_w
            ax.bar(x + offset, grid[i], bar_w, color=colors[i], label=m,
                   edgecolor="white", linewidth=0.4,
                   hatch=_model_hatch(m), zorder=3)
        ax.set_xticks(x)
        # Rotated labels need right-alignment to keep the tick under the
        # *end* of the text; unrotated labels should sit centered under the tick.
        ax.set_xticklabels(x_labels, fontsize=23, rotation=rot,
                           ha="right" if rot else "center")
        ax.tick_params(axis="y", labelsize=26)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_ylabel(ylabel, fontsize=27, fontweight="bold")
        if show_zero_axis:
            ax.axhline(0, color="#444", linewidth=0.7, zorder=1)
            grid_min = float(grid.min()) if grid.size else 0
            grid_max = float(grid.max()) if grid.size else 0
            lo = min(0, grid_min) - 5
            hi = max(0, grid_max) + 5
            ax.set_ylim(lo if ymin_lim is None else min(lo, ymin_lim), hi)
        else:
            ax.set_ylim(0, 108)
        if divider_after is not None:
            positions = (divider_after if isinstance(divider_after, (list, tuple))
                         else [divider_after])
            for p in positions:
                ax.axvline(p + 0.5, color="#888", linestyle="--",
                           alpha=0.4, linewidth=0.8)
        # Now that ylim is finalised, paint "0" labels above any cells that
        # came in as exactly zero. Zero-height bars would otherwise vanish.
        ylo, yhi = ax.get_ylim()
        offset_px = (yhi - ylo) * zero_offset_fraction
        for i, m in enumerate(model_labels):
            x_off = (i - (len(model_labels) - 1) / 2) * bar_w
            for j, v in enumerate(grid[i]):
                if abs(v) < 1e-9:
                    ax.text(j + x_off, offset_px, "0",
                            ha="center", va="bottom",
                            fontsize=23, color=colors[i], zorder=4)

    def _print_grid(title, col_labels, grid, fmt="{:6.1f}"):
        print(f"\n=== plot values: {title} ===")
        if grid.shape[1] == 0:
            print("  (no data)")
            return
        col_w = max(8, *(len(c) for c in col_labels))
        m_w = max(len("Model"), *(len(m) for m in model_labels))
        print("Model".ljust(m_w) + " | " + " | ".join(c.center(col_w) for c in col_labels))
        print("-" * m_w + "-+-" + "-+-".join("-" * col_w for _ in col_labels))
        for m, row in zip(model_labels, grid):
            cells = [fmt.format(v).center(col_w) for v in row]
            print(m.ljust(m_w) + " | " + " | ".join(cells))

    # ── Platforms plot ────────────────────────────────────────────────────
    if output_path_platforms and plat_grid.size:
        _print_grid("Money-Transfer Platforms (FPR %; Δ₁=for−gen, Δ₂=trad−for)",
                    plat_labels, plat_grid)
        divider_positions = [
            len(generic_order) - 1,
            len(generic_order) + len(foreign_order) - 1,
            len(generic_order) + len(foreign_order) + len(traditional_order) - 1,
            len(generic_order) + len(foreign_order) + len(traditional_order),
        ]
        fig, ax = plt.subplots(1, 1, figsize=(15, 4.8))
        _draw_bars(ax, plat_grid, plat_labels,
                   "False-Positive Rate (%)",
                   divider_after=divider_positions, rot=30, show_zero_axis=True)
        ax.set_title("Money-Transfer Platforms — FPR by category "
                     f"(averaged across {len(q_keys)} questions)",
                     fontsize=30, fontweight="bold")
        handles, labels_ = ax.get_legend_handles_labels()
        plt.tight_layout(rect=[0, 0.10, 1, 1])
        fig.legend(handles, labels_, loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), ncol=_legend_ncol(model_labels),
                   fontsize=21, frameon=False)
        plt.savefig(output_path_platforms, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote money-transfer platforms plot to {output_path_platforms}")

    # ── Per-question deltas: two-panel figure (Δ₁ top, Δ₂ bottom) ─────────
    if output_path_questions and delta1_grid.size:
        _print_grid("Per-Question Δ₁ (foreign − generic FPR %)",
                    q_labels, delta1_grid, fmt="{:+6.1f}")
        _print_grid("Per-Question Δ₂ (traditional − foreign FPR %)",
                    q_labels, delta2_grid, fmt="{:+6.1f}")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        div = [len(q_labels) - 2] if len(q_labels) > 1 else None
        _draw_bars(ax1, delta1_grid, q_labels,
                   "Δ₁ FPR (Foreign − Generic, %)",
                   divider_after=div, rot=30, show_zero_axis=True)
        ax1.set_title("Foreign vs Generic — per-question FPR gap",
                      fontsize=30, fontweight="bold")
        _draw_bars(ax2, delta2_grid, q_labels,
                   "Δ₂ FPR (Traditional − Foreign, %)",
                   divider_after=div, rot=30, show_zero_axis=True)
        ax2.set_title("Traditional vs Foreign — per-question FPR gap",
                      fontsize=30, fontweight="bold")
        handles, labels_ = ax1.get_legend_handles_labels()
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        fig.legend(handles, labels_, loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), ncol=_legend_ncol(model_labels),
                   fontsize=21, frameon=False)
        plt.savefig(output_path_questions, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote money-transfer per-question plot to {output_path_questions}")

    # ── Per-category summary (US P2P / Non-US P2P / IVTSs + deltas) ───────
    # One bar per (model, column) where columns are the three pooled-FPR
    # categories plus two pairwise deltas:
    #   Δ₁ = Non-US P2P − US P2P
    #   Δ₂ = IVTSs      − Non-US P2P
    # Pooling means the cells equal what the significance test sees.
    if output_path_categories:
        cat_display = {
            "generic":     "US EPS",
            "foreign":     "Non-US\nEPS",
            "traditional": "IVTS",
        }
        cat_keys = list(_MT_ORIGINS)

        ids_by_origin = {o: [] for o in cat_keys}
        for tc_id, lab in labels.items():
            o = lab.get("platform_origin")
            if o in ids_by_origin:
                ids_by_origin[o].append(tc_id)

        rate_grid = np.array([
            [_rate_or_zero(rm, ids_by_origin[o]) for o in cat_keys]
            for _, rm in model_retrieved
        ])
        # Δ columns sit alongside the categories so the gap is readable in
        # the same panel. Indices: 0=US, 1=Non-US, 2=IVTSs.
        d1 = (rate_grid[:, 1] - rate_grid[:, 0]).reshape(-1, 1)
        d2 = (rate_grid[:, 2] - rate_grid[:, 1]).reshape(-1, 1)
        cat_grid = np.hstack([rate_grid, d1, d2])
        cat_labels_x = [cat_display[o] for o in cat_keys] + ["Δ₁", "Δ₂"]

        _print_grid("Money-Transfer Category Averages (FPR %; Δ in pp)",
                    cat_labels_x, cat_grid)

        fig, ax = plt.subplots(1, 1, figsize=(12.7, 4.5))
        # show_zero_axis=True so negative Δ bars render below the baseline.
        _draw_bars(ax, cat_grid, cat_labels_x,
                   "Recall Rate",
                   divider_after=[len(cat_keys) - 1],
                   rot=0, show_zero_axis=True)
        handles, labels_ = ax.get_legend_handles_labels()
        _fit_bottom_legend(fig, handles, labels_, model_labels,
                           fontsize=20, plot_h=3.5, min_width=12.7)
        plt.savefig(output_path_categories, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote money-transfer category-average plot to {output_path_categories}")


_MT_ORIGINS = ("generic", "foreign", "traditional")

# Column headers for the per-platform LaTeX table. Long brand names are
# abbreviated so all sixteen platforms fit across a `table*`.
_MT_PLATFORM_SHORT = {
    "Western Union": "WUnion",
    "Orange Money":  "Orange",
    "Mercado Pago":  "MercPago",
}


def build_money_transfer_latex_table(labels, runs):
    """Return a LaTeX string: per-(model, platform) false-positive rate for the
    money-transfer study, grouped US EPS / non-US EPS / IVTS, with the two
    aggregate gaps Δ₁ (non-US EPS − US EPS) and Δ₂ (IVTS − non-US EPS).

    Every cell is a false-positive rate: ground truth is the empty list for
    each of these injected transactions, so any non-zero value is the platform
    name alone driving the model to flag it.
    """
    from data_mutator import DataMutator
    groups = [
        ("US EPS",     "generic",     list(DataMutator.BIAS_STUDY_MT_PLATFORMS_GENERIC)),
        ("Non-US EPS", "foreign",     list(DataMutator.BIAS_STUDY_MT_PLATFORMS_FOREIGN)),
        ("IVTS",       "traditional", list(DataMutator.BIAS_STUDY_MT_PLATFORMS_TRADITIONAL)),
    ]

    by_platform, by_origin = {}, {}
    for tc_id, lab in labels.items():
        by_platform.setdefault((lab["platform_origin"], lab["platform"]),
                               []).append(tc_id)
        by_origin.setdefault(lab["platform_origin"], []).append(tc_id)

    # Drop platforms with no cases left after the saturated-question filter, so
    # the table never carries an all-`-` column.
    groups = [(title, origin, [p for p in plats if (origin, p) in by_platform])
              for title, origin, plats in groups]
    groups = [g for g in groups if g[2]]
    if not groups or not runs:
        return ""

    model_retrieved = [(col, retrieval_per_run(labels, path))
                       for col, _, _, path in runs]

    def fmt(rate):
        return "-" if rate is None else f"{rate:.0f}"

    def delta(retrieved, a, b):
        ra = _retrieval_rate(retrieved, by_origin.get(a, []))
        rb = _retrieval_rate(retrieved, by_origin.get(b, []))
        if ra is None or rb is None:
            return "-"
        return f"{ra - rb:+.0f}"

    # ── Column spec / headers ─────────────────────────────────────────────
    # A small gap separates each origin group, and a wider one sets the two
    # delta columns apart from the raw rates.
    col_spec = "l" + "@{\\hspace{6pt}}".join("c" * len(p) for _, _, p in groups)
    col_spec += "@{\\hspace{8pt}}cc"

    hdr_parts = [r"\multicolumn{" + str(len(plats)) + r"}{c}{" + title + "}"
                 for title, _, plats in groups]
    header_row = (r"\multirow{2}{*}{Model} & " + " & ".join(hdr_parts)
                  + r" & \multicolumn{2}{c}{Gap} \\")

    cmid_parts, col_idx = [], 2
    for _, _, plats in groups:
        cmid_parts.append(r"\cmidrule(lr){" + str(col_idx) + "-"
                          + str(col_idx + len(plats) - 1) + "}")
        col_idx += len(plats)
    cmid_parts.append(r"\cmidrule(lr){" + str(col_idx) + "-"
                      + str(col_idx + 1) + "}")
    cmid_row = " ".join(cmid_parts)

    subhdr_cells = [_latex_escape(_MT_PLATFORM_SHORT.get(p, p))
                    for _, _, plats in groups for p in plats]
    subhdr = (" & " + " & ".join(subhdr_cells)
              + r" & $\Delta_1$ & $\Delta_2$ \\")

    body_rows = []
    for col, retrieved in model_retrieved:
        cells = [_latex_escape(col)]
        for _, origin, plats in groups:
            for p in plats:
                cells.append(fmt(_retrieval_rate(
                    retrieved, by_platform.get((origin, p), []))))
        cells.append(delta(retrieved, "foreign", "generic"))
        cells.append(delta(retrieved, "traditional", "foreign"))
        body_rows.append(" & ".join(cells) + r" \\")

    # Nineteen columns overrun even a full-width float at \footnotesize, so the
    # tabular is scaled to \textwidth rather than hand-tuned per platform count.
    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{" + col_spec + "}\n"
        "\\toprule\n"
        + header_row + "\n"
        + cmid_row + "\n"
        + subhdr + "\n"
        "\\midrule\n"
        + "\n".join(body_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\caption{Retrieval rate for money transfers as a function of transfer type. $\\Delta_1$ = non-US EPS $-$ US EPS "
        "and $\\Delta_2$ = IVTS $-$ non-US EPS, each pooled over all platforms "
        "in the group.}\n"
        "\\label{tab:bias_money_transfer}\n"
        "\\end{table*}\n"
    )


def build_money_transfer_table(labels, runs):
    """
    Aggregate the money-transfer-platform bias dataset across three origin
    buckets — generic (US-branded), foreign (modern foreign platforms), and
    traditional (informal value-transfer systems with strong cultural
    associations).

    Returns (question_keys, columns, rate_grid, count_grid) where each cell
    is keyed by (question_key, (model_col, origin)). A non-zero rate is
    name-driven bias: the model flagged an `ACH Credit / ACH Debit -
    <platform>` transaction tagged "default" as matching the question's
    criterion, even though ground truth is the empty list.
    """
    by_question = defaultdict(lambda: defaultdict(list))
    for tc_id, label in labels.items():
        by_question[label["question_key"]][label["platform_origin"]].append(tc_id)

    try:
        from data_mutator import DataMutator
        registry_order = [q["key"] for q in DataMutator.BIAS_STUDY_MT_QUESTIONS]
        question_keys = [k for k in registry_order if k in by_question]
    except Exception:
        question_keys = sorted(by_question)

    rate_grid = {q: {} for q in question_keys}
    count_grid = {q: {} for q in question_keys}
    columns = [r[0] for r in runs]

    for run in runs:
        col, path = run[0], run[3]
        retrieved = retrieval_per_run(labels, path)
        for q in question_keys:
            for origin in _MT_ORIGINS:
                ids = by_question[q].get(origin, [])
                scored = [retrieved[i] for i in ids if i in retrieved]
                total = len(scored)
                hits = sum(1 for x in scored if x)
                rate_grid[q][(col, origin)] = (hits / total) if total else None
                count_grid[q][(col, origin)] = (hits, total)

    return question_keys, columns, rate_grid, count_grid


def print_money_transfer_table(question_keys, columns, rate_grid, count_grid):
    """
    Per-model, per-question FPR across the three origins with two delta
    columns:
      Δ₁ = foreign     − generic        (modern foreign vs US baseline)
      Δ₂ = traditional − foreign        (traditional informal vs modern)
    Cell shorthand: G=generic, F=foreign, T=traditional.
    """
    print()
    print("=== Money-transfer platform bias (FPR; Δ₁=for−gen, Δ₂=trad−for) ===")
    if not columns:
        print("  (no runs found)")
        return
    if not question_keys:
        print("  (no cases)")
        return

    q_w = max(len("Question"), *(len(q) for q in question_keys))
    rate_w = 8
    delta_w = 7

    header_cells = []
    cell_widths = []
    for col in columns:
        header_cells.append(f"{col} G".center(rate_w))
        header_cells.append(f"{col} F".center(rate_w))
        header_cells.append(f"{col} T".center(rate_w))
        header_cells.append(f"{col} Δ₁".center(delta_w))
        header_cells.append(f"{col} Δ₂".center(delta_w))
        cell_widths.extend([rate_w, rate_w, rate_w, delta_w, delta_w])
    sep = "-" * q_w + "-+-" + "-+-".join("-" * w for w in cell_widths)
    print("Question".ljust(q_w) + " | " + " | ".join(header_cells))
    print(sep)

    def _rate_cell(rate, width):
        return ("--" if rate is None else f"{rate*100:5.1f}%").center(width)

    def _delta_cell(a, b, width):
        if a is None or b is None:
            return "--".center(width)
        return f"{(a - b)*100:+5.1f}".center(width)

    for q in question_keys:
        cells = []
        for col in columns:
            g = rate_grid[q].get((col, "generic"))
            f = rate_grid[q].get((col, "foreign"))
            t = rate_grid[q].get((col, "traditional"))
            cells.append(_rate_cell(g, rate_w))
            cells.append(_rate_cell(f, rate_w))
            cells.append(_rate_cell(t, rate_w))
            cells.append(_delta_cell(f, g, delta_w))
            cells.append(_delta_cell(t, f, delta_w))
        print(q.ljust(q_w) + " | " + " | ".join(cells))

    # ALL row — aggregate across questions per (model, origin).
    cells = []
    for col in columns:
        sums = {o: [sum(count_grid[q][(col, o)][0] for q in question_keys),
                    sum(count_grid[q][(col, o)][1] for q in question_keys)]
                for o in _MT_ORIGINS}
        rates = {o: (sums[o][0] / sums[o][1]) if sums[o][1] else None
                 for o in _MT_ORIGINS}
        cells.append(_rate_cell(rates["generic"], rate_w))
        cells.append(_rate_cell(rates["foreign"], rate_w))
        cells.append(_rate_cell(rates["traditional"], rate_w))
        cells.append(_delta_cell(rates["foreign"], rates["generic"], delta_w))
        cells.append(_delta_cell(rates["traditional"], rates["foreign"], delta_w))
    print(sep)
    print("ALL".ljust(q_w) + " | " + " | ".join(cells))


def print_money_transfer_significance(labels, runs):
    """
    For each model, run two 2×2 contingency tests:
        • generic vs foreign       (Δ₁ — is foreign branding flagged more than US?)
        • foreign vs traditional   (Δ₂ — does the gap widen for traditional names?)
    Reports the foreign-vs-generic χ² + p, then a Fisher's exact p for each
    pairwise comparison side-by-side. Rows where either comparison clears
    α=0.05 (Fisher's) are marked with `*`.
    """
    try:
        from scipy.stats import chi2_contingency, fisher_exact
    except ImportError:
        print("\n=== Significance test skipped — scipy not installed ===")
        return

    print("\n=== Statistical significance: per-model pairwise FPR tests ===")
    print("H0 (per comparison): the two trap-retrieval rates are equal.")
    print(f"Operating on {len(labels)} cases (after dropping saturated questions).")
    if not runs:
        print("  (no runs)")
        return

    header = (f"{'Model':<14} | "
              f"{'Gen':>10} | {'For':>10} | {'Trad':>10} | "
              f"{'p (F vs G)':>11} | {'p (T vs F)':>11} |")
    print(header)
    print("-" * len(header))

    def _pct(hits, total):
        return f"{hits}/{total} ({hits/total*100:.1f}%)" if total else "  --   "

    def _fisher_p(hits_a, n_a, hits_b, n_b):
        if not n_a or not n_b:
            return None
        try:
            _, p = fisher_exact([[hits_a, n_a - hits_a],
                                 [hits_b, n_b - hits_b]])
            return p
        except ValueError:
            return None

    def _fmt_p(p):
        return f"{p:11.4f}" if p is not None else "    --     "

    for col, _, _, path in runs:
        retrieved = retrieval_per_run(labels, path)
        cnt = {o: [0, 0] for o in _MT_ORIGINS}  # origin -> [hits, total]
        for tc_id, lab in labels.items():
            if tc_id not in retrieved:
                continue
            hit = 1 if retrieved[tc_id] else 0
            o = lab["platform_origin"]
            if o in cnt:
                cnt[o][0] += hit
                cnt[o][1] += 1

        p_fg = _fisher_p(cnt["foreign"][0], cnt["foreign"][1],
                         cnt["generic"][0], cnt["generic"][1])
        p_tf = _fisher_p(cnt["traditional"][0], cnt["traditional"][1],
                         cnt["foreign"][0], cnt["foreign"][1])
        sig = " *" if (p_fg is not None and p_fg < 0.05) or (p_tf is not None and p_tf < 0.05) else "  "

        print(f"{col:<14} | "
              f"{_pct(*cnt['generic']):>10} | {_pct(*cnt['foreign']):>10} | "
              f"{_pct(*cnt['traditional']):>10} | "
              f"{_fmt_p(p_fg)} | {_fmt_p(p_tf)} |{sig}")

    print("  (Significant at α=0.05 in either pairwise test marked with *; "
          "Fisher's exact used throughout — robust to small expected counts.)")


def _safe_load(path):
    """Return labels dict or None if path doesn't exist."""
    if not os.path.exists(path):
        print(f"  (skipping — {path} not found)")
        return None
    return load_bias_labels(path)


def _saturated_mt_question_keys(labels, runs):
    """
    Return the set of money-transfer question_keys that are saturated for
    every model — i.e. every model retrieves the trap on 0% of that question's
    cases, or every model retrieves it on 100%. Saturated rows carry no bias
    signal and just clutter the table / plot, so we drop them upstream.

    Returns an empty set when there are no runs (nothing to judge against).
    """
    if not runs:
        return set()
    by_q = defaultdict(list)
    for tc_id, lab in labels.items():
        by_q[lab["question_key"]].append(tc_id)

    model_retrieved = [retrieval_per_run(labels, path) for _, _, _, path in runs]
    saturated = set()
    for q, ids in by_q.items():
        rates = []
        for retrieved in model_retrieved:
            scored = [retrieved[i] for i in ids if i in retrieved]
            if scored:
                rates.append(sum(1 for x in scored if x) / len(scored))
        if rates and (all(r == 0.0 for r in rates) or all(r == 1.0 for r in rates)):
            saturated.add(q)
    return saturated


def main():
    # ── Names dataset (foreign-origin question) ─────────────────────────────
    print("=== bias_study_names ===")
    names_labels = _safe_load(BIAS_LABELS_PATH_NAMES)
    names_runs = discover_runs(RESULTS_GLOB_NAMES) if names_labels else []
    if names_labels and names_runs:
        print(f"Loaded {len(names_labels)} cases; "
              f"{len(names_runs)} run(s): {[r[0] for r in names_runs]}")
        for mod_type, title in (
            ("company", "Foreign-origin — company-name modification"),
            ("name",    "Foreign-origin — personal-name modification"),
        ):
            langs, cols, rates, counts = build_table(names_labels, names_runs, mod_type)
            print_table(title, langs, cols, rates, counts)
        foreign_tbl, _ = build_latex_tables(names_labels, names_runs)
    else:
        foreign_tbl = ""
        if names_labels:
            print(f"No result files matching {RESULTS_GLOB_NAMES}")

    # ── Savings dataset (private-savings-club question) ─────────────────────
    print("\n=== bias_study_savings ===")
    savings_labels = _safe_load(BIAS_LABELS_PATH_SAVINGS)
    savings_runs = discover_runs(RESULTS_GLOB_SAVINGS) if savings_labels else []
    if savings_labels and savings_runs:
        print(f"Loaded {len(savings_labels)} cases; "
              f"{len(savings_runs)} run(s): {[r[0] for r in savings_runs]}")
        langs, cols, rates, counts = build_table(savings_labels, savings_runs,
                                                  "savings_org")
        print_table("Savings-club — fund-name modification",
                    langs, cols, rates, counts)
        _, savings_tbl = build_latex_tables(savings_labels, savings_runs)
    else:
        savings_tbl = ""
        if savings_labels:
            print(f"No result files matching {RESULTS_GLOB_SAVINGS}")

    # ── Money-transfer dataset (per-question, per-platform-origin) ─────────
    print("\n=== bias_study_money_transfer ===")
    mt_labels = _safe_load(BIAS_LABELS_PATH_MT)
    mt_runs = discover_runs(RESULTS_GLOB_MT) if mt_labels else []
    if mt_labels and mt_runs:
        print(f"Loaded {len(mt_labels)} cases; "
              f"{len(mt_runs)} run(s): {[r[0] for r in mt_runs]}")
        # Drop questions that are at 0% or 100% across every model — they
        # have no bias signal and only inflate the table. The filtered
        # `mt_labels` flows through to the plot as well.
        saturated = _saturated_mt_question_keys(mt_labels, mt_runs)
        if saturated:
            print(f"  Dropping {len(saturated)} saturated question(s) "
                  f"(0% or 100% across all models): {sorted(saturated)}")
            mt_labels = {tc: lab for tc, lab in mt_labels.items()
                         if lab["question_key"] not in saturated}
            print(f"  → {len(mt_labels)} cases remain after filtering")
        qkeys, cols, rates, counts = build_money_transfer_table(mt_labels, mt_runs)
        print_money_transfer_table(qkeys, cols, rates, counts)
        mt_tbl = build_money_transfer_latex_table(mt_labels, mt_runs)
    else:
        mt_tbl = ""
        if mt_labels:
            print(f"No result files matching {RESULTS_GLOB_MT}")

    # ── LaTeX output ───────────────────────────────────────────────────────
    # `bias_tables.tex` holds the two experiments the paper reports (names and
    # money transfer) and is \input from the appendix. The savings study is not
    # in the paper, so its table goes to a separate file that nothing includes.
    if foreign_tbl or mt_tbl:
        out_path = os.path.join(LATEX_DIR, "bias_tables.tex")
        with open(out_path, "w") as f:
            f.write("% Bias-study LaTeX tables — generated by bias_analysis.py\n\n")
            if foreign_tbl:
                f.write(foreign_tbl)
                f.write("\n")
            if mt_tbl:
                f.write(mt_tbl)
        print(f"\nWrote LaTeX tables to {out_path}")
    if savings_tbl:
        out_path = os.path.join(LATEX_DIR, "bias_tables_savings.tex")
        with open(out_path, "w") as f:
            f.write("% Savings-study LaTeX table — generated by bias_analysis.py\n"
                    "% Not \\input anywhere: the savings study is not in the paper.\n\n")
            f.write(savings_tbl)
        print(f"Wrote savings LaTeX table to {out_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    # Written straight into the LaTeX source tree so a rerun updates the paper
    # with no manual copying; `\includegraphics{figs/…}` in acl_latex.tex
    # resolves relative to LATEX_DIR, hence the flat `figs/` layout.
    figs_dir = LATEX_FIGS_DIR
    os.makedirs(figs_dir, exist_ok=True)
    # Figures take the CRIT-filtered run list (see PLOT_CRIT_MODELS); the LaTeX
    # tables above were built from the full list and are unaffected.
    if names_labels and names_runs:
        build_bias_plot(names_labels, _figure_runs(names_runs),
                        output_path_foreign=os.path.join(figs_dir,
                                                         "bias_foreign_plot.png"),
                        output_path_savings_positives=None,
                        output_path_savings_negatives=None)
    if savings_labels and savings_runs:
        build_bias_plot(savings_labels, savings_runs,
                        output_path_foreign=None,
                        output_path_savings_positives=os.path.join(
                            figs_dir, "bias_savings_positives.png"),
                        output_path_savings_negatives=os.path.join(
                            figs_dir, "bias_savings_negatives.png"))
    if mt_labels and mt_runs:
        build_money_transfer_plot(
            mt_labels, _figure_runs(mt_runs),
            output_path_platforms=os.path.join(
                figs_dir, "bias_money_transfer_platforms.png"),
            output_path_questions=os.path.join(
                figs_dir, "bias_money_transfer_questions.png"),
            output_path_categories=os.path.join(
                figs_dir, "bias_money_transfer_categories.png"))
        print_money_transfer_significance(mt_labels, mt_runs)


if __name__ == "__main__":
    main()
