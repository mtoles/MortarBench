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
RESULTS_GLOB_NAMES = "results/bias_study_names/rephrased_question/*/*/*/*_results.jsonl"
RESULTS_GLOB_SAVINGS = "results/bias_study_savings/rephrased_question/*/*/*/*_results.jsonl"

# Back-compat shim — the old single-dataset path. Kept as the names-dataset
# alias so callers that read these still get the foreign-origin labels.
BIAS_LABELS_PATH = BIAS_LABELS_PATH_NAMES
RESULTS_GLOB = RESULTS_GLOB_NAMES

MODEL_DISPLAY = {
    "claude-sonnet-4-5": "Claude 4.5",
    "claude-sonnet-4-6": "Claude 4.6",
    "gpt-5": "GPT-5.5",
    "gemini-3.1-pro-preview": "Gemini 3.1",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def discover_runs(glob_pat: str = RESULTS_GLOB):
    """Return list of (column_label, model_id, model_type, path).
    The latest timestamped directory wins when multiple exist per
    (model_id, model_type)."""
    by_key = {}
    for path in glob.glob(glob_pat):
        # .../<model_id>/<model_type>/<timestamp>/<file>
        parts = path.split(os.sep)
        model_type = parts[-3]
        model_id = parts[-4]
        timestamp = parts[-2]
        key = (model_id, model_type)
        prev = by_key.get(key)
        if prev is None or os.path.basename(os.path.dirname(prev)) < timestamp:
            by_key[key] = path

    runs = []
    for (model_id, model_type), path in by_key.items():
        disp = MODEL_DISPLAY.get(model_id, model_id)
        if model_type == "threshold":
            column = "CRIT (ours)"
        elif model_type == "baseline":
            column = disp
        else:
            column = f"{disp} {model_type}"
        runs.append((column, model_id, model_type, path))

    order = {"Claude 4.6": 0, "Claude 4.5": 0, "GPT-5.5": 1,
             "Gemini 3.1": 2, "CRIT (ours)": 3}
    runs.sort(key=lambda r: (order.get(r[0], 99), r[0]))
    return runs


def retrieval_per_run(labels, results_path):
    """Return {test_case_id: bool_retrieved_bias_txn}."""
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
        injected = set(label["bias_injected_transaction_ids"] or [])
        pred = _parse_id_list(r.get("pred"))
        out[tc_id] = bool(injected & pred)
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
    # Each group: en (slightly separated) + 8 non-English language variants.
    n_per_group = 1 + len(_NONEN_LANG_ORDER)
    # `c@{\hspace{8pt}}` adds a small gap between the English baseline column
    # and the non-English variants within the same multicolumn group.
    group_spec = "c@{\\hspace{8pt}}" + "c" * len(_NONEN_LANG_ORDER)
    tabular_spec = "l" + group_spec + group_spec

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
    subhdr = " & EN & " + " & ".join(
        _lang_header(l) for l in _NONEN_LANG_ORDER
    ) + " & EN & " + " & ".join(
        _lang_header(l) for l in _NONEN_LANG_ORDER
    ) + r" \\"

    body_rows = []
    for col, retrieved in model_retrieved:
        cells = [_latex_escape(col),
                 fmt(_retrieval_rate(retrieved, en_company_ids))]
        for lang in _NONEN_LANG_ORDER:
            cells.append(fmt(_retrieval_rate(
                retrieved, by_mod_lang.get(("company", lang), []))))
        cells.append(fmt(_retrieval_rate(retrieved, en_person_ids)))
        for lang in _NONEN_LANG_ORDER:
            cells.append(fmt(_retrieval_rate(
                retrieved, by_mod_lang.get(("name", lang), []))))
        body_rows.append(" & ".join(cells) + r" \\")

    foreign_tbl = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{" + tabular_spec + "}\n"
        "\\toprule\n"
        + header_cells + "\n"
        + cmid + "\n"
        + subhdr + "\n"
        "\\midrule\n"
        + "\n".join(body_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Foreign-origin trap-transaction retrieval rate "
        "(\\% of bias-study cases where the model's predicted ID list "
        "included the injected ACH-credit transaction; ground truth is empty "
        "for every column). Columns are language of the injected entity name "
        "(two-letter abbreviation; $^*$ = Latin transliteration of the "
        "non-Latin script). `EN' is the English baseline for each group.}\n"
        "\\label{tab:bias_foreign}\n"
        "\\end{table}\n"
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
    # Consistent per-model colors; CRIT (ours) gets a distinctive accent.
    palette = {
        "Claude 4.6":  "#d97757",   # warm orange
        "GPT-5.5":     "#10a37f",   # green
        "Gemini 3.1":  "#4285f4",   # blue
        "CRIT (ours)": "#7b3ff2",   # purple
    }
    colors = [palette.get(m, f"C{i}") for i, m in enumerate(model_labels)]

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
                   zorder=3)
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
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        fig1.legend(handles, labels_, loc="lower center",
                    bbox_to_anchor=(0.5, 0.0),
                    ncol=len(model_labels), fontsize=15, frameon=False)
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
                       ncol=len(model_labels), fontsize=15, frameon=False)
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
        ax_neg.set_ylabel("False-positive rate (%)",
                          fontsize=15, fontweight="bold")
        handles, labels_ = ax_neg.get_legend_handles_labels()
        plt.tight_layout(rect=[0, 0.10, 1, 1])
        fig_neg.legend(handles, labels_, loc="lower center",
                       bbox_to_anchor=(0.5, 0.0),
                       ncol=len(model_labels), fontsize=15, frameon=False)
        plt.savefig(output_path_savings_negatives, dpi=300, bbox_inches="tight")
        plt.close(fig_neg)
        print(f"Wrote savings-negatives plot to {output_path_savings_negatives}")


def _safe_load(path):
    """Return labels dict or None if path doesn't exist."""
    if not os.path.exists(path):
        print(f"  (skipping — {path} not found)")
        return None
    return load_bias_labels(path)


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

    # ── Combined LaTeX output ──────────────────────────────────────────────
    if foreign_tbl or savings_tbl:
        out_path = "bias_tables.tex"
        with open(out_path, "w") as f:
            f.write("% Bias-study LaTeX tables — generated by bias_analysis.py\n\n")
            if foreign_tbl:
                f.write(foreign_tbl)
                f.write("\n")
            if savings_tbl:
                f.write(savings_tbl)
        print(f"\nWrote LaTeX tables to {out_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    figs_dir = os.path.join("paper", "figs", "bias")
    os.makedirs(figs_dir, exist_ok=True)
    if names_labels and names_runs:
        build_bias_plot(names_labels, names_runs,
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


if __name__ == "__main__":
    main()
