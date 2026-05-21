import argparse
import concurrent.futures
import json
import os
import re
from pathlib import Path

from eval import flatten_plaid_transactions
from llm import call_llm_wrapper


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


def default_input_path(question_col="question", model="gpt-5", model_type="baseline"):
    """Resolve the latest paperN dir and the latest timestamped run inside it."""
    paper = latest_paper_dir()
    run_parent = os.path.join("results", paper, question_col, model, model_type)
    timestamps = sorted(os.listdir(run_parent))
    if not timestamps:
        raise FileNotFoundError(f"No run timestamps found under {run_parent}")
    return os.path.join(run_parent, timestamps[-1])

compare_prompt = """Question:
{question}

Bank Statement (Transactions):
{bank_statement}

ULAD DU:
{ulad_du}

Ground Truth Answer:
{ground_truth_answer}

Predicted Answer:
{predicted_answer}

Predicted Answer Reasoning:
{predicted_answer_reasoning}

Try to find the root cause of the wrong answer. Analyze the reasoning as well as the difference in the output. Note that there are three different types of questions: boolean, transaction list, and account list. Transaction and account list questions require that the ground truth and predicted answers have the exact same set of items. When they do not, analyze each missing or extra item individually. Try to guess what would have caused the model to inaccurately include or exclude an item. Respond with a 1-sentence summary of the root cause of the wrong answer.
"""

# summaries_prompt = """List of summaries:

# {summaries}

# Analyze the summaries and categorize them into root cause categories. Try to use 3-6 categories, including one "Unknown" category.

# Respond with a JSON array where each element corresponds to the category for each summary in order. Format:
# ["Category 1", "Category 2", "Category 3", ...]

# Only output the JSON array, nothing else.
# """

summaries_prompt = """List of summaries:

{summaries}

Analyze the summaries and categorize them into root causes. Possible root causes are:

Value Matching
Transaction Classification
Domain Knowledge
Prompt Constraint Misinterpretation
Unknown
Other

Respond with a JSON array where each element corresponds to the category for each summary in order. Format:
["Category 1", "Category 2", "Category 3", ...]

Only output the JSON array, nothing else.

"""

def _load_polarity_lookup(dataset_path="generated_data/test_cases_official/test_cases"):
    """Build {question_id (1-based): polarity} from questions.csv.

    eval.load_dataset assigns question_id = (CSV row index + 1), so this
    lookup gives us the ground-truth polarity without inference.
    """
    import pandas as pd
    csv_path = os.path.join(dataset_path, "questions.csv")
    df = pd.read_csv(csv_path)
    return {str(idx + 1): row["polarity"] for idx, row in df.iterrows()}


def pair_failure_correlation(rows, polarity_lookup):
    """For groups of same-question + same-polarity instances (expected 2 each),
    compute the conditional probability of getting the 2nd instance wrong given
    the 1st was wrong vs. right.

    Groups by (question_text, polarity) where polarity is read from
    `polarity_lookup` keyed on the row's `question_id`. Returns a dict of stats.
    Uses EM (>=1.0 == correct) for the binary signal.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    dropped_no_polarity = 0
    for r in rows:
        em = r.get("exact_match")
        if isinstance(em, list):
            em = em[0]
        if em is None:
            continue
        pol = polarity_lookup.get(str(r.get("question_id")))
        if pol is None:
            dropped_no_polarity += 1
            continue
        key = (r.get("question"), pol)
        groups[key].append(bool(em))

    # Count ordered (1st, 2nd) outcomes across all ordered pairs within each
    # group. With 2 instances per group, this is (a,b) and (b,a) — symmetric.
    n_pairs = 0          # ordered pairs (i, j), i != j
    n_1st_wrong = 0
    n_1st_right = 0
    n_2nd_wrong_given_1st_wrong = 0
    n_2nd_wrong_given_1st_right = 0
    groups_used = 0
    for outcomes in groups.values():
        if len(outcomes) < 2:
            continue
        groups_used += 1
        for i, a in enumerate(outcomes):
            for j, b in enumerate(outcomes):
                if i == j:
                    continue
                n_pairs += 1
                if not a:                       # 1st wrong
                    n_1st_wrong += 1
                    if not b:
                        n_2nd_wrong_given_1st_wrong += 1
                else:                           # 1st right
                    n_1st_right += 1
                    if not b:
                        n_2nd_wrong_given_1st_right += 1

    total_instances = sum(len(v) for v in groups.values())
    base_error_rate = (
        sum(1 for v in groups.values() for x in v if not x) / total_instances
        if total_instances else 0.0
    )

    p_2w_g_1w = (n_2nd_wrong_given_1st_wrong / n_1st_wrong) if n_1st_wrong else 0.0
    p_2w_g_1r = (n_2nd_wrong_given_1st_right / n_1st_right) if n_1st_right else 0.0
    return {
        "groups_used": groups_used,
        "groups_dropped_singleton": sum(1 for v in groups.values() if len(v) < 2),
        "groups_with_no_polarity": dropped_no_polarity,
        "ordered_pairs": n_pairs,
        "base_error_rate": base_error_rate,
        "p_2nd_wrong_given_1st_wrong": p_2w_g_1w,
        "p_2nd_wrong_given_1st_right": p_2w_g_1r,
        "lift": (p_2w_g_1w / p_2w_g_1r) if p_2w_g_1r else None,
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


def _build_failure_mode_latex(distribution, total, txn_only=False):
    """Build a paper-style LaTeX table (with plain-text comment block) for the
    failure-mode breakdown. Mirrors the format used in plot_results.py.

    `distribution` is a list of (category, count) tuples already sorted as desired.
    """
    rows = [(cat, n, n / total * 100) for cat, n in distribution]

    plain_header = ["Category", "Count", "% of Total"]
    plain_rows = [[cat, str(n), f"{pct:.1f}%"] for cat, n, pct in rows]
    widths = [
        max(len(plain_header[i]), *(len(r[i]) for r in plain_rows))
        for i in range(len(plain_header))
    ]
    def line(cells):
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
    sep = "-+-".join("-" * w for w in widths)
    plain_comment_lines = [
        "% Plain-text rendering of the table above:",
        "%   " + line(plain_header),
        "%   " + sep,
    ] + ["%   " + line(r) for r in plain_rows]
    plain_comment = "\n".join(plain_comment_lines) + "\n"

    body = "\n".join(
        f"{_latex_escape(cat)} & {n} & {pct:.1f}\\% \\\\"
        for cat, n, pct in rows
    )

    subset_note = " on transaction-list questions" if txn_only else ""
    caption = (
        f"Failure-mode breakdown over $N={total}$ wrong predictions{subset_note}."
    )
    label = "tab:failure_modes" + ("_txn" if txn_only else "")

    return (
        plain_comment +
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lrr}\n"
        "\\toprule\n"
        "Category & Count & \\% of Total \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


def analyze_failures(input_path: str, model_id: str = "gemini-3-pro-preview", limit: int = None, txn_only: bool = False):
    """
    Analyze failures from a JSONL results file.

    Args:
        input_path: Path to the directory containing gpt-5_results.jsonl
        model_id: Model to use for analysis (default: gemini-3-pro-preview)
        limit: Optional limit on number of failures to analyze
        txn_only: If True, only analyze failures with answer_type == "txn_id_list".
    """
    # Resolve the results jsonl regardless of which model produced it
    # (e.g. gpt-5_results.jsonl, gemini-3.1-pro-preview_results.jsonl).
    input_dir = Path(input_path)
    jsonl_candidates = sorted(input_dir.glob("*_results.jsonl"))
    if not jsonl_candidates:
        raise FileNotFoundError(f"No *_results.jsonl under {input_dir}")
    jsonl_path = jsonl_candidates[0]
    output_path = input_dir / "failure_analysis.md"

    print(f"Reading failures from: {jsonl_path}")

    # Load every row (the pair-correlation stats need both correct and wrong
    # answers), then split out the failures we'll send to the analyzer LLM.
    all_rows = []
    failures = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if txn_only and data.get('answer_type') != 'txn_id_list':
                continue
            all_rows.append(data)
            if data['f1_score'][0] is not None and data['f1_score'][0] < 1.0:
                failures.append(data)
    polarity_lookup = _load_polarity_lookup()
    pair_stats = pair_failure_correlation(all_rows, polarity_lookup)

    print(f"Found {len(failures)} failures (f1 < 1.0)" + (" [txn_id_list only]" if txn_only else ""))

    # Apply limit if specified
    if limit is not None and limit > 0:
        failures = failures[:limit]
        print(f"Limited to {len(failures)} failures")

    # Analyze each failure with the analyzer model in parallel. Modeled on the
    # ThreadPoolExecutor pattern in eval.evaluate_model.
    def analyze_one(failure):
        bank_statement_obj = failure.get('bank_statement')
        if bank_statement_obj:
            transactions_flat = flatten_plaid_transactions(bank_statement_obj)
            bank_statement_str = json.dumps(transactions_flat, indent=2)
        else:
            bank_statement_str = "No bank statement available"

        ulad_du_str = failure.get('ulad_du', "No ULAD DU available")

        prompt = compare_prompt.format(
            question=failure['question'],
            bank_statement=bank_statement_str,
            ulad_du=ulad_du_str,
            ground_truth_answer=failure['answer'],
            predicted_answer=failure['pred'][0],
            predicted_answer_reasoning=failure['raw_answer'][0]
        )
        messages = [{"role": "user", "content": prompt}]
        summary, _, _ = call_llm_wrapper(model_id, messages)
        return {
            'question_id': failure['question_id'],
            'question': failure['question'],
            'ground_truth': failure['answer'],
            'predicted': failure['pred'][0],
            'reasoning': failure['raw_answer'][0],
            'f1_score': failure['f1_score'][0],
            'summary': summary,
        }

    results = [None] * len(failures)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(10, len(failures))
    ) as executor:
        future_to_idx = {
            executor.submit(analyze_one, failure): idx
            for idx, failure in enumerate(failures)
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1
            print(f"Analyzed failure {completed}/{len(failures)}")

    # Generate categories for all summaries
    print("Generating categories for all summaries...")
    all_summaries = "\n".join([f"{i+1}. {r['summary']}" for i, r in enumerate(results)])
    category_prompt = summaries_prompt.format(summaries=all_summaries)
    messages = [{"role": "user", "content": category_prompt}]
    categories_text, _, _ = call_llm_wrapper(model_id, messages)

    # Parse categories from JSON using regex
    # Look for JSON array pattern in the response
    json_match = re.search(r'\[.*\]', categories_text, re.DOTALL)

    if json_match:
        try:
            categories = json.loads(json_match.group(0))
            print(f"Successfully parsed {len(categories)} categories from JSON")
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON: {e}")
            print(f"Response was: {categories_text}")
            # Fallback: use "Unknown" for all
            categories = ["Unknown"] * len(results)
    else:
        print(f"Warning: No JSON array found in response: {categories_text}")
        # Fallback: use "Unknown" for all
        categories = ["Unknown"] * len(results)

    # Handle mismatch between number of categories and summaries
    if len(categories) != len(results):
        print(f"Warning: Expected {len(results)} categories but got {len(categories)}")
        # Pad with Unknown if we have fewer categories
        while len(categories) < len(results):
            categories.append("Unknown")
        # Truncate if we have more categories
        categories = categories[:len(results)]

    # Category distribution (descending by count, ties broken alphabetically).
    from collections import Counter
    counts = Counter(categories)
    total = len(categories) or 1
    distribution = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    latex_table = _build_failure_mode_latex(distribution, total, txn_only=txn_only)

    # Write results to markdown
    print(f"Writing results to: {output_path}")
    with open(output_path, 'w') as f:
        f.write("# Failure Analysis\n\n")
        f.write(f"**Source:** `{input_path}`\n\n")
        f.write(f"**Total Failures Analyzed:** {len(results)}\n\n")

        f.write("## Pair Failure Correlation\n\n")
        f.write(
            "Each question appears 4× in the dataset (2 positive + 2 negative "
            "cases). Among groups of same-question + same-polarity instances, "
            "given the model got the *first* instance wrong, how often does it "
            "also get the second wrong (vs. when it got the first right)?\n\n"
        )
        ps = pair_stats
        lift_str = f"{ps['lift']:.2f}×" if ps["lift"] is not None else "n/a"
        f.write(
            f"- Groups used: {ps['groups_used']}\n"
            f"- Ordered pairs counted: {ps['ordered_pairs']}\n"
            f"- Base error rate (any instance): {ps['base_error_rate']*100:.1f}%\n"
            f"- P(2nd wrong | 1st wrong): "
            f"{ps['p_2nd_wrong_given_1st_wrong']*100:.1f}%\n"
            f"- P(2nd wrong | 1st right): "
            f"{ps['p_2nd_wrong_given_1st_right']*100:.1f}%\n"
            f"- Lift (clustering of failures): {lift_str}\n\n"
        )

        f.write("## Failure Mode Breakdown (LaTeX)\n\n")
        f.write("```latex\n")
        f.write(latex_table)
        f.write("```\n\n")
        f.write("## Category Distribution\n\n")
        f.write("| Category | Count | % of Total |\n")
        f.write("|---|---:|---:|\n")
        for cat, n in distribution:
            f.write(f"| {cat} | {n} | {n / total * 100:.1f}% |\n")
        f.write("\n---\n\n")

        for i, result in enumerate(results):
            f.write(f"## Failure {i+1}\n\n")
            f.write(f"**Question ID:** {result['question_id']}\n\n")
            f.write(f"**Question:** {result['question']}\n\n")
            f.write(f"**Ground Truth Answer:** {result['ground_truth']}\n\n")
            f.write(f"**Predicted Answer:** {result['predicted']}\n\n")
            f.write(f"**Predicted Answer Reasoning:**\n```\n{result['reasoning']}\n```\n\n")
            f.write(f"**F1 Score:** {result['f1_score']}\n\n")
            f.write(f"**Root Cause Summary:** {result['summary']}\n\n")
            f.write(f"**Category:** {categories[i]}\n\n")
            f.write("---\n\n")

    print(f"✓ Analysis complete! Results written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze failures from GPT-5 results')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of failures to analyze')
    parser.add_argument('--model', type=str, default="gemini-3-pro-preview",
                        help='Model to use for analysis (default: gemini-3-pro-preview)')
    parser.add_argument('--input-path', type=str, default=None,
                        help='Path to the directory containing gpt-5_results.jsonl. '
                             'Defaults to the latest run under the highest-numbered paperN directory.')
    parser.add_argument('--txn-only', action='store_true', default=False,
                        help='Only analyze failures with answer_type == "txn_id_list".')

    args = parser.parse_args()
    input_path = args.input_path or default_input_path()
    print(f"Using input path: {input_path}")
    analyze_failures(input_path, model_id=args.model, limit=args.limit, txn_only=args.txn_only)
