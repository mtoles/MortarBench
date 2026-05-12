import argparse
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

summaries_prompt = """List of summaries:

{summaries}

Analyze the summaries and categorize them into root cause categories. Try to use 3-6 categories, including one "Unknown" category.

Respond with a JSON array where each element corresponds to the category for each summary in order. Format:
["Category 1", "Category 2", "Category 3", ...]

Only output the JSON array, nothing else.
"""

def analyze_failures(input_path: str, model_id: str = "gemini-3-pro-preview", limit: int = None):
    """
    Analyze failures from a JSONL results file.

    Args:
        input_path: Path to the directory containing gpt-5_results.jsonl
        model_id: Model to use for analysis (default: gemini-3-pro-preview)
        limit: Optional limit on number of failures to analyze
    """
    # Construct paths
    input_dir = Path(input_path)
    jsonl_path = input_dir / "gpt-5_results.jsonl"
    output_path = input_dir / "failure_analysis.md"

    print(f"Reading failures from: {jsonl_path}")

    # Load and filter failures
    failures = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Filter for f1_score < 1
            if data['f1_score'][0] is not None and data['f1_score'][0] < 1.0:
                failures.append(data)

    print(f"Found {len(failures)} failures (f1 < 1.0)")

    # Apply limit if specified
    if limit is not None and limit > 0:
        failures = failures[:limit]
        print(f"Limited to {len(failures)} failures")

    # Analyze each failure with Gemini
    results = []
    for i, failure in enumerate(failures):
        print(f"Analyzing failure {i+1}/{len(failures)}...")

        # Format bank statement as transactions list
        bank_statement_obj = failure.get('bank_statement')
        if bank_statement_obj:
            transactions_flat = flatten_plaid_transactions(bank_statement_obj)
            bank_statement_str = json.dumps(transactions_flat, indent=2)
        else:
            bank_statement_str = "No bank statement available"

        # Get ULAD DU
        ulad_du_str = failure.get('ulad_du', "No ULAD DU available")

        # Format the compare prompt
        prompt = compare_prompt.format(
            question=failure['question'],
            bank_statement=bank_statement_str,
            ulad_du=ulad_du_str,
            ground_truth_answer=failure['answer'],
            predicted_answer=failure['pred'][0],
            predicted_answer_reasoning=failure['raw_answer'][0]
        )

        # Call Gemini
        messages = [{"role": "user", "content": prompt}]
        summary, _, _ = call_llm_wrapper(model_id, messages)

        # Store results
        results.append({
            'question_id': failure['question_id'],
            'question': failure['question'],
            'ground_truth': failure['answer'],
            'predicted': failure['pred'][0],
            'reasoning': failure['raw_answer'][0],
            'f1_score': failure['f1_score'][0],
            'summary': summary
        })

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

    # Write results to markdown
    print(f"Writing results to: {output_path}")
    with open(output_path, 'w') as f:
        f.write("# Failure Analysis\n\n")
        f.write(f"**Source:** `{input_path}`\n\n")
        f.write(f"**Total Failures Analyzed:** {len(results)}\n\n")
        f.write("---\n\n")

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

    args = parser.parse_args()
    input_path = args.input_path or default_input_path()
    print(f"Using input path: {input_path}")
    analyze_failures(input_path, model_id=args.model, limit=args.limit)
