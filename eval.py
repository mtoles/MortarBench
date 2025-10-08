"""
Evaluate foundational models on the dataset.jsonl file.

Creates a .md summary of the results.

Models to test:
GPT-5
... will add more later, make this generalizable.

Options:
downsample_size: int
    number of samples to use for evaluation

model_id: str
    model id to use for evaluation

output_path: str
    path to save the results


"""

import datetime
import os
import json
import argparse
import random
import pandas as pd
from llm import call_llm_wrapper


domain_expertise = "A large deposit is defined as exceeding 50% of the borrower's total monthly qualifying income."

prompt = "Question: {question}\n\nBank Statement: {context}\n\nULAD DU: {ulad_du}\n\nAnswer the question. {domain_expertise}Do not think out loud. {answer_type}."

answer_type_dict = {
    "id_list": 'Answer with a valid JSON list of transaction/account IDs, e.g. `["d2rf4l6kq23ndu9seg6g", "d2rf4l6kq23ndu9seg60"]`, or [] if there are no IDs. When reporting accounts, use the AccountNumber, not the BankStatementAccountID.',
    "boolean": "Answer with yes or no.",
}

test_case_map = {6: 86744679655, 7: 81613557991, 8: 84192307554, 9: 83352063666, 10: 81301535410}


def load_test_case_json(test_case_number):
    """Load the JSON file for a given test case number."""
    json_path = f"data/Test Case {test_case_number} Docs/test_case_{test_case_number}_bank_statement_solo.json"

    if not os.path.exists(json_path):
        print(f"Warning: JSON file not found: {json_path}")
        return None

    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def load_ulad_xml(test_case_number):
    """Load the ULAD DU XML file for a given test case number."""
    xml_path = f"data/Test Case {test_case_number} Docs/ulad.xml"
    with open(xml_path, "r") as f:
        return f.read()


def preprocess_data(
    excel_path="data/Labeled Questions and Answers.xlsx",
    output_path="data/preprocessed_data.jsonl",
):
    """Preprocess the Excel data and test case docs into JSONL format."""
    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found: {excel_path}")
        return False

    try:
        df = pd.read_excel(excel_path, sheet_name="6-10", header=1)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return False

    print(f"Loaded {len(df)} rows from Excel file")
    print(f"Columns: {list(df.columns)}")

    with open(output_path, "w") as outfile:
        for idx, row in df.iterrows():
            # Extract common columns
            answer = row["Revised Answer V2"]
            test_case_number = row["Test Case Number"]
            answer_type = row["Answer Type"]
            question = row["Rephrased Question"]

            # Validate answer types and skip invalid ones
            if answer_type not in ["id_list", "boolean"]:
                raise ValueError(f"Invalid answer type: {answer_type}")

            # Skip if question is NaN, empty, or "n/a"/"N/A"
            if pd.isna(question) or str(question).strip().lower() in [
                "",
                "n/a",
                "na",
            ]:
                continue

            # Load corresponding JSON file
            bank_statement = load_test_case_json(int(test_case_number))
            if bank_statement is None:
                raise ValueError(
                    f"No bank statement data for test case {test_case_number}"
                )

            # Load corresponding ULAD DU XML file
            try:
                ulad_du = load_ulad_xml(int(test_case_number))
            except Exception as e:
                print(f"Warning: Skipping row {idx} - no ULAD DU data: {e}")
                continue

            # Create JSONL entry
            entry = {
                "loan_id": test_case_map[int(test_case_number)],
                "question": str(question).strip(),
                "answer": answer,
                "bank_statement": bank_statement,
                "ulad_du": ulad_du,
                "answer_type": answer_type,
            }

            # Write to JSONL file
            outfile.write(json.dumps(entry) + "\n")

    print(f"Successfully created {output_path}")
    return True


def load_dataset(dataset_path="data/preprocessed_data.jsonl"):
    """Load the JSONL dataset as a DataFrame."""
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            dataset.append(data)
    return pd.DataFrame(dataset)


def evaluate_model(model_id, df, use_domain_expertise, downsample_size=None):
    """Evaluate a model on the dataset."""
    # Keep a copy of the original dataset
    domain_expertise_str = (
        f"Domain Expertise: {domain_expertise}\n\n" if use_domain_expertise else ""
    )
    if downsample_size:
        df = df.sample(n=min(downsample_size, len(df)), random_state=42).reset_index(
            drop=True
        )

    results = []
    correct_count = 0
    total_count = len(df)

    for i, row in df.iterrows():
        loan_id = row["loan_id"]
        question = row["question"]
        answer = row["answer"]
        answer_type = row["answer_type"]
        bank_statement = json.dumps(row["bank_statement"], indent=2)
        ulad_du = row["ulad_du"]
        # Format the prompt
        formatted_prompt = prompt.format(
            question=question,
            context=bank_statement,
            ulad_du=ulad_du,
            domain_expertise=domain_expertise_str,
            answer_type=answer_type_dict[answer_type],
        )

        # Call the model
        if model_id == "solo":
            predicted_answer = call_llm_wrapper(
                model_id=model_id,
                messages=[{"role": "user", "content": question}],
                loan_id=loan_id,
            )
        else:
            predicted_answer = call_llm_wrapper(
                model_id=model_id,
                messages=[{"role": "user", "content": formatted_prompt}],
                loan_id=loan_id,
            )

        # Check if correct
        correct = is_correct(predicted_answer, answer_type, answer)
        if correct:
            correct_count += 1

        result = {
            "question": question,
            "true_answer": answer,
            "predicted_answer": predicted_answer,
            "answer_type": answer_type,
            "correct": correct,
        }
        results.append(result)

        print(f"Progress: {i+1}/{total_count} - Accuracy: {correct_count/(i+1):.3f}")

    accuracy = correct_count / total_count
    return results, accuracy, df


def is_correct(predicted_answer, answer_type, answer):
    answer = str(answer)
    if answer_type == "boolean":
        # Convert answer to yes/no format for comparison
        answer_str = answer.lower()
        if answer_str in ["yes", "true", "1"]:
            expected = "yes"
        elif answer_str in ["no", "false", "0"]:
            expected = "no"
        else:
            print(f"Warning: Invalid boolean answer value: {answer}")
            return False
        return predicted_answer.lower() == expected
    elif answer_type == "id_list":
        try:
            pred_as_list = json.loads(predicted_answer)
        except json.JSONDecodeError:
            print(f"warning: invalid json: {predicted_answer} for answer: {answer}")
            return False
        answer = answer.replace(" ", "").split(",")
        if isinstance(answer, str) and answer.lower() == "none":
            answer = []
        return set(pred_as_list) == set(answer)
    else:
        raise ValueError(f"Invalid answer type: {answer_type}")


def save_results(
    results,
    accuracy,
    model_id,
    output_dir,
    downsample_size=None,
    df=None,
):
    """Save evaluation results."""
    # Save results as JSONL with original data plus pred and correct columns
    if df is not None:
        # Add prediction and correctness columns to the dataframe
        df_with_results = df.copy()
        df_with_results["pred"] = [result["predicted_answer"] for result in results]
        df_with_results["correct"] = [result["correct"] for result in results]

        jsonl_path = f"{output_dir}/{model_id}_results.jsonl"
        df_with_results.to_json(jsonl_path, orient="records", lines=True)
        print(f"JSONL results saved to {jsonl_path}")

    # Create markdown summary
    summary_path = f"{output_dir}/{model_id}_summary.md"
    with open(summary_path, "w") as f:
        f.write(f"# Evaluation Results: {model_id}\n\n")
        f.write(
            f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write(f"**Model:** {model_id}\n\n")
        f.write(f"**Dataset Size:** {len(results)}")
        if downsample_size:
            f.write(f" (downsampled from original)")
        f.write("\n\n")
        f.write(f"**Overall Accuracy:** {accuracy:.3f} ({accuracy*100:.1f}%)\n\n")

        # Accuracy by answer type
        f.write("## Accuracy by Answer Type\n\n")
        answer_type_stats = {}
        for result in results:
            answer_type = result["answer_type"]
            if answer_type not in answer_type_stats:
                answer_type_stats[answer_type] = {"correct": 0, "total": 0}
            answer_type_stats[answer_type]["total"] += 1
            if result["correct"]:
                answer_type_stats[answer_type]["correct"] += 1

        for answer_type, stats in answer_type_stats.items():
            acc = stats["correct"] / stats["total"]
            f.write(
                f"- **{answer_type}:** {acc:.3f} ({acc*100:.1f}%) - {stats['correct']}/{stats['total']}\n"
            )

        f.write("\n## Sample Errors\n\n")
        errors = [r for r in results if not r["correct"]][:10]
        for i, error in enumerate(errors):
            f.write(f"### Error {i+1}\n")
            f.write(f"**Question:** {error['question']}\n")
            f.write(f"**Expected:** {error['true_answer']} ({error['answer_type']})\n")
            f.write(f"**Predicted:** {error['predicted_answer']}\n\n")

    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on bank statement QA dataset"
    )
    parser.add_argument(
        "--model_id", type=str, default="gpt-5", help="Model ID to evaluate"
    )
    parser.add_argument("--downsample_size", type=int, help="Number of samples to use")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/preprocessed_data.jsonl",
        help="Path to dataset",
    )
    parser.add_argument(
        "--use_domain_expertise", action="store_true", help="Use domain expertise"
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        default="data/Labeled Questions and Answers.xlsx",
        help="Path to Excel file for preprocessing",
    )
    parser.add_argument(
        "--skip_preprocessing", action="store_true", help="Skip preprocessing step"
    )

    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"results/{now}"
    os.makedirs(output_dir, exist_ok=True)

    # Create single dataset
    if not args.skip_preprocessing or not os.path.exists(args.dataset_path):
        print("Preprocessing data...")
        if not preprocess_data(args.excel_path, args.dataset_path):
            print("Preprocessing failed. Exiting.")
            return
    else:
        print(
            f"Preprocessed data found at {args.dataset_path}. Skipping preprocessing."
        )

    print(f"\n{'='*50}")
    print(f"Running evaluation")
    print(f"{'='*50}")

    print(f"Loading dataset...")
    dataset = load_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} samples")

    if len(dataset) == 0:
        print(f"No samples found. Exiting.")
        return

    if args.downsample_size:
        print(f"Downsampling to {args.downsample_size} samples")
        random.seed(42)  # For reproducibility

    print(f"Evaluating model: {args.model_id}")
    results, accuracy, df = evaluate_model(
        args.model_id, dataset, args.use_domain_expertise, args.downsample_size
    )

    print(f"\nFinal Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

    # Save results
    save_results(
        results,
        accuracy,
        args.model_id,
        output_dir,
        args.downsample_size,
        df,
    )

    print(f"Results saved in {output_dir}")

    print(f"\n{'='*50}")
    print("Evaluation completed!")
    print(f"Results saved in {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
