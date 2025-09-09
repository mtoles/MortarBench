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
from openai import OpenAI


prompt = 'Question: {question}\n\nBank Statement: {context}\n\nULAD DU: {ulad_du}\n\nAnswer the question. Do not think out loud. Return ONLY yes, none, a number, or a valid JSON list of transaction/account IDs, e.g. `["d2rf4l6kq23ndu9seg6g", "d2rf4l6kq23ndu9seg60"]` Your JSON list must be strings enclosed by "" contained within [] and separated by commas.'

client = OpenAI()


def load_dataset(dataset_path="data/preprocessed_data.jsonl"):
    """Load the JSONL dataset as a DataFrame."""
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return pd.DataFrame(dataset)


def evaluate_model(model_id, df, downsample_size=None):
    """Evaluate a model on the dataset."""
    # Keep a copy of the original dataset
    original_df = df.copy()

    if downsample_size:
        df = df.sample(n=min(downsample_size, len(df)), random_state=42).reset_index(
            drop=True
        )

    results = []
    correct_count = 0
    total_count = len(df)

    for i, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]
        answer_type = row["answer_type"]
        bank_statement = json.dumps(row["bank_statement"], indent=2)
        ulad_du = row["ulad_du"]
        # Format the prompt
        formatted_prompt = prompt.format(
            question=question, context=bank_statement, ulad_du=ulad_du
        )

        # Call the model
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": formatted_prompt}],
        )

        predicted_answer = response.choices[0].message.content.strip()

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
    if answer_type == "yes":
        return predicted_answer == "yes"
    elif answer_type == "no":
        return predicted_answer == "no"
    elif answer_type == "none":
        return predicted_answer in ["none", "[]"]
    elif answer_type == "id_list":
        try:
            pred_as_list = json.loads(predicted_answer)
        except json.JSONDecodeError:
            print(f"warning: invalid json: {predicted_answer} for answer: {answer}")
            return False
        answer_options = answer.replace(" ", "").split(",")
        return set(pred_as_list) == set(answer)
    else:
        raise ValueError(f"Invalid answer type: {answer_type}")


def save_results(
    results, accuracy, model_id, output_dir, downsample_size=None, df=None
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
            f.write(f"**Question:** {error['question']}\n\n")
            f.write(
                f"**Expected:** {error['true_answer']} ({error['answer_type']})\n\n"
            )
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

    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"results/{now}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} samples")

    if args.downsample_size:
        print(f"Downsampling to {args.downsample_size} samples")
        random.seed(42)  # For reproducibility

    print(f"Evaluating model: {args.model_id}")
    results, accuracy, df = evaluate_model(args.model_id, dataset, args.downsample_size)

    print(f"\nFinal Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

    save_results(results, accuracy, args.model_id, output_dir, args.downsample_size, df)


if __name__ == "__main__":
    main()
