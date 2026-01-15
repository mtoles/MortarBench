import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import argparse
import re


def referenced_anything(s: str) -> bool:
    return "referenced the following data:\n\n{}" not in s.lower()


def calculate_metric_stats(series):
    """
    Calculate average of averages and average standard deviation for a metric series.
    Handles both single values and lists (for trials > 1).
    
    For each question:
    - Calculate mean across trials
    - Calculate std across trials
    Then:
    - Average the means (gives overall mean)
    - Average the stds (gives average variability per question)
    
    Returns: (mean_of_means, mean_of_stds)
    """
    if len(series) == 0:
        return 0.0, 0.0
    
    per_example_means = []
    per_example_stds = []
    
    for value in series:
        if isinstance(value, list):
            # For lists, calculate mean and std across trials
            per_example_means.append(np.mean(value))
            per_example_stds.append(np.std(value, ddof=1) if len(value) > 1 else 0.0)
        else:
            # For single values, use as is with 0 std
            per_example_means.append(value)
            per_example_stds.append(0.0)
    
    mean_of_means = np.mean(per_example_means)
    mean_of_stds = np.mean(per_example_stds)
    return mean_of_means, mean_of_stds


def get_model_name_from_dir(dir_path):
    """Extract model name from summary .md file in directory."""
    md_files = glob.glob(os.path.join(dir_path, "*_summary.md"))
    if not md_files:
        return None
    
    with open(md_files[0], 'r') as f:
        first_line = f.readline().strip()
        if first_line.startswith("# Evaluation Results:"):
            return first_line.replace("# Evaluation Results:", "").strip()
    
    return None


def parse_pii_false_f1_from_md(md_file_path):
    """
    Parse the .md summary file to extract F1 Accuracy (PII==False) value.
    Returns the F1 score as a float, or None if not found.
    """
    if not os.path.exists(md_file_path):
        return None
    
    with open(md_file_path, 'r') as f:
        content = f.read()
        
    # Look for "## Metrics for PII==False Subset" section
    if "## Metrics for PII==False Subset" not in content:
        return None
    
    # Extract the F1 Accuracy line
    pattern = r'\*\*F1 Accuracy \(PII==False\):\*\* ([\d.]+)'
    match = re.search(pattern, content)
    if match:
        return float(match.group(1))
    
    return None


def process_subdirectory(final_dir, subdir_name):
    """
    Process a subdirectory (Question or Rephrased_Question) and return metrics for each model.
    Returns: dict mapping model_name -> (f1_all, f1_pii_false)
    """
    subdir_path = os.path.join(final_dir, subdir_name)
    if not os.path.exists(subdir_path):
        print(f"Subdirectory {subdir_path} does not exist, skipping...")
        return {}
    
    model_metrics = {}
    
    # Get all timestamped directories
    timestamp_dirs = [
        d for d in os.listdir(subdir_path)
        if os.path.isdir(os.path.join(subdir_path, d))
    ]
    
    for timestamp_dir in timestamp_dirs:
        dir_path = os.path.join(subdir_path, timestamp_dir)
        
        # Get model name from summary file
        md_files = glob.glob(os.path.join(dir_path, "*_summary.md"))
        if not md_files:
            continue
        
        model_name = get_model_name_from_dir(dir_path)
        if not model_name:
            continue
        
        # Find results file
        jsonl_files = glob.glob(os.path.join(dir_path, "*.jsonl"))
        if not jsonl_files:
            continue
        
        results_file = jsonl_files[0]
        
        df = pd.read_json(results_file, lines=True)
        
        # Calculate F1 for all data
        f1_all, f1_all_std = calculate_metric_stats(df["f1_score"])
        
        # Get PII==False F1 from .md file
        f1_pii_false = parse_pii_false_f1_from_md(md_files[0])
        if f1_pii_false is None:
            # Fallback: try to calculate from data if pii column exists
            if "pii" in df.columns:
                pii_false_df = df[df["pii"] == False]
                if len(pii_false_df) > 0:
                    f1_pii_false, _ = calculate_metric_stats(pii_false_df["f1_score"])
                else:
                    f1_pii_false = 0.0
            else:
                f1_pii_false = 0.0
        
        # We don't have std for PII==False from .md file, so set to 0
        f1_pii_false_std = 0.0
        
        model_metrics[model_name] = {
            "f1_all": f1_all,
            "f1_all_std": f1_all_std,
            "f1_pii_false": f1_pii_false,
            "f1_pii_false_std": f1_pii_false_std,
        }
    
    return model_metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze model results and create comparison plots")
    parser.add_argument(
        "--final_dir",
        type=str,
        default="results/presentation",
        help="Base directory containing Question and Rephrased_Question subdirectories (default: results/presentation)"
    )
    args = parser.parse_args()
    
    final_dir = args.final_dir
    
    # Process both subdirectories
    question_metrics = process_subdirectory(final_dir, "Question")
    rephrased_metrics = process_subdirectory(final_dir, "Rephrased_Question")
    
    # Get all unique model names
    all_models = set(question_metrics.keys()) | set(rephrased_metrics.keys())
    
    if not all_models:
        print("No models found in either subdirectory!")
        return
    
    print(f"Found {len(all_models)} models: {sorted(all_models)}")
    
    # Collect data for plotting
    plot_data = []
    csv_data = []
    
    for model_name in sorted(all_models):
        q_data = question_metrics.get(model_name, {})
        r_data = rephrased_metrics.get(model_name, {})
        
        f1_question = q_data.get("f1_all", 0.0)
        f1_rephrased = r_data.get("f1_all", 0.0)
        f1_question_pii_false = q_data.get("f1_pii_false", 0.0)
        f1_rephrased_pii_false = r_data.get("f1_pii_false", 0.0)
        
        plot_data.append({
            "model": model_name,
            "f1_question": f1_question,
            "f1_rephrased": f1_rephrased,
            "f1_question_pii_false": f1_question_pii_false,
            "f1_rephrased_pii_false": f1_rephrased_pii_false,
        })
        
        csv_data.append({
            "Model": model_name,
            "F1 Question": f1_question,
            "F1 Rephrased_Question": f1_rephrased,
            "F1 Question PII==False": f1_question_pii_false,
            "F1 Rephrased_Question PII==False": f1_rephrased_pii_false,
        })
    
    # Create analysis directory if it doesn't exist
    analysis_dir = "analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Create DataFrame for CSV
    csv_df = pd.DataFrame(csv_data)
    csv_output_path = os.path.join(analysis_dir, "results_comparison.csv")
    csv_df.to_csv(csv_output_path, index=False)
    print(f"\nCSV saved to {csv_output_path}")
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    models = [d["model"] for d in plot_data]
    x = np.arange(len(models))
    width = 0.2
    
    f1_question_vals = [d["f1_question"] for d in plot_data]
    f1_rephrased_vals = [d["f1_rephrased"] for d in plot_data]
    f1_question_pii_vals = [d["f1_question_pii_false"] for d in plot_data]
    f1_rephrased_pii_vals = [d["f1_rephrased_pii_false"] for d in plot_data]
    
    bars1 = ax.bar(x - 1.5*width, f1_question_vals, width, label="F1 Question", color="#1f77b4")
    bars2 = ax.bar(x - 0.5*width, f1_rephrased_vals, width, label="F1 Rephrased_Question", color="#ff7f0e")
    bars3 = ax.bar(x + 0.5*width, f1_question_pii_vals, width, label="F1 Question PII==False", color="#2ca02c")
    bars4 = ax.bar(x + 1.5*width, f1_rephrased_pii_vals, width, label="F1 Rephrased_Question PII==False", color="#d62728")
    
    ax.set_ylabel("F1 Score")
    ax.set_title("Model F1 Score Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plot_output_path = os.path.join(analysis_dir, "results_comparison.png")
    plt.savefig(plot_output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_output_path}")
    plt.show()


if __name__ == "__main__":
    main()
