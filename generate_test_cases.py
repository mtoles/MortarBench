"""
Test Case Generator for Mortgage Application Dataset

This script reads questions.csv and generates mutated bank statement and ULAD files
for each question, creating a complete test dataset with ground truth answers.
"""

import json
import os
import pandas as pd
from typing import Dict, Any
from data_mutator import DataMutator
import argparse
import random


# Mapping from tag keywords in questions to mutation functions
TAG_TO_MUTATION = {
    "BNPL": "mutate_bnpl_transactions",
    "large deposits": "mutate_large_deposits",
    "rental payments": "mutate_rental_payments",
    "cryptocurrency": "mutate_crypto_deposits",
    "overdraft": "mutate_overdraft_fees",
    "NSF": "mutate_overdraft_fees",
    "payday loan": "mutate_payday_loans",
    "foreign": "mutate_foreign_deposits",
    "secured loan": "mutate_secured_loan_deposits",
    "cash deposits": "mutate_cash_deposits",
    "unexplained deposits": "mutate_unexplained_deposits",
    "undisclosed income": "mutate_undisclosed_income",
    "undisclosed housing": "mutate_undisclosed_housing_payments",
    "withdrawal": "mutate_withdrawal_transactions",
    "earnest money": "mutate_withdrawal_transactions",
    "retirement": "mutate_retirement_accounts",
    "custodial": "mutate_custodial_accounts",
    "additional account holder": "mutate_additional_account_holder",
}


def detect_mutation_function(question: str, rephrased_question: str) -> str:
    """
    Detect which mutation function to use based on the question text.
    
    Args:
        question: Original question text
        rephrased_question: Rephrased question text
        
    Returns:
        Name of the mutation function to use, or None if not found
    """
    combined_text = f"{question} {rephrased_question}".lower()
    
    # Check for each tag keyword
    for keyword, func_name in TAG_TO_MUTATION.items():
        if keyword.lower() in combined_text:
            return func_name
    
    return None


def generate_test_case(
    row: pd.Series,
    mutator: DataMutator,
    output_dir: str,
    test_case_id: int
) -> Dict[str, Any]:
    """
    Generate a test case for a specific question.
    
    Args:
        row: Row from questions.csv
        mutator: DataMutator instance
        output_dir: Directory to save generated files
        test_case_id: Unique ID for this test case
        
    Returns:
        Dictionary with test case metadata
    """
    question = row['question']
    rephrased_question = row['rephrased_question']
    answer_type = row['answer_type']
    need_bank_statement = row['need_bank_statement']
    need_ulad = row['need_ulad']
    test_case_number = row['test_case_number']
    label = row['label']
    
    # Detect mutation function
    mutation_func_name = detect_mutation_function(question, rephrased_question)
    
    if not mutation_func_name:
        print(f"  ⚠️  Could not detect mutation function for: {rephrased_question[:60]}...")
        return None
    
    # Get the mutation function
    mutation_func = getattr(mutator, mutation_func_name, None)
    if not mutation_func:
        print(f"  ⚠️  Mutation function {mutation_func_name} not found")
        return None
    
    print(f"  ✓ Using mutation: {mutation_func_name}")
    
    # Generate mutation with random number of transactions
    try:
        mutated_bank, answer = mutation_func()
    except Exception as e:
        print(f"  ✗ Error in mutation: {e}")
        return None
    
    # Create test case directory
    test_case_dir = os.path.join(output_dir, f"test_case_{test_case_id:04d}")
    os.makedirs(test_case_dir, exist_ok=True)
    
    # Save mutated bank statement if needed
    if need_bank_statement == 1:
        bank_path = os.path.join(test_case_dir, "bank_statement.json")
        with open(bank_path, 'w') as f:
            json.dump(mutated_bank, f, indent=2)
    
    # Save ULAD if needed (currently just copy base)
    if need_ulad == 1:
        ulad_path = os.path.join(test_case_dir, "ulad.json")
        with open(ulad_path, 'w') as f:
            json.dump(mutator.base_ulad, f, indent=2)
    
    # Create metadata
    metadata = {
        "test_case_id": test_case_id,
        "test_case_number": test_case_number,
        "question": question,
        "rephrased_question": rephrased_question,
        "answer_type": answer_type,
        "label": label,
        "mutation_function": mutation_func_name,
        "ground_truth_answer": answer,
        "need_bank_statement": bool(need_bank_statement),
        "need_ulad": bool(need_ulad),
        "bank_statement_path": "bank_statement.json" if need_bank_statement else None,
        "ulad_path": "ulad.json" if need_ulad else None
    }
    
    # Save metadata
    metadata_path = os.path.join(test_case_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate test cases from questions.csv")
    parser.add_argument(
        "--questions",
        default="data/questions.csv",
        help="Path to questions.csv"
    )
    parser.add_argument(
        "--bank-statement",
        default="generated_data/dataset_generated-test-7a8d6178.json",
        help="Path to base bank statement JSON"
    )
    parser.add_argument(
        "--ulad",
        default="data/ulad.json",
        help="Path to base ULAD JSON"
    )
    parser.add_argument(
        "--output",
        default="test_cases",
        help="Output directory for test cases"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test cases to generate (for testing)"
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Only generate test cases for specific tags (e.g., 'BNPL' 'large deposits')"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load questions
    print(f"Loading questions from {args.questions}...")
    questions_df = pd.read_csv(args.questions)
    
    # Filter by tags if specified
    if args.tags:
        print(f"Filtering for tags: {', '.join(args.tags)}")
        mask = questions_df.apply(
            lambda row: any(
                tag.lower() in f"{row['question']} {row['rephrased_question']}".lower()
                for tag in args.tags
            ),
            axis=1
        )
        questions_df = questions_df[mask]
    
    # Apply limit if specified
    if args.limit:
        questions_df = questions_df.head(args.limit)
    
    print(f"Processing {len(questions_df)} questions...")
    
    # Initialize mutator
    print(f"Initializing mutator with:")
    print(f"  Bank statement: {args.bank_statement}")
    print(f"  ULAD: {args.ulad}")
    
    mutator = DataMutator(args.bank_statement, args.ulad)
    
    # Generate test cases
    test_cases = []
    success_count = 0
    skip_count = 0
    
    for idx, row in questions_df.iterrows():
        test_case_id = idx + 1
        print(f"\n[{test_case_id}/{len(questions_df)}] Processing test case {row['test_case_number']}...")
        print(f"  Question: {row['rephrased_question'][:80]}...")
        
        metadata = generate_test_case(row, mutator, args.output, test_case_id)
        
        if metadata:
            test_cases.append(metadata)
            success_count += 1
            print(f"  ✓ Generated test case {test_case_id}")
        else:
            skip_count += 1
    
    # Save summary
    summary = {
        "total_questions": len(questions_df),
        "successful_generations": success_count,
        "skipped": skip_count,
        "test_cases": test_cases
    }
    
    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"  Total: {len(questions_df)} questions")
    print(f"  Successful: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Output directory: {args.output}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
