"""
Test Case Generator for Mortgage Application Dataset

Reads questions.csv and generates mutated bank statement (+) ULAD files for each
question, creating a complete test dataset with ground truth answers.

Mutation dispatch:
  - Bank-only mutations (mutate_transaction / mutate_account) return (bank, answer)
  - ULAD cross-document mutations return (bank, ulad, answer)
"""

import json
import os
import copy
import pandas as pd
from typing import Dict, Any, Optional
from data_mutator import DataMutator
import argparse


# ==================== MUTATION RULES ====================
# Ordered list of (keyword, spec).  First match wins.
# Specs:
#   {"type": "transaction", "mutation_type": "<key in TRANSACTION_CONFIGS>"}
#   {"type": "account",     "mutation_type": "<key in ACCOUNT_CONFIGS>"}
#   {"type": "ulad",        "fn": "<DataMutator method name>"}
#
# ULAD rules are listed first so specific phrases always take priority over
# shorter bank-only keywords that might accidentally match the same text.

MUTATION_RULES = [
    # ── ULAD cross-document mutations ──────────────────────────────────────
    ("employer names stated in employment history",
     {"type": "ulad", "fn": "mutate_employer_payroll_consistency"}),

    ("payroll deposit entries match",
     {"type": "ulad", "fn": "mutate_employer_payroll_consistency"}),

    ("payroll deposits in the bank statements are consistent with the income and employment",
     {"type": "ulad", "fn": "mutate_employer_payroll_consistency"}),

    ("payroll deposits from an employer not listed on the loan application",
     {"type": "ulad", "fn": "mutate_payroll_undisclosed_employer"}),

    ("employer not listed on the loan application",
     {"type": "ulad", "fn": "mutate_payroll_undisclosed_employer"}),

    ("potential undisclosed employment income",
     {"type": "ulad", "fn": "mutate_payroll_undisclosed_employer"}),

    ("address on the account statement",
     {"type": "ulad", "fn": "mutate_address_match"}),

    ("current address on the loan application",
     {"type": "ulad", "fn": "mutate_address_match"}),

    ("mailing address provided on the loan application",
     {"type": "ulad", "fn": "mutate_address_match"}),

    ("borrower's address on their bank statements aligns",
     {"type": "ulad", "fn": "mutate_address_match"}),

    ("gift amount stated on the loan application",
     {"type": "ulad", "fn": "mutate_gift_deposit"}),

    ("deposit matching the gift amount",
     {"type": "ulad", "fn": "mutate_gift_deposit"}),

    ("child support, alimony",
     {"type": "ulad", "fn": "mutate_child_support_disclosure"}),

    ("alimony, or wage garnishment",
     {"type": "ulad", "fn": "mutate_child_support_disclosure"}),

    ("recurring payments for child support",
     {"type": "ulad", "fn": "mutate_child_support_disclosure"}),

    ("recurring debt payments that were not disclosed on the loan application",
     {"type": "ulad", "fn": "mutate_undisclosed_liabilities"}),

    ("recurring debt payments that are not disclosed on the loan application",
     {"type": "ulad", "fn": "mutate_undisclosed_liabilities"}),

    ("payments to creditors that are not listed on the credit report or the loan application",
     {"type": "ulad", "fn": "mutate_undisclosed_liabilities"}),

    ("undisclosed other income source",
     {"type": "ulad", "fn": "mutate_undisclosed_income_source"}),

    ("consistent pattern of deposits that could represent an undisclosed income source",
     {"type": "ulad", "fn": "mutate_undisclosed_income_source"}),

    ("consistent pattern of deposits on the asset statement that could represent an undisclosed",
     {"type": "ulad", "fn": "mutate_undisclosed_income_source"}),

    ("rental income reported for the property",
     {"type": "ulad", "fn": "mutate_rental_income_consistency"}),

    ("rental income deposits on the bank statements align",
     {"type": "ulad", "fn": "mutate_rental_income_consistency"}),

    ("gross rental income reported on the loan application",
     {"type": "ulad", "fn": "mutate_rental_income_consistency"}),

    ("deposits match the properties monthly gross rental income",
     {"type": "ulad", "fn": "mutate_rental_income_consistency"}),

    ("joint accounts where one or more account holders are not listed as borrowers",
     {"type": "ulad", "fn": "mutate_joint_account_holder"}),

    ("joint account",
     {"type": "ulad", "fn": "mutate_joint_account_holder"}),

    ("net pay amounts on any of the borrower",
     {"type": "ulad", "fn": "mutate_payroll_paystub_consistency"}),

    ("net pay amounts",
     {"type": "ulad", "fn": "mutate_payroll_paystub_consistency"}),

    ("pay stubs provided",
     {"type": "ulad", "fn": "mutate_payroll_paystub_consistency"}),

    ("payroll deposits on the bank statements exactly match",
     {"type": "ulad", "fn": "mutate_payroll_paystub_consistency"}),

    # ── Bank-transaction mutations ─────────────────────────────────────────
    ("bnpl",
     {"type": "transaction", "mutation_type": "bnpl"}),

    ("buy now pay later",
     {"type": "transaction", "mutation_type": "bnpl"}),

    ("klarna",
     {"type": "transaction", "mutation_type": "bnpl"}),

    ("afterpay",
     {"type": "transaction", "mutation_type": "bnpl"}),

    ("affirm",
     {"type": "transaction", "mutation_type": "bnpl"}),

    ("large deposit",
     {"type": "transaction", "mutation_type": "large_deposits"}),

    ("irregular deposit",
     {"type": "transaction", "mutation_type": "large_deposits"}),

    ("unusually large",
     {"type": "transaction", "mutation_type": "large_deposits"}),

    ("rental payment",
     {"type": "transaction", "mutation_type": "rental_payments"}),

    ("evidence of rental payment",
     {"type": "transaction", "mutation_type": "rental_payments"}),

    ("cryptocurrency source",
     {"type": "transaction", "mutation_type": "crypto_deposits"}),

    ("cryptocurrency",
     {"type": "transaction", "mutation_type": "crypto_deposits"}),

    ("originate from a cryptocurrency",
     {"type": "transaction", "mutation_type": "crypto_deposits"}),

    ("overdraft",
     {"type": "transaction", "mutation_type": "overdraft_fees"}),

    ("non-sufficient funds",
     {"type": "transaction", "mutation_type": "overdraft_fees"}),

    ("nsf",
     {"type": "transaction", "mutation_type": "overdraft_fees"}),

    ("payday loan",
     {"type": "transaction", "mutation_type": "payday_loans"}),

    ("high-interest lending source",
     {"type": "transaction", "mutation_type": "payday_loans"}),

    ("foreign origin",
     {"type": "transaction", "mutation_type": "foreign_deposits"}),

    ("deposits that could be of foreign origin",
     {"type": "transaction", "mutation_type": "foreign_deposits"}),

    ("secured loan",
     {"type": "transaction", "mutation_type": "secured_loan_deposits"}),

    ("cash deposit",
     {"type": "transaction", "mutation_type": "cash_deposits"}),

    ("excessive cash deposit",
     {"type": "transaction", "mutation_type": "cash_deposits"}),

    ("unexplained deposit",
     {"type": "transaction", "mutation_type": "unexplained_deposits"}),

    ("unsecured borrowed funds",
     {"type": "transaction", "mutation_type": "unexplained_deposits"}),

    ("private savings club",
     {"type": "transaction", "mutation_type": "savings_club"}),

    ("savings club",
     {"type": "transaction", "mutation_type": "savings_club"}),

    ("informal arrangement",
     {"type": "transaction", "mutation_type": "savings_club"}),

    ("sou-sou",
     {"type": "transaction", "mutation_type": "undisclosed_income"}),

    ("undisclosed income",
     {"type": "transaction", "mutation_type": "undisclosed_income"}),

    ("undisclosed housing",
     {"type": "transaction", "mutation_type": "undisclosed_housing_payments"}),

    ("earnest money",
     {"type": "transaction", "mutation_type": "withdrawals"}),

    ("withdrawal matching the earnest money",
     {"type": "transaction", "mutation_type": "withdrawals"}),

    ("mortgage payment needed to confirm current payment history",
     {"type": "transaction", "mutation_type": "mortgage_payments"}),

    ("additional account holder",
     {"type": "transaction", "mutation_type": "additional_account_holder"}),

    # ── Bank-account mutations ─────────────────────────────────────────────
    ("retirement account",
     {"type": "account", "mutation_type": "retirement"}),

    ("retirement assets",
     {"type": "account", "mutation_type": "retirement"}),

    ("custodial account",
     {"type": "account", "mutation_type": "custodial"}),

    ("business account",
     {"type": "account", "mutation_type": "business"}),
]


def detect_mutation(question: str, rephrased: str) -> Optional[Dict]:
    """
    Return the first matching mutation spec from MUTATION_RULES.
    Combined text is lower-cased for case-insensitive matching.
    """
    combined = f"{question} {rephrased}".lower()
    for keyword, spec in MUTATION_RULES:
        if keyword.lower() in combined:
            return spec
    return None


def execute_mutation(mutator: DataMutator, spec: Dict, answer_type: str):
    """
    Execute a mutation and return a normalised 3-tuple (bank, ulad, answer).
    Bank-only mutations pad ulad with a deep copy of the base.
    """
    mtype = spec["type"]

    if mtype == "transaction":
        bank, answer = mutator.mutate_transaction(spec["mutation_type"], answer_type=answer_type)
        ulad = copy.deepcopy(mutator.base_ulad)
        return bank, ulad, answer

    if mtype == "account":
        bank, answer = mutator.mutate_account(spec["mutation_type"], answer_type=answer_type)
        ulad = copy.deepcopy(mutator.base_ulad)
        return bank, ulad, answer

    if mtype == "ulad":
        fn = getattr(mutator, spec["fn"])
        # Try to call with answer_type parameter
        try:
            result = fn(answer_type=answer_type)
        except TypeError:
            # Function doesn't support answer_type parameter
            result = fn()
        # All ULAD mutation functions return (bank, ulad, answer)
        return result

    raise ValueError(f"Unknown mutation type: {mtype}")


def generate_test_case(
    row: pd.Series,
    mutator: DataMutator,
    output_dir: str,
    test_case_id: int,
) -> Optional[Dict[str, Any]]:
    """Generate one test case directory and return its metadata dict."""
    question = row["question"]
    rephrased = row["rephrased_question"]
    answer_type = row["answer_type"]
    need_bank = int(row["need_bank_statement"])
    need_ulad = int(row["need_ulad"])
    tc_number = row["test_case_number"]
    label = row["label"]
    gid = row["gid"]
    old_id = row["old_id"]
    

    spec = detect_mutation(question, rephrased)
    if spec is None:
        print(f"  ⚠  No mutation rule matched: {rephrased[:70]}...")
        return None

    print(f"  ✓ Rule matched: type={spec['type']}, "
          f"key={spec.get('mutation_type') or spec.get('fn')}")

    try:
        # Use answer_type from CSV (both questions.csv and unique_questions.csv have this column)
        answer_type_param = row['answer_type']
        mutated_bank, mutated_ulad, answer = execute_mutation(mutator, spec, answer_type_param)
    except Exception as exc:
        print(f"  ✗ Mutation error: {exc}")
        return None

    # ── Write files ────────────────────────────────────────────────────────
    tc_dir = os.path.join(output_dir, f"test_case_{test_case_id:04d}")
    os.makedirs(tc_dir, exist_ok=True)

    bank_rel_path = None
    ulad_rel_path = None

    if need_bank:
        bank_rel_path = "bank_statement.json"
        with open(os.path.join(tc_dir, bank_rel_path), "w") as f:
            json.dump(mutated_bank, f, indent=2)

    if need_ulad:
        ulad_rel_path = "ulad.json"
        with open(os.path.join(tc_dir, ulad_rel_path), "w") as f:
            json.dump(mutated_ulad, f, indent=2)

    mutation_key = spec.get("mutation_type") or spec.get("fn")
    metadata = {
        "test_case_id": test_case_id,
        "test_case_number": tc_number,
        "question": question,
        "rephrased_question": rephrased,
        "answer_type": answer_type,
        "label": label,
        "mutation_type": spec["type"],
        "mutation_key": mutation_key,
        "ground_truth_answer": answer,
        "need_bank_statement": bool(need_bank),
        "need_ulad": bool(need_ulad),
        "bank_statement_path": bank_rel_path,
        "ulad_path": ulad_rel_path,
        "gid": gid,
        "old_id": old_id,
    }

    with open(os.path.join(tc_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate test cases from questions.csv")
    parser.add_argument("--questions", default="data/questions.csv")
    parser.add_argument(
        "--bank-statement",
        default="generated_data/bank_statement.json",
    )
    parser.add_argument(
        "--ulad",
        default="generated_data/ulad.json",
    )
    parser.add_argument("--output", default="test_cases")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max test cases to generate (for quick testing)")
    parser.add_argument("--tags", nargs="+", default=None,
                        help="Only process questions containing these keywords")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading questions from {args.questions}...")
    df = pd.read_csv(args.questions)

    if args.tags:
        print(f"Filtering for tags: {', '.join(args.tags)}")
        mask = df.apply(
            lambda r: any(
                t.lower() in f"{r['question']} {r['rephrased_question']}".lower()
                for t in args.tags
            ),
            axis=1,
        )
        df = df[mask]

    if args.limit:
        df = df.head(args.limit)

    print(f"Processing {len(df)} questions...")
    print(f"  Bank statement : {args.bank_statement}")
    print(f"  ULAD           : {args.ulad}")

    mutator = DataMutator(args.bank_statement, args.ulad)

    results, success, skipped = [], 0, 0

    for idx, row in df.iterrows():
        tc_id = idx + 1
        print(f"\n[{tc_id}/{len(df)}] {row['test_case_number']} - {row['rephrased_question'][:70]}...")

        meta = generate_test_case(row, mutator, args.output, tc_id)
        if meta:
            results.append(meta)
            success += 1
            print(f"  → saved to test_case_{tc_id:04d}/")
        else:
            skipped += 1

    summary = {
        "total_questions": len(df),
        "successful": success,
        "skipped": skipped,
        "test_cases": results,
    }
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"  Total     : {len(df)}")
    print(f"  Successful: {success}")
    print(f"  Skipped   : {skipped}")
    print(f"  Output    : {args.output}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
