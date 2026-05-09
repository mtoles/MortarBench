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
import random
import pandas as pd
from typing import Dict, Any, Optional
import data_mutator
from data_mutator import DataMutator
from dataset_generator import PlaidGenerator
from ulad_generator import UladGenerator
import argparse


def generate_profile(ulad_template="data/ulad_template.json", months=3, user_name="John Homeowner", statement_type="personal"):
    """Generate a fresh random bank statement + ULAD profile (as dicts)."""
    plaid_gen = PlaidGenerator(num_months=months, user_name=user_name, statement_type=statement_type)
    bank_data = plaid_gen.generate_single_dataset()
    ulad_gen = UladGenerator(ulad_template, bank_data, ".")
    ulad_data = ulad_gen.generate_ulad()
    return bank_data, ulad_data


def mutator_from_dicts(bank, ulad, bank_2=None, ulad_2=None):
    """Create a DataMutator from in-memory dicts instead of file paths."""
    m = object.__new__(DataMutator)
    m.base_bank_statement = bank
    m.base_ulad = ulad
    m.base_bank_statement_2 = bank_2
    m.base_ulad_2 = ulad_2
    m.transaction_counter = 1000
    return m


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

    ("match any employer listed for the primary borrower",
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

    ("recurring debits indicate liabilities not listed on the loan application",
     {"type": "ulad", "fn": "mutate_undisclosed_liabilities"}),

    ("recurring debits on the bank statements indicate any liabilities not listed",
     {"type": "ulad", "fn": "mutate_undisclosed_liabilities"}),

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

    ("rental income deposits, if any, support the gross rental income",
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

    # ── Recurring income / expense matching ─────────────────────────────────
    ("recurring deposits match the claimed alimony",
     {"type": "ulad", "fn": "mutate_recurring_income_match"}),

    ("recurring deposits match the claimed",
     {"type": "ulad", "fn": "mutate_recurring_income_match"}),

    ("recurring deposits, if any, match the claimed",
     {"type": "ulad", "fn": "mutate_recurring_income_match"}),

    ("recurring debit match the claimed alimony",
     {"type": "ulad", "fn": "mutate_recurring_expense_match"}),

    ("recurring debits match the claimed alimony",
     {"type": "ulad", "fn": "mutate_recurring_expense_match"}),

    ("recurring debits match the claimed",
     {"type": "ulad", "fn": "mutate_recurring_expense_match"}),

    ("recurring debits, if any, match the claimed",
     {"type": "ulad", "fn": "mutate_recurring_expense_match"}),

    # ── Eligible income ──────────────────────────────────────────────────
    ("do not support employment income sources disclosed",
     {"type": "ulad", "fn": "mutate_eligible_income"}),

    ("eligible income",
     {"type": "ulad", "fn": "mutate_eligible_income"}),

    ("what is the eligible income",
     {"type": "ulad", "fn": "mutate_eligible_income"}),

    # ── Two-borrower mutations ─────────────────────────────────────────────
    # Inverted (not-documented) rules must come first to win the first-match
    ("large deposits, if any, are not documented",
     {"type": "two_borrower", "fn": "mutate_undocumented_large_deposit"}),

    ("large deposit documented by a corresponding debit from the other borrower",
     {"type": "two_borrower", "fn": "mutate_large_deposit_corresponding_debit"}),

    ("corresponding debit from the other borrower's account within the correct time window",
     {"type": "two_borrower", "fn": "mutate_large_deposit_corresponding_debit"}),

    ("is the large deposit documented",
     {"type": "two_borrower", "fn": "mutate_large_deposit_corresponding_debit"}),

    ("large deposits, if any, are documented",
     {"type": "two_borrower", "fn": "mutate_large_deposit_corresponding_debit"}),

    # ── Auto loan third party payment mutations ────────────────────────────
    ("automobile loan be excluded from the borrower's debt because a third party has been paying",
     {"type": "auto_loan", "fn": "mutate_auto_loan_third_party_payment"}),

    ("third party has been paying it for the last 12 months",
     {"type": "auto_loan", "fn": "mutate_auto_loan_third_party_payment"}),

    ("made the last 12 month auto payment",
     {"type": "auto_loan", "fn": "mutate_auto_loan_third_party_payment"}),

    # ── Credit card debt exclusion mutations (commented out per PR review - row 30 deleted) ──
    # ("account be excluded from the debt because the borrower pays the full balance each month",
    #  {"type": "ulad", "fn": "mutate_credit_card_full_balance_payment"}),
    #
    # ("pays the full balance each month",
    #  {"type": "ulad", "fn": "mutate_credit_card_full_balance_payment"}),

    # ── Statement staleness (end date > 45 days before application date) ─────
    ("end date more than 45 days before the initial loan application date",
     {"type": "bank_special", "fn": "mutate_statement_staleness"}),

    ("account statement's end date more than 45 days",
     {"type": "bank_special", "fn": "mutate_statement_staleness"}),

    # ── Unexplained large deposits (after excluding payroll/tax refunds) ─────
    # Must precede the generic "large deposit" rule to win first-match.
    ("unexplained after excluding payroll, tax refunds",
     {"type": "bank_special", "fn": "mutate_unexplained_large_deposits"}),

    ("large deposits remain unexplained after excluding payroll",
     {"type": "bank_special", "fn": "mutate_unexplained_large_deposits"}),

    # ── Multiple-employer payroll (two-year window) ────────────────────────
    ("payroll deposits from multiple employers",
     {"type": "bank_special", "fn": "mutate_multiple_employer_payroll"}),

    ("multiple employers during the most recent two-year employment history",
     {"type": "bank_special", "fn": "mutate_multiple_employer_payroll"}),

    # ── Employment gap in last 12 months (payroll-based) ──────────────────
    ("employment gap greater than one month in the most recent 12 months",
     {"type": "bank_special", "fn": "mutate_employment_gap"}),

    ("employment gap greater than one month",
     {"type": "bank_special", "fn": "mutate_employment_gap"}),

    # ── Bank-only special mutations ────────────────────────────────────────
    ("missing transactions",
     {"type": "bank_special", "fn": "mutate_missing_transactions"}),

    ("any missing transactions",
     {"type": "bank_special", "fn": "mutate_missing_transactions"}),

    ("are there any missing transactions",
     {"type": "bank_special", "fn": "mutate_missing_transactions"}),

    ("missing or non-consecutive bank statement records",
     {"type": "bank_special", "fn": "mutate_missing_transactions"}),

    ("missing date",
     {"type": "bank_special", "fn": "mutate_missing_date"}),

    ("any missing date",
     {"type": "bank_special", "fn": "mutate_missing_date"}),

    # ── Bank-transaction mutations ─────────────────────────────────────────
    # Negation variants must come before the generic BNPL / secured-loan keywords
    ("not made to known bnpl providers",
     {"type": "transaction", "mutation_type": "regular_recurring_debits"}),

    ("are not made to known bnpl",
     {"type": "transaction", "mutation_type": "regular_recurring_debits"}),

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

    ("doesn't appear to be from a secured loan",
     {"type": "transaction", "mutation_type": "regular_deposits"}),

    ("does not appear to be from a secured loan",
     {"type": "transaction", "mutation_type": "regular_deposits"}),

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


def execute_mutation(mutator: DataMutator, spec: Dict, answer_type: str, positive: bool = True):
    """
    Execute a mutation and return a normalised 3-tuple (bank, ulad, answer).
    Bank-only mutations pad ulad with a deep copy of the base.
    After mutation, rebuilds bank statement metadata (Transactions,
    BankStatementAccounts, BankStatements, AggregateFigures) so the
    output matches the format produced by dataset_generator.py.

    Polarity is controlled via data_mutator.BOOLEAN_FIXED_VALUE, which
    _resolve_boolean() checks before falling back to random.
    """
    data_mutator.BOOLEAN_FIXED_VALUE = "Yes" if positive else "No"

    mtype = spec["type"]

    if mtype == "transaction":
        bank, answer = mutator.mutate_transaction(spec["mutation_type"], answer_type=answer_type)
        ulad = copy.deepcopy(mutator.base_ulad)
        result = (bank, ulad, answer)

    elif mtype == "account":
        bank, answer = mutator.mutate_account(spec["mutation_type"], answer_type=answer_type)
        ulad = copy.deepcopy(mutator.base_ulad)
        result = (bank, ulad, answer)

    elif mtype == "bank_special":
        fn = getattr(mutator, spec["fn"])
        bank, answer = fn(answer_type=answer_type)
        ulad = copy.deepcopy(mutator.base_ulad)
        result = (bank, ulad, answer)

    elif mtype == "ulad":
        fn = getattr(mutator, spec["fn"])
        try:
            result = fn(answer_type=answer_type)
        except TypeError:
            result = fn()

    elif mtype == "two_borrower":
        fn = getattr(mutator, spec["fn"])
        result = fn(answer_type=answer_type)

    elif mtype == "auto_loan":
        fn = getattr(mutator, spec["fn"])
        result = fn(answer_type=answer_type)

    else:
        raise ValueError(f"Unknown mutation type: {mtype}")

    # Rebuild bank statement metadata so Transactions / BankStatementAccounts /
    # BankStatements / AggregateFigures stay consistent with override_accounts.
    # The rebuild also honours the _monthly_statements flag if set by the mutation.
    if mtype in ("two_borrower", "auto_loan"):
        # result is (bank_a, bank_b, ulad, answer)
        mutator._rebuild_bank_metadata(result[0])
        mutator._rebuild_bank_metadata(result[1])
    else:
        # result is (bank, ulad, answer)
        mutator._rebuild_bank_metadata(result[0])

    # Apply any post-rebuild actions (e.g. removing a monthly statement for Q2)
    banks_to_process = (
        [result[0], result[1]] if mtype in ("two_borrower", "auto_loan")
        else [result[0]]
    )
    for b in banks_to_process:
        actions = b.pop("_post_rebuild_actions", None)
        if not actions:
            continue
        if actions.get("remove_statement_month"):
            month_prefix = actions["remove_statement_month"]  # "YYYY-MM"
            b["BankStatements"] = [
                s for s in b["BankStatements"]
                if not s["StartDate"].startswith(month_prefix)
            ]

    return result


def generate_test_case(
    row: pd.Series,
    mutator: DataMutator,
    output_dir: str,
    test_case_id: int,
    positive: bool = True,
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
        result = execute_mutation(mutator, spec, answer_type_param, positive=positive)
        
        # Handle different return formats
        if spec["type"] in ["two_borrower", "auto_loan"]:
            # Two-borrower and auto loan mutations return (bank_a, bank_b, ulad, answer)
            mutated_bank_a, mutated_bank_b, mutated_ulad, answer = result
            mutated_bank = mutated_bank_a  # Primary bank statement
        else:
            # Standard mutations return (bank, ulad, answer)
            mutated_bank, mutated_ulad, answer = result
            mutated_bank_b = None
            
    except Exception as exc:
        print(f"  ✗ Mutation error: {exc}")
        return None

    # ── Write files ────────────────────────────────────────────────────────
    tc_dir = os.path.join(output_dir, f"test_case_{test_case_id:04d}")
    os.makedirs(tc_dir, exist_ok=True)

    bank_rel_path = None
    bank_b_rel_path = None
    ulad_rel_path = None

    if need_bank:
        bank_rel_path = "bank_statement.json"
        with open(os.path.join(tc_dir, bank_rel_path), "w") as f:
            json.dump(mutated_bank, f, indent=2)
        
        # Write second bank statement if it exists (two-borrower or auto loan case)
        if mutated_bank_b is not None:
            if spec["type"] == "auto_loan":
                bank_b_rel_path = "bank_statement_third_party.json"
            else:
                bank_b_rel_path = "bank_statement_b.json"
            with open(os.path.join(tc_dir, bank_b_rel_path), "w") as f:
                json.dump(mutated_bank_b, f, indent=2)

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
        "bank_statement_b_path": bank_b_rel_path,
        "ulad_path": ulad_rel_path,
        "gid": gid,
        "old_id": old_id,
    }

    with open(os.path.join(tc_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate test cases from questions.csv")
    parser.add_argument("--questions", default="data/questions_unique_generated.csv")
    parser.add_argument("--dataset_path", default="test_cases_official",
                        help="Dataset name under generated_data/ (e.g. 'default', 'test_cases_official'). "
                             "Must match eval.py --dataset_path.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max test cases to generate (for quick testing)")
    parser.add_argument("--tags", nargs="+", default=None,
                        help="Only process questions containing these keywords")
    parser.add_argument("--seed", type=str, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--cases_per_question", type=int, default=4,
                        help="How many independent profile pairs to generate per question. "
                             "Each pair yields both polarities, so total cases = 2 * cases_per_question.")
    args = parser.parse_args()

    dataset_dir = os.path.join("generated_data", args.dataset_path)
    output_dir = os.path.join(dataset_dir, "test_cases")

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(output_dir, exist_ok=True)

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

    cases_per_q = args.cases_per_question
    rows_per_q = 2 * cases_per_q
    total_cases = len(df) * rows_per_q

    # Duplicate each row (cases_per_question × 2 polarities) so the CSV aligns with test case numbering
    df_expanded = df.loc[df.index.repeat(rows_per_q)].reset_index(drop=True)
    df_expanded["polarity"] = ["positive", "negative"] * (len(df) * cases_per_q)
    df_expanded.to_csv(os.path.join(output_dir, "questions.csv"), index=False)
    print(f"Saved {len(df_expanded)} rows ({len(df)} questions × {cases_per_q} profile pairs × 2 polarities) "
          f"to {output_dir}/questions.csv")

    print(f"Processing {len(df)} questions ({cases_per_q} unique profile pairs per question)...")

    results, success, skipped = [], 0, 0
    tc_id = 0

    for _, row in df.iterrows():
        for case_idx in range(cases_per_q):
            # Generate a unique profile for this (question, case) pair
            bank, ulad = generate_profile()
            bank_2, ulad_2 = generate_profile()
            mutator = mutator_from_dicts(bank, ulad, bank_2, ulad_2)

            for positive in (True, False):
                tc_id += 1
                polarity = "positive" if positive else "negative"
                print(f"\n[{tc_id}/{total_cases}] {row['test_case_number']} "
                      f"(case {case_idx + 1}/{cases_per_q}, {polarity}) - {row['rephrased_question'][:60]}...")

                meta = generate_test_case(row, mutator, output_dir, tc_id, positive=positive)
                if meta:
                    meta["polarity"] = polarity
                    meta["case_index"] = case_idx
                    results.append(meta)
                    success += 1
                    print(f"  → saved to test_case_{tc_id:04d}/")
                else:
                    skipped += 1

    summary = {
        "total_questions": len(df),
        "cases_per_question": cases_per_q,
        "total_test_cases": total_cases,
        "successful": success,
        "skipped": skipped,
        "test_cases": results,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"  Questions : {len(df)}")
    print(f"  Test cases: {total_cases} ({len(df)} × {cases_per_q} profile pairs × 2 polarities)")
    print(f"  Successful: {success}")
    print(f"  Skipped   : {skipped}")
    print(f"  Output    : {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
