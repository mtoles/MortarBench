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


def generate_profile(
    ulad_template="data/ulad_template.json",
    months=3,
    user_name="John Homeowner",
    statement_type="personal",
):
    """Generate a fresh random bank statement + ULAD profile (as dicts)."""
    plaid_gen = PlaidGenerator(
        num_months=months, user_name=user_name, statement_type=statement_type
    )
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
    (
        "employer names stated in employment history",
        {"type": "ulad", "fn": "mutate_employer_payroll_consistency"},
    ),
    (
        "payroll deposit entries match",
        {"type": "ulad", "fn": "mutate_employer_payroll_consistency"},
    ),
    (
        "payroll deposits in the bank statements are consistent with the income and employment",
        {"type": "ulad", "fn": "mutate_employer_payroll_consistency"},
    ),
    (
        "match any employer listed for the primary borrower",
        {"type": "ulad", "fn": "mutate_employer_payroll_consistency"},
    ),
    (
        "payroll deposits from an employer not listed on the loan application",
        {"type": "ulad", "fn": "mutate_payroll_undisclosed_employer"},
    ),
    (
        "employer not listed on the loan application",
        {"type": "ulad", "fn": "mutate_payroll_undisclosed_employer"},
    ),
    (
        "potential undisclosed employment income",
        {"type": "ulad", "fn": "mutate_payroll_undisclosed_employer"},
    ),
    (
        "address on the account statement",
        {"type": "ulad", "fn": "mutate_address_match"},
    ),
    (
        "current address on the loan application",
        {"type": "ulad", "fn": "mutate_address_match"},
    ),
    (
        "mailing address provided on the loan application",
        {"type": "ulad", "fn": "mutate_address_match"},
    ),
    (
        "borrower's address on their bank statements aligns",
        {"type": "ulad", "fn": "mutate_address_match"},
    ),
    (
        "gift amount stated on the loan application",
        {"type": "ulad", "fn": "mutate_gift_deposit"},
    ),
    ("deposit matching the gift amount", {"type": "ulad", "fn": "mutate_gift_deposit"}),
    (
        "child support, alimony",
        {"type": "ulad", "fn": "mutate_child_support_disclosure"},
    ),
    (
        "alimony, or wage garnishment",
        {"type": "ulad", "fn": "mutate_child_support_disclosure"},
    ),
    (
        "recurring payments for child support",
        {"type": "ulad", "fn": "mutate_child_support_disclosure"},
    ),
    (
        "recurring debits indicate liabilities not listed on the loan application",
        {"type": "ulad", "fn": "mutate_undisclosed_liabilities"},
    ),
    (
        "recurring debits on the bank statements indicate any liabilities not listed",
        {"type": "ulad", "fn": "mutate_undisclosed_liabilities"},
    ),
    (
        "recurring debt payments that were not disclosed on the loan application",
        {"type": "ulad", "fn": "mutate_undisclosed_liabilities"},
    ),
    (
        "recurring debt payments that are not disclosed on the loan application",
        {"type": "ulad", "fn": "mutate_undisclosed_liabilities"},
    ),
    (
        "payments to creditors that are not listed on the loan application",
        {"type": "ulad", "fn": "mutate_undisclosed_liabilities"},
    ),
    (
        "undisclosed other income source",
        {"type": "ulad", "fn": "mutate_undisclosed_income_source"},
    ),
    (
        "consistent pattern of deposits that could represent an undisclosed income source",
        {"type": "ulad", "fn": "mutate_undisclosed_income_source"},
    ),
    (
        "consistent pattern of deposits on the asset statement that could represent an undisclosed",
        {"type": "ulad", "fn": "mutate_undisclosed_income_source"},
    ),
    (
        "rental income reported for the property",
        {"type": "ulad", "fn": "mutate_rental_income_consistency"},
    ),
    (
        "rental income deposits on the bank statements align",
        {"type": "ulad", "fn": "mutate_rental_income_consistency"},
    ),
    (
        "gross rental income reported on the loan application",
        {"type": "ulad", "fn": "mutate_rental_income_consistency"},
    ),
    (
        "deposits match the properties monthly gross rental income",
        {"type": "ulad", "fn": "mutate_rental_income_consistency"},
    ),
    (
        "rental income deposits, if any, support the gross rental income",
        {"type": "ulad", "fn": "mutate_rental_income_consistency"},
    ),
    (
        "joint accounts where one or more account holders are not listed as borrowers",
        {"type": "ulad", "fn": "mutate_joint_account_holder"},
    ),
    ("joint account", {"type": "ulad", "fn": "mutate_joint_account_holder"}),
    (
        "Do the payroll deposits on the bank statements exactly match the income reported on the ULAD",
        {"type": "ulad", "fn": "mutate_payroll_paystub_consistency"},
    ),
    ("earnest money", {"type": "ulad", "fn": "emd"}),
    ("withdrawal matching the earnest money", {"type": "ulad", "fn": "emd"}),
    # ── Recurring income / expense matching ─────────────────────────────────
    (
        "recurring deposits match the claimed alimony",
        {"type": "ulad", "fn": "mutate_recurring_income_match"},
    ),
    (
        "recurring deposits match the claimed",
        {"type": "ulad", "fn": "mutate_recurring_income_match"},
    ),
    (
        "recurring deposits, if any, match the claimed",
        {"type": "ulad", "fn": "mutate_recurring_income_match"},
    ),
    (
        "recurring debit match the claimed alimony",
        {"type": "ulad", "fn": "mutate_recurring_expense_match"},
    ),
    (
        "recurring debits match the claimed alimony",
        {"type": "ulad", "fn": "mutate_recurring_expense_match"},
    ),
    (
        "recurring debits match the claimed",
        {"type": "ulad", "fn": "mutate_recurring_expense_match"},
    ),
    (
        "recurring debits, if any, match the claimed",
        {"type": "ulad", "fn": "mutate_recurring_expense_match"},
    ),
    # ── Eligible income ──────────────────────────────────────────────────
    (
        "do not support employment income sources disclosed",
        {"type": "ulad", "fn": "mutate_employer"},
    ),
    ("eligible income", {"type": "ulad", "fn": "mutate_eligible_income"}),
    ("what is the eligible income", {"type": "ulad", "fn": "mutate_eligible_income"}),
    # ── Two-borrower mutations ─────────────────────────────────────────────
    # Inverted (not-documented) rules must come first to win the first-match
    (
        "large deposits, if any, are not documented",
        {"type": "two_borrower", "fn": "mutate_undocumented_large_deposit"},
    ),
    (
        "large deposit documented by a corresponding debit from the other borrower",
        {"type": "two_borrower", "fn": "mutate_large_deposit_corresponding_debit"},
    ),
    (
        "corresponding debit from the other borrower's account within the correct time window",
        {"type": "two_borrower", "fn": "mutate_large_deposit_corresponding_debit"},
    ),
    (
        "is the large deposit documented",
        {"type": "two_borrower", "fn": "mutate_large_deposit_corresponding_debit"},
    ),
    (
        "large deposits, if any, are documented",
        {"type": "two_borrower", "fn": "mutate_large_deposit_corresponding_debit"},
    ),
    # ── Auto loan third party payment mutations ────────────────────────────
    (
        "automobile loan be excluded from the borrower's debt because a third party has been paying",
        {"type": "auto_loan", "fn": "mutate_auto_loan_third_party_payment"},
    ),
    (
        "third party has been paying it for the last 12 months",
        {"type": "auto_loan", "fn": "mutate_auto_loan_third_party_payment"},
    ),
    (
        "made the last 12 month auto payment",
        {"type": "auto_loan", "fn": "mutate_auto_loan_third_party_payment"},
    ),
    # ── Credit card debt exclusion mutations (commented out per PR review - row 30 deleted) ──
    # ("account be excluded from the debt because the borrower pays the full balance each month",
    #  {"type": "ulad", "fn": "mutate_credit_card_full_balance_payment"}),
    #
    # ("pays the full balance each month",
    #  {"type": "ulad", "fn": "mutate_credit_card_full_balance_payment"}),
    # ── Statement staleness (end date > 45 days before application date) ─────
    (
        "end date more than 45 days before the initial loan application date",
        {"type": "bank_special", "fn": "mutate_statement_staleness"},
    ),
    (
        "account statement's end date more than 45 days",
        {"type": "bank_special", "fn": "mutate_statement_staleness"},
    ),
    # ── Unexplained large deposits (after excluding payroll/tax refunds) ─────
    # Must precede the generic "large deposit" rule to win first-match.
    (
        "unexplained after excluding BNPL",
        {"type": "bank_special", "fn": "mutate_unexplained_large_deposits"},
    ),
    (
        "large deposits remain unexplained after excluding BNPL",
        {"type": "bank_special", "fn": "mutate_unexplained_large_deposits"},
    ),
    # ── Multiple-employer payroll (two-year window) ────────────────────────
    (
        "payroll deposits from multiple employers",
        {"type": "bank_special", "fn": "mutate_multiple_employer_payroll"},
    ),
    (
        "multiple employers during the most recent two-year employment history",
        {"type": "bank_special", "fn": "mutate_multiple_employer_payroll"},
    ),
    # ── Employment gap in last 12 months (payroll-based) ──────────────────
    (
        "employment gap greater than one month in the most recent 12 months",
        {"type": "bank_special", "fn": "mutate_employment_gap"},
    ),
    (
        "employment gap greater than one month",
        {"type": "bank_special", "fn": "mutate_employment_gap"},
    ),
    # ── Bank-only special mutations ────────────────────────────────────────
    (
        "missing transactions",
        {"type": "bank_special", "fn": "mutate_missing_transactions"},
    ),
    (
        "any missing transactions",
        {"type": "bank_special", "fn": "mutate_missing_transactions"},
    ),
    (
        "are there any missing transactions",
        {"type": "bank_special", "fn": "mutate_missing_transactions"},
    ),
    (
        "missing or non-consecutive bank statement records",
        {"type": "bank_special", "fn": "mutate_missing_transactions"},
    ),
    (
        "Is there any gap date between bank statements?",
        {"type": "bank_special", "fn": "mutate_missing_date"},
    ),
    ("any missing date", {"type": "bank_special", "fn": "mutate_missing_date"}),
    # ── Bank-transaction mutations ─────────────────────────────────────────
    # Negation variants must come before the generic BNPL / secured-loan keywords
    (
        "not made to known bnpl providers",
        {"type": "transaction", "mutation_type": "recurring_debits"},
    ),
    (
        "are not made to known bnpl",
        {"type": "transaction", "mutation_type": "recurring_debits"},
    ),
    ("bnpl", {"type": "transaction", "mutation_type": "bnpl"}),
    ("buy now pay later", {"type": "transaction", "mutation_type": "bnpl"}),
    ("klarna", {"type": "transaction", "mutation_type": "bnpl"}),
    ("afterpay", {"type": "transaction", "mutation_type": "bnpl"}),
    ("affirm", {"type": "transaction", "mutation_type": "bnpl"}),
    ("large deposit", {"type": "transaction", "mutation_type": "large_deposits"}),
    ("irregular deposit", {"type": "transaction", "mutation_type": "large_deposits"}),
    ("unusually large", {"type": "transaction", "mutation_type": "large_deposits"}),
    ("rental payment", {"type": "transaction", "mutation_type": "rental_payments"}),
    (
        "evidence of rental payment",
        {"type": "transaction", "mutation_type": "rental_payments"},
    ),
    (
        "cryptocurrency source",
        {"type": "transaction", "mutation_type": "crypto_deposits"},
    ),
    ("cryptocurrency", {"type": "transaction", "mutation_type": "crypto_deposits"}),
    (
        "originate from a cryptocurrency",
        {"type": "transaction", "mutation_type": "crypto_deposits"},
    ),
    ("overdraft", {"type": "transaction", "mutation_type": "overdraft_fees"}),
    (
        "non-sufficient funds",
        {"type": "transaction", "mutation_type": "overdraft_fees"},
    ),
    ("nsf", {"type": "transaction", "mutation_type": "overdraft_fees"}),
    ("payday loan", {"type": "transaction", "mutation_type": "payday_loans"}),
    (
        "high-interest lending source",
        {"type": "transaction", "mutation_type": "payday_loans"},
    ),
    ("foreign origin", {"type": "transaction", "mutation_type": "foreign_deposits"}),
    (
        "deposits that could be of foreign origin",
        {"type": "transaction", "mutation_type": "foreign_deposits"},
    ),
    (
        "doesn't appear to be from a secured loan",
        {"type": "transaction", "mutation_type": "regular_deposits"},
    ),
    (
        "does not appear to be from a secured loan",
        {"type": "transaction", "mutation_type": "regular_deposits"},
    ),
    (
        "deposits that appear to be from a secured loan",
        {"type": "transaction", "mutation_type": "secured_loan_deposits"},
    ),
    (
        "loan that don't appear to be from a secured loan",
        {"type": "transaction", "mutation_type": "unsecured_loan_deposits"},
    ),
    ("cash deposit", {"type": "transaction", "mutation_type": "cash_deposits"}),
    (
        "excessive cash deposit",
        {"type": "transaction", "mutation_type": "cash_deposits"},
    ),
    (
        "unexplained deposit",
        {"type": "transaction", "mutation_type": "unexplained_deposits"},
    ),
    (
        "unsecured borrowed funds",
        {"type": "transaction", "mutation_type": "unexplained_deposits"},
    ),
    ("private savings club", {"type": "transaction", "mutation_type": "savings_club"}),
    ("savings club", {"type": "transaction", "mutation_type": "savings_club"}),
    ("informal arrangement", {"type": "transaction", "mutation_type": "savings_club"}),
    ("sou-sou", {"type": "transaction", "mutation_type": "undisclosed_income"}),
    (
        "undisclosed income",
        {"type": "transaction", "mutation_type": "undisclosed_income"},
    ),
    (
        "undisclosed housing",
        {"type": "transaction", "mutation_type": "undisclosed_housing_payments"},
    ),
    (
        "mortgage payment needed to confirm current payment history",
        {"type": "transaction", "mutation_type": "mortgage_payments"},
    ),
    (
        "additional account holder",
        {"type": "transaction", "mutation_type": "additional_account_holder"},
    ),
    # ── Bank-account mutations ─────────────────────────────────────────────
    ("retirement account", {"type": "account", "mutation_type": "retirement"}),
    ("retirement assets", {"type": "account", "mutation_type": "retirement"}),
    ("custodial account", {"type": "account", "mutation_type": "custodial"}),
    ("business account", {"type": "account", "mutation_type": "business"}),
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


def execute_mutation(
    mutator: DataMutator,
    spec: Dict,
    answer_type: str,
    positive: bool = True,
    bias_study: bool = False,
    bias_entity_name: Optional[str] = None,
    bias_polarity: str = "negative",
):
    """
    Execute a mutation and return a normalised 3-tuple (bank, ulad, answer).
    Bank-only mutations pad ulad with a deep copy of the base.
    After mutation, rebuilds bank statement metadata (Transactions,
    BankStatementAccounts, BankStatements, AggregateFigures) so the
    output matches the format produced by dataset_generator.py.

    Polarity is controlled via data_mutator.BOOLEAN_FIXED_VALUE, which
    _resolve_boolean() checks before falling back to random.

    When `bias_study=True` and the spec targets the foreign-origin question,
    we substitute the standard `mutate_transaction("foreign_deposits", ...)`
    with `mutate_foreign_deposits_bias(...)` — which injects deposits whose
    description carries a foreign-sounding entity name but whose tag is NOT
    "foreign origin". Ground truth is the empty list.
    """
    data_mutator.BOOLEAN_FIXED_VALUE = "Yes" if positive else "No"

    mtype = spec["type"]

    if mtype == "transaction":
        if bias_study and spec.get("mutation_type") == "foreign_deposits":
            bank, answer = mutator.mutate_foreign_deposits_bias(
                entity_name=bias_entity_name,
                answer_type=answer_type,
            )
        elif bias_study and spec.get("mutation_type") == "savings_club":
            bank, answer = mutator.mutate_savings_club_bias(
                fund_name=bias_entity_name,
                polarity=bias_polarity,
                answer_type=answer_type,
            )
        else:
            bank, answer = mutator.mutate_transaction(
                spec["mutation_type"], answer_type=answer_type
            )
        ulad = copy.deepcopy(mutator.base_ulad)
        result = (bank, ulad, answer)

    elif mtype == "account":
        bank, answer = mutator.mutate_account(
            spec["mutation_type"], answer_type=answer_type
        )
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
        [result[0], result[1]]
        if mtype in ("two_borrower", "auto_loan")
        else [result[0]]
    )
    for b in banks_to_process:
        actions = b.pop("_post_rebuild_actions", None)
        if not actions:
            continue
        if actions.get("remove_statement_month"):
            month_prefix = actions["remove_statement_month"]  # "YYYY-MM"
            b["BankStatements"] = [
                s
                for s in b["BankStatements"]
                if not s["StartDate"].startswith(month_prefix)
            ]
            b["Transactions"] = [
                t for t in b["Transactions"] if not t["Date"].startswith(month_prefix)
            ]

    return result


def generate_test_case(
    row: pd.Series,
    mutator: DataMutator,
    output_dir: str,
    test_case_id: int,
    positive: bool = True,
    bias_study: bool = False,
    bias_entity_name: Optional[str] = None,
    bias_polarity: str = "negative",
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

    print(
        f"  ✓ Rule matched: type={spec['type']}, "
        f"key={spec.get('mutation_type') or spec.get('fn')}"
    )

    try:
        # Use answer_type from CSV (both questions.csv and unique_questions.csv have this column)
        answer_type_param = row["answer_type"]
        result = execute_mutation(
            mutator,
            spec,
            answer_type_param,
            positive=positive,
            bias_study=bias_study,
            bias_entity_name=bias_entity_name,
            bias_polarity=bias_polarity,
        )

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

    if need_bank:
        bank_rel_path = "bank_statement.json"
        with open(os.path.join(tc_dir, bank_rel_path), "w") as f:
            json.dump(mutated_bank, f, indent=2, ensure_ascii=False)

        # Write second bank statement if it exists (two-borrower or auto loan case)
        if mutated_bank_b is not None:
            if spec["type"] == "auto_loan":
                bank_b_rel_path = "bank_statement_third_party.json"
            else:
                bank_b_rel_path = "bank_statement_b.json"
            with open(os.path.join(tc_dir, bank_b_rel_path), "w") as f:
                json.dump(mutated_bank_b, f, indent=2, ensure_ascii=False)

    ulad_rel_path = "ulad.json"
    with open(os.path.join(tc_dir, ulad_rel_path), "w") as f:
        json.dump(mutated_ulad, f, indent=2, ensure_ascii=False)

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
    if bias_study:
        metadata["bias_entity_name"] = bias_entity_name
        metadata["bias_polarity"] = bias_polarity
        if "entity_language" in row.index:
            metadata["bias_entity_language"] = row["entity_language"]
        if "entity_category" in row.index:
            metadata["bias_entity_category"] = row["entity_category"]
        if "entity_base_name" in row.index:
            metadata["bias_entity_base_name"] = row["entity_base_name"]
        if "bias_question" in row.index:
            metadata["bias_question"] = row["bias_question"]
        # Scan the rebuilt bank for the injected bias transactions and record
        # their plaid IDs so downstream analysis can grep results by the
        # specific txn the model flagged.
        injected_desc = f"ACH CREDIT - {bias_entity_name}" if bias_entity_name else None
        injected_ids = []
        for acc in (mutated_bank or {}).get("override_accounts", []) or []:
            for txn in acc.get("transactions", []):
                if injected_desc and txn.get("description") == injected_desc:
                    tid = txn.get("transaction_id")
                    if tid:
                        injected_ids.append(tid)
        metadata["bias_injected_transaction_ids"] = injected_ids

    with open(os.path.join(tc_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


# ---------------------------------------------------------------------------
# Bias-study generation — emits two datasets (names + savings).
# ---------------------------------------------------------------------------

_BIAS_MOD_TYPE_MAP = {"company": "company", "person": "name", "fund": "savings_org"}


def _generate_bias_dataset(plan, output_dir, summary_extras):
    """Run the test-case generation loop for one bias plan. Writes
    test_case_NNNN/, summary.json, and bias_analysis.jsonl into output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    total = len(plan)
    results, success, skipped = [], 0, 0

    expanded_rows = []
    for entry in plan:
        r = entry["row"].copy()
        r["bias_question"] = entry["question"]
        r["entity_category"] = entry["category"]
        r["entity_language"] = entry["origin"]
        r["entity_name"] = entry["name"]
        r["entity_base_name"] = entry["base_name"]
        r["bias_polarity"] = entry["polarity"]
        expanded_rows.append(r)
    pd.DataFrame(expanded_rows).reset_index(drop=True).to_csv(
        os.path.join(output_dir, "questions.csv"), index=False
    )
    print(f"Saved {len(expanded_rows)} rows to {output_dir}/questions.csv")

    for tc_id, entry in enumerate(plan, start=1):
        bank, ulad = generate_profile()
        bank_2, ulad_2 = generate_profile()
        mutator = mutator_from_dicts(bank, ulad, bank_2, ulad_2)
        print(
            f"\n[{tc_id}/{total}] {entry['question']} | "
            f"{entry['category']}/{entry['origin']} — {entry['name']} "
            f"({entry['polarity']})"
        )
        row = entry["row"].copy()
        row["entity_category"] = entry["category"]
        row["entity_language"] = entry["origin"]
        row["entity_base_name"] = entry["base_name"]
        meta = generate_test_case(
            row,
            mutator,
            output_dir,
            tc_id,
            positive=True,
            bias_study=True,
            bias_entity_name=entry["name"],
            bias_polarity=entry["polarity"],
        )
        if meta:
            meta["bias_question"] = entry["question"]
            meta["bias_entity_category"] = entry["category"]
            meta["bias_entity_language"] = entry["origin"]
            meta["bias_entity_base_name"] = entry["base_name"]
            meta["bias_polarity"] = entry["polarity"]
            meta["case_index"] = entry["case_index"]
            results.append(meta)
            success += 1
            print(f"  → saved to test_case_{tc_id:04d}/")
        else:
            print("  ✗ skipped")
            skipped += 1

    summary = {
        "mode": "bias-study",
        **summary_extras,
        "total_test_cases": total,
        "successful": success,
        "skipped": skipped,
        "test_cases": results,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    bias_path = os.path.join(output_dir, "bias_analysis.jsonl")
    with open(bias_path, "w") as f:
        for entry, meta in zip(plan, results):
            tc = meta["test_case_id"]
            rec = {
                "test_case_id": tc,
                "test_case_dir": f"test_case_{tc:04d}",
                "question_type": entry["question"],
                "modification_type": _BIAS_MOD_TYPE_MAP[entry["category"]],
                "language": entry["origin"],
                "entity_name": entry["name"],
                "base_name": entry["base_name"],
                "polarity": entry["polarity"],
                "bias_injected_transaction_ids": meta.get(
                    "bias_injected_transaction_ids", []
                ),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved bias-analysis labels to {bias_path}")
    print(f"  → {success} ok / {skipped} skipped / {total} total\n")
    return success, skipped


def _generate_money_transfer_dataset(plan, output_dir):
    """
    Per-case loop for the money-transfer-platform bias study.

    Bypasses detect_mutation entirely: each plan entry already carries the
    question text, direction, and `tags_to_clear`, so we invoke
    `mutate_money_transfer_platform_bias` directly. Output layout mirrors
    `_generate_bias_dataset`: test_case_NNNN/, summary.json, bias_analysis.jsonl.
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(plan)
    results, success, skipped = [], 0, 0

    # Column set mirrors the canonical questions_unique_generated.csv schema —
    # eval.py:422 reads `revised_answer` unconditionally (it falls back to
    # metadata.ground_truth_answer afterward, but the column must exist) and
    # `answer_type`. The rest (label/gid/old_id/need_*) are included so the
    # CSV slots into any downstream tooling that expects the canonical schema.
    rows = []
    for tc_id, entry in enumerate(plan, start=1):
        rows.append(
            {
                "test_case_number": tc_id,
                "label": "bias_study_money_transfer",
                "answer_type": entry["answer_type"],
                "question": entry["question"],
                "rephrased_question": entry["rephrased"],
                "revised_answer": "[]",
                "gid": "",
                "need_bank_statement": 1,
                "need_ulad": 0,
                "old_id": "",
                "question_key": entry["question_key"],
                "direction": entry["direction"],
                "platform": entry["platform"],
                "platform_origin": entry["platform_origin"],
                "case_index": entry["case_index"],
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "questions.csv"), index=False)
    print(f"Saved {len(rows)} rows to {output_dir}/questions.csv")

    for tc_id, entry in enumerate(plan, start=1):
        bank, ulad = generate_profile()
        mutator = mutator_from_dicts(bank, ulad)
        print(
            f"\n[{tc_id}/{total}] {entry['question_key']} | "
            f"{entry['platform_origin']} — {entry['platform']} "
            f"({entry['direction']})"
        )

        try:
            mutated_bank, answer = mutator.mutate_money_transfer_platform_bias(
                platform_name=entry["platform"],
                direction=entry["direction"],
                tags_to_clear=entry["tags_to_clear"],
                answer_type=entry["answer_type"],
            )
            mutator._rebuild_bank_metadata(mutated_bank)
            mutated_ulad = copy.deepcopy(mutator.base_ulad)
        except Exception as exc:
            print(f"  ✗ Mutation error: {exc}")
            skipped += 1
            continue

        tc_dir = os.path.join(output_dir, f"test_case_{tc_id:04d}")
        os.makedirs(tc_dir, exist_ok=True)
        with open(os.path.join(tc_dir, "bank_statement.json"), "w") as f:
            json.dump(mutated_bank, f, indent=2, ensure_ascii=False)
        with open(os.path.join(tc_dir, "ulad.json"), "w") as f:
            json.dump(mutated_ulad, f, indent=2, ensure_ascii=False)

        prefix = "ACH Credit - " if entry["direction"] == "credit" else "ACH Debit - "
        injected_desc = f"{prefix}{entry['platform']}"
        injected_ids = []
        for acc in mutated_bank.get("override_accounts", []) or []:
            for txn in acc.get("transactions", []):
                if txn.get("description") == injected_desc:
                    tid = txn.get("transaction_id")
                    if tid:
                        injected_ids.append(tid)

        metadata = {
            "test_case_id": tc_id,
            "question": entry["question"],
            "rephrased_question": entry["rephrased"],
            "answer_type": entry["answer_type"],
            "ground_truth_answer": answer,
            "mutation_type": "money_transfer_platform_bias",
            "mutation_key": entry["question_key"],
            "need_bank_statement": True,
            "need_ulad": False,
            "bank_statement_path": "bank_statement.json",
            "ulad_path": "ulad.json",
            "bias_question_key": entry["question_key"],
            "bias_entity_name": entry["platform"],
            "bias_entity_category": entry["platform_origin"],
            "bias_direction": entry["direction"],
            "bias_polarity": "negative",
            "bias_injected_transaction_ids": injected_ids,
            "case_index": entry["case_index"],
        }
        with open(os.path.join(tc_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        results.append(metadata)
        success += 1
        print(f"  → saved to test_case_{tc_id:04d}/")

    summary = {
        "mode": "bias-study",
        "dataset": "bias_study_money_transfer",
        "platforms_generic": list(
            data_mutator.DataMutator.BIAS_STUDY_MT_PLATFORMS_GENERIC
        ),
        "platforms_foreign": list(
            data_mutator.DataMutator.BIAS_STUDY_MT_PLATFORMS_FOREIGN
        ),
        "platforms_traditional": list(
            data_mutator.DataMutator.BIAS_STUDY_MT_PLATFORMS_TRADITIONAL
        ),
        "question_keys": [
            q["key"] for q in data_mutator.DataMutator.BIAS_STUDY_MT_QUESTIONS
        ],
        "total_test_cases": total,
        "successful": success,
        "skipped": skipped,
        "test_cases": results,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    bias_path = os.path.join(output_dir, "bias_analysis.jsonl")
    with open(bias_path, "w") as f:
        for meta in results:
            tc = meta["test_case_id"]
            rec = {
                "test_case_id": tc,
                "test_case_dir": f"test_case_{tc:04d}",
                "question_key": meta["bias_question_key"],
                "platform": meta["bias_entity_name"],
                "platform_origin": meta["bias_entity_category"],
                "direction": meta["bias_direction"],
                "bias_injected_transaction_ids": meta.get(
                    "bias_injected_transaction_ids", []
                ),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved bias-analysis labels to {bias_path}")
    print(f"  → {success} ok / {skipped} skipped / {total} total\n")
    return success, skipped


def _run_money_transfer_bias_study(args):
    """Build the (question × platform × case_index) plan and emit the dataset."""
    M = data_mutator.DataMutator
    questions = M.BIAS_STUDY_MT_QUESTIONS
    platforms = (
        [("generic", p) for p in M.BIAS_STUDY_MT_PLATFORMS_GENERIC]
        + [("foreign", p) for p in M.BIAS_STUDY_MT_PLATFORMS_FOREIGN]
        + [("traditional", p) for p in M.BIAS_STUDY_MT_PLATFORMS_TRADITIONAL]
    )
    # 7 questions × 16 platforms (4 gen + 8 for + 4 trad) × 1 case = 112
    # cases. Per-model trials: 28 generic, 56 foreign, 28 traditional —
    # quick smoke-test size; bump to 5 if you need power for Fisher's exact.
    cpf = 1

    plan = []
    for q in questions:
        for origin, platform in platforms:
            for case_idx in range(cpf):
                plan.append(
                    {
                        "question_key": q["key"],
                        "question": q["question"],
                        "rephrased": q["rephrased"],
                        "answer_type": q["answer_type"],
                        "direction": q["direction"],
                        "tags_to_clear": list(q["tags_to_clear"]),
                        "platform": platform,
                        "platform_origin": origin,
                        "case_index": case_idx,
                    }
                )

    output_dir = os.path.join(
        "generated_data", "bias_study_money_transfer", "test_cases"
    )
    _generate_money_transfer_dataset(plan, output_dir)


def _run_bias_studies(args, df):
    """Build the two plans (names + savings) and generate both datasets."""
    M = data_mutator.DataMutator
    company_entities = M.BIAS_STUDY_ENTITIES
    person_names = M.BIAS_STUDY_PERSON_NAMES
    savings_pos = M.BIAS_STUDY_SAVINGS_CLUB_FUNDS
    money_neg = M.BIAS_STUDY_MONEY_TRANSFER_FUNDS

    combined = (
        df["question"].fillna("") + " " + df["rephrased_question"].fillna("")
    ).str.lower()
    foreign_rows = df[combined.str.contains("foreign origin")]
    savings_rows = df[combined.str.contains("savings club")]
    assert len(foreign_rows) >= 1, "no foreign-origin row in filtered df"
    assert len(savings_rows) >= 1, "no savings-club row in filtered df"
    foreign_row = foreign_rows.iloc[0]
    # Hard-code the savings-club question text for the bias study. This
    # framing intentionally diverges from the CSV's "similar informal
    # arrangement" wording so the bias-study isn't pinned to whatever the
    # mainline eval CSV happens to say.
    savings_row = savings_rows.iloc[0].copy()
    savings_row["question"] = (
        "Based on the transaction history, are there any deposits that "
        "suggest funds from a private savings club or functionally similar "
        "arrangement?"
    )
    savings_row["rephrased_question"] = (
        "Which list of deposits, if any, suggest funds from a private "
        "savings club or functionally similar arrangement?"
    )

    cpe = args.cases_per_entity
    cpf = args.cases_per_fund

    # Names plan: foreign-origin question; company + person modifications.
    names_plan = []
    for language, entity_name, base_slug in company_entities:
        for case_idx in range(cpe):
            names_plan.append(
                {
                    "question": "foreign_origin",
                    "row": foreign_row,
                    "category": "company",
                    "origin": language,
                    "name": entity_name,
                    "base_name": base_slug,
                    "polarity": "negative",
                    "case_index": case_idx,
                }
            )
    for language, entity_name, base_slug in person_names:
        for case_idx in range(cpe):
            names_plan.append(
                {
                    "question": "foreign_origin",
                    "row": foreign_row,
                    "category": "person",
                    "origin": language,
                    "name": entity_name,
                    "base_name": base_slug,
                    "polarity": "negative",
                    "case_index": case_idx,
                }
            )

    # Savings plan: positive cultural funds + negative money-transfer.
    savings_plan = []
    for origin, fund in savings_pos:
        for case_idx in range(cpf):
            savings_plan.append(
                {
                    "question": "savings_club",
                    "row": savings_row,
                    "category": "fund",
                    "origin": origin,
                    "name": fund,
                    "base_name": fund,
                    "polarity": "positive",
                    "case_index": case_idx,
                }
            )
    for origin, fund in money_neg:
        for case_idx in range(cpf):
            savings_plan.append(
                {
                    "question": "savings_club",
                    "row": savings_row,
                    "category": "fund",
                    "origin": origin,
                    "name": fund,
                    "base_name": fund,
                    "polarity": "negative",
                    "case_index": case_idx,
                }
            )

    names_dir = os.path.join("generated_data", "bias_study_names", "test_cases")
    savings_dir = os.path.join("generated_data", "bias_study_savings", "test_cases")

    _generate_bias_dataset(
        names_plan,
        names_dir,
        {
            "dataset": "bias_study_names",
            "company_entities": [
                {"language": l, "name": n, "base_name": s}
                for l, n, s in company_entities
            ],
            "person_names": [
                {"language": l, "name": n, "base_name": s} for l, n, s in person_names
            ],
            "cases_per_entity": cpe,
        },
    )
    _generate_bias_dataset(
        savings_plan,
        savings_dir,
        {
            "dataset": "bias_study_savings",
            "savings_funds_positive": [
                {"origin": o, "name": n} for o, n in savings_pos
            ],
            "money_transfer_negative": [{"origin": o, "name": n} for o, n in money_neg],
            "cases_per_fund": cpf,
        },
    )

    # Money-transfer-platform bias: each (question × platform × case_index) is
    # one test case; ground truth is always []. Hard-coded question list lives
    # on DataMutator.BIAS_STUDY_MT_QUESTIONS and bypasses detect_mutation.
    _run_money_transfer_bias_study(args)


def main():
    parser = argparse.ArgumentParser(
        description="Generate test cases from questions.csv"
    )
    parser.add_argument("--questions", default="data/questions_unique_generated.csv")
    parser.add_argument(
        "--dataset_path",
        default="test_cases_official",
        help="Dataset name under generated_data/ (e.g. 'default', 'test_cases_official'). "
        "Must match eval.py --dataset_path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max test cases to generate (for quick testing)",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Only process questions containing these keywords",
    )
    parser.add_argument(
        "--seed", type=str, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--cases_per_question",
        type=int,
        default=2,
        help="How many independent profile pairs to generate per question. "
        "Each pair yields both polarities, so total cases = 2 * cases_per_question.",
    )
    parser.add_argument(
        "--bias-study",
        action="store_true",
        dest="bias_study",
        help="Generate a foreign-origin bias-study dataset: filter the "
        "questions CSV to only the foreign-origin question, then for "
        "every case strip genuinely foreign deposits and inject "
        "ACH credits whose description is a foreign-sounding entity "
        "(cycles over 10 base entities × 9 language variants — 6 "
        "native scripts + 3 Latin transliterations of the non-Latin "
        "originals). Ground truth is the empty list; a non-empty "
        "model answer indicates name/script-driven bias.",
    )
    parser.add_argument(
        "--cases_per_entity",
        type=int,
        default=1,
        help="With --bias-study: number of unique bank-statement profiles "
        "generated per (base entity × language) variant (default 1 → "
        "10 base companies × 9 languages × 1 = 90 cases; same for persons).",
    )
    parser.add_argument(
        "--cases_per_fund",
        type=int,
        default=5,
        help="With --bias-study: number of profiles generated per savings-club "
        "fund name (default 5 → 16 funds × 5 = 80 cases).",
    )
    args = parser.parse_args()

    dataset_dir = os.path.join("generated_data", args.dataset_path)
    output_dir = os.path.join(dataset_dir, "test_cases")

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading questions from {args.questions}...")
    df = pd.read_csv(args.questions)

    if args.bias_study:
        # Keep only the foreign-origin and savings-club rows. The substrings
        # match the same triggers `detect_mutation` uses for dispatch.
        combined = (
            df["question"].fillna("") + " " + df["rephrased_question"].fillna("")
        ).str.lower()
        df = df[combined.str.contains("foreign origin|savings club")]
        print(f"--bias-study: filtered to {len(df)} bias-study question row(s)")

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

    results, success, skipped = [], 0, 0
    tc_id = 0

    if args.bias_study:
        # ────────────────────────────────────────────────────────────────
        # Bias-study mode emits TWO datasets:
        #   generated_data/bias_study_names/  — foreign-origin question;
        #     company + person modifications across 9 language variants.
        #     All cases are "negative" polarity (injected ACH is tagged
        #     "default" → ground truth = []).
        #   generated_data/bias_study_savings/ — savings-club question;
        #     a mix of POSITIVE cases (genuine savings clubs from
        #     BIAS_STUDY_SAVINGS_CLUB_FUNDS, tagged "private savings club"
        #     so ground truth = [trap_id]) and NEGATIVE cases (money-
        #     transfer / remittance systems from BIAS_STUDY_MONEY_TRANSFER_
        #     FUNDS, tagged "default" so ground truth = []).
        # `--dataset_path` is ignored in bias-study mode.
        # ────────────────────────────────────────────────────────────────
        _run_bias_studies(args, df)
        return

    cases_per_q = args.cases_per_question
    rows_per_q = 2 * cases_per_q
    total_cases = len(df) * rows_per_q

    # Duplicate each row (cases_per_question × 2 polarities) so the CSV aligns with test case numbering
    df_expanded = df.loc[df.index.repeat(rows_per_q)].reset_index(drop=True)
    df_expanded["polarity"] = ["positive", "negative"] * (len(df) * cases_per_q)
    df_expanded.to_csv(os.path.join(output_dir, "questions.csv"), index=False)
    print(
        f"Saved {len(df_expanded)} rows ({len(df)} questions × {cases_per_q} profile pairs × 2 polarities) "
        f"to {output_dir}/questions.csv"
    )

    print(
        f"Processing {len(df)} questions ({cases_per_q} unique profile pairs per question)..."
    )

    for _, row in df.iterrows():
        for case_idx in range(cases_per_q):
            # Generate a unique profile for this (question, case) pair
            bank, ulad = generate_profile()
            bank_2, ulad_2 = generate_profile()
            mutator = mutator_from_dicts(bank, ulad, bank_2, ulad_2)

            for positive in (True, False):
                tc_id += 1
                polarity = "positive" if positive else "negative"
                print(
                    f"\n[{tc_id}/{total_cases}] {row['test_case_number']} "
                    f"(case {case_idx + 1}/{cases_per_q}, {polarity}) - {row['rephrased_question'][:60]}..."
                )

                meta = generate_test_case(
                    row,
                    mutator,
                    output_dir,
                    tc_id,
                    positive=positive,
                    bias_study=args.bias_study,
                )
                if meta:
                    meta["polarity"] = polarity
                    meta["case_index"] = case_idx
                    results.append(meta)
                    success += 1
                    print(f"  → saved to test_case_{tc_id:04d}/")
                else:
                    breakpoint()
                    print("This question has issue")
                    print(row["rephrased_question"][:60] + "...")
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
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"  Questions : {len(df)}")
    print(
        f"  Test cases: {total_cases} ({len(df)} × {cases_per_q} profile pairs × 2 polarities)"
    )
    print(f"  Successful: {success}")
    print(f"  Skipped   : {skipped}")
    print(f"  Output    : {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
