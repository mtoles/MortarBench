# Data Mutation Guide

## Overview

A data mutation system that generates test cases from a list of questions CSV.
The system mutates both bank statements and ULAD files, then records a ground truth answer for each question.

---

## Quick Start (End-to-End)

End-to-end process to generate the test cases:
```bash
# Full pipeline: generate templates + mutate test cases
./run_generate_test_cases.sh

# Reuse existing templates (skip dataset_generator.py)
./run_generate_test_cases.sh --skip-generate

# Custom seed for reproducibility
./run_generate_test_cases.sh --seed my-seed-123

# Use a different questions file
./run_generate_test_cases.sh --questions data/questions.csv --output generated_data/test_cases
```

### What the script does

1. **Step 1** — Runs `dataset_generator.py` twice to produce:
   - `generated_data/bank_statement.json` + `generated_data/ulad.json` (primary borrower)
   - `generated_data/bank_statement_2.json` + `generated_data/ulad_2.json` (second borrower, for two-borrower scenarios)
2. **Step 2** — Runs `generate_test_cases.py` with `data/questions_unique_generated.csv` to produce mutated test cases in `generated_data/test_cases_unique/`.

To regenerate more synthetic data, simply run the script again (optionally with a different `--seed`).

---

## Boolean Probability Control

All boolean-answer mutations are controlled by two global variables in `data_mutator.py`:

```python
BOOLEAN_PROB = 0.5          # Probability of "Yes" answer (0.0–1.0)
BOOLEAN_FIXED_VALUE = None  # Set to "Yes" or "No" to force all answers
```

| Setting | Effect |
|---------|--------|
| `BOOLEAN_PROB = 0.5` | 50/50 chance of Yes/No (default) |
| `BOOLEAN_PROB = 1.0` | Always Yes |
| `BOOLEAN_PROB = 0.0` | Always No |
| `BOOLEAN_FIXED_VALUE = "Yes"` | Force all boolean answers to Yes (overrides `BOOLEAN_PROB`) |
| `BOOLEAN_FIXED_VALUE = "No"` | Force all boolean answers to No |

Individual mutation functions still accept an explicit `match=True/False` parameter that takes highest priority.

---

## Components

### `data_mutator.py`
Core mutation library with three public interfaces:

| Method | Returns | Use when |
|--------|---------|----------|
| `mutate_transaction(type, num=None, answer_type="id_list")` | `(bank, answer)` | Adding/removing tagged transactions |
| `mutate_account(type, has=None, answer_type="id_list")` | `(bank, answer)` | Adding/removing whole accounts |
| `mutate_*(answer_type="boolean")` | `(bank, ulad, answer)` | Cross-document ULAD consistency checks |

**Answer Format Types:**
- `"boolean"`: Returns `"Yes"` or `"No"`
- `"id_list"`: Returns `["transaction_id1", "transaction_id2", ...]` or `[]`
- `"id_list_account"`: Returns `["account_num1", "account_num2", ...]` or `[]`

### `generate_test_cases.py`
Batch processor that reads any questions CSV file, detects the right mutation via
`MUTATION_RULES`, runs it, and writes each test case to its own directory.
Automatically handles answer formatting based on `answer_type` column when present.

### `run_generate_test_cases.sh`
End-to-end shell script that runs the full pipeline (template generation + test case mutation) in one command.

---

## Mutation Types

### Bank Transaction Mutations (`mutate_transaction`)

| Key | Tag | Description |
|-----|-----|-------------|
| `bnpl` | `BNPL transactions` | Klarna, Afterpay, Affirm, etc. |
| `large_deposits` | `large deposits` | Wire gifts, ACH credit |
| `rental_payments` | `rental payments` | Monthly rent (spaced 30 days) |
| `crypto_deposits` | `deposit from cryptocurrency source` | Coinbase, Gemini, etc. |
| `overdraft_fees` | `overdraft or NSF` | $25/$35/$50 discrete fees |
| `payday_loans` | `payday loan or high-interest lending source` | Speedy Cash, BEFOREPAY, etc. |
| `foreign_deposits` | `foreign origin` | INTL WIRE IN CREDIT |
| `secured_loan_deposits` | `secured loan` | 401K LOAN, Fidelity |
| `cash_deposits` | `excessive cash deposits` | ATM Cash Deposit |
| `unexplained_deposits` | `unexplained deposits` | LendingClub, LIGHTSTREAM, etc. |
| `undisclosed_income` | `undisclosed income source` | SSA, side gig (recurring) |
| `undisclosed_housing_payments` | `undisclosed housing payments` | Monthly housing |
| `withdrawals` | `withdrawal` | Earnest money, wire out |
| `additional_account_holder` | `additional account holder` | Transfer from joint holder |
| `mortgage_payments` | `mortgage payments` | ACH DEBIT mortgage payment |
| `savings_club` | `private savings club` | Community savings club funds |

### Bank Account Mutations (`mutate_account`)

| Key | Type | Description |
|-----|------|-------------|
| `retirement` | `investment / 401k` | Add/remove 401k with contributions |
| `custodial` | `depository / money market` | Add/remove UTMA custodial account |
| `business` | `depository / checking (class=business)` | Add/remove business checking |

### ULAD Cross-Document Mutations (return `bank, ulad, answer` unless noted)

| Method | What it mutates | Default answer_type |
|--------|----------------|-------------|
| `mutate_employer_payroll_consistency` | ULAD employer + bank payroll deposits | `boolean` |
| `mutate_address_match` | Bank identity address vs ULAD residence address | `boolean` |
| `mutate_gift_deposit` | ULAD PURCHASE_CREDITS gift amount + matching bank deposit | `id_list` |
| `mutate_child_support_disclosure` | Bank: recurring child-support payments not in ULAD | `id_list` |
| `mutate_undisclosed_liabilities` | Bank: recurring BNPL / alimony / Venmo rent payments **present on the bank statement but intentionally absent from ULAD LIABILITIES** | `id_list` |
| `mutate_rental_income_consistency` | ULAD REO rental income + bank deposits | `boolean` |
| `mutate_joint_account_holder` | Bank: joint account with non-borrower; ULAD: single borrower | `id_list_account` |
| `mutate_payroll_paystub_consistency` | Bank payroll deposits vs hypothetical paystub amount | `boolean` |
| `mutate_payroll_undisclosed_employer` | ULAD employer A, bank payroll from employer B | `id_list` |
| `mutate_undisclosed_income_source` | Bank: recurring SSA/side-gig deposits absent from ULAD | `id_list` |
| `mutate_recurring_income_match` | ULAD income items (alimony, child support, Social Security, etc.) + matching or mismatching recurring deposits to bank; supports `disclosed=False` | `boolean` |
| `mutate_recurring_expense_match` | ULAD EXPENSES (alimony, child support, SSA) + matching or mismatching recurring debits to bank; supports `disclosed=False` | `boolean` |
| `mutate_eligible_income` | 12 months of categorized transactions (qualifying deposits, non-qualifying deposits, obligations); answer = qualifying minus obligations | `dollar_amount` |
| `mutate_large_deposit_corresponding_debit` | **Two-bank-statement + ULAD**: large deposit in Borrower A's account and (optional) matching debit in Borrower B's account within a 3-day window | `boolean` / `id_list` / detailed |
| `mutate_auto_loan_third_party_payment` | **Two-bank-statement + ULAD**: auto loan liability in ULAD; third party pays the auto loan for >=12 months in their own (non-joint) account | `boolean` / `id_list` / detailed |

### Bank-Only Special Mutations

These return only a bank statement and an answer (no ULAD changes).

| Method | What it mutates | Default answer_type |
|--------|----------------|-------------|
| `mutate_missing_transactions` | Adjusts ending balance so that `starting_balance + sum(transactions) != end_balance`, creating an apparent gap / missing transactions | `boolean` / detailed |
| `mutate_missing_date` | Enables monthly `BankStatements`, then removes one middle month's statement to create a gap in date coverage | `boolean` / detailed |

**Note:** All ULAD functions now accept an `answer_type` parameter to control output format.

---

## Examples

```bash
# End-to-end: generate templates + test cases
./run_generate_test_cases.sh

# Generate ALL test cases from unique questions
python generate_test_cases.py \
  --questions data/questions_unique_generated.csv \
  --bank-statement generated_data/bank_statement.json \
  --ulad generated_data/ulad.json \
  --bank-statement-2 generated_data/bank_statement_2.json \
  --ulad-2 generated_data/ulad_2.json \
  --output generated_data/test_cases_unique

# Generate first 10 for a quick look
python generate_test_cases.py --limit 10

# Only BNPL-related questions
python generate_test_cases.py --tags "BNPL"

# Multiple tags
python generate_test_cases.py --tags "BNPL" "large deposit" "payday loan"
```

---

## Bank Statement Format

Each generated `bank_statement.json` contains the following top-level keys:

| Key | Description |
|-----|-------------|
| `seed` | Unique dataset identifier |
| `override_accounts` | Raw account objects with nested transactions (internal format) |
| `SearchParams` | Sort/pagination params (`SortField`, `SortOrder`, `Size`) |
| `Transactions` | Flat list of all transactions across accounts (eval-ready format) |
| `BankStatementAccounts` | Per-account metadata (balances, totals, masked account numbers) |
| `BankStatements` | Per-account statement metadata (date ranges, client info) |
| `AggregateFigures` | Overall credit/debit totals and transaction count |

The `Transactions` list uses absolute `Amount` values with a `Type` field (`"credit"` or `"debit"`), while `override_accounts` uses signed amounts.

After mutation, `generate_test_cases.py` calls `_rebuild_bank_metadata()` to regenerate `Transactions`, `BankStatementAccounts`, `BankStatements`, and `AggregateFigures` from `override_accounts`, keeping all sections consistent.

---

## Output Structure

```
generated_data/test_cases_unique/
├── summary.json               <- aggregate stats + all metadata
├── test_case_0001/
│   ├── metadata.json          <- question, answer, mutation info
│   ├── bank_statement.json    <- mutated Plaid JSON (if need_bank_statement=1)
│   └── ulad.json              <- mutated ULAD JSON   (if need_ulad=1)
├── test_case_0002/
│   └── ...
└── ...
```

---

### TRANSACTION_CONFIGS fields

| Field | Type | Description |
|-------|------|-------------|
| `tag` | str | Transaction tag for remove/insert |
| `keywords` | list[str] | Provider / description keywords |
| `description_template` | str | `"{keyword}"` or `"ACH DEBIT - {keyword} PMT"` |
| `amount_range` | tuple | `(min, max)` uniform random |
| `amount_discrete` | list | Use instead of range (e.g. NSF fees) |
| `amount_sign` | str | `"positive"` or `"negative"` |
| `default_count_range` | tuple | `(min, max)` transaction count |
| `date_spacing` | str/None | `"monthly"`, `"bi-weekly"`, `"weekly"`, or `None` |
| `recurring` | bool | All transactions share the same keyword + base amount |
