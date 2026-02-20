# Data Mutation Guide

## Overview

A data mutation system that generates test cases from a list of questions CSV.  
The system mutates both bank statements and ULAD files, then records a ground truth answer for each question.

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
| `mortgage_payments` | (missing) | ACH DEBIT mortgage payment |
| `savings_club` | (missing) | Community savings club funds |

### Bank Account Mutations (`mutate_account`)

| Key | Type | Description |
|-----|------|-------------|
| `retirement` | `investment / 401k` | Add/remove 401k with contributions |
| `custodial` | `depository / money market` | Add/remove UTMA custodial account |
| `business`(missing)| `depository / checking (class=business)` | Add/remove business checking |

### ULAD Cross-Document Mutations (return `bank, ulad, answer`)

| Method | What it mutates | Default answer_type |
|--------|----------------|-------------|
| `mutate_employer_payroll_consistency` | ULAD employer + bank payroll deposits | `boolean` |
| `mutate_address_match` | Bank identity address vs ULAD residence address | `boolean` |
| `mutate_gift_deposit` | ULAD PURCHASE_CREDITS gift amount + matching bank deposit | `id_list` |
| `mutate_child_support_disclosure` | Bank: recurring child-support payments not in ULAD | `id_list` |
| `mutate_undisclosed_liabilities` | Bank: creditor payments absent from ULAD LIABILITIES | `id_list` |
| `mutate_rental_income_consistency` | ULAD REO rental income + bank deposits | `boolean` |
| `mutate_joint_account_holder` | Bank: joint account with non-borrower; ULAD: single borrower | `id_list_account` |
| `mutate_payroll_paystub_consistency` | Bank payroll deposits vs hypothetical paystub amount | `boolean` |
| `mutate_payroll_undisclosed_employer` | ULAD employer A, bank payroll from employer B | `id_list` |
| `mutate_undisclosed_income_source` | Bank: recurring SSA/side-gig deposits absent from ULAD | `id_list` |

**Note:** All ULAD functions now accept an `answer_type` parameter to control output format.

---

## Examples

```bash
# Generate ALL test cases (88/90 covered)
python generate_test_cases.py

# Generate first 10 for a quick look
python generate_test_cases.py --limit 10

# Only BNPL-related questions
python generate_test_cases.py --tags "BNPL"

# Multiple tags
python generate_test_cases.py --tags "BNPL" "large deposit" "payday loan"

# Custom file paths
python generate_test_cases.py \
  --bank-statement generated_data/bank_statement_template.json \
  --ulad generated_data/ulad_template.json \
  --output my_test_cases
```

---

## Output Structure

```
test_cases/
├── summary.json               ← aggregate stats + all metadata
├── test_case_0001/
│   ├── metadata.json          ← question, answer, mutation info
│   ├── bank_statement.json    ← mutated Plaid JSON (if need_bank_statement=1)
│   └── ulad.json              ← mutated ULAD JSON   (if need_ulad=1)
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

