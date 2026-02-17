# Data Mutation Guide

## Overview

A data mutation system that generates test cases from `questions.csv` file. 
The system automatically mutates bank statements and ULAD data, and generates ground truth answers in your expected format  


## Components

### 1. `data_mutator.py`
Core mutation library with unified mutation functions and configuration dictionaries.

**Key Features:**
- Removes existing transactions by tag
- Adds new randomized transactions
- Generates formatted ground truth answers
- Maintains transaction consistency (dates, IDs, amounts)

**Functions:**
- `mutate_transaction(transaction_type, num_transactions)` - Handles all transaction mutations
- `mutate_account(account_type, has_account)` - Handles all account mutations

**Available Mutation Types:**

**Transaction Types (14):**
- `bnpl` - BNPL payments (Klarna, Afterpay, etc.)
- `large_deposits` - Large deposits (wires, gifts)
- `rental_payments` - Rental/housing payments
- `crypto_deposits` - Cryptocurrency deposits
- `overdraft_fees` - Overdraft and NSF fees
- `payday_loans` - Payday loan deposits
- `foreign_deposits` - International wire transfers
- `secured_loan_deposits` - Secured loan proceeds (401k loans)
- `cash_deposits` - Excessive cash deposits
- `unexplained_deposits` - Unsecured loan deposits
- `undisclosed_income` - Side income, consulting, SSA
- `undisclosed_housing_payments` - Undisclosed mortgage/housing
- `withdrawals` - General withdrawals, earnest money
- `additional_account_holder` - Transactions from joint holders

**Account Types (2):**
- `retirement` - Add/remove retirement accounts
- `custodial` - Add/remove custodial accounts

### 2. `generate_test_cases.py`
Batch processor that reads questions.csv and generates complete test cases.

**Features:**
- Automatically detects which mutation to apply based on question text
- Generates separate directories for each test case
- Creates metadata files with ground truth answers
- Supports filtering by tag keywords
- Generates summary report

## Quick Start

### Generate All Test Cases

```bash
python generate_test_cases.py
```

This will:
1. Read all questions from `data/questions.csv`
2. Generate test cases in `test_cases/` directory
3. Each test case gets its own folder: `test_case_0001/`, `test_case_0002/`, etc.

### Generate Limited Test Cases

```bash
python generate_test_cases.py --limit 10
```

### Generate Only Specific Tags

```bash
# Only BNPL-related questions
python generate_test_cases.py --tags "BNPL"

# Multiple tags
python generate_test_cases.py --tags "BNPL" "large deposits" "cryptocurrency"
```

### Custom Paths

```bash
python generate_test_cases.py \
  --questions data/questions.csv \
  --bank-statement generated_data/dataset_generated-test-7a8d6178.json \
  --ulad data/ulad.json \
  --output my_test_cases
```

## Output Structure

After running the generator, you'll have:

```
test_cases/
├── summary.json                    # Overall summary of generation
├── test_case_0001/
│   ├── metadata.json              # Test case info and ground truth
│   ├── bank_statement.json        # Mutated bank statement
│   └── ulad.json                  # ULAD data (if needed)
├── test_case_0002/
│   ├── metadata.json
│   ├── bank_statement.json
│   └── ulad.json
└── ...
```

### metadata.json Structure

```json
{
  "test_case_id": 1,
  "test_case_number": "1",
  "question": "Original question text",
  "rephrased_question": "Rephrased question text",
  "answer_type": "id_list",
  "label": "has_any",
  "mutation_function": "mutate_bnpl_transactions",
  "ground_truth_answer": "3 transactions:\n1) 2026-01-15 - ACH DEBIT - Klarna PMT - $62.00\n...",
  "need_bank_statement": true,
  "need_ulad": false,
  "bank_statement_path": "bank_statement.json",
  "ulad_path": null
}
```

## Usage

```python
from data_mutator import DataMutator

# Initialize mutator
mutator = DataMutator(
    "generated_data/dataset_generated-test-7a8d6178.json",
    "data/ulad.json"
)

# Transaction mutations using unified function
bank, answer = mutator.mutate_transaction("bnpl", num_transactions=3)
bank, answer = mutator.mutate_transaction("large_deposits", num_deposits=5)
bank, answer = mutator.mutate_transaction("crypto_deposits")  # Random count

# Account mutations using unified function
bank, answer = mutator.mutate_account("retirement", has_account=True)
bank, answer = mutator.mutate_account("custodial")  # Random decision
```

### Mutation Function Parameters

Most mutation functions accept an optional parameter to control randomness:

```python
# Random number of transactions (uses default_count_range from config)
bank, answer = mutator.mutate_transaction("bnpl")

# Exactly 3 transactions
bank, answer = mutator.mutate_transaction("bnpl", num_transactions=3)

# No transactions (negative test case)
bank, answer = mutator.mutate_transaction("bnpl", num_transactions=0)
```

For account-based mutations:

```python
# Random decision
bank, answer = mutator.mutate_account("retirement")

# Force presence
bank, answer = mutator.mutate_account("retirement", has_account=True)

# Force absence
bank, answer = mutator.mutate_account("retirement", has_account=False)
```

## Configuration System

### TRANSACTION_CONFIGS

All transaction mutations are defined in the `TRANSACTION_CONFIGS` dictionary in `data_mutator.py`:

```python
TRANSACTION_CONFIGS = {
    "bnpl": {
        "tag": "BNPL transactions",
        "keywords": ["Klarna", "Afterpay", "Affirm", ...],
        "description_template": "ACH DEBIT - {keyword} PMT",
        "amount_range": (25, 150),
        "amount_sign": "negative",
        "default_count_range": (0, 4),
        "date_spacing": None,
    },
    # ... 13 more transaction types
}
```

**Configuration Fields:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `tag` | str | Transaction tag for filtering | `"BNPL transactions"` |
| `keywords` | list[str] | Provider/description keywords | `["Klarna", "Afterpay"]` |
| `description_template` | str | Template for transaction description | `"ACH DEBIT - {keyword} PMT"` |
| `amount_range` | tuple | (min, max) amount range | `(25, 150)` |
| `amount_discrete` | list[float] | Discrete amounts (alternative to range) | `[25, 35, 50]` |
| `amount_sign` | str | `"positive"` or `"negative"` | `"negative"` |
| `default_count_range` | tuple | (min, max) default transaction count | `(0, 4)` |
| `date_spacing` | str/None | Date spacing pattern | `"monthly"`, `"bi-weekly"`, `None` |
| `recurring` | bool | Use same keyword for all transactions | `True` |

### ACCOUNT_CONFIGS

All account mutations are defined in the `ACCOUNT_CONFIGS` dictionary:

```python
ACCOUNT_CONFIGS = {
    "retirement": {
        "tag": "retirement accounts",
        "type": "investment",
        "subtype": "401k",
        "balance_range": (20000, 150000),
        "transaction_tag": "retirement assets",
        "transaction_description": "Contribution",
        "transaction_amount": 500,
        "official_name": None,
        "answer_format": "Retirement account #{account_num} - ${balance:,.2f}",
    },
    # ... more account types
}
```


## Advanced Configuration Features

### Date Spacing

Control how transactions are spaced in time:

```python
# Random dates (default)
"date_spacing": None

# Monthly spacing (30 days apart)
"date_spacing": "monthly"

# Bi-weekly spacing (15 days apart)
"date_spacing": "bi-weekly"

# Weekly spacing (7 days apart)
"date_spacing": "weekly"
```

### Recurring Transactions

For recurring income/payments that should use the same provider:

```python
{
    "tag": "undisclosed income source",
    "keywords": ["Consulting Income", "Side Gig", "Freelance Payment"],
    "recurring": True,  # All transactions will use same keyword
    "amount_range": (200, 2500),
    # Each transaction gets amount with slight variance
}
```

### Discrete Amounts

For fees with specific amounts:

```python
{
    "tag": "overdraft or NSF",
    "keywords": ["Overdraft Fee", "NSF Fee"],
    "amount_discrete": [25, 35, 50],  # Instead of range
    "amount_sign": "negative",
}
```

### Description Templates

Flexible templates for transaction descriptions:

```python
# Use keyword as-is
"description_template": "{keyword}"
# Result: "WIRE IN CREDIT - GIFT FROM RELATIVE"

# Add prefix/suffix
"description_template": "ACH DEBIT - {keyword} PMT"
# Result: "ACH DEBIT - Klarna PMT"

# Complex formatting
"description_template": "Transfer from {keyword}"
# Result: "Transfer from Alice Homeowner"
```

## Tag Mapping

The system automatically detects which mutation to use based on keywords in questions:

| Question Keywords | Mutation Type |
|------------------|---------------|
| "BNPL", "buy now pay later", "Klarna" | `bnpl` |
| "large deposits", "irregular deposits" | `large_deposits` |
| "rental payments", "rent" | `rental_payments` |
| "cryptocurrency", "crypto" | `crypto_deposits` |
| "overdraft", "NSF" | `overdraft_fees` |
| "payday loan" | `payday_loans` |
| "foreign", "international wire" | `foreign_deposits` |
| "secured loan", "401k loan" | `secured_loan_deposits` |
| "cash deposits" | `cash_deposits` |
| "unexplained deposits", "unsecured" | `unexplained_deposits` |
| "undisclosed income" | `undisclosed_income` |
| "undisclosed housing" | `undisclosed_housing_payments` |
| "withdrawal", "earnest money" | `withdrawals` |
| "retirement" | `retirement` |
| "custodial" | `custodial` |