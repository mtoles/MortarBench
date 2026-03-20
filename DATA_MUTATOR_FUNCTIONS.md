# data_mutator.py Function Reference

## Public Mutation Functions

### Bank-Statement-Only Transaction Mutations

These use `mutate_transaction(type)` and return `(bank, answer)`.

| Function call | Question | What it does |
|---|---|---|
| `mutate_transaction("bnpl")` | Scan for recurring BNPL payments (Klarna, Afterpay, etc.) | Removes existing BNPL-tagged transactions, inserts 0-4 new ones with known provider names |
| `mutate_transaction("large_deposits")` | Identify any large deposits | Removes/inserts large wire or ACH deposits ($5k-$60k) |
| `mutate_transaction("rental_payments")` | Show evidence of rental payments | Removes/inserts monthly-spaced rent payments ($1.2k-$3k) |
| `mutate_transaction("crypto_deposits")` | Deposits from a cryptocurrency source | Removes/inserts deposits from Coinbase, Gemini, etc. |
| `mutate_transaction("overdraft_fees")` | Any overdraft or NSF fees | Removes/inserts discrete fee amounts ($25/$35/$50) |
| `mutate_transaction("payday_loans")` | Deposits from payday loan or high-interest lender | Removes/inserts deposits from Speedy Cash, MoneyLion, etc. |
| `mutate_transaction("foreign_deposits")` | Deposits of foreign origin | Removes/inserts international wire transfers |
| `mutate_transaction("secured_loan_deposits")` | Deposits from a secured loan | Removes/inserts 401K loan or secured loan proceeds |
| `mutate_transaction("cash_deposits")` | Excessive cash deposits | Removes/inserts ATM/branch cash deposits ($500-$12k) |
| `mutate_transaction("unexplained_deposits")` | Unexplained deposits or unsecured borrowed funds | Removes/inserts LendingClub, SoFi, etc. proceeds |
| `mutate_transaction("undisclosed_income")` | Undisclosed income source | Removes/inserts recurring side-gig or SSA deposits (bi-weekly spacing) |
| `mutate_transaction("undisclosed_housing_payments")` | Undisclosed housing payments | Removes/inserts monthly housing payment debits |
| `mutate_transaction("withdrawals")` | Withdrawal matching earnest money deposit | Removes/inserts earnest money, wire out, or ATM withdrawals |
| `mutate_transaction("additional_account_holder")` | Additional account holder transfers | Removes/inserts transfers from a named co-holder |
| `mutate_transaction("mortgage_payments")` | Mortgage payment for current payment history | Removes/inserts monthly mortgage payment debits |
| `mutate_transaction("savings_club")` | Deposits from private savings club or informal arrangement | Removes/inserts community savings club deposits |

### Bank-Statement-Only Account Mutations

These use `mutate_account(type)` and return `(bank, answer)`.

| Function call | Question | What it does |
|---|---|---|
| `mutate_account("retirement")` | Retirement accounts as source of mortgage funds | Removes/adds a 401k investment account with contribution transaction |
| `mutate_account("custodial")` | Custodial account designated | Removes/adds a money-market custodial account (UTMA) |
| `mutate_account("business")` | Business account instead of personal | Removes/adds a business checking account with business identity |

### Bank-Only Special Mutations

Return `(bank, answer)`. Dispatched via `bank_special` type.

| Function | Question | What it does |
|---|---|---|
| `mutate_missing_transactions` | Are there any missing transactions? | Creates a balance discrepancy: sets `end_balance` so `starting_balance + sum(transactions) != end_balance`, indicating missing transactions |
| `mutate_missing_date` | Is there any missing date? | Enables monthly BankStatements then removes one middle month's statement to create a gap in date coverage between consecutive periods |

### ULAD Cross-Document Mutations

Return `(bank, ulad, answer)`. Dispatched via `ulad` type.

| Function | Question | What it does |
|---|---|---|
| `mutate_employer_payroll_consistency` | Do payroll deposits match the employer on the loan application? | Sets ULAD employer and bank payroll to same (match) or different (mismatch) company |
| `mutate_address_match` | Does the bank statement address match the loan application address? | Sets bank identity address to match or differ from ULAD residence address |
| `mutate_gift_deposit` | Is there a deposit matching the gift amount on the loan application? | Sets a gift amount in ULAD PURCHASE_CREDITS and adds a matching or non-matching deposit to the bank |
| `mutate_child_support_disclosure` | Are there recurring child support/alimony payments not disclosed on the application? | Adds recurring child support debits to bank that are NOT in ULAD liabilities |
| `mutate_undisclosed_liabilities` | Are there recurring debt payments not disclosed on the loan application? | Adds BNPL, alimony, or Venmo rent payments to bank that are absent from ULAD liabilities |
| `mutate_rental_income_consistency` | Do rental income deposits align with gross rental income on the application? | Adds REO property to ULAD with a rental income amount, then adds matching or mismatching deposits |
| `mutate_joint_account_holder` | Are there joint accounts where a holder is not a borrower on the application? | Adds a joint checking account with a non-borrower co-holder; ULAD keeps single borrower |
| `mutate_payroll_paystub_consistency` | Do payroll deposits match the net pay amounts on paystubs? | Injects payroll deposits and compares to a hypothetical paystub net-pay (match or mismatch) |
| `mutate_payroll_undisclosed_employer` | Are there payroll deposits from an employer not on the application? | Sets ULAD employer to Company A but injects bank payroll from Company B |
| `mutate_undisclosed_income_source` | Is there a consistent pattern of deposits representing undisclosed income? | Injects recurring deposits (SSA, side gig, consulting) not declared in ULAD |
| `mutate_recurring_income_match` | Do recurring deposits match claimed alimony, child support, or Social Security income? | Adds income items to ULAD and matching/mismatching recurring deposits to bank. Supports `disclosed=False` variant for undisclosed income |
| `mutate_recurring_expense_match` | Do recurring debits match claimed alimony, child support, or Social Security expenses? | Adds liabilities to ULAD and matching/mismatching recurring debits to bank. Supports `disclosed=False` variant for undisclosed expenses |
| `mutate_eligible_income` | What is the eligible income? | Generates 12 months of categorized transactions (qualifying deposits, non-qualifying deposits, obligations) and computes `eligible_income = qualifying_deposits - obligations`. Answer is a dollar amount |
| `mutate_credit_card_full_balance_payment` | Can the credit card be excluded because the borrower pays full balance monthly? | Adds credit card liability to ULAD and payment transactions to bank. High varying payments = full balance (excludable); low consistent = minimums (not excludable) |

### Two-Borrower Mutations

Return `(bank_a, bank_b, ulad, answer)`. Dispatched via `two_borrower` type.

| Function | Question | What it does |
|---|---|---|
| `mutate_large_deposit_corresponding_debit` | Is the large deposit documented by a corresponding debit from the other borrower? | Creates two borrower identities and bank statements. Places a large deposit in one account and optionally a matching debit in the other within a time window. Enables monthly BankStatements for both borrowers |

### Auto Loan Third-Party Mutations

Return `(borrower_bank, third_party_bank, ulad, answer)`. Dispatched via `auto_loan` type.

| Function | Question | What it does |
|---|---|---|
| `mutate_auto_loan_third_party_payment` | Can the auto loan be excluded because a third party paid it for 12 months? | Creates borrower + third-party bank statements. Adds auto loan to ULAD. If third party pays consistently (all 12 months on a non-joint account), the loan is excludable. Enables monthly BankStatements for the third-party bank |

---

## Helper Functions (Private)

### Transaction Helpers

| Function | Purpose |
|---|---|
| `_generate_txn_id(dataset_id)` | Generate unique transaction ID like `plaid-{id}-{counter}` |
| `_get_random_date(start_days_ago, end_days_ago)` | Random (date_transacted, date_posted) tuple within a day range |
| `_get_spaced_dates(num_dates, spacing)` | Evenly spaced dates (monthly/bi-weekly/weekly) |
| `_remove_transactions_by_tag(bank, tag)` | Remove all transactions with a given tag from all accounts |
| `_remove_transactions_by_description(bank, keywords)` | Remove transactions whose description contains any keyword |
| `_add_transaction_to_checking(bank, txn)` | Add a transaction to the primary (non-business) checking account |
| `_format_transaction_list(transactions, answer_type)` | Format transactions as boolean, id_list, or detailed text |

### ULAD Helpers

| Function | Purpose |
|---|---|
| `_get_deal(ulad)` | Navigate to DEAL node in ULAD |
| `_get_primary_borrower_party(ulad)` | Find the first PARTY with a BORROWER role |
| `_get_employer_name(ulad)` / `_set_employer_name(ulad, name)` | Read/write employer FullName |
| `_get_borrower_address(ulad)` / `_set_borrower_address(ulad, addr)` | Read/write borrower residence address |
| `_set_gift_amount(ulad, amount)` | Set PURCHASE_CREDITS gift amount |
| `_get_bank_identity_address(bank)` / `_set_bank_identity_address(bank, ...)` | Read/write bank statement identity address |
| `_set_bank_identity_name(bank, name, email)` | Set identity name and email on all accounts |
| `_add_borrower_to_ulad(ulad, first, last, email, seq)` | Add a second borrower PARTY to the ULAD |
| `_add_auto_loan_to_ulad(ulad, creditor, payment, balance)` | Shorthand for adding an Installment liability |
| `_add_liability_to_ulad(ulad, creditor, type, payment, balance)` | Add any liability type (Installment, Revolving, Other) to ULAD |
| `_add_income_item_to_ulad(ulad, income_type, amount, employment_indicator)` | Add a CURRENT_INCOME_ITEM (e.g. Alimony, SocialSecurity) to the borrower |
| `_add_credit_card_to_ulad(ulad, card_name, payment, balance)` | Add a Revolving credit card liability |

### Bank Metadata Helpers

| Function | Purpose |
|---|---|
| `_get_month_boundaries(year, month)` | Return (first_day, last_day) strings for a calendar month |
| `_months_in_range(start_date, end_date)` | List of (year, month) tuples spanning a date range |
| `_rebuild_bank_metadata(bank)` | Rebuild `Transactions`, `BankStatementAccounts`, `BankStatements`, `AggregateFigures` from `override_accounts`. Supports `_monthly_statements` flag for per-month statements |
| `_calculate_bank_statement_months(bank)` | Count months spanned by transaction dates (minimum 3) |
