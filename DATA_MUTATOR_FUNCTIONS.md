# data_mutator.py Function Reference

Each row below maps a question from `data/questions.csv` to the mutation function that synthesizes its test data.

---

## Questions and Their Mutation Functions

| old_id | Question (from CSV) | answer_type | Mutation Function | What it does |
|---|---|---|---|---|
| 1-1 | Scan the provided bank statements for any recurring payments or debit transactions made to known BNPL providers. | id_list | `mutate_transaction("bnpl")` | Removes existing BNPL-tagged transactions, inserts 0-4 new ones with provider names (Klarna, Afterpay, Affirm, etc.) |
| 1-2 | Do the payroll deposit entries match with the primary borrower's employer names stated in employment history? | boolean | `mutate_employer_payroll_consistency` | Sets ULAD employer and bank payroll deposits to same (match) or different (mismatch) company |
| 1-3 | Identify any large deposits on the borrower's bank statements. | id_list | `mutate_transaction("large_deposits")` | Removes/inserts large wire or ACH deposits ($5k-$60k) |
| 1-4 | Do the bank statements show evidence of rental payments? | id_list | `mutate_transaction("rental_payments")` | Removes/inserts monthly-spaced rent payments ($1.2k-$3k) |
| 1-6 | Scan the provided bank statements for any recurring payments that indicate a liability for child support, alimony, or wage garnishment. If found, confirm whether the payments were specifically disclosed on the loan application. | id_list | `mutate_child_support_disclosure` | Adds recurring child support/alimony debits to bank that are NOT reflected in ULAD liabilities |
| 1-7 | Verify if a deposit matching the gift amount stated on the loan application is reflected in the borrower's bank statements. | id_list | `mutate_gift_deposit` | Sets a gift amount in ULAD PURCHASE_CREDITS and adds a matching or non-matching deposit to the bank |
| 1-8 | Review the account activity for any deposits that appear to be from a secured loan. | id_list | `mutate_transaction("secured_loan_deposits")` | Removes/inserts 401K loan or secured loan proceeds ($1k-$60k) |
| 1-9 | Verify that the payroll deposits on the bank statements exactly match the net pay amounts on any of the borrower's corresponding pay stubs provided. | boolean | `mutate_payroll_paystub_consistency` | Injects payroll deposits and compares to a hypothetical paystub net-pay amount (match or mismatch) |
| 1-10 | Which deposits on the bank statement, if any, appear to originate from a cryptocurrency source? | id_list | `mutate_transaction("crypto_deposits")` | Removes/inserts deposits from Coinbase, Gemini, Kraken, etc. |
| 2-10 | Is there a consistent pattern of deposits on the asset statement that could represent an undisclosed other income source, not including cash deposits? | id_list | `mutate_undisclosed_income_source` | Injects recurring deposits (SSA, side gig, consulting) not declared in ULAD |
| 2-9 | Does the account reflect any large cash deposits or show a pattern of cash deposits that appears excessive? | id_list | `mutate_transaction("cash_deposits")` | Removes/inserts ATM/branch cash deposits ($500-$12k) |
| 2-6 | Are there any overdraft or NSF (non-sufficient funds) fees present in the account activity? | id_list | `mutate_transaction("overdraft_fees")` | Removes/inserts discrete fee amounts ($25/$35/$50) |
| 3-7 | Check the bank account activity for a withdrawal that matches the earnest money deposit (EMD) amount specified in the purchase contract. | id_list | `mutate_transaction("withdrawals")` | Removes/inserts earnest money, wire out, or ATM withdrawals |
| 3-9 | Compare the borrower's employer name to any payroll deposits on the bank statements. Note any potential undisclosed employment income. | id_list | `mutate_payroll_undisclosed_employer` | Sets ULAD employer to Company A but injects bank payroll from Company B |
| 3-8 | Scan the borrower's bank statements for any payments to creditors that are not listed on either the credit report or the loan application. | id_list | `mutate_undisclosed_liabilities` | Adds BNPL, alimony, or Venmo rent payments to bank that are absent from ULAD liabilities |
| 3-6 | Review all asset statements to determine if any account is a joint account. If a joint account is identified, confirm whether all listed account holders are also listed as borrowers on the loan application. | id_list_account | `mutate_joint_account_holder` | Adds a joint checking account with a non-borrower co-holder; ULAD keeps single borrower |
| 4-10 | Examine the account records for any deposits that could be of foreign origin. | id_list | `mutate_transaction("foreign_deposits")` | Removes/inserts international wire transfers |
| 4-9 | Review the account statements for any accounts designated as a custodial account. | id_list_account | `mutate_account("custodial")` | Removes/adds a money-market custodial account (UTMA) |
| 4-8 | Scan the provided account activity for any deposits originating from a payday loan or similar high-interest lending source. | id_list | `mutate_transaction("payday_loans")` | Removes/inserts deposits from Speedy Cash, MoneyLion, etc. |
| 4-7 | Based on the transaction history, are there any deposits that suggest funds from a private savings club or similar informal arrangement? | id_list | `mutate_transaction("savings_club")` | Removes/inserts community savings club deposits |
| 4-6 | What accounts are business accounts instead of personal accounts? | id_list_account | `mutate_account("business")` | Removes/adds a business checking account with business identity |
| 5-7 | Are there any provided retirement accounts that can be used as a source of funds for the mortgage? | id_list_account | `mutate_account("retirement")` | Removes/adds a 401k investment account with contribution transaction |
| 5-9 | Does the statement reflect the mortgage payment needed to confirm current payment history? | boolean | `mutate_transaction("mortgage_payments")` | Removes/inserts monthly mortgage payment debits |
| 5-8 | Do the rental income deposits shown on the bank statements align with the gross rental income reported for the property(ies) listed on the loan application? | boolean | `mutate_rental_income_consistency` | Adds REO property to ULAD with rental income amount, then adds matching or mismatching deposits |
| 5-5 | Verify that the borrower's address on their bank statements aligns with the current address or mailing address provided on the loan application. | boolean | `mutate_address_match` | Sets bank identity address to match or differ from ULAD residence address |
| — | Are there any missing transactions? | boolean | `mutate_missing_transactions` | Creates balance discrepancy: sets `end_balance` so `starting_balance + sum(transactions) != end_balance` |
| — | Is the large deposit documented? | boolean | `mutate_large_deposit_corresponding_debit` | Two borrower bank statements + ULAD. Places large deposit in one account, optionally adds matching debit in the other within a time window. Generates monthly BankStatements for both |
| — | Do you see the third party made the last 12 month auto payment? | boolean | `mutate_auto_loan_third_party_payment` | Borrower + third-party bank statements. Adds auto loan to ULAD. Third party pays consistently for 12 months on a non-joint account. Generates monthly BankStatements for third party |
| — | Can this account be excluded from the debt because the borrower pays the full balance each month? | boolean | `mutate_credit_card_full_balance_payment` | Adds credit card liability to ULAD. High varying payments = full balance (excludable); low consistent = minimums (not excludable) |
| — | Is there any missing date? | boolean | `mutate_missing_date` | Enables monthly BankStatements, then removes one middle month's statement to create a gap in date coverage |
| — | Do the recurring deposits match the claimed alimony, child support, or Social Security income? | boolean | `mutate_recurring_income_match` | Adds income items (Alimony/ChildSupport/SocialSecurity) to ULAD and matching or mismatching recurring deposits to bank. Supports `disclosed=False` for undisclosed variant |
| — | Do the recurring debits match the claimed alimony, child support, or Social Security expenses? | boolean | `mutate_recurring_expense_match` | Adds liabilities to ULAD and matching or mismatching recurring debits to bank. Supports `disclosed=False` for undisclosed variant |
| — | What is the eligible income? | dollar_amount | `mutate_eligible_income` | Generates 12 months of categorized transactions (qualifying deposits, non-qualifying deposits, obligations). Computes `eligible_income = sum(qualifying) - sum(obligations)`. Answer is a dollar amount |

---

## Helper Functions

### Transaction Helpers

| Function | Purpose |
|---|---|
| `_generate_txn_id(dataset_id)` | Generate unique transaction ID like `plaid-{id}-{counter}` |
| `_get_random_date(start_days_ago, end_days_ago)` | Random `(date_transacted, date_posted)` tuple within a day range |
| `_get_spaced_dates(num_dates, spacing)` | Evenly spaced dates (`monthly` / `bi-weekly` / `weekly`) |
| `_remove_transactions_by_tag(bank, tag)` | Remove all transactions with a given tag from all accounts |
| `_remove_transactions_by_description(bank, keywords)` | Remove transactions whose description contains any keyword |
| `_add_transaction_to_checking(bank, txn)` | Add a transaction to the primary non-business checking account |
| `_format_transaction_list(transactions, answer_type)` | Format transactions as `boolean`, `id_list`, or detailed text |

### ULAD Helpers

| Function | Purpose |
|---|---|
| `_get_deal(ulad)` | Navigate to the DEAL node in ULAD |
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
| `_get_month_boundaries(year, month)` | Return `(first_day, last_day)` strings for a calendar month |
| `_months_in_range(start_date, end_date)` | List of `(year, month)` tuples spanning a date range |
| `_rebuild_bank_metadata(bank)` | Rebuild `Transactions`, `BankStatementAccounts`, `BankStatements`, `AggregateFigures` from `override_accounts`. Supports `_monthly_statements` flag for per-month statements |
| `_calculate_bank_statement_months(bank)` | Count months spanned by transaction dates (minimum 3) |
