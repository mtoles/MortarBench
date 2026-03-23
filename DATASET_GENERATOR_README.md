# Dataset Generator Documentation

## Overview
The `dataset_generator.py` script is a Python utility designed to generate synthetic financial datasets in a JSON format consistent with Plaid API responses. These datasets are specifically tailored to answer a set of "id_list" type questions found in the `Labeled Questions and Answers.xlsx` file.

The generator ensures that every generated dataset contains specific transaction patterns (scenarios) that trigger positive identification for various mortgage underwriting checks, such as undisclosed debts, recurring payments, or income irregularities.

## How It Works

### 1. Initialization and Uniqueness
- **Seed & UUID**: Each run or dataset instance is initialized with a random seed (if provided) or generates a unique dataset ID using `uuid.uuid4()`. This ensures that every generated file has a unique identifier in its metadata and filenames.
- **Unique Identifiers**: 
    - **Transactions**: Uses a format `plaid-{dataset_id}-{sequence_number}` (e.g., `plaid-c0ab3026-00001`) to guarantee global uniqueness.
    - **Accounts**: Generates 8-digit account numbers and checks against a local usage set to prevent duplicates within a single dataset.

### 2. Scenario Injection
To ensure the dataset can answer all target questions, the generator explicitly injects transaction records corresponding to specific keywords. The script uses a dictionary of `KEYWORDS` mapped to categories:

- **Liabilities**: BNPL (Klarna, Afterpay), Child Support, Undisclosed Debt (Credit Cards, Auto Loans), Unsecured Loans (LendingClub).
- **Income/Assets**: Payroll, rental income, large deposits, gifts, cash deposits, Undisclosed Income (Side gigs), Undisclosed Employment (Secondary payroll).
- **Risk Indicators**: Overdraft fees, payday loans, erratic cash flow, crypto purchases.
- **Account Types**: Identifies and creates Business, Custodial, Joint, and Retirement accounts based on specific properties (e.g., ownership, official names).

### 3. Account Structure
The generator creates a user "John Homeowner" and populates a list of `override_accounts`. 
**Mandatory Accounts:**
1.  **Main Checking**: Always present. Contains the bulk of daily transactions, payroll, and injected risk scenarios.

**Optional/Probabilistic Accounts:** (To mimic real-life scenarios, these appear with varying probabilities)
2.  **Savings** (~80% chance): Contains specific withdrawals/deposits. If missing, critical transactions (like Secured Loan deposits) are moved to the Checking account to ensure coverage.
3.  **Retirement** (~60% chance): A standard 401k investment account.
4.  **Joint Account** (~40% chance): A checking account shared with "Alice Homeowner".
5.  **Custodial Account** (~40% chance): A simulated custodial account (e.g., for a minor).
6.  **Business Account** (~30% chance): An account clearly linked to a business entity ("John's Business LLC").

### 4. Logic Flow
1.  **Date Handling**: All dates are generated relative to "today", ensuring the data looks recent and consistent (e.g., `date_posted` is always 1 day after `date_transacted`).
2.  **Transaction Mixing**: While payroll and standard behavior are standardized, the "Scenario Injection" phase appends specific transactions to the main checking account.
3.  **Output**: The final structure wraps these accounts in a root object containing the `seed` and `override_accounts` list, as well as structurally restructured data including `Transactions`, `BankStatementAccounts`, `BankStatements`, and `AggregateFigures` arrays to mimic a formalized bank statement layout expected by the revised benchmark test cases.

## Usage

### Interactive Mode
Simply run the script without arguments to be prompted for input (number of datasets, output directory, months of statements to generate, and borrower's name):
```bash
python dataset_generator.py
```

### Command Line Mode
Run the script with arguments for automation:
```bash
python dataset_generator.py -n <number_of_datasets> -o <output_directory> -m <number_of_months> -u <user_name>
```

**Arguments:**
- `--ulad-template`: The path to the ULAD template file (default: `data/ulad_template.json`).
- `-n`, `--number`: The number of unique dataset files to generate (default: 1).
- `-o`, `--output`: The directory where JSON files will be saved (default: `generated_data`).
- `-m`, `--months`: The number of months of statements to generate (default: 3).
- `-u`, `--user_name`: The borrower's name to use in the generated statements (default: `John Homeowner`).

**Example:**
```bash
python dataset_generator.py -n 5 -m 6 -o ./data/test_batch_1 --ulad-template ./data/ulad_template.json
```
This will create 5 JSON files in the `./data/test_batch_1` folder, each with unique transaction and account IDs, spanning 6 months of data. It will also create 5 ULAD JSON files in the `./data/test_batch_1` folder, each with unique ULAD data.

## Question Coverage

The following table details how specific `id_list` questions from the benchmark are covered by the generator's scenario injections:

| ID List Question Category | Generator Implementation |
| :--- | :--- |
| **Recurring BNPL Payments** | Injects transactions with keywords: "Klarna", "Afterpay", "Affirm", etc. |
| **Rental Payments** | Injects transactions with keywords: "Rent Payment", "Landlord", etc. |
| **Child Support/Alimony** | Injects "Child Support", "Alimony", or "Wage Garnishment" transactions. |
| **Crypto Source Deposits** | Injects deposits from "Coinbase", "Gemini", "Binance". |
| **Overdraft/NSF Fees** | Injects "Overdraft Fee" or "NSF Fee" transactions. |
| **Undisclosed Debt Payments** | Injects payments to "Chase Credit Card", "Auto Loan", etc. |
| **Large/Irregular Deposits** | Injects "Wire Transfer - Gift" and other large lump-sum deposits (> $5,000). |
| **Cash Deposits** | Injects "ATM Cash Deposit", "Branch Deposit". |
| **Unsecured Loans** | Injects deposits from "LendingClub", "SoFi", "Upstart". |
| **Undisclosed Income** | Injects recurring small deposits from "Consulting Income", "Side Gig". |
| **Undisclosed Employment** | Injects "ACH CREDIT PAYROLL" from a SECOND employer different from the primary one. |
| **Earnest Money Withdrawal** | Injects a specific "Earnest Money Deposit" withdrawal transaction. |
| **Foreign Deposits** | Injects "International Wire", "Swift Transfer". |
| **Payday Loans** | Injects deposits/payments from "Speedy Cash", "Check 'n Go". |
| **Private Savings Club** | Injects "Savings Club", "Sou-sou" transactions. |
| **Secured Loan Deposits** | Injects "Secured Loan Deposit" or "Loan Proceeds" into a Savings account. |
| **Joint Accounts** | Generates a specific account with joint ownership ("Alice Homeowner"). |
| **Custodial Accounts** | Generates an account named "Custodial Account for Jr". |
| **Business Accounts** | Generates an account owned by "John's Business LLC". |
| **Retirement Funds** | Generates a standard 401k investment account. |


## Transaction Tags

Each transaction in the generated dataset is classified with a `tag` field to support automated analysis. The following tags are available:

- `large deposits`: Significant lump-sum deposits (e.g., Wire Transfers, Gifts).
- `rental payments`: Payments to landlords or property management.
- `BNPL transactions`: Buy Now, Pay Later services (Klarna, Affirm, etc.).
- `secured loan`: Deposits from secured loans.
- `deposit from cryptocurrency source`: Inflows from crypto exchanges.
- `overdraft or NSF`: Fees indicating insufficient funds.
- `withdrawal`: General withdrawals including crypto purchases and earnest money.
- `payday loan or high-interest lending source`: Transactions involving payday lenders.
- `custodial account`: Transfers or indicators of custodial accounts.
- `foreign origin`: International wires or transfers.
- `retirement accounts`: Indicators of 401k or IRA accounts.
- `retirement assets`: Contributions or identifying transactions for retirement assets.
- `additional account holder`: Transactions suggesting another person is using the account.
- `undisclosed housing payments`: Mortgage or housing payments not disclosed.
- `undisclosed income source`: Side gigs or consulting income.
- `unexplained deposits`: Deposits from unsecured loans or unknown sources.
- `excessive cash deposits`: Frequent or large cash deposits (relative to income).

