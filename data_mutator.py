"""
Data Mutator for Mortgage Application Dataset Generation

This module provides mutation functions to create test cases from base bank statement
and ULAD files. Each mutation function corresponds to specific transaction tags and
generates appropriate ground truth answers.

Mutation functions:
  - mutate_transaction(type)  → (bank, answer)          # bank-statement-only mutations
  - mutate_account(type)      → (bank, answer)          # account add/remove mutations
  - mutate_*()                → (bank, ulad, answer)    # ULAD cross-document mutations
"""

import json
import random
import copy
import calendar
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd


# ==================== GLOBAL BOOLEAN PROBABILITY CONTROL ====================
# BOOLEAN_PROB: Single probability value controlling Yes/No outcomes for all boolean
# questions. 0.5 = equal chance of Yes/No, 1.0 = always Yes, 0.0 = always No.
# BOOLEAN_FIXED_VALUE: Set to "Yes" or "No" to force a specific answer for all boolean
# questions, overriding BOOLEAN_PROB. Set to None to use probability-based generation.
BOOLEAN_PROB = 0.5
BOOLEAN_FIXED_VALUE = None  # Set to "Yes" or "No" to override probability


def _resolve_boolean(explicit_value=None):
    """
    Resolve a boolean decision for mutation functions.

    Priority:
      1. explicit_value (passed directly to the mutation function)
      2. BOOLEAN_FIXED_VALUE (global override)
      3. BOOLEAN_PROB (global probability)

    Returns True for "Yes" outcome, False for "No" outcome.
    """
    if explicit_value is not None:
        return explicit_value
    if BOOLEAN_FIXED_VALUE is not None:
        return BOOLEAN_FIXED_VALUE.lower().startswith("y")
    return random.random() < BOOLEAN_PROB


# ==================== TRANSACTION CONFIGS ====================

TRANSACTION_CONFIGS = {
    "bnpl": {
        "tag": "BNPL transactions",
        "keywords": ["Klarna", "Afterpay", "Affirm", "Sezzle", "Zip Co", "PayPal in 4"],
        "description_template": "ACH DEBIT - {keyword} PMT",
        "amount_range": (25, 150),
        "amount_sign": "negative",
        "default_count_range": (0, 4),
        "date_spacing": None,
    },
    "large_deposits": {
        "tag": "large deposits",
        "keywords": [
            "WIRE IN CREDIT - GIFT FROM RELATIVE",
            "WIRE TRANSFER IN",
            "ACH CREDIT - GIFT",
            "WIRE IN CREDIT - MARY HOMEOWNER GIFT",
            "ACH CREDIT - FIDELITY INVESTMENTS DEPOSIT"
        ],
        "description_template": "{keyword}",
        "amount_range": (5000, 60000),
        "amount_sign": "positive",
        "default_count_range": (0, 4),
        "date_spacing": None,
    },
    "rental_payments": {
        "tag": "rental payments",
        "keywords": ["Rent Payment", "Landlord Payment", "Property Management", "Apt Rent",
                     "ZELLE PAYMENT - LANDLORD", "ZELLE PAYMENT TO"],
        "description_template": "{keyword}",
        "amount_range": (1200, 3000),
        "amount_sign": "negative",
        "default_count_range": (0, 3),
        "date_spacing": "monthly",
    },
    "crypto_deposits": {
        "tag": "deposit from cryptocurrency source",
        "keywords": ["Coinbase", "Crypto.com", "Binance", "Gemini", "Kraken"],
        "description_template": "ACH CREDIT - {keyword}",
        "amount_range": (50, 20000),
        "amount_sign": "positive",
        "default_count_range": (0, 3),
        "date_spacing": None,
    },
    "overdraft_fees": {
        "tag": "overdraft or NSF",
        "keywords": ["Overdraft Fee", "NSF Fee", "Non-Sufficient Funds", "NSF FEE"],
        "description_template": "{keyword}",
        "amount_range": None,
        "amount_discrete": [25, 35, 50],
        "amount_sign": "negative",
        "default_count_range": (0, 2),
        "date_spacing": None,
    },
    "payday_loans": {
        "tag": "payday loan or high-interest lending source",
        "keywords": ["Speedy Cash", "Check 'n Go", "Ace Cash Express", "BEFOREPAY", "MoneyLion"],
        "description_template": "ACH CREDIT - {keyword}",
        "amount_range": (200, 2500),
        "amount_sign": "positive",
        "default_count_range": (0, 2),
        "date_spacing": None,
    },
    "foreign_deposits": {
        "tag": "foreign origin",
        "keywords": ["INTL WIRE IN CREDIT - FOREIGN FUNDS TRANSFER"],
        "description_template": "{keyword}",
        "amount_range": (500, 50000),
        "amount_sign": "positive",
        "default_count_range": (0, 2),
        "date_spacing": None,
    },
    "secured_loan_deposits": {
        "tag": "secured loan",
        "keywords": ["Loan Proceeds", "Secured Loan Deposit", "401K LOAN", "FIDELITY"],
        "description_template": "ACH CREDIT - {keyword}",
        "amount_range": (1000, 60000),
        "amount_sign": "positive",
        "default_count_range": (0, 2),
        "date_spacing": None,
    },
    "cash_deposits": {
        "tag": "excessive cash deposits",
        "keywords": ["ATM Cash Deposit", "Cash Deposit - ATM", "Branch Deposit", "ATM CASH DEPOSIT"],
        "description_template": "{keyword}",
        "amount_range": (500, 12000),
        "amount_sign": "positive",
        "default_count_range": (0, 3),
        "date_spacing": None,
    },
    "unexplained_deposits": {
        "tag": "unexplained deposits",
        "keywords": ["LendingClub", "SoFi", "Upstart", "LIGHTSTREAM", "Prosper"],
        "description_template": "{keyword} Proceeds",
        "amount_range": (2000, 50000),
        "amount_sign": "positive",
        "default_count_range": (0, 2),
        "date_spacing": None,
    },
    "undisclosed_income": {
        "tag": "undisclosed income source",
        "keywords": ["Consulting Income", "Side Gig", "Freelance Payment", "SSA US TREASURY", "Venmo Cash Out"],
        "description_template": "{keyword}",
        "amount_range": (200, 2500),
        "amount_sign": "positive",
        "default_count_range": (0, 6),
        "date_spacing": "bi-weekly",
        "recurring": True,
    },
    "undisclosed_housing_payments": {
        "tag": "undisclosed housing payments",
        "keywords": ["Housing Payment"],
        "description_template": "{keyword}",
        "amount_range": (800, 2500),
        "amount_sign": "negative",
        "default_count_range": (0, 3),
        "date_spacing": "monthly",
    },
    "withdrawals": {
        "tag": "withdrawal",
        "keywords": ["Earnest Money Deposit", "Wire Out", "Purchase at Kraken", "ATM Withdrawal"],
        "description_template": "{keyword}",
        "amount_range": (100, 5000),
        "amount_sign": "negative",
        "default_count_range": (0, 3),
        "date_spacing": None,
    },
    "additional_account_holder": {
        "tag": "additional account holder",
        "keywords": ["Alice Homeowner", "Bob Smith", "Jane Doe"],
        "description_template": "Transfer from {keyword}",
        "amount_range": (200, 1000),
        "amount_sign": "positive",
        "default_count_range": (1, 2),
        "date_spacing": None,
    },
    "mortgage_payments": {
        "tag": "mortgage payments",
        "keywords": ["CALLABLE MORTGAGE PAYMENT", "BEST EVER MORTGAGE PAYMENT", "MORTGAGE PAYMENT"],
        "description_template": "ACH DEBIT - {keyword}",
        "amount_range": (800, 3000),
        "amount_sign": "negative",
        "default_count_range": (2, 4),
        "date_spacing": "monthly",
    },
    "savings_club": {
        "tag": "private savings club",
        "keywords": ["COMMUNITY SAVINGS CLUB FUNDS", "SAVINGS CLUB DEPOSIT", "INFORMAL SAVINGS GROUP"],
        "description_template": "ACH CREDIT - {keyword}",
        "amount_range": (1000, 5000),
        "amount_sign": "positive",
        "default_count_range": (1, 3),
        "date_spacing": "monthly",
    },
    "regular_recurring_debits": {
        "tag": "regular recurring debits",
        "keywords": ["Netflix", "Spotify", "AT&T", "Comcast", "Gym Membership", "Electric Company", "Water Bill"],
        "description_template": "ACH DEBIT - {keyword}",
        "amount_range": (15, 250),
        "amount_sign": "negative",
        "default_count_range": (0, 4),
        "date_spacing": "monthly",
    },
    "regular_deposits": {
        "tag": "regular deposits",
        "keywords": ["DIRECT DEPOSIT - EMPLOYER", "PAYROLL DIRECT DEPOSIT", "ACH CREDIT - TRANSFER IN", "CHECK DEPOSIT"],
        "description_template": "{keyword}",
        "amount_range": (500, 5000),
        "amount_sign": "positive",
        "default_count_range": (0, 3),
        "date_spacing": None,
    },
}

# ==================== ACCOUNT CONFIGS ====================

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
    "custodial": {
        "tag": "custodial account",
        "type": "depository",
        "subtype": "money market",
        "balance_range": (500, 10000),
        "transaction_tag": "custodial account",
        "transaction_description": "Transfer In",
        "transaction_amount": 50,
        "official_name": "Custodial Account for Jr",
        "answer_format": "Custodial Account #{account_num}",
    },
    "business": {
        "tag": "business account",
        "type": "depository",
        "subtype": "checking",
        "class": "business",
        "balance_range": (1000, 100000),
        "transaction_tag": "business account",
        "transaction_description": "Business Transfer",
        "transaction_amount": 100,
        "official_name": "John's Business LLC",
        "answer_format": "Business account #{account_num} - {official_name}",
    },
}

# ==================== ULAD MUTATION CONFIGS ====================

ULAD_MUTATION_CONFIGS = {
    "employer_payroll_consistency": {
        "employers": [
            "NY MTA", "Amazon", "Google", "Microsoft", "City Hospital",
            "State University", "Wawa", "AAA", "Acme Corp", "Global Tech",
        ],
        "payroll_range": (2000, 5000),
        "num_payroll": 3,
    },
    "address_match": {
        "mismatch_addresses": [
            {"AddressLineText": "456 Oak Ave", "CityName": "Boston",
             "CountryCode": "US", "PostalCode": "02101", "StateCode": "MA"},
            {"AddressLineText": "9991 Warford St", "CityName": "Dawson",
             "CountryCode": "US", "PostalCode": "50066", "StateCode": "IA"},
            {"AddressLineText": "321 Elm St", "CityName": "Miami",
             "CountryCode": "US", "PostalCode": "33101", "StateCode": "FL"},
            {"AddressLineText": "4321 Cul de Sac St", "CityName": "Someplace",
             "CountryCode": "US", "PostalCode": "10001", "StateCode": "NY"},
            {"AddressLineText": "73 Dwight Drive", "CityName": "Ocean",
             "CountryCode": "US", "PostalCode": "07712", "StateCode": "NJ"},
        ],
    },
    "gift_deposit": {
        "amount_range": (5000, 60000),
        "donor_names": ["Mary Homeowner", "Dad Homeowner", "Aunt Jane", "Uncle Bob"],
    },
    "child_support_disclosure": {
        "amount_range": (300, 2000),
        "num_payments": 2,
        "keywords": ["DC OAG", "NJFSPC", "Child Support"],
        "description_template": "ACH DEBIT {keyword}",
    },
    "undisclosed_liabilities": {
        "liability_types": [
            {
                "type": "bnpl",
                "providers": ["Klarna", "Afterpay", "Affirm", "Sezzle", "Zip Co"],
                "description_template": "ACH DEBIT - {provider} PMT",
                "amount_range": (25, 150),
                "recurring_count": (2, 4),
                "date_spacing": "monthly"
            },
            {
                "type": "alimony",
                "providers": ["DC OAG", "NJFSPC", "Child Support Services", "Family Court"],
                "description_template": "ACH DEBIT {provider}",
                "amount_range": (300, 1500),
                "recurring_count": (2, 3),
                "date_spacing": "monthly"
            },
            {
                "type": "rent_venmo",
                "providers": ["LANDLORD", "PROPERTY MGT", "RENTAL PAYMENT", "HOUSING PMT"],
                "description_template": "VENMO PAYMENT TO {provider}",
                "amount_range": (800, 2500),
                "recurring_count": (2, 3),
                "date_spacing": "monthly"
            }
        ],
        "num_liability_types": (1, 2),  # How many different liability types to add
    },
    "rental_income_consistency": {
        "rental_amounts": [2000, 2400, 3000, 3500, 5000, 8000],
        "payer_names": ["Jane Doe", "John Smith", "Robert Johnson"],
    },
    "joint_account_holder": {
        "joint_names": ["Alice Homeowner", "DAD FIRSTIMER", "Jane Smith", "Bob Homeowner"],
        "balance_range": (5000, 50000),
    },
    "payroll_paystub_consistency": {
        "employer": "AAA",
        "base_payroll_range": (2000, 5000),
        "mismatch_diffs": [300, 500, 600, -200, -300],
    },
    "payroll_undisclosed_employer": {
        "ulad_employers": ["Amazon", "Google", "Microsoft", "Wawa", "City Hospital"],
        "bank_employers": ["NY MTA", "State University", "Acme Corp", "Global Tech", "AAA"],
    },
    "undisclosed_income_source": {
        "income_sources": ["SSA US TREASURY", "Consulting Income", "Side Gig", "Venmo Cash Out"],
        "amount_range": (200, 2500),
        "num_sources": 2,
        "date_spacing": "monthly",
    },
    "large_deposit_corresponding_debit": {
        "borrower_names": [
            {"first": "John", "last": "Homeowner", "email": "john.homeowner@testmail.com"},
            {"first": "Alice", "last": "Homeowner", "email": "alice.homeowner@testmail.com"}
        ],
        "deposit_amount_range": (5000, 50000),
        "time_window_days": 3,  # Debit must be within 3 days of deposit
        "description_templates": {
            "deposit": "WIRE IN CREDIT - TRANSFER FROM {borrower_name}",
            "debit": "WIRE OUT - TRANSFER TO {borrower_name}"
        }
    },
    "auto_loan_third_party_payment": {
        "borrower_names": [
            {"first": "John", "last": "Homeowner", "email": "john.homeowner@testmail.com"},
            {"first": "Sarah", "last": "Parent", "email": "sarah.parent@testmail.com"}  # Third party (parent)
        ],
        "auto_loan": {
            "creditors": ["Toyota Financial", "Honda Finance", "Ford Credit", "Chase Auto", "Capital One Auto"],
            "monthly_payment_range": (250, 800),
            "description_template": "ACH DEBIT - {creditor} AUTO LOAN",
            "liability_amount_range": (15000, 45000)
        },
        "months_required": 12,
    },
    # "credit_card_full_balance_payment": {  # Commented out per PR review - row 30 deleted
    #     "credit_cards": [
    #         {"name": "Chase Sapphire", "account_suffix": "4532"},
    #         {"name": "Capital One Venture", "account_suffix": "8901"},
    #         {"name": "American Express Gold", "account_suffix": "1234"},
    #         {"name": "Citi Double Cash", "account_suffix": "5678"},
    #         {"name": "Discover It", "account_suffix": "9012"}
    #     ],
    #     "payment_amount_range": (800, 4500),
    #     "minimum_payment_range": (25, 150),
    #     "full_balance_probability": 0.7,
    #     "liability_balance_range": (2000, 15000),
    #     "description_template": "ACH DEBIT - {card_name} PAYMENT"
    # },
    "missing_date": {},
    "recurring_income_match": {
        "income_types": [
            {
                "type": "Alimony",
                "ulad_income_type": "Alimony",
                "keywords": ["ALIMONY INCOME", "SPOUSAL SUPPORT DEPOSIT"],
                "description_template": "ACH CREDIT - {keyword}",
                "amount_range": (500, 3000),
            },
            {
                "type": "ChildSupport",
                "ulad_income_type": "ChildSupport",
                "keywords": ["CHILD SUPPORT INCOME", "STATE CHILD SUPPORT DEPOSIT"],
                "description_template": "ACH CREDIT - {keyword}",
                "amount_range": (300, 2000),
            },
            {
                "type": "SocialSecurity",
                "ulad_income_type": "SocialSecurity",
                "keywords": ["SSA US TREASURY", "SOC SEC BENEFIT"],
                "description_template": "ACH CREDIT - {keyword}",
                "amount_range": (800, 3500),
            },
        ],
        "num_income_types": (1, 2),
        "num_months": 3,
    },
    "recurring_expense_match": {
        "expense_types": [
            {
                "type": "Alimony",
                "ulad_expense_type": "Alimony",
                "keywords": ["ALIMONY PAYMENT", "SPOUSAL SUPPORT"],
                "description_template": "ACH DEBIT - {keyword}",
                "amount_range": (500, 3000),
            },
            {
                "type": "ChildSupport",
                "ulad_expense_type": "ChildSupport",
                "keywords": ["CHILD SUPPORT PMT", "DC OAG", "STATE CHILD SUPPORT"],
                "description_template": "ACH DEBIT - {keyword}",
                "amount_range": (300, 2000),
            },
            {
                "type": "SocialSecurity",
                "ulad_expense_type": "SeparateMaintenanceExpense",
                "keywords": ["SSA REPAYMENT", "SOC SEC OVERPAYMENT"],
                "description_template": "ACH DEBIT - {keyword}",
                "amount_range": (200, 1000),
            },
        ],
        "num_expense_types": (1, 2),
        "num_months": 3,
    },
    "eligible_income": {
        "personal_months": 12,
        "business_months": 24,
        "qualifying_deposits": [
            {"description_template": "ACH CREDIT PAYROLL - {employer}", "tag": "payroll", "amount_range": (3000, 8000)},
            {"description_template": "ACH CREDIT - SSA US TREASURY", "tag": "ssa_income", "amount_range": (800, 3500)},
        ],
        "non_qualifying_deposits": [
            {"description_template": "WIRE IN CREDIT - GIFT", "tag": "gift", "amount_range": (5000, 30000)},
            {"description_template": "ACH CREDIT - {crypto}", "tag": "crypto", "amount_range": (500, 10000)},
            {"description_template": "ATM Cash Deposit", "tag": "cash", "amount_range": (500, 5000)},
        ],
        "obligations": [
            {"description_template": "ACH DEBIT - {creditor} AUTO LOAN", "tag": "auto_loan", "amount_range": (200, 800)},
            {"description_template": "ACH DEBIT - MORTGAGE PAYMENT", "tag": "mortgage", "amount_range": (1000, 3000)},
            {"description_template": "ACH DEBIT - CHILD SUPPORT", "tag": "child_support", "amount_range": (300, 1500)},
        ],
        "employers": ["Acme Corp", "Global Tech", "State University"],
        "creditors": ["Toyota Financial", "Wells Fargo", "Chase Auto"],
        "crypto_sources": ["Coinbase", "Gemini", "Kraken"],
        "include_business_probability": 0.5,
    },
}


class DataMutator:
    """Handles mutation of bank statements and ULAD data for test case generation."""

    def __init__(self, bank_statement_path: str, ulad_path: str, bank_statement_2_path: str = None, ulad_2_path: str = None):
        with open(bank_statement_path, 'r') as f:
            self.base_bank_statement = json.load(f)
        with open(ulad_path, 'r') as f:
            self.base_ulad = json.load(f)
        
        # Optional second bank statement and ULAD for two-borrower scenarios
        self.base_bank_statement_2 = None
        self.base_ulad_2 = None
        if bank_statement_2_path:
            with open(bank_statement_2_path, 'r') as f:
                self.base_bank_statement_2 = json.load(f)
        if ulad_2_path:
            with open(ulad_2_path, 'r') as f:
                self.base_ulad_2 = json.load(f)
                
        self.transaction_counter = 1000

    def _generate_txn_id(self, dataset_id: str) -> str:
        self.transaction_counter += 1
        return f"plaid-{dataset_id}-{self.transaction_counter:05d}"

    def _get_random_date(self, start_days_ago: int = 90, end_days_ago: int = 0) -> Tuple[str, str]:
        days_ago = random.randint(end_days_ago, start_days_ago)
        date_transacted = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        date_posted = (datetime.now() - timedelta(days=days_ago - 1)).strftime("%Y-%m-%d")
        return date_transacted, date_posted

    def _get_spaced_dates(self, num_dates: int, spacing: str) -> List[Tuple[str, str]]:
        spacing_days = {"monthly": 30, "bi-weekly": 15, "weekly": 7}
        days_between = spacing_days.get(spacing, 30)
        dates = []
        for i in range(num_dates):
            start = 90 - (i * days_between)
            end = max(0, start - days_between + 10)
            dates.append(self._get_random_date(start, end))
        return dates

    def _detect_monthly_income(self, bank: Dict) -> float:
        """Estimate monthly income from the most-frequent recurring payroll deposit."""
        amounts = []
        for account in bank["override_accounts"]:
            if "transactions" not in account:
                continue
            for txn in account["transactions"]:
                # Match by description since the primary payroll uses the "general
                # transaction" tag, not a payroll-specific tag.
                if txn["amount"] > 0 and "PAYROLL" in txn["description"].upper():
                    amounts.append(round(float(txn["amount"]), 2))
        if not amounts:
            return 0.0
        counts = Counter(amounts)
        # Mode of payroll amounts: filters out the one-off undisclosed-employment paycheck.
        return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def _split_payroll_to_weekly(self, bank: Dict, monthly_amount: float) -> None:
        """Replace each monthly payroll deposit at monthly_amount with 4 weekly
        deposits of monthly_amount/4, so individual paychecks fall below the
        50%-of-monthly-income threshold without changing total income."""
        if monthly_amount <= 0:
            return
        weekly_amount = round(monthly_amount / 4, 2)
        dataset_id = bank["seed"].split("-")[-1]
        for account in bank["override_accounts"]:
            if "transactions" not in account:
                continue
            new_txns = []
            for txn in account["transactions"]:
                # Only split the recurring primary payroll; one-off undisclosed-employment
                # paychecks have a different amount and are left as-is.
                if ("PAYROLL" in txn["description"].upper()
                        and abs(txn["amount"] - monthly_amount) < 0.01):
                    base_date = datetime.strptime(txn["date_transacted"], "%Y-%m-%d")
                    for week in range(4):
                        new_dt = base_date + timedelta(days=7 * week)
                        new_txns.append({
                            **txn,
                            "amount": weekly_amount,
                            "date_transacted": new_dt.strftime("%Y-%m-%d"),
                            "date_posted": (new_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
                            "transaction_id": self._generate_txn_id(dataset_id),
                        })
                else:
                    new_txns.append(txn)
            account["transactions"] = new_txns

    def _clamp_large_deposits(self, bank: Dict, monthly_income: float) -> None:
        """Reduce every non-payroll deposit above 0.5 * monthly_income to a random
        value strictly less than that threshold. Identified payroll deposits are
        excluded so that the bank statement remains internally consistent (the
        Selling Guide also excludes clearly identified payroll from large-deposit
        review)."""
        if monthly_income <= 0:
            return
        threshold = 0.5 * monthly_income
        floor = min(50.0, threshold * 0.1)
        # 0.99x keeps the new amount strictly below the threshold even after rounding.
        ceiling = threshold * 0.99
        for account in bank["override_accounts"]:
            if "transactions" not in account:
                continue
            for txn in account["transactions"]:
                # Skip payroll: clamping it would shift the monthly-income baseline.
                if "PAYROLL" in txn["description"].upper():
                    continue
                if txn["amount"] > threshold:
                    txn["amount"] = round(random.uniform(floor, ceiling), 2)

    def _remove_transactions_by_tag(self, bank: Dict, tag: str) -> List[Dict]:
        removed = []
        for account in bank["override_accounts"]:
            if "transactions" not in account:
                continue
            kept, gone = [], []
            for txn in account["transactions"]:
                (gone if txn.get("tag") == tag else kept).append(txn)
            account["transactions"] = kept
            removed.extend(gone)
        return removed

    def _remove_transactions_by_description(self, bank: Dict, keywords: List[str]) -> None:
        kw_lower = [k.lower() for k in keywords]
        for account in bank["override_accounts"]:
            if "transactions" not in account:
                continue
            account["transactions"] = [
                t for t in account["transactions"]
                if not any(kw in t.get("description", "").lower() for kw in kw_lower)
            ]

    def _add_transaction_to_checking(self, bank: Dict, txn: Dict) -> None:
        dataset_id = bank["seed"].split("-")[-1]
        txn["transaction_id"] = self._generate_txn_id(dataset_id)
        for account in bank["override_accounts"]:
            if (account.get("type") == "depository"
                    and account.get("subtype") == "checking"
                    and account.get("class") != "business"):
                account["transactions"].append(txn)
                account["transactions"].sort(key=lambda x: x["date_transacted"], reverse=True)
                return

    def _format_transaction_list(self, transactions: List[Dict], answer_type: str = "id_list") -> str:
        if not transactions:
            if answer_type == "boolean":
                return "No"
            elif answer_type in ["id_list", "id_list_account"]:
                return "[]"  # Empty list for no transactions/accounts
            return "None"
        
        if answer_type == "boolean":
            return "Yes"
        elif answer_type in ["id_list", "id_list_account"]:
            # Return list of transaction IDs as strings
            return str([txn.get("transaction_id", "") for txn in transactions])
        else:
            # Default format for backward compatibility
            result = f"{len(transactions)} transaction{'s' if len(transactions) > 1 else ''}:\n"
            for i, txn in enumerate(transactions, 1):
                date = txn.get("date_transacted", "")
                desc = txn.get("description", "")
                amount = txn.get("amount", 0)
                result += f"{i}) {date} - {desc} - ${abs(amount):,.2f}\n"
            return result.strip()

    # ==================== ULAD HELPERS ====================

    def _get_deal(self, ulad: Dict) -> Dict:
        return ulad["MESSAGE"]["DEAL_SETS"]["DEAL_SET"]["DEALS"]["DEAL"]

    def _get_primary_borrower_party(self, ulad: Dict) -> Optional[Dict]:
        deal = self._get_deal(ulad)
        parties = deal["PARTIES"]["PARTY"]
        if not isinstance(parties, list):
            parties = [parties]
        for party in parties:
            role = party.get("ROLES", {}).get("ROLE", {})
            if "BORROWER" in role:
                return party
        return None

    def _get_employer_name(self, ulad: Dict) -> Optional[str]:
        party = self._get_primary_borrower_party(ulad)
        if not party:
            return None
        employers = party["ROLES"]["ROLE"]["BORROWER"].get("EMPLOYERS", "")
        if not employers or employers == "":
            return None
        employer = employers.get("EMPLOYER", {})
        if isinstance(employer, list):
            employer = employer[0]
        return employer.get("LEGAL_ENTITY", {}).get("LEGAL_ENTITY_DETAIL", {}).get("FullName")

    def _set_employer_name(self, ulad: Dict, name: str) -> None:
        party = self._get_primary_borrower_party(ulad)
        if not party:
            return
        employers = party["ROLES"]["ROLE"]["BORROWER"].get("EMPLOYERS", "")
        if employers and employers != "":
            employer = employers.get("EMPLOYER", {})
            if isinstance(employer, list):
                employer = employer[0]
            employer["LEGAL_ENTITY"]["LEGAL_ENTITY_DETAIL"]["FullName"] = name

    def _get_borrower_address(self, ulad: Dict) -> Optional[Dict]:
        party = self._get_primary_borrower_party(ulad)
        if not party:
            return None
        residences = party["ROLES"]["ROLE"]["BORROWER"].get("RESIDENCES", "")
        if not residences or residences == "":
            return None
        residence = residences.get("RESIDENCE", {})
        if isinstance(residence, list):
            residence = residence[0]
        return residence.get("ADDRESS", {})

    def _set_borrower_address(self, ulad: Dict, address: Dict) -> None:
        party = self._get_primary_borrower_party(ulad)
        if not party:
            return
        residences = party["ROLES"]["ROLE"]["BORROWER"].get("RESIDENCES", "")
        if residences and residences != "":
            residence = residences.get("RESIDENCE", {})
            if isinstance(residence, list):
                residence = residence[0]
            residence["ADDRESS"] = address

    def _set_gift_amount(self, ulad: Dict, amount: float) -> None:
        deal = self._get_deal(ulad)
        loans = deal.get("LOANS", {}).get("LOAN", {})
        loan = loans[0] if isinstance(loans, list) else loans
        loan["PURCHASE_CREDITS"] = {
            "PURCHASE_CREDIT": {
                "PurchaseCreditAmount": f"{amount:.2f}",
                "PurchaseCreditType": "GiftFunds",
            }
        }

    def _get_bank_identity_address(self, bank: Dict) -> Optional[Dict]:
        for account in bank["override_accounts"]:
            identity = account.get("identity", {})
            addresses = identity.get("addresses", [])
            if addresses:
                return addresses[0].get("data", {})
        return None

    def _set_bank_identity_address(self, bank: Dict, street: str, city: str,
                                   region: str, postal_code: str, country: str = "US") -> None:
        for account in bank["override_accounts"]:
            identity = account.get("identity", {})
            for addr_entry in identity.get("addresses", []):
                addr_entry["data"] = {
                    "street": street,
                    "city": city,
                    "region": region,
                    "postal_code": postal_code,
                    "country": country,
                }

    def _set_bank_identity_name(self, bank: Dict, name: str, email: str) -> None:
        """Set the identity name and email for all accounts in a bank statement."""
        for account in bank["override_accounts"]:
            identity = account.get("identity", {})
            identity["names"] = [name]
            if "emails" in identity and identity["emails"]:
                identity["emails"][0]["data"] = email
            else:
                identity["emails"] = [{"data": email, "primary": True, "type": "primary"}]

    def _add_borrower_to_ulad(self, ulad: Dict, first_name: str, last_name: str, email: str, sequence_num: int) -> None:
        """Add a second borrower to the ULAD PARTIES section."""
        deal = self._get_deal(ulad)
        parties = deal["PARTIES"]["PARTY"]
        
        if not isinstance(parties, list):
            parties = [parties]
            deal["PARTIES"]["PARTY"] = parties
        
        # Create new borrower party
        new_borrower = {
            "INDIVIDUAL": {
                "CONTACT_POINTS": {
                    "CONTACT_POINT": [
                        {
                            "CONTACT_POINT_TELEPHONE": {
                                "ContactPointTelephoneValue": "9999999998"
                            },
                            "CONTACT_POINT_DETAIL": {
                                "ContactPointRoleType": "Mobile"
                            }
                        },
                        {
                            "CONTACT_POINT_EMAIL": {
                                "ContactPointEmailValue": email
                            }
                        }
                    ]
                },
                "NAME": {
                    "FirstName": first_name,
                    "LastName": last_name
                }
            },
            "ROLES": {
                "ROLE": {
                    "BORROWER": {
                        "BORROWER_DETAIL": {
                            "BorrowerBirthDate": "1990-01-01",
                            "CommunityPropertyStateResidentIndicator": "false",
                            "DependentCount": "0",
                            "MaritalStatusType": "Married"
                        },
                        "CURRENT_INCOME": {
                            "CURRENT_INCOME_ITEMS": {
                                "CURRENT_INCOME_ITEM": {
                                    "CURRENT_INCOME_ITEM_DETAIL": {
                                        "CurrentIncomeMonthlyTotalAmount": "5500",
                                        "EmploymentIncomeIndicator": "true",
                                        "IncomeType": "Base"
                                    },
                                    "_SequenceNumber": "1",
                                    "_xlink:label": f"CURRENT_INCOME_ITEM_{sequence_num}_1"
                                }
                            }
                        },
                        "DECLARATION": {
                            "DECLARATION_DETAIL": {
                                "BankruptcyIndicator": "false",
                                "CitizenshipResidencyType": "USCitizen",
                                "HomeownerPastThreeYearsType": "No",
                                "IntentToOccupyType": "Yes",
                                "OutstandingJudgmentsIndicator": "false",
                                "PartyToLawsuitIndicator": "false",
                                "PresentlyDelinquentIndicator": "false",
                                "PriorPropertyDeedInLieuConveyedIndicator": "false",
                                "PriorPropertyForeclosureCompletedIndicator": "false",
                                "PriorPropertyShortSaleCompletedIndicator": "false",
                                "PropertyProposedCleanEnergyLienIndicator": "false",
                                "UndisclosedBorrowedFundsIndicator": "false",
                                "UndisclosedComakerOfNoteIndicator": "false",
                                "UndisclosedCreditApplicationIndicator": "false",
                                "UndisclosedMortgageApplicationIndicator": "false",
                                "EXTENSION": {
                                    "OTHER": {
                                        "DECLARATION_DETAIL_EXTENSION": {
                                            "SpecialBorrowerSellerRelationshipIndicator": {
                                                "__prefix": "ULAD",
                                                "__text": "false"
                                            },
                                            "__prefix": "ULAD"
                                        }
                                    }
                                }
                            }
                        },
                        "DEPENDENTS": "",
                        "EMPLOYERS": {
                            "EMPLOYER": {
                                "LEGAL_ENTITY": {
                                    "LEGAL_ENTITY_DETAIL": {
                                        "FullName": "Tech Corp"
                                    }
                                },
                                "EMPLOYMENT": {
                                    "EmploymentBorrowerSelfEmployedIndicator": "false",
                                    "EmploymentClassificationType": "Primary",
                                    "EmploymentPositionDescription": "Software Engineer",
                                    "EmploymentStartDate": "2015-01-01",
                                    "EmploymentStatusType": "Current",
                                    "EmploymentTimeInLineOfWorkMonthsCount": "120",
                                    "SpecialBorrowerEmployerRelationshipIndicator": "false",
                                    "EXTENSION": {
                                        "OTHER": {
                                            "EMPLOYMENT_EXTENSION": {
                                                "ForeignIncomeIndicator": {
                                                    "__prefix": "DU",
                                                    "__text": "false"
                                                },
                                                "SeasonalIncomeIndicator": {
                                                    "__prefix": "DU",
                                                    "__text": "false"
                                                },
                                                "__prefix": "DU"
                                            }
                                        }
                                    }
                                },
                                "_SequenceNumber": "1",
                                "_xlink:label": f"EMPLOYER_{sequence_num}_1"
                            }
                        },
                        "GOVERNMENT_BORROWER": "",
                        "GOVERNMENT_MONITORING": {
                            "GOVERNMENT_MONITORING_DETAIL": {
                                "HMDAEthnicityCollectedBasedOnVisualObservationOrSurnameIndicator": "false",
                                "HMDAEthnicityRefusalIndicator": "false",
                                "HMDAGenderCollectedBasedOnVisualObservationOrNameIndicator": "false",
                                "HMDAGenderRefusalIndicator": "false",
                                "HMDARaceCollectedBasedOnVisualObservationOrSurnameIndicator": "false",
                                "HMDARaceRefusalIndicator": "false",
                                "EXTENSION": {
                                    "OTHER": {
                                        "GOVERNMENT_MONITORING_DETAIL_EXTENSION": {
                                            "__prefix": "ULAD"
                                        }
                                    }
                                }
                            }
                        },
                        "RESIDENCES": {
                            "RESIDENCE": {
                                "ADDRESS": {
                                    "AddressLineText": "175 13th St",
                                    "CityName": "Washington",
                                    "CountryCode": "US",
                                    "PostalCode": "20013",
                                    "StateCode": "DC"
                                },
                                "LANDLORD": {
                                    "LANDLORD_DETAIL": {
                                        "MonthlyRentAmount": "3000.00"
                                    }
                                },
                                "RESIDENCE_DETAIL": {
                                    "BorrowerResidencyBasisType": "Rent",
                                    "BorrowerResidencyDurationMonthsCount": "120",
                                    "BorrowerResidencyType": "Current"
                                }
                            }
                        }
                    },
                    "ROLE_DETAIL": {
                        "PartyRoleType": "Borrower"
                    },
                    "_SequenceNumber": sequence_num,
                    "_xlink:label": f"BORROWER_{sequence_num}"
                }
            },
            "TAXPAYER_IDENTIFIERS": {
                "TAXPAYER_IDENTIFIER": {
                    "TaxpayerIdentifierType": "SocialSecurityNumber",
                    "TaxpayerIdentifierValue": "991919992"
                }
            }
        }
        
        parties.append(new_borrower)
        
        # Update loan detail to reflect multiple borrowers
        loans = deal.get("LOANS", {}).get("LOAN", {})
        loan = loans[0] if isinstance(loans, list) else loans
        loan["LOAN_DETAIL"]["BorrowerCount"] = str(len(parties) - 1)  # Subtract 1 for property owner

    def _add_auto_loan_to_ulad(self, ulad: Dict, creditor: str, monthly_payment: float, balance: float) -> None:
        """Add an auto loan liability to the ULAD LIABILITIES section."""
        self._add_liability_to_ulad(ulad, creditor, "Installment", monthly_payment, balance)

    def _add_liability_to_ulad(self, ulad: Dict, creditor: str, liability_type: str,
                               monthly_payment: float, balance: float) -> None:
        """Add a liability to the ULAD LIABILITIES section."""
        deal = self._get_deal(ulad)

        if "LIABILITIES" not in deal:
            deal["LIABILITIES"] = {"LIABILITY": []}

        liabilities = deal["LIABILITIES"].get("LIABILITY", [])
        if not isinstance(liabilities, list):
            liabilities = [liabilities] if liabilities else []

        prefix = liability_type[:4].upper()
        entry = {
            "LIABILITY_DETAIL": {
                "LiabilityAccountIdentifier": f"{prefix}{random.randint(100000, 999999)}",
                "LiabilityExclusionIndicator": "false",
                "LiabilityMonthlyPaymentAmount": f"{monthly_payment:.2f}",
                "LiabilityPayoffStatusIndicator": "false",
                "LiabilityRemainingTermMonthsCount": str(random.randint(24, 72)),
                "LiabilityType": liability_type,
                "LiabilityUnpaidBalanceAmount": f"{balance:.2f}"
            },
            "LIABILITY_HOLDER": {
                "NAME": {
                    "FullName": creditor
                }
            },
            "_SequenceNumber": str(len(liabilities) + 1),
            "_xlink:label": f"LIABILITY_{len(liabilities) + 1}"
        }

        liabilities.append(entry)
        deal["LIABILITIES"]["LIABILITY"] = liabilities

    def _add_expense_to_ulad(self, ulad: Dict, expense_type: str,
                              monthly_payment: float) -> None:
        """Add an expense to the ULAD EXPENSES section under the primary borrower.

        In the MISMO/ULAD format, recurring obligations like alimony, child support,
        and Social Security repayments belong in BORROWER > EXPENSES > EXPENSE, not
        in DEAL > LIABILITIES.

        Args:
            ulad: The ULAD data structure.
            expense_type: MISMO ExpenseType (e.g. "Alimony", "ChildSupport",
                          "SeparateMaintenanceExpense").
            monthly_payment: Monthly payment amount.
        """
        party = self._get_primary_borrower_party(ulad)
        if not party:
            return
        borrower = party["ROLES"]["ROLE"]["BORROWER"]

        expenses = borrower.get("EXPENSES")
        if not expenses or expenses == "":
            expenses = {"EXPENSE": []}
            borrower["EXPENSES"] = expenses

        expense_list = expenses.get("EXPENSE", [])
        if not isinstance(expense_list, list):
            expense_list = [expense_list] if expense_list else []

        seq = len(expense_list) + 1
        expense_list.append({
            "EXPENSE_DETAIL": {
                "ExpenseMonthlyPaymentAmount": f"{monthly_payment:.2f}",
                "ExpenseType": expense_type,
            },
            "_SequenceNumber": str(seq),
            "_xlink:label": f"EXPENSE_{seq}",
        })
        borrower["EXPENSES"]["EXPENSE"] = expense_list

    def _add_income_item_to_ulad(self, ulad: Dict, income_type: str,
                                 monthly_amount: float,
                                 employment_indicator: str = "false") -> None:
        """Add a CURRENT_INCOME_ITEM to the primary borrower in ULAD."""
        party = self._get_primary_borrower_party(ulad)
        if not party:
            return
        borrower = party["ROLES"]["ROLE"]["BORROWER"]
        income = borrower.get("CURRENT_INCOME")
        if not income or income == "":
            income = {"CURRENT_INCOME_ITEMS": {"CURRENT_INCOME_ITEM": []}}
            borrower["CURRENT_INCOME"] = income

        items = income["CURRENT_INCOME_ITEMS"].get("CURRENT_INCOME_ITEM", [])
        if not isinstance(items, list):
            items = [items] if items else []

        seq = len(items) + 1
        items.append({
            "CURRENT_INCOME_ITEM_DETAIL": {
                "CurrentIncomeMonthlyTotalAmount": f"{monthly_amount:.0f}",
                "EmploymentIncomeIndicator": employment_indicator,
                "IncomeType": income_type,
            },
            "_SequenceNumber": str(seq),
            "_xlink:label": f"CURRENT_INCOME_ITEM_1_{seq}",
        })
        income["CURRENT_INCOME_ITEMS"]["CURRENT_INCOME_ITEM"] = items

    # ==================== BANK METADATA REBUILD ====================

    @staticmethod
    def _get_month_boundaries(year: int, month: int) -> Tuple[str, str]:
        """Return (first_day, last_day) as YYYY-MM-DD strings for a calendar month."""
        first_day = f"{year:04d}-{month:02d}-01"
        last_day_num = calendar.monthrange(year, month)[1]
        last_day = f"{year:04d}-{month:02d}-{last_day_num:02d}"
        return first_day, last_day

    @staticmethod
    def _months_in_range(start_date: str, end_date: str) -> List[Tuple[int, int]]:
        """Return sorted list of (year, month) tuples covering start_date to end_date."""
        sd = datetime.strptime(start_date[:10], "%Y-%m-%d")
        ed = datetime.strptime(end_date[:10], "%Y-%m-%d")
        months = []
        cur = sd.replace(day=1)
        while cur <= ed:
            months.append((cur.year, cur.month))
            if cur.month == 12:
                cur = cur.replace(year=cur.year + 1, month=1)
            else:
                cur = cur.replace(month=cur.month + 1)
        return months

    def _rebuild_bank_metadata(self, bank: Dict) -> None:
        """Rebuild Transactions, BankStatementAccounts, BankStatements, and
        AggregateFigures from override_accounts to keep them consistent after
        mutation.  Mirrors the output format added to dataset_generator.py.

        If bank["_monthly_statements"] is truthy, generates one BankStatement
        per calendar month per account instead of a single spanning statement.
        The flag is consumed (removed) by this method."""
        monthly_mode = bank.pop("_monthly_statements", False)

        dataset_id = bank.get("seed", "unknown").split("-")[-1]
        doc_id = "doc-" + dataset_id
        vdoc_id = "vdoc-" + dataset_id
        today = datetime.now().strftime("%Y-%m-%d")

        # Determine statement date range from transaction dates
        all_dates = []
        for account in bank.get("override_accounts", []):
            for txn in account.get("transactions", []):
                d = txn.get("date_transacted")
                if d:
                    all_dates.append(d)
        stmt_start_date = min(all_dates) if all_dates else today
        stmt_end_date = max(all_dates) if all_dates else today

        transactions_out: List[Dict] = []
        bank_statement_accounts_out: List[Dict] = []
        bank_statements_out: List[Dict] = []

        for account in bank.get("override_accounts", []):
            acc_num = account.get("numbers", {}).get("account", "0000")
            acc_id = "acc-" + acc_num

            client_names = account.get("identity", {}).get("names", ["Unknown"])

            client_full_address = ""
            addrs = account.get("identity", {}).get("addresses", [])
            if addrs and isinstance(addrs, list) and len(addrs) > 0:
                addr_data = addrs[0].get("data", {})
                if isinstance(addr_data, dict):
                    client_full_address = (
                        f"{addr_data.get('street', '')} "
                        f"{addr_data.get('city', '')} "
                        f"{addr_data.get('region', '')}"
                    )

            acc_type = account.get("subtype", account.get("type", ""))
            txns = account.get("transactions", [])
            total_credit = sum(
                t.get("amount", 0) for t in txns if t.get("amount", 0) >= 0
            )
            total_debit = sum(
                t.get("amount", 0) for t in txns if t.get("amount", 0) < 0
            )

            bsa = {
                "AccountNumber": f"xxxx{acc_num[-4:]}",
                "AccountType": acc_type,
                "BankName": "Generated Bank",
                "BankFullAddress": "",
                "ClientNames": client_names,
                "ClientName": "",
                "StartBalance": account.get("starting_balance", 0.0),
                "EndBalance": account.get("end_balance", 0.0),
                "TotalCreditAmount": total_credit,
                "TotalDebitAmount": total_debit,
                "BankStatementAccountID": acc_id,
                "ID": random.randint(100, 999),
                "CreatedAt": today + "T00:00:00Z",
                "UpdatedAt": today + "T00:00:00Z",
                "LoanID": "generated-loan",
                "AccountID": acc_id,
                "ClientID": "generated-client",
            }
            bank_statement_accounts_out.append(bsa)

            if monthly_mode and txns:
                # --- Monthly BankStatements: one per calendar month ---
                acc_dates = [t.get("date_transacted", "") for t in txns if t.get("date_transacted")]
                if not acc_dates:
                    acc_dates = [stmt_start_date, stmt_end_date]
                months = self._months_in_range(min(acc_dates), max(acc_dates))

                # Group transactions by (year, month)
                txns_by_month: Dict[Tuple[int, int], List[Dict]] = {m: [] for m in months}
                for txn in txns:
                    d = txn.get("date_transacted", "")
                    if d:
                        dt = datetime.strptime(d[:10], "%Y-%m-%d")
                        key = (dt.year, dt.month)
                        if key in txns_by_month:
                            txns_by_month[key].append(txn)
                        else:
                            # Transaction outside expected range; add to closest month
                            txns_by_month.setdefault(key, []).append(txn)
                            if key not in months:
                                months.append(key)
                                months.sort()

                rolling_balance = account.get("starting_balance", 0.0)
                for yr, mo in months:
                    m_start, m_end = self._get_month_boundaries(yr, mo)
                    month_stmt_id = f"stmt-{acc_num}-{yr:04d}-{mo:02d}"
                    month_txns = txns_by_month.get((yr, mo), [])
                    month_flow = sum(t.get("amount", 0.0) for t in month_txns)
                    month_credit = sum(t.get("amount", 0) for t in month_txns if t.get("amount", 0) >= 0)
                    month_debit = sum(t.get("amount", 0) for t in month_txns if t.get("amount", 0) < 0)
                    month_end_balance = round(rolling_balance + month_flow, 2)

                    bs = {
                        "BankName": "Generated Bank",
                        "BankFullAddress": "",
                        "ClientNames": client_names,
                        "ClientFullAddress": client_full_address,
                        "ClientName": "",
                        "StatementDate": m_end,
                        "StartDate": m_start,
                        "EndDate": m_end,
                        "StartBalance": round(rolling_balance, 2),
                        "EndBalance": month_end_balance,
                        "TotalCreditAmount": month_credit,
                        "TotalDebitAmount": month_debit,
                        "BankStatementID": month_stmt_id,
                        "BorrowerID": "",
                        "BorrowerItemID": "",
                        "DocumentID": doc_id,
                        "VirtualDocumentID": vdoc_id,
                        "BankStatementAccountID": acc_id,
                        "IsPossibleDuplicate": False,
                        "ID": random.randint(1000, 9999),
                        "CreatedAt": today + "T00:00:00Z",
                        "UpdatedAt": today + "T00:00:00Z",
                        "LoanID": "generated-loan",
                        "AccountID": acc_id,
                        "ClientID": "generated-client",
                    }
                    bank_statements_out.append(bs)
                    rolling_balance = month_end_balance

                    # Build Transaction entries for this month
                    for txn in month_txns:
                        amt = txn.get("amount", 0.0)
                        txn_type = "credit" if amt >= 0 else "debit"
                        txn_obj = {
                            "Date": txn.get("date_transacted", ""),
                            "ParsedDate": txn.get("date_transacted", "") + "T00:00:00Z",
                            "Type": txn_type,
                            "Description": txn.get("description", ""),
                            "Amount": abs(amt),
                            "Category": txn.get("tag", ""),
                            "TransactionID": txn.get("transaction_id", ""),
                            "DocumentID": doc_id,
                            "VirtualDocumentID": vdoc_id,
                            "BankStatementID": month_stmt_id,
                            "BankStatementAccountID": acc_id,
                            "PageNumber": 1,
                            "BoundingBox": {"X0": 0, "X1": 0, "Y0": 0, "Y1": 0},
                            "TaggedAt": today + "T00:00:00Z",
                            "ID": random.randint(10000, 99999),
                            "CreatedAt": today + "T00:00:00Z",
                            "UpdatedAt": today + "T00:00:00Z",
                            "LoanID": "generated-loan",
                            "AccountID": acc_id,
                            "ClientID": "generated-client",
                        }
                        transactions_out.append(txn_obj)
            else:
                # --- Single-period BankStatement (default) ---
                stmt_id = "stmt-" + acc_num
                bs = {
                    "BankName": "Generated Bank",
                    "BankFullAddress": "",
                    "ClientNames": client_names,
                    "ClientFullAddress": client_full_address,
                    "ClientName": "",
                    "StatementDate": stmt_end_date,
                    "StartDate": stmt_start_date,
                    "EndDate": stmt_end_date,
                    "StartBalance": account.get("starting_balance", 0.0),
                    "EndBalance": account.get("end_balance", 0.0),
                    "TotalCreditAmount": total_credit,
                    "TotalDebitAmount": total_debit,
                    "BankStatementID": stmt_id,
                    "BorrowerID": "",
                    "BorrowerItemID": "",
                    "DocumentID": doc_id,
                    "VirtualDocumentID": vdoc_id,
                    "BankStatementAccountID": acc_id,
                    "IsPossibleDuplicate": False,
                    "ID": random.randint(1000, 9999),
                    "CreatedAt": today + "T00:00:00Z",
                    "UpdatedAt": today + "T00:00:00Z",
                    "LoanID": "generated-loan",
                    "AccountID": acc_id,
                    "ClientID": "generated-client",
                }
                bank_statements_out.append(bs)

                for txn in txns:
                    amt = txn.get("amount", 0.0)
                    txn_type = "credit" if amt >= 0 else "debit"
                    txn_obj = {
                        "Date": txn.get("date_transacted", ""),
                        "ParsedDate": txn.get("date_transacted", "") + "T00:00:00Z",
                        "Type": txn_type,
                        "Description": txn.get("description", ""),
                        "Amount": abs(amt),
                        "Category": txn.get("tag", ""),
                        "TransactionID": txn.get("transaction_id", ""),
                        "DocumentID": doc_id,
                        "VirtualDocumentID": vdoc_id,
                        "BankStatementID": stmt_id,
                        "BankStatementAccountID": acc_id,
                        "PageNumber": 1,
                        "BoundingBox": {"X0": 0, "X1": 0, "Y0": 0, "Y1": 0},
                        "TaggedAt": today + "T00:00:00Z",
                        "ID": random.randint(10000, 99999),
                        "CreatedAt": today + "T00:00:00Z",
                        "UpdatedAt": today + "T00:00:00Z",
                        "LoanID": "generated-loan",
                        "AccountID": acc_id,
                        "ClientID": "generated-client",
                    }
                    transactions_out.append(txn_obj)

        overall_credit = sum(
            t["Amount"] for t in transactions_out if t["Type"] == "credit"
        )
        overall_debit = sum(
            -t["Amount"] for t in transactions_out if t["Type"] == "debit"
        )

        bank["SearchParams"] = {
            "SortField": "Date",
            "SortOrder": "desc",
            "Size": 1000000,
        }
        bank["Transactions"] = transactions_out
        bank["BankStatementAccounts"] = bank_statement_accounts_out
        bank["BankStatements"] = bank_statements_out
        bank["AggregateFigures"] = {
            "Overall": {
                "TotalAmount": overall_credit + overall_debit,
                "CreditAmount": overall_credit,
                "DebitAmount": overall_debit,
                "Count": len(transactions_out),
            }
        }

    # ==================== BANK STATEMENT-ONLY MUTATION FUNCTIONS ====================

    def mutate_transaction(self, transaction_type: str, num_transactions: int = None, answer_type: str = "id_list") -> Tuple[Dict, str]:
        """
        Transaction mutation to remove existing tag and insert fresh transactions.

        Args:
            transaction_type: Type of transaction to mutate (from TRANSACTION_CONFIGS)
            num_transactions: Number of transactions to add (None for random)
            answer_type: Format of answer - "boolean", "id_list", or "default"

        Returns:
            (mutated_bank_statement, answer_string)
        """
        if transaction_type not in TRANSACTION_CONFIGS:
            raise ValueError(f"Unknown transaction type: {transaction_type}")

        config = TRANSACTION_CONFIGS[transaction_type]
        bank = copy.deepcopy(self.base_bank_statement)
        self._remove_transactions_by_tag(bank, config["tag"])

        # Tag-based removal alone leaves payroll/loan/crypto deposits that exceed
        # the 50%-of-monthly-income threshold the question actually tests against.
        monthly_income = 0.0
        if transaction_type == "large_deposits":
            monthly_income = self._detect_monthly_income(bank)
            self._split_payroll_to_weekly(bank, monthly_income)
            self._clamp_large_deposits(bank, monthly_income)

        if num_transactions is None:
            if _resolve_boolean():
                low, high = config["default_count_range"]
                num_transactions = random.randint(max(low, 1), high)
            else:
                num_transactions = 0

        added = []

        # Recurring: all transactions share the same keyword + base amount
        if config.get("recurring") and num_transactions > 0:
            keyword = random.choice(config["keywords"])
            base_amount = round(random.uniform(*config["amount_range"]), 2)
        else:
            keyword = None
            base_amount = None

        dates = (self._get_spaced_dates(num_transactions, config["date_spacing"])
                 if config.get("date_spacing") and num_transactions > 0
                 else [self._get_random_date() for _ in range(num_transactions)])

        for i in range(num_transactions):
            cur_keyword = keyword if keyword else random.choice(config["keywords"])

            # Scale to monthly income so injected deposits are guaranteed above the threshold.
            if transaction_type == "large_deposits" and monthly_income > 0:
                amount = round(random.uniform(0.5 * monthly_income + 1, 3 * monthly_income), 2)
            elif config.get("amount_discrete"):
                amount = float(random.choice(config["amount_discrete"]))
            elif config.get("recurring"):
                amount = base_amount + round(random.uniform(-50, 50), 2)
            else:
                amount = round(random.uniform(*config["amount_range"]), 2)

            amount = -abs(amount) if config["amount_sign"] == "negative" else abs(amount)
            description = config["description_template"].format(keyword=cur_keyword)
            date_transacted, date_posted = dates[i]

            txn = {
                "description": description,
                "amount": amount,
                "currency": "USD",
                "transaction_id": "",
                "tag": config["tag"],
                "date_transacted": date_transacted,
                "date_posted": date_posted,
            }
            self._add_transaction_to_checking(bank, txn)
            added.append(txn)

        return bank, self._format_transaction_list(added, answer_type)

    def mutate_account(self, account_type: str, has_account: bool = None, answer_type: str = "id_list") -> Tuple[Dict, str]:
        """
        Account mutation to remove existing account of that type and optionally add one.

        Args:
            account_type: Type of account to mutate (from ACCOUNT_CONFIGS)
            has_account: Whether to add account (None for random)
            answer_type: Format of answer - "boolean", "id_list", or "default"

        Returns:
            (mutated_bank_statement, answer_string)
        """
        if account_type not in ACCOUNT_CONFIGS:
            raise ValueError(f"Unknown account type: {account_type}")

        config = ACCOUNT_CONFIGS[account_type]
        bank = copy.deepcopy(self.base_bank_statement)

        # Remove existing accounts of this type
        if account_type == "retirement":
            bank["override_accounts"] = [
                a for a in bank["override_accounts"]
                if not (a.get("type") == config["type"] and a.get("subtype") == config["subtype"])
            ]
        elif account_type == "custodial":
            bank["override_accounts"] = [
                a for a in bank["override_accounts"]
                if "Custodial" not in a.get("official_name", "")
            ]
        elif account_type == "business":
            bank["override_accounts"] = [
                a for a in bank["override_accounts"]
                if a.get("class") != "business"
            ]

        if has_account is None:
            has_account = _resolve_boolean()

        if not has_account:
            if answer_type == "boolean":
                return bank, "No"
            elif answer_type in ["id_list", "id_list_account"]:
                return bank, "[]"  # Empty list for no accounts
            return bank, "None"

        dataset_id = bank["seed"].split("-")[-1]
        account_num = f"{random.randint(10000000, 99999999)}"
        balance = round(random.uniform(*config["balance_range"]), 2)

        account = {
            "type": config["type"],
            "subtype": config["subtype"],
            "starting_balance": balance,
            "currency": "USD",
            "numbers": {"account": account_num},
            "transactions": [{
                "description": config["transaction_description"],
                "amount": config["transaction_amount"],
                "currency": "USD",
                "transaction_id": self._generate_txn_id(dataset_id),
                "tag": config["transaction_tag"],
                "date_transacted": self._get_random_date()[0],
                "date_posted": self._get_random_date()[1],
            }],
            "identity": bank["override_accounts"][0]["identity"],
        }

        if config.get("official_name"):
            account["official_name"] = config["official_name"]
        if config.get("class"):
            account["class"] = config["class"]
            # Business accounts use business name in identity
            account["identity"] = {
                "names": [config["official_name"]],
                "emails": [{"data": "business@testmail.com", "primary": True, "type": "work"}],
                "addresses": bank["override_accounts"][0]["identity"]["addresses"],
            }
        if account_type == "retirement":
            account["tags"] = [config["tag"]]

        bank["override_accounts"].append(account)

        if answer_type == "boolean":
            answer = "Yes"
        elif answer_type in ["id_list", "id_list_account"]:
            # For accounts, return the account number as a list
            answer = str([account_num])
        else:
            # Default format
            answer = config["answer_format"].format(
                account_num=account_num,
                balance=balance,
                official_name=config.get("official_name", ""),
            )
        return bank, answer

    # ==================== ULAD MUTATION FUNCTIONS ====================

    def mutate_employer_payroll_consistency(self, match: bool = None, answer_type: str = "id_list") -> Tuple[Dict, Dict, str]:
        """
        Set payroll deposits in bank from a specific employer; set ULAD employer to
        match (consistent) or differ (mismatch).

        Returns:
            (mutated_bank, mutated_ulad, answer)
        """
        config = ULAD_MUTATION_CONFIGS["employer_payroll_consistency"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)

        match = _resolve_boolean(match)

        # Pick ULAD employer
        ulad_employer = random.choice(config["employers"])
        self._set_employer_name(ulad, ulad_employer)

        # Remove existing payroll transactions from bank
        self._remove_transactions_by_description(bank, ["PAYROLL", "ACH CREDIT PAYROLL"])

        # Pick bank payroll employer
        if match:
            bank_employer = ulad_employer
        else:
            others = [e for e in config["employers"] if e != ulad_employer]
            bank_employer = random.choice(others)

        # Add payroll deposits
        payroll_amount = round(random.uniform(*config["payroll_range"]), 2)
        added = []
        for i in range(config["num_payroll"]):
            date_transacted, date_posted = self._get_random_date(90 - i * 30, max(0, 60 - i * 30))
            txn = {
                "description": f"ACH CREDIT PAYROLL - {bank_employer}",
                "amount": payroll_amount,
                "currency": "USD",
                "transaction_id": "",
                "tag": "general transaction",
                "date_transacted": date_transacted,
                "date_posted": date_posted,
            }
            self._add_transaction_to_checking(bank, txn)
            added.append(txn)

        if answer_type == "boolean":
            answer = "Yes" if match else "No"
        elif answer_type == "id_list":
            if match:
                answer = str([txn.get("transaction_id", "") for txn in added])
            else:
                answer = "[]"
        else:
            if match:
                answer = (f"Yes - payroll deposits from \"{bank_employer}\" match the employer "
                          f"stated on the loan application.")
            else:
                answer = (f"Mismatch: Payroll deposits are from \"{bank_employer}\" but the loan "
                          f"application states employer is \"{ulad_employer}\".")

        return bank, ulad, answer

    def mutate_address_match(self, match: bool = None, answer_type: str = "boolean") -> Tuple[Dict, Dict, str]:
        """
        Set bank statement identity address and ULAD residence address to match
        (consistent) or differ (mismatch).

        Returns:
            (mutated_bank, mutated_ulad, answer)
        """
        config = ULAD_MUTATION_CONFIGS["address_match"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)

        match = _resolve_boolean(match)

        # Get the ULAD address
        ulad_addr = self._get_borrower_address(ulad) or {}
        ulad_addr_str = (
            f"{ulad_addr.get('AddressLineText', '')}, "
            f"{ulad_addr.get('CityName', '')}, "
            f"{ulad_addr.get('StateCode', '')} "
            f"{ulad_addr.get('PostalCode', '')}"
        ).strip(", ")

        if match:
            # Set bank identity to ULAD address
            self._set_bank_identity_address(
                bank,
                street=ulad_addr.get("AddressLineText", ""),
                city=ulad_addr.get("CityName", ""),
                region=ulad_addr.get("StateCode", ""),
                postal_code=ulad_addr.get("PostalCode", ""),
                country=ulad_addr.get("CountryCode", "US"),
            )
            if answer_type == "boolean":
                answer = "Yes"
            else:
                answer = "Yes - the bank statement address matches the current address on the loan application."
        else:
            mismatch = random.choice(config["mismatch_addresses"])
            bank_addr_str = (
                f"{mismatch['AddressLineText']}, "
                f"{mismatch['CityName']}, "
                f"{mismatch['StateCode']} "
                f"{mismatch['PostalCode']}"
            )
            self._set_bank_identity_address(
                bank,
                street=mismatch["AddressLineText"],
                city=mismatch["CityName"],
                region=mismatch["StateCode"],
                postal_code=mismatch["PostalCode"],
                country=mismatch.get("CountryCode", "US"),
            )
            if answer_type == "boolean":
                answer = "No"
            else:
                answer = (f"No - mismatch.\n"
                          f"Bank statement address: {bank_addr_str}\n"
                          f"Loan application current address: {ulad_addr_str}")

        return bank, ulad, answer

    def mutate_gift_deposit(self, match: bool = None, answer_type: str = "id_list") -> Tuple[Dict, Dict, str]:
        """
        Set a gift amount in ULAD PURCHASE_CREDITS and add a corresponding deposit
        in the bank statement that matches (or does not match) that amount.

        Returns:
            (mutated_bank, mutated_ulad, answer)
        """
        config = ULAD_MUTATION_CONFIGS["gift_deposit"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)

        match = _resolve_boolean(match)

        gift_amount = round(random.uniform(*config["amount_range"]), 2)
        donor = random.choice(config["donor_names"])
        self._set_gift_amount(ulad, gift_amount)

        # Remove existing large deposits
        self._remove_transactions_by_tag(bank, "large deposits")
        date_transacted, date_posted = self._get_random_date()

        if match:
            deposit_amount = gift_amount
            desc = f"WIRE IN CREDIT - {donor.upper()} GIFT"
            txn = {
                "description": desc,
                "amount": deposit_amount,
                "currency": "USD",
                "transaction_id": "",
                "tag": "large deposits",
                "date_transacted": date_transacted,
                "date_posted": date_posted,
            }
            self._add_transaction_to_checking(bank, txn)
            
            if answer_type == "id_list":
                answer = str([txn.get("transaction_id", "")])
            else:
                answer = (f"Matching deposit:\n"
                          f"{date_transacted} {desc} ${gift_amount:,.2f}")
        else:
            # Deposit with a different amount
            diff = random.choice([500, 1000, 2000, 5000, -500, -1000])
            deposit_amount = max(0, round(gift_amount + diff, 2))
            while deposit_amount == gift_amount:
                diff = random.choice([500, 1000, 2000, 5000, -500, -1000])
                deposit_amount = max(0, round(gift_amount + diff, 2))
            desc = "WIRE IN CREDIT"
            txn = {
                "description": desc,
                "amount": deposit_amount,
                "currency": "USD",
                "transaction_id": "",
                "tag": "large deposits",
                "date_transacted": date_transacted,
                "date_posted": date_posted,
            }
            self._add_transaction_to_checking(bank, txn)
            
            if answer_type == "id_list":
                answer = "[]"  # No matching deposit
            else:
                answer = (f"No deposit matching the gift amount of ${gift_amount:,.2f} stated on the application.\n"
                          f"Deposit found: {date_transacted} {desc} ${deposit_amount:,.2f}")

        return bank, ulad, answer

    def mutate_child_support_disclosure(self, answer_type: str = "id_list") -> Tuple[Dict, Dict, str]:
        """
        Add recurring child support / alimony payments to the bank statement that
        are NOT reflected as disclosed liabilities in the ULAD.

        Returns:
            (mutated_bank, mutated_ulad, answer)
        """
        config = ULAD_MUTATION_CONFIGS["child_support_disclosure"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)

        # Remove any pre-existing child-support-like transactions
        cs_keywords = ["child support", "alimony", "dc oag", "njfspc", "wage garnishment"]
        self._remove_transactions_by_description(bank, cs_keywords)

        added = []
        if not _resolve_boolean():
            return bank, ulad, self._format_transaction_list(added, answer_type)

        keyword = random.choice(config["keywords"])
        amount = round(random.uniform(*config["amount_range"]), 2)

        for i in range(config["num_payments"]):
            date_transacted, date_posted = self._get_random_date(90 - i * 30, max(0, 60 - i * 30))
            txn = {
                "description": config["description_template"].format(keyword=keyword),
                "amount": -amount,
                "currency": "USD",
                "transaction_id": "",
                "tag": "general transaction",
                "date_transacted": date_transacted,
                "date_posted": date_posted,
            }
            self._add_transaction_to_checking(bank, txn)
            added.append(txn)

        if answer_type == "id_list":
            answer = str([t.get("transaction_id", "") for t in added])
        else:
            lines = "\n".join(
                f"{t['date_transacted']}: ${abs(t['amount']):,.2f} - {t['description']}"
                for t in added
            )
            answer = f"Recurring payment identified - not disclosed on loan application:\n{lines}"

        return bank, ulad, answer

    def mutate_undisclosed_liabilities(self, num_liability_types: int = None, answer_type: str = "id_list") -> Tuple[Dict, Dict, str]:
        """
        Add recurring payments for undisclosed liabilities (BNPL, alimony, rent from Venmo) 
        to the bank statement that do NOT appear in ULAD liabilities.

        Strategy:
        - Bank statement: includes recurring payments for BNPL, alimony, rent from Venmo
        - ULAD: does NOT include these undisclosed liabilities
        
        Returns:
            (mutated_bank, mutated_ulad, answer)
        """
        config = ULAD_MUTATION_CONFIGS["undisclosed_liabilities"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)

        if num_liability_types is None:
            if _resolve_boolean():
                num_liability_types = random.randint(*config["num_liability_types"])
            else:
                num_liability_types = 0

        # Remove any existing transactions that might conflict with our undisclosed liabilities
        self._remove_transactions_by_description(bank, ["klarna", "afterpay", "affirm", "venmo payment", "dc oag", "njfspc"])

        # Get existing ULAD creditors to ensure we don't accidentally match them
        deal = self._get_deal(ulad)
        liabilities = deal.get("LIABILITIES", {}).get("LIABILITY", [])
        if not isinstance(liabilities, list):
            liabilities = [liabilities]
        existing_creditors = {
            lib.get("LIABILITY_HOLDER", {}).get("NAME", {}).get("FullName", "").lower()
            for lib in liabilities if isinstance(lib, dict)
        }

        # Select liability types to add (ensuring they don't overlap with ULAD)
        available_types = []
        for liability_type in config["liability_types"]:
            # Check if any providers overlap with existing ULAD creditors
            non_overlapping_providers = [
                p for p in liability_type["providers"] 
                if p.lower() not in existing_creditors
            ]
            if non_overlapping_providers:
                liability_type_copy = liability_type.copy()
                liability_type_copy["providers"] = non_overlapping_providers
                available_types.append(liability_type_copy)

        if not available_types:
            # Fallback: use all types if no overlap found
            available_types = config["liability_types"]

        selected_types = random.sample(available_types, min(num_liability_types, len(available_types)))
        added = []

        for liability_type in selected_types:
            provider = random.choice(liability_type["providers"])
            base_amount = round(random.uniform(*liability_type["amount_range"]), 2)
            num_payments = random.randint(*liability_type["recurring_count"])
            
            # Generate recurring payment dates
            dates = self._get_spaced_dates(num_payments, liability_type["date_spacing"])
            
            for i in range(num_payments):
                # Add slight variation to amount for realism
                amount = round(base_amount + random.uniform(-10, 10), 2)
                date_transacted, date_posted = dates[i]
                
                txn = {
                    "description": liability_type["description_template"].format(provider=provider),
                    "amount": -abs(amount),  # Ensure negative (payment out)
                    "currency": "USD",
                    "transaction_id": "",
                    "tag": "undisclosed liability",
                    "date_transacted": date_transacted,
                    "date_posted": date_posted,
                }
                self._add_transaction_to_checking(bank, txn)
                added.append(txn)

        # IMPORTANT: Do NOT add these liabilities to ULAD - they remain undisclosed
        # The ULAD is returned unchanged to maintain the "undisclosed" nature

        answer = self._format_transaction_list(added, answer_type)
        return bank, ulad, answer

    def mutate_rental_income_consistency(self, match: bool = None, answer_type: str = "id_list") -> Tuple[Dict, Dict, str]:
        """
        Add a REO property to ULAD with a set gross rental income amount, then add
        rental income deposits to the bank statement that match (or differ).

        Returns:
            (mutated_bank, mutated_ulad, answer)
        """
        config = ULAD_MUTATION_CONFIGS["rental_income_consistency"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)

        match = _resolve_boolean(match)

        rental_amount = random.choice(config["rental_amounts"])
        payer = random.choice(config["payer_names"])

        # Inject REO asset into ULAD if not present
        deal = self._get_deal(ulad)
        assets = deal.get("ASSETS", {}).get("ASSET", [])
        if not isinstance(assets, list):
            assets = [assets]
        has_reo = any(
            isinstance(a, dict) and a.get("ASSET_DETAIL", {}).get("AssetType") == "RealEstateOwned"
            for a in assets
        )
        if not has_reo:
            reo_asset = {
                "ASSET_DETAIL": {
                    "AssetType": "RealEstateOwned",
                    "AssetCashOrMarketValueAmount": f"{random.randint(200000, 800000)}.00",
                },
                "OWNED_PROPERTY": {
                    "ADDRESS": {
                        "AddressLineText": "100 Rental Ln",
                        "CityName": "Newark",
                        "CountryCode": "US",
                        "PostalCode": "07102",
                        "StateCode": "NJ",
                    },
                    "OWNED_PROPERTY_DETAIL": {
                        "OwnedPropertyDispositionStatusType": "Retain",
                        "OwnedPropertyRentalIncomeGrossAmount": f"{rental_amount:.2f}",
                    },
                },
                "_SequenceNumber": str(len(assets) + 1),
                "_xlink:label": f"ASSET_{len(assets) + 1}",
            }
            if isinstance(deal["ASSETS"]["ASSET"], list):
                deal["ASSETS"]["ASSET"].append(reo_asset)
            else:
                deal["ASSETS"]["ASSET"] = [deal["ASSETS"]["ASSET"], reo_asset]
        else:
            for a in assets:
                if isinstance(a, dict) and a.get("ASSET_DETAIL", {}).get("AssetType") == "RealEstateOwned":
                    a.setdefault("OWNED_PROPERTY", {}).setdefault("OWNED_PROPERTY_DETAIL", {})[
                        "OwnedPropertyRentalIncomeGrossAmount"
                    ] = f"{rental_amount:.2f}"

        # Remove any existing rental-income deposits (positive amounts tagged rental)
        self._remove_transactions_by_description(bank, ["RENTAL"])

        deposit_amount = rental_amount if match else rental_amount + random.choice([-500, 500, 1000, -1000])
        deposit_amount = max(100, deposit_amount)

        added = []
        for i in range(2):
            date_transacted, date_posted = self._get_random_date(90 - i * 30, max(0, 60 - i * 30))
            txn = {
                "description": f"RENTAL INCOME FROM {payer.upper()}",
                "amount": deposit_amount,
                "currency": "USD",
                "transaction_id": "",
                "tag": "general transaction",
                "date_transacted": date_transacted,
                "date_posted": date_posted,
            }
            self._add_transaction_to_checking(bank, txn)
            added.append(txn)

        if answer_type == "boolean":
            answer = "Yes" if match else "No"
        elif answer_type == "id_list":
            if match:
                answer = str([txn.get("transaction_id", "") for txn in added])
            else:
                answer = "[]"
        else:
            if match:
                answer = (f"Yes - rental income deposits of ${deposit_amount:,.2f}/month align with "
                          f"gross rental income of ${rental_amount:,.2f} reported on the application.")
            else:
                answer = (f"Mismatch: Bank shows rental deposits of ${deposit_amount:,.2f}/month "
                          f"but application reports gross rental income of ${rental_amount:,.2f}/month.")

        return bank, ulad, answer

    def mutate_joint_account_holder(self, joint_name: str = None, answer_type: str = "default") -> Tuple[Dict, Dict, str]:
        """
        Add a joint checking account with a non-borrower co-holder to the bank
        statement. The ULAD retains only the primary borrower.

        Returns:
            (mutated_bank, mutated_ulad, answer)
        """
        config = ULAD_MUTATION_CONFIGS["joint_account_holder"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)

        # Remove any pre-existing joint accounts
        bank["override_accounts"] = [
            a for a in bank["override_accounts"]
            if len(a.get("identity", {}).get("names", [])) <= 1
        ]

        if not _resolve_boolean():
            if answer_type == "id_list_account":
                return bank, ulad, "[]"
            return bank, ulad, "None"

        if joint_name is None:
            joint_name = random.choice(config["joint_names"])

        dataset_id = bank["seed"].split("-")[-1]
        account_num = f"{random.randint(10000000, 99999999)}"
        balance = round(random.uniform(*config["balance_range"]), 2)
        primary_name = bank["override_accounts"][0]["identity"]["names"][0]

        joint_account = {
            "type": "depository",
            "subtype": "checking",
            "starting_balance": balance,
            "currency": "USD",
            "numbers": {"account": account_num},
            "transactions": [{
                "description": "Transfer",
                "amount": 200,
                "currency": "USD",
                "transaction_id": self._generate_txn_id(dataset_id),
                "tag": "general transaction",
                "date_transacted": self._get_random_date()[0],
                "date_posted": self._get_random_date()[1],
            }],
            "identity": {
                "names": [primary_name, joint_name],
                "emails": bank["override_accounts"][0]["identity"]["emails"],
                "addresses": bank["override_accounts"][0]["identity"]["addresses"],
            },
            "end_balance": balance,
        }
        bank["override_accounts"].append(joint_account)

        if answer_type == "id_list_account":
            answer = str([account_num])
        else:
            answer = (f"Account #{account_num}:\n"
                      f"  - Account holders: {primary_name} and {joint_name}\n"
                      f"  - {joint_name} is not listed as a borrower on the loan application")

        return bank, ulad, answer

    def mutate_payroll_paystub_consistency(self, match: bool = None, answer_type: str = "boolean") -> Tuple[Dict, Dict, str]:
        """
        Inject payroll deposits into the bank statement and generate an answer
        comparing them to a hypothetical paystub net-pay amount (match or mismatch).

        Returns:
            (mutated_bank, mutated_ulad, answer)
        """
        config = ULAD_MUTATION_CONFIGS["payroll_paystub_consistency"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)

        match = _resolve_boolean(match)

        employer = config["employer"]
        bank_deposit = round(random.uniform(*config["base_payroll_range"]), 2)

        self._remove_transactions_by_description(bank, ["PAYROLL", "ACH DEPOSIT"])

        added = []
        for i in range(2):
            date_transacted, date_posted = self._get_random_date(90 - i * 30, max(0, 60 - i * 30))
            txn = {
                "description": f"ACH DEPOSIT {employer} INC PAYROLL",
                "amount": bank_deposit,
                "currency": "USD",
                "transaction_id": "",
                "tag": "general transaction",
                "date_transacted": date_transacted,
                "date_posted": date_posted,
            }
            self._add_transaction_to_checking(bank, txn)
            added.append(txn)

        if answer_type == "boolean":
            answer = "Yes" if match else "No"
        else:
            if match:
                answer = (f"Yes - payroll deposits of ${bank_deposit:,.2f} match the net pay amounts "
                          f"shown on {employer} paystubs.")
            else:
                diff = random.choice(config["mismatch_diffs"])
                paystub_amount = round(bank_deposit + diff, 2)
                answer = (f"Mismatch:\n"
                          f"Paystub from {employer} shows net pay of ${paystub_amount:,.2f} but the "
                          f"corresponding bank deposit shows ${bank_deposit:,.2f}. "
                          f"This represents a ${abs(diff):,.2f} difference.")

        return bank, ulad, answer

    def mutate_payroll_undisclosed_employer(self, answer_type: str = "id_list") -> Tuple[Dict, Dict, str]:
        """
        Set ULAD employer to Company A, but inject bank payroll from Company B,
        simulating employment not disclosed on the loan application.

        Returns:
            (mutated_bank, mutated_ulad, answer)
        """
        config = ULAD_MUTATION_CONFIGS["payroll_undisclosed_employer"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)

        self._remove_transactions_by_description(bank, ["PAYROLL", "ACH CREDIT PAYROLL"])

        if not _resolve_boolean():
            return bank, ulad, self._format_transaction_list([], answer_type)

        ulad_employer = self._get_employer_name(ulad) or random.choice(config["ulad_employers"])
        self._set_employer_name(ulad, ulad_employer)

        bank_employers = [e for e in config["bank_employers"] if e != ulad_employer]
        bank_employer = random.choice(bank_employers)

        payroll_amount = round(random.uniform(2000, 5000), 2)
        added = []
        for i in range(3):
            date_transacted, date_posted = self._get_random_date(90 - i * 30, max(0, 60 - i * 30))
            txn = {
                "description": f"ACH CREDIT PAYROLL - {bank_employer}",
                "amount": payroll_amount,
                "currency": "USD",
                "transaction_id": "",
                "tag": "undisclosed income source",
                "date_transacted": date_transacted,
                "date_posted": date_posted,
            }
            self._add_transaction_to_checking(bank, txn)
            added.append(txn)

        if answer_type == "id_list":
            answer = str([txn.get("transaction_id", "") for txn in added])
        else:
            answer = (f"Borrower employment stated as \"{ulad_employer}\" but payroll deposits "
                      f"are from \"{bank_employer} DIR DEP\".")

        return bank, ulad, answer

    def mutate_undisclosed_income_source(self, answer_type: str = "id_list") -> Tuple[Dict, Dict, str]:
        """
        Inject recurring deposits from an income source not disclosed in ULAD
        (e.g. SSA, side gig, consulting).

        Returns:
            (mutated_bank, mutated_ulad, answer)
        """
        config = ULAD_MUTATION_CONFIGS["undisclosed_income_source"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)

        self._remove_transactions_by_tag(bank, "undisclosed income source")

        if not _resolve_boolean():
            return bank, ulad, self._format_transaction_list([], answer_type)

        income_source = random.choice(config["income_sources"])
        base_amount = round(random.uniform(*config["amount_range"]), 2)
        num = config["num_sources"]
        dates = self._get_spaced_dates(num, config["date_spacing"])

        added = []
        for i in range(num):
            amount = round(base_amount + random.uniform(-30, 30), 2)
            date_transacted, date_posted = dates[i]
            txn = {
                "description": income_source,
                "amount": amount,
                "currency": "USD",
                "transaction_id": "",
                "tag": "undisclosed income source",
                "date_transacted": date_transacted,
                "date_posted": date_posted,
            }
            self._add_transaction_to_checking(bank, txn)
            added.append(txn)

        answer = self._format_transaction_list(added, answer_type)
        return bank, ulad, answer

    def mutate_missing_transactions(self, has_missing: bool = None, answer_type: str = "id_list_account") -> Tuple[Dict, str]:
        """
        Create a scenario where starting balance + transactions don't equal ending balance,
        indicating missing transactions in the bank statement.

        This function will:
        1. Calculate the actual balance based on starting balance + all transactions
        2. If has_missing=True, set ending balance to a different value to create a discrepancy
        3. If has_missing=False, set ending balance to match (no missing transactions)

        Args:
            has_missing: If True, create discrepancy. If False, balances match.
                        Controlled by BOOLEAN_PROB / BOOLEAN_FIXED_VALUE when None.
            answer_type: Format of answer - "boolean", "id_list", or "default"

        Returns:
            (mutated_bank_statement, answer_string)
        """
        bank = copy.deepcopy(self.base_bank_statement)

        has_missing = _resolve_boolean(has_missing)

        # Find the primary checking account to modify
        primary_account = None
        for account in bank["override_accounts"]:
            if (account.get("type") == "depository"
                and account.get("subtype") == "checking"
                and account.get("class") != "business"):
                primary_account = account
                break

        if not primary_account:
            # Fallback to first account if no checking account found
            primary_account = bank["override_accounts"][0]

        # Calculate what the ending balance should be based on transactions
        starting_balance = primary_account.get("starting_balance", 0)
        transaction_sum = sum(txn.get("amount", 0) for txn in primary_account.get("transactions", []))
        calculated_ending_balance = starting_balance + transaction_sum

        if has_missing:
            # Create a discrepancy by setting ending balance to a different value
            discrepancy_amounts = [500, 750, 1000, 1250, 1500, 2000, -300, -500, -750, -1000]
            discrepancy = random.choice(discrepancy_amounts)
            primary_account["end_balance"] = calculated_ending_balance + discrepancy
        else:
            # No missing transactions - ending balance matches
            discrepancy = 0
            primary_account["end_balance"] = calculated_ending_balance

        if answer_type == "boolean":
            answer = "Yes" if has_missing else "No"
        elif answer_type in ["id_list_account", "id_list"]:
            if has_missing:
                acc_num = primary_account.get("numbers", {}).get("account", "0000")
                answer = str([acc_num])
            else:
                answer = "[]"
        else:
            if has_missing:
                answer = (f"Yes - Balance discrepancy detected.\n"
                         f"Starting balance: ${starting_balance:,.2f}\n"
                         f"Sum of transactions: ${transaction_sum:,.2f}\n"
                         f"Expected ending balance: ${calculated_ending_balance:,.2f}\n"
                         f"Actual ending balance: ${primary_account['end_balance']:,.2f}\n"
                         f"Discrepancy: ${discrepancy:,.2f} (indicates missing transactions)")
            else:
                answer = (f"No - All transactions accounted for.\n"
                         f"Starting balance + transactions = ending balance")

        return bank, answer

    def mutate_missing_date(self, gap: bool = None, answer_type: str = "boolean") -> Tuple[Dict, str]:
        """
        Create a scenario where monthly bank statements either have a contiguous
        date range (no gap) or a missing month (gap).

        Sets _monthly_statements flag so _rebuild_bank_metadata produces per-month
        BankStatements.  When gap=True, sets _post_rebuild_actions to remove one
        middle month after rebuild.

        Returns:
            (mutated_bank_statement, answer_string)
        """
        bank = copy.deepcopy(self.base_bank_statement)
        bank["_monthly_statements"] = True

        gap = _resolve_boolean(gap)

        if gap:
            # Determine which months are covered and pick a middle one to remove
            all_dates = []
            for account in bank.get("override_accounts", []):
                for txn in account.get("transactions", []):
                    d = txn.get("date_transacted")
                    if d:
                        all_dates.append(d)
            if all_dates:
                months = self._months_in_range(min(all_dates), max(all_dates))
                if len(months) >= 3:
                    # Pick a middle month (not first or last)
                    middle = months[len(months) // 2]
                    month_str = f"{middle[0]:04d}-{middle[1]:02d}"
                    bank["_post_rebuild_actions"] = {"remove_statement_month": month_str}

        if answer_type == "boolean":
            answer = "Yes" if gap else "No"
        else:
            answer = "Yes - gap detected in bank statement date coverage" if gap else "No - statement dates are contiguous"

        return bank, answer

    def mutate_recurring_income_match(self, match: bool = None, disclosed: bool = True,
                                     answer_type: str = "id_list") -> Tuple[Dict, Dict, str]:
        """
        Add recurring income deposits (alimony, child support, SSA) to the bank
        statement and optionally declare corresponding income on the ULAD.

        Args:
            match: If True, bank deposits match ULAD declared amount. If False, mismatch.
            disclosed: If True, income is declared on ULAD. If False, deposits exist but
                       no corresponding ULAD income item (undisclosed income).
            answer_type: "boolean" or "id_list"

        Returns:
            (mutated_bank, mutated_ulad, answer)
        """
        config = ULAD_MUTATION_CONFIGS["recurring_income_match"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)

        match = _resolve_boolean(match)

        num_types = random.randint(*config["num_income_types"])
        selected = random.sample(config["income_types"], min(num_types, len(config["income_types"])))

        # Remove existing income-related deposits
        income_kws = []
        for it in config["income_types"]:
            income_kws.extend(it["keywords"])
        self._remove_transactions_by_description(bank, income_kws)

        added = []
        for income_type in selected:
            declared_amount = round(random.uniform(*income_type["amount_range"]), 2)
            keyword = random.choice(income_type["keywords"])

            if disclosed:
                self._add_income_item_to_ulad(ulad, income_type["ulad_income_type"], declared_amount)

            if not disclosed or match:
                deposit_amount = declared_amount
            else:
                # Mismatch: different amount
                diff = random.choice([200, 500, -200, -300, 800])
                deposit_amount = max(50, round(declared_amount + diff, 2))

            # Add recurring monthly deposits
            dates = self._get_spaced_dates(config["num_months"], "monthly")
            for i in range(config["num_months"]):
                amt = round(deposit_amount + random.uniform(-20, 20), 2)
                date_transacted, date_posted = dates[i]
                txn = {
                    "description": income_type["description_template"].format(keyword=keyword),
                    "amount": abs(amt),
                    "currency": "USD",
                    "transaction_id": "",
                    "tag": "recurring income",
                    "date_transacted": date_transacted,
                    "date_posted": date_posted,
                }
                self._add_transaction_to_checking(bank, txn)
                added.append(txn)

        if answer_type == "boolean":
            if not disclosed:
                answer = "No"  # undisclosed means no match by definition
            else:
                answer = "Yes" if match else "No"
        elif answer_type == "id_list":
            if match and disclosed:
                answer = self._format_transaction_list(added, answer_type)
            else:
                answer = "[]"
        else:
            answer = self._format_transaction_list(added, answer_type)

        return bank, ulad, answer

    def mutate_recurring_expense_match(self, match: bool = None, disclosed: bool = True,
                                      answer_type: str = "id_list") -> Tuple[Dict, Dict, str]:
        """
        Add recurring expense debits (alimony, child support, SSA repayment) to the
        bank statement and optionally declare corresponding liabilities on the ULAD.

        Args:
            match: If True, bank debits match ULAD declared liability amount.
            disclosed: If True, liability is declared on ULAD. If False, debits exist
                       but no corresponding ULAD liability (undisclosed expense).
            answer_type: "boolean" or "id_list"

        Returns:
            (mutated_bank, mutated_ulad, answer)
        """
        config = ULAD_MUTATION_CONFIGS["recurring_expense_match"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)

        match = _resolve_boolean(match)

        num_types = random.randint(*config["num_expense_types"])
        selected = random.sample(config["expense_types"], min(num_types, len(config["expense_types"])))

        # Remove existing expense-related debits
        expense_kws = []
        for et in config["expense_types"]:
            expense_kws.extend(et["keywords"])
        self._remove_transactions_by_description(bank, expense_kws)

        added = []
        for expense_type in selected:
            declared_amount = round(random.uniform(*expense_type["amount_range"]), 2)
            keyword = random.choice(expense_type["keywords"])

            if disclosed:
                self._add_expense_to_ulad(
                    ulad,
                    expense_type["ulad_expense_type"],
                    declared_amount,
                )

            if not disclosed or match:
                debit_amount = declared_amount
            else:
                diff = random.choice([200, 500, -200, -300, 800])
                debit_amount = max(50, round(declared_amount + diff, 2))

            dates = self._get_spaced_dates(config["num_months"], "monthly")
            for i in range(config["num_months"]):
                amt = round(debit_amount + random.uniform(-20, 20), 2)
                date_transacted, date_posted = dates[i]
                txn = {
                    "description": expense_type["description_template"].format(keyword=keyword),
                    "amount": -abs(amt),
                    "currency": "USD",
                    "transaction_id": "",
                    "tag": "recurring expense",
                    "date_transacted": date_transacted,
                    "date_posted": date_posted,
                }
                self._add_transaction_to_checking(bank, txn)
                added.append(txn)

        if answer_type == "boolean":
            if not disclosed:
                answer = "No"
            else:
                answer = "Yes" if match else "No"
        elif answer_type == "id_list":
            if match and disclosed:
                answer = self._format_transaction_list(added, answer_type)
            else:
                answer = "[]"
        else:
            answer = self._format_transaction_list(added, answer_type)

        return bank, ulad, answer

    def mutate_eligible_income(self, include_business: bool = None,
                               answer_type: str = "id_list") -> Tuple[Dict, Dict, str]:
        """
        Generate 12 months of personal bank statement data (and optionally 24 months
        of business data) with qualifying deposits and obligations, then compute
        eligible income = qualifying_deposits - obligations.

        Returns:
            (mutated_bank, mutated_ulad, answer)  where answer is a dollar amount string.
        """
        config = ULAD_MUTATION_CONFIGS["eligible_income"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)
        bank["_monthly_statements"] = True

        if include_business is None:
            include_business = _resolve_boolean()

        # Remove existing payroll / income transactions
        self._remove_transactions_by_description(bank, ["PAYROLL", "ACH CREDIT PAYROLL", "SSA US TREASURY"])

        total_qualifying = 0.0
        total_obligations = 0.0
        non_qualifying_added = []
        employer = random.choice(config["employers"])
        creditor = random.choice(config["creditors"])
        crypto = random.choice(config["crypto_sources"])

        # Generate 12 months of personal transactions
        for month_offset in range(config["personal_months"]):
            base_date = datetime.now() - timedelta(days=30 * month_offset + random.randint(5, 25))
            date_str = base_date.strftime("%Y-%m-%d")
            date_posted = (base_date + timedelta(days=1)).strftime("%Y-%m-%d")

            # Qualifying deposits
            for dep_cfg in config["qualifying_deposits"]:
                amt = round(random.uniform(*dep_cfg["amount_range"]), 2)
                desc = dep_cfg["description_template"].format(employer=employer)
                txn = {
                    "description": desc,
                    "amount": amt,
                    "currency": "USD",
                    "transaction_id": "",
                    "tag": dep_cfg["tag"],
                    "date_transacted": date_str,
                    "date_posted": date_posted,
                }
                self._add_transaction_to_checking(bank, txn)
                total_qualifying += amt

            # Obligations
            for obl_cfg in config["obligations"]:
                amt = round(random.uniform(*obl_cfg["amount_range"]), 2)
                desc = obl_cfg["description_template"].format(creditor=creditor)
                txn = {
                    "description": desc,
                    "amount": -amt,
                    "currency": "USD",
                    "transaction_id": "",
                    "tag": obl_cfg["tag"],
                    "date_transacted": date_str,
                    "date_posted": date_posted,
                }
                self._add_transaction_to_checking(bank, txn)
                total_obligations += amt

            # Scatter some non-qualifying deposits (every few months)
            if month_offset % 3 == 0:
                nq = random.choice(config["non_qualifying_deposits"])
                amt = round(random.uniform(*nq["amount_range"]), 2)
                desc = nq["description_template"].format(crypto=crypto)
                txn = {
                    "description": desc,
                    "amount": amt,
                    "currency": "USD",
                    "transaction_id": "",
                    "tag": nq["tag"],
                    "date_transacted": date_str,
                    "date_posted": date_posted,
                }
                self._add_transaction_to_checking(bank, txn)
                non_qualifying_added.append(txn)
                # Non-qualifying — NOT added to total_qualifying

        eligible = round(total_qualifying - total_obligations, 2)

        if answer_type == "id_list":
            answer = str([txn.get("transaction_id", "") for txn in non_qualifying_added])
        elif answer_type == "dollar_amount":
            answer = f"${eligible:,.2f}"
        elif answer_type == "boolean":
            answer = "Yes" if eligible > 0 else "No"
        else:
            answer = f"${eligible:,.2f}"

        return bank, ulad, answer

    def mutate_large_deposit_corresponding_debit(self, match: bool = None, answer_type: str = "id_list") -> Tuple[Dict, Dict, Dict, str]:
        """
        Create a scenario with two borrowers where a large deposit in one borrower's account
        has (or doesn't have) a corresponding debit in the other borrower's account within
        the correct time window.
        
        Strategy:
        - Two bank statements: BorrowerA and BorrowerB with different names
        - ULAD: Contains both borrowers in PARTIES section
        - Large deposit in BorrowerA's account
        - Corresponding debit in BorrowerB's account (if match=True) within time window
        
        Returns:
            (mutated_bank_A, mutated_bank_B, mutated_ulad, answer)
        """
        if self.base_bank_statement_2 is None:
            raise ValueError("Second bank statement template required for two-borrower mutation")
            
        config = ULAD_MUTATION_CONFIGS["large_deposit_corresponding_debit"]
        
        # Create copies of both bank statements and ULAD
        bank_a = copy.deepcopy(self.base_bank_statement)
        bank_b = copy.deepcopy(self.base_bank_statement_2)
        ulad = copy.deepcopy(self.base_ulad)
        bank_a["_monthly_statements"] = True
        bank_b["_monthly_statements"] = True

        match = _resolve_boolean(match)

        # Set up borrower identities
        borrower_a = config["borrower_names"][0]  # John Homeowner
        borrower_b = config["borrower_names"][1]  # Alice Homeowner
        
        # Update bank statement identities
        borrower_a_name = f"{borrower_a['first']} {borrower_a['last']}"
        borrower_b_name = f"{borrower_b['first']} {borrower_b['last']}"
        
        self._set_bank_identity_name(bank_a, borrower_a_name, borrower_a["email"])
        self._set_bank_identity_name(bank_b, borrower_b_name, borrower_b["email"])
        
        # Add both borrowers to ULAD
        # First borrower is already in the base ULAD, update their name
        party = self._get_primary_borrower_party(ulad)
        if party:
            party["INDIVIDUAL"]["NAME"]["FirstName"] = borrower_a["first"]
            party["INDIVIDUAL"]["NAME"]["LastName"] = borrower_a["last"]
            # Update email in contact points
            contact_points = party["INDIVIDUAL"]["CONTACT_POINTS"]["CONTACT_POINT"]
            for cp in contact_points:
                if "CONTACT_POINT_EMAIL" in cp:
                    cp["CONTACT_POINT_EMAIL"]["ContactPointEmailValue"] = borrower_a["email"]
        
        # Add second borrower to ULAD
        self._add_borrower_to_ulad(ulad, borrower_b["first"], borrower_b["last"], borrower_b["email"], 2)
        
        # Remove existing large deposits from both accounts
        self._remove_transactions_by_tag(bank_a, "large deposits")
        self._remove_transactions_by_tag(bank_b, "large deposits")
        
        # Generate deposit amount and dates
        deposit_amount = round(random.uniform(*config["deposit_amount_range"]), 2)
        deposit_date = self._get_random_date(30, 5)  # Deposit 5-30 days ago
        
        # Create large deposit in BorrowerA's account
        deposit_txn = {
            "description": config["description_templates"]["deposit"].format(borrower_name=borrower_b_name),
            "amount": deposit_amount,
            "currency": "USD",
            "transaction_id": "",
            "tag": "large deposits",
            "date_transacted": deposit_date[0],
            "date_posted": deposit_date[1],
        }
        self._add_transaction_to_checking(bank_a, deposit_txn)
        
        debit_txn = None
        if match:
            # Create corresponding debit in BorrowerB's account within time window
            # Debit should be 1-3 days before the deposit
            days_before = random.randint(1, config["time_window_days"])
            debit_date_obj = datetime.strptime(deposit_date[0], "%Y-%m-%d") - timedelta(days=days_before)
            debit_date = (debit_date_obj.strftime("%Y-%m-%d"), 
                         (debit_date_obj + timedelta(days=1)).strftime("%Y-%m-%d"))
            
            debit_txn = {
                "description": config["description_templates"]["debit"].format(borrower_name=borrower_a_name),
                "amount": -deposit_amount,  # Same amount, negative
                "currency": "USD",
                "transaction_id": "",
                "tag": "withdrawal",
                "date_transacted": debit_date[0],
                "date_posted": debit_date[1],
            }
            self._add_transaction_to_checking(bank_b, debit_txn)
        
        # Format answer
        if answer_type == "boolean":
            answer = "Yes" if match else "No"
        elif answer_type == "id_list":
            if match and debit_txn:
                # Return the deposit transaction ID — the question asks which deposits are documented
                answer = str([deposit_txn.get("transaction_id", "")])
            else:
                answer = "[]"
        else:
            # Default detailed format
            if match and debit_txn:
                answer = (f"Yes - Corresponding debit found.\n"
                         f"Deposit: {deposit_txn['date_transacted']} - {borrower_a_name} received ${deposit_amount:,.2f} from {borrower_b_name}\n"
                         f"Debit: {debit_txn['date_transacted']} - {borrower_b_name} sent ${deposit_amount:,.2f} to {borrower_a_name}\n"
                         f"Time difference: {days_before} day(s) (within {config['time_window_days']}-day window)")
            else:
                answer = (f"No - No corresponding debit found.\n"
                         f"Deposit: {deposit_txn['date_transacted']} - {borrower_a_name} received ${deposit_amount:,.2f}\n"
                         f"No matching debit transaction found in {borrower_b_name}'s account within the {config['time_window_days']}-day time window.")
        
        return bank_a, bank_b, ulad, answer

    def mutate_undocumented_large_deposit(self, answer_type: str = "id_list") -> Tuple[Dict, Dict, Dict, str]:
        """
        Inverted variant of mutate_large_deposit_corresponding_debit.

        Positive case (is_undocumented=True): the large deposit has NO corresponding debit
            → the deposit is undocumented → answer = [deposit_id]
        Negative case (is_undocumented=False): the large deposit HAS a corresponding debit
            → the deposit is documented → answer = []
        """
        is_undocumented = _resolve_boolean()
        # Pass the inverted boolean so the base function creates the scenario we need
        bank_a, bank_b, ulad, _ = self.mutate_large_deposit_corresponding_debit(
            match=not is_undocumented,
            answer_type=answer_type,
        )

        if answer_type == "id_list":
            if is_undocumented:
                deposit_id = ""
                for account in bank_a.get("override_accounts", []):
                    for txn in account.get("transactions", []):
                        if txn.get("tag") == "large deposits":
                            deposit_id = txn.get("transaction_id", "")
                            break
                    if deposit_id:
                        break
                answer = str([deposit_id]) if deposit_id else "[]"
            else:
                answer = "[]"
        elif answer_type == "boolean":
            answer = "Yes" if is_undocumented else "No"
        else:
            answer = str(is_undocumented)

        return bank_a, bank_b, ulad, answer

    def mutate_auto_loan_third_party_payment(self, third_party_pays: bool = None, answer_type: str = "boolean") -> Tuple[Dict, Dict, Dict, str]:
        """
        Create a scenario where an auto loan liability exists in ULAD and check if a third party
        has been consistently paying it for the last 12 months.
        
        Strategy:
        - Borrower's bank statement: May or may not have auto loan payments (assumes 12+ months of data)
        - Third party's bank statement: Has consistent auto loan payments (if third_party_pays=True)
        - ULAD: Contains auto loan liability
        - Verify third party account is not joint with borrower
        
        Returns:
            (mutated_borrower_bank, mutated_third_party_bank, mutated_ulad, answer)
        """
        if self.base_bank_statement_2 is None:
            raise ValueError("Second bank statement template required for auto loan mutation (third party)")
            
        config = ULAD_MUTATION_CONFIGS["auto_loan_third_party_payment"]
        
        # Create copies
        borrower_bank = copy.deepcopy(self.base_bank_statement)
        third_party_bank = copy.deepcopy(self.base_bank_statement_2)
        ulad = copy.deepcopy(self.base_ulad)
        third_party_bank["_monthly_statements"] = True
        
        third_party_pays = _resolve_boolean(third_party_pays)

        # Set up identities
        borrower = config["borrower_names"][0]  # borrower
        third_party = config["borrower_names"][1]  # third party
        
        borrower_name = f"{borrower['first']} {borrower['last']}"
        third_party_name = f"{third_party['first']} {third_party['last']}"
        
        # Update bank statement identities
        self._set_bank_identity_name(borrower_bank, borrower_name, borrower["email"])
        self._set_bank_identity_name(third_party_bank, third_party_name, third_party["email"])
        
        # Ensure third party account is NOT joint with borrower
        for account in third_party_bank["override_accounts"]:
            identity = account.get("identity", {})
            names = identity.get("names", [])
            if len(names) > 1:  # Remove any joint names
                identity["names"] = [third_party_name]
        
        # Set up auto loan details
        creditor = random.choice(config["auto_loan"]["creditors"])
        monthly_payment = round(random.uniform(*config["auto_loan"]["monthly_payment_range"]), 2)
        loan_balance = round(random.uniform(*config["auto_loan"]["liability_amount_range"]), 2)
        
        # Add auto loan to ULAD
        self._add_auto_loan_to_ulad(ulad, creditor, monthly_payment, loan_balance)
        
        # Remove existing auto loan payments from both accounts
        auto_keywords = ["auto", "car", "vehicle", creditor.lower()]
        self._remove_transactions_by_description(borrower_bank, auto_keywords)
        self._remove_transactions_by_description(third_party_bank, auto_keywords)
        
        # Generate 12 months of auto loan payment history
        months_with_payments = 0
        payment_months = []
        
        if third_party_pays:
            # Third party pays consistently (all 12 months)
            payment_months = list(range(config["months_required"]))  # All 12 months
            months_with_payments = len(payment_months)
            
            for month_offset in payment_months:
                # Calculate payment date (going back from current date)
                payment_date_obj = datetime.now() - timedelta(days=30 * month_offset + random.randint(5, 25))
                payment_date = (payment_date_obj.strftime("%Y-%m-%d"), 
                               (payment_date_obj + timedelta(days=1)).strftime("%Y-%m-%d"))
                
                # Add slight variation to payment amount
                payment_amount = round(monthly_payment + random.uniform(-10, 10), 2)
                
                txn = {
                    "description": config["auto_loan"]["description_template"].format(creditor=creditor),
                    "amount": -abs(payment_amount),
                    "currency": "USD",
                    "transaction_id": "",
                    "tag": "auto loan payment",
                    "date_transacted": payment_date[0],
                    "date_posted": payment_date[1],
                }
                self._add_transaction_to_checking(third_party_bank, txn)
        else:
            # Third party doesn't pay consistently (or borrower pays themselves)
            # Add sporadic payments from borrower or no payments at all
            sporadic_months = random.randint(0, 3)  # 0-3 months only
            payment_months = random.sample(range(config["months_required"]), sporadic_months)
            months_with_payments = len(payment_months)
            
            for month_offset in payment_months:
                payment_date_obj = datetime.now() - timedelta(days=30 * month_offset + random.randint(5, 25))
                payment_date = (payment_date_obj.strftime("%Y-%m-%d"), 
                               (payment_date_obj + timedelta(days=1)).strftime("%Y-%m-%d"))
                
                payment_amount = round(monthly_payment + random.uniform(-10, 10), 2)
                
                # Randomly choose who makes the sporadic payment
                if random.random() < 0.5:
                    # Borrower makes payment
                    txn = {
                        "description": config["auto_loan"]["description_template"].format(creditor=creditor),
                        "amount": -abs(payment_amount),
                        "currency": "USD",
                        "transaction_id": "",
                        "tag": "auto loan payment",
                        "date_transacted": payment_date[0],
                        "date_posted": payment_date[1],
                    }
                    self._add_transaction_to_checking(borrower_bank, txn)
                else:
                    # Third party makes sporadic payment
                    txn = {
                        "description": config["auto_loan"]["description_template"].format(creditor=creditor),
                        "amount": -abs(payment_amount),
                        "currency": "USD",
                        "transaction_id": "",
                        "tag": "auto loan payment",
                        "date_transacted": payment_date[0],
                        "date_posted": payment_date[1],
                    }
                    self._add_transaction_to_checking(third_party_bank, txn)
        
        # Determine if loan can be excluded (third party paid for exactly 12 months or above)
        can_exclude = (third_party_pays and 
                      months_with_payments >= config["months_required"])
        
        # Format answer
        if answer_type == "boolean":
            answer = "Yes" if can_exclude else "No"
        elif answer_type == "id_list":
            # Return transaction IDs of third party payments
            third_party_payments = []
            for account in third_party_bank["override_accounts"]:
                if account.get("type") == "depository" and account.get("subtype") in ["checking", "savings"]:
                    payments = [txn for txn in account.get("transactions", []) if txn.get("tag") == "auto loan payment"]
                    third_party_payments.extend(payments)
            answer = str([txn.get("transaction_id", "") for txn in third_party_payments])
        else:
            # Default detailed format
            if can_exclude:
                answer = (f"Yes - Auto loan can be excluded.\n"
                         f"Third party ({third_party_name}) made {months_with_payments} payments out of {config['months_required']} months.\n"
                         f"Creditor: {creditor}\n"
                         f"Monthly payment: ${monthly_payment:,.2f}\n"
                         f"Third party account is not joint with borrower.")
            else:
                answer = (f"No - Auto loan cannot be excluded.\n"
                         f"Third party ({third_party_name}) made only {months_with_payments} payments out of {config['months_required']} months.\n"
                         f"Minimum required: {config['months_required']} months.\n"
                         f"Creditor: {creditor}")
        
        return borrower_bank, third_party_bank, ulad, answer

    def mutate_statement_staleness(self, stale: bool = None, answer_type: str = "boolean") -> Tuple[Dict, str]:
        """
        Create a scenario where account statement end dates are either stale (more than
        45 days before the loan application date, treated as today) or fresh (within 45 days).

        When stale=True, all transaction dates are shifted backward so that the newest
        transaction falls 46-75 days in the past, making the derived statement EndDate stale.

        Returns:
            (mutated_bank_statement, answer_string)
        """
        bank = copy.deepcopy(self.base_bank_statement)
        stale = _resolve_boolean(stale)

        if stale:
            offset_days = random.randint(46, 75)
            for account in bank["override_accounts"]:
                new_txns = []
                for txn in account.get("transactions", []):
                    try:
                        d = datetime.strptime(txn["date_transacted"], "%Y-%m-%d")
                        dp = datetime.strptime(txn["date_posted"], "%Y-%m-%d")
                        new_txns.append({
                            **txn,
                            "date_transacted": (d - timedelta(days=offset_days)).strftime("%Y-%m-%d"),
                            "date_posted": (dp - timedelta(days=offset_days)).strftime("%Y-%m-%d"),
                        })
                    except (ValueError, KeyError):
                        new_txns.append(txn)
                account["transactions"] = new_txns
        # else: base data already has recent transactions (end date within 45 days)

        answer = "Yes" if stale else "No"
        return bank, answer

    def mutate_unexplained_large_deposits(self, answer_type: str = "id_list") -> Tuple[Dict, str]:
        """
        Add large deposits to the bank statement: some explained (payroll bonus, tax
        refund, internal transfer) and, depending on polarity, some unexplained.
        The answer is the transaction IDs of the unexplained deposits only.

        Returns:
            (mutated_bank_statement, answer_string)
        """
        bank = copy.deepcopy(self.base_bank_statement)

        monthly_income = self._detect_monthly_income(bank)
        self._split_payroll_to_weekly(bank, monthly_income)
        self._clamp_large_deposits(bank, monthly_income)
        self._remove_transactions_by_tag(bank, "large deposits")

        threshold = max(0.5 * monthly_income, 1000.0) if monthly_income > 0 else 5000.0

        explained_items = [
            ("PAYROLL BONUS - {employer}", ["Acme Corp", "Global Tech", "State University"]),
            ("TAX REFUND - IRS", [None]),
            ("ACH CREDIT - TRANSFER FROM OWN SAVINGS", [None]),
        ]
        unexplained_descs = [
            "WIRE IN CREDIT - UNKNOWN SOURCE",
            "ACH CREDIT - UNNAMED SENDER",
            "WIRE TRANSFER IN",
        ]

        # Always add 1-2 explained large deposits for context
        num_explained = random.randint(1, 2)
        for i in range(num_explained):
            template, names = random.choice(explained_items)
            employer = random.choice(names)
            desc = template.format(employer=employer) if employer else template
            amount = round(random.uniform(threshold * 1.1, threshold * 3), 2)
            date_t, date_p = self._get_random_date(90 - i * 20, max(0, 70 - i * 20))
            txn = {
                "description": desc,
                "amount": amount,
                "currency": "USD",
                "transaction_id": "",
                "tag": "explained large deposit",
                "date_transacted": date_t,
                "date_posted": date_p,
            }
            self._add_transaction_to_checking(bank, txn)

        added_unexplained = []
        if _resolve_boolean():
            for i in range(random.randint(1, 3)):
                desc = random.choice(unexplained_descs)
                amount = round(random.uniform(threshold * 1.1, threshold * 4), 2)
                date_t, date_p = self._get_random_date(85 - i * 20, max(0, 65 - i * 20))
                txn = {
                    "description": desc,
                    "amount": amount,
                    "currency": "USD",
                    "transaction_id": "",
                    "tag": "unexplained large deposit",
                    "date_transacted": date_t,
                    "date_posted": date_p,
                }
                self._add_transaction_to_checking(bank, txn)
                added_unexplained.append(txn)

        return bank, self._format_transaction_list(added_unexplained, answer_type)

    def mutate_multiple_employer_payroll(self, multi_employer: bool = None, answer_type: str = "boolean") -> Tuple[Dict, str]:
        """
        Create a scenario where payroll deposits over the two-year window come from
        multiple employers (Yes) or a single employer (No).

        Returns:
            (mutated_bank_statement, answer_string)
        """
        bank = copy.deepcopy(self.base_bank_statement)
        multi_employer = _resolve_boolean(multi_employer)

        self._remove_transactions_by_description(bank, ["PAYROLL", "ACH CREDIT PAYROLL"])

        employers = ["Acme Corp", "Global Tech", "State University", "NY MTA", "City Hospital"]

        if multi_employer:
            selected = random.sample(employers, random.randint(2, 3))
            for emp_idx, employer in enumerate(selected):
                amount = round(random.uniform(2000, 5000), 2)
                for month_offset in range(2):
                    start = 90 - emp_idx * 15 - month_offset * 30
                    end = max(0, start - 20)
                    date_t, date_p = self._get_random_date(start, end)
                    txn = {
                        "description": f"ACH CREDIT PAYROLL - {employer}",
                        "amount": round(amount + random.uniform(-50, 50), 2),
                        "currency": "USD",
                        "transaction_id": "",
                        "tag": "general transaction",
                        "date_transacted": date_t,
                        "date_posted": date_p,
                    }
                    self._add_transaction_to_checking(bank, txn)
        else:
            employer = random.choice(employers)
            amount = round(random.uniform(2000, 5000), 2)
            for i in range(3):
                date_t, date_p = self._get_random_date(90 - i * 30, max(0, 60 - i * 30))
                txn = {
                    "description": f"ACH CREDIT PAYROLL - {employer}",
                    "amount": round(amount + random.uniform(-50, 50), 2),
                    "currency": "USD",
                    "transaction_id": "",
                    "tag": "general transaction",
                    "date_transacted": date_t,
                    "date_posted": date_p,
                }
                self._add_transaction_to_checking(bank, txn)

        answer = "Yes" if multi_employer else "No"
        return bank, answer

    def mutate_employment_gap(self, gap: bool = None, answer_type: str = "boolean") -> Tuple[Dict, str]:
        """
        Generate 12 months of payroll deposits with either a gap greater than one
        month (gap=True) or a continuous monthly pattern (gap=False).

        When gap=True, two consecutive months are skipped, creating a >30-day
        window with no payroll, which represents an employment gap.

        Returns:
            (mutated_bank_statement, answer_string)
        """
        bank = copy.deepcopy(self.base_bank_statement)
        gap = _resolve_boolean(gap)

        self._remove_transactions_by_description(bank, ["PAYROLL", "ACH CREDIT PAYROLL"])

        employer = random.choice(["Acme Corp", "Global Tech", "State University"])
        base_amount = round(random.uniform(2000, 5000), 2)

        skip_start = random.randint(2, 7) if gap else -1
        skip_end = skip_start + 1  # two-month gap → > one month

        for month_offset in range(12):
            if gap and skip_start <= month_offset <= skip_end:
                continue  # Leave these months without payroll
            days_ago = month_offset * 30 + random.randint(5, 15)
            date_obj = datetime.now() - timedelta(days=days_ago)
            date_t = date_obj.strftime("%Y-%m-%d")
            date_p = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
            txn = {
                "description": f"ACH CREDIT PAYROLL - {employer}",
                "amount": round(base_amount + random.uniform(-50, 50), 2),
                "currency": "USD",
                "transaction_id": "",
                "tag": "general transaction",
                "date_transacted": date_t,
                "date_posted": date_p,
            }
            self._add_transaction_to_checking(bank, txn)

        answer = "Yes" if gap else "No"
        return bank, answer

    # NOTE: mutate_credit_card_full_balance_payment and its helpers (_add_credit_card_to_ulad,
    # _calculate_bank_statement_months) have been commented out per PR review - row 30 deleted.
    #
    # def mutate_credit_card_full_balance_payment(self, pays_full_balance=None, answer_type="boolean"):
    #     ...
    # def _add_credit_card_to_ulad(self, ulad, card_name, monthly_payment, balance):
    #     ...
    # def _calculate_bank_statement_months(self, bank):
    #     ...


def main():
    """Example usage"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python data_mutator.py <bank_statement.json> <ulad.json>")
        sys.exit(1)

    bank_statement_path = sys.argv[1]
    ulad_path = sys.argv[2]

    mutator = DataMutator(bank_statement_path, ulad_path)

    print("=== Mutate BNPL Transactions (id_list) ===")
    bank, answer = mutator.mutate_transaction("bnpl", num_transactions=3, answer_type="id_list")
    print(answer)

    print("\n=== Mutate BNPL Transactions (boolean) ===")
    bank, answer = mutator.mutate_transaction("bnpl", num_transactions=3, answer_type="boolean")
    print(answer)

    print("\n=== Mutate Employer Payroll Consistency (boolean) ===")
    bank, ulad, answer = mutator.mutate_employer_payroll_consistency(answer_type="boolean")
    print(answer)

    print("\n=== Mutate Address Match (boolean) ===")
    bank, ulad, answer = mutator.mutate_address_match(answer_type="boolean")
    print(answer)

    print("\nDone!")


if __name__ == "__main__":
    main()
