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
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd


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
    # "mortgage_payments": {
    #     "tag": "missing",
    #     "keywords": ["CALLABLE MORTGAGE PAYMENT", "BEST EVER MORTGAGE PAYMENT", "MORTGAGE PAYMENT"],
    #     "description_template": "ACH DEBIT - {keyword}",
    #     "amount_range": (800, 3000),
    #     "amount_sign": "negative",
    #     "default_count_range": (2, 4),
    #     "date_spacing": "monthly",
    # },
    # "savings_club": {
    #     "tag": "missing",
    #     "keywords": ["COMMUNITY SAVINGS CLUB FUNDS", "SAVINGS CLUB DEPOSIT", "INFORMAL SAVINGS GROUP"],
    #     "description_template": "ACH CREDIT - {keyword}",
    #     "amount_range": (1000, 5000),
    #     "amount_sign": "positive",
    #     "default_count_range": (1, 3),
    #     "date_spacing": "monthly",
    # },
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
    # "business": {
    #     "tag": "missing",
    #     "type": "depository",
    #     "subtype": "checking",
    #     "class": "business",
    #     "balance_range": (1000, 100000),
    #     "transaction_tag": "general transaction",
    #     "transaction_description": "Business Transfer",
    #     "transaction_amount": 100,
    #     "official_name": "John's Business LLC",
    #     "answer_format": "Business account #{account_num} - {official_name}",
    # },
}

# ==================== ULAD MUTATION CONFIGS ====================

ULAD_MUTATION_CONFIGS = {
    "employer_payroll_consistency": {
        "employers": [
            "NY MTA", "Amazon", "Google", "Microsoft", "City Hospital",
            "State University", "Wawa", "AAA", "Acme Corp", "Global Tech",
        ],
        "match_probability": 0.5,
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
        "match_probability": 0.5,
    },
    "gift_deposit": {
        "amount_range": (5000, 60000),
        "donor_names": ["Mary Homeowner", "Dad Homeowner", "Aunt Jane", "Uncle Bob"],
        "match_probability": 0.7,
    },
    "child_support_disclosure": {
        "amount_range": (300, 2000),
        "num_payments": 2,
        "keywords": ["DC OAG", "NJFSPC", "Child Support"],
        "description_template": "ACH DEBIT {keyword}",
    },
    "undisclosed_liabilities": {
        "amount_range": (50, 500),
        "creditors": [
            "Capital One", "American Express", "Discover", "Ally Bank",
            "Chase Credit Card", "AMEX", "Visa Credit Card", "No Nonsense Credit Card",
        ],
        "num_range": (2, 5),
    },
    "rental_income_consistency": {
        "rental_amounts": [2000, 2400, 3000, 3500, 5000, 8000],
        "payer_names": ["Jane Doe", "John Smith", "Robert Johnson"], # can add more payer names here
        "match_probability": 0.5,
    },
    "joint_account_holder": {
        "joint_names": ["Alice Homeowner", "DAD FIRSTIMER", "Jane Smith", "Bob Homeowner"],
        "balance_range": (5000, 50000),
    },
    "payroll_paystub_consistency": {
        "employer": "AAA",
        "base_payroll_range": (2000, 5000),
        "mismatch_diffs": [300, 500, 600, -200, -300],
        "match_probability": 0.5,
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
}


class DataMutator:
    """Handles mutation of bank statements and ULAD data for test case generation."""

    def __init__(self, bank_statement_path: str, ulad_path: str):
        with open(bank_statement_path, 'r') as f:
            self.base_bank_statement = json.load(f)
        with open(ulad_path, 'r') as f:
            self.base_ulad = json.load(f)
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

        if num_transactions is None:
            num_transactions = random.randint(*config["default_count_range"])

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

            if config.get("amount_discrete"):
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
            has_account = random.random() < 0.5

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
    # All return (mutated_bank, mutated_ulad, answer_string)

    def mutate_employer_payroll_consistency(self, match: bool = None, answer_type: str = "boolean") -> Tuple[Dict, Dict, str]:
        """
        Set payroll deposits in bank from a specific employer; set ULAD employer to
        match (consistent) or differ (mismatch).

        Returns:
            (mutated_bank, mutated_ulad, answer)
        """
        config = ULAD_MUTATION_CONFIGS["employer_payroll_consistency"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)

        if match is None:
            match = random.random() < config["match_probability"]

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

        if answer_type == "boolean":
            answer = "Yes" if match else "No"
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

        if match is None:
            match = random.random() < config["match_probability"]

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

        if match is None:
            match = random.random() < config["match_probability"]

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

        keyword = random.choice(config["keywords"])
        amount = round(random.uniform(*config["amount_range"]), 2)
        added = []

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

    def mutate_undisclosed_liabilities(self, num_undisclosed: int = None, answer_type: str = "id_list") -> Tuple[Dict, Dict, str]:
        """
        Add recurring payments to creditors in the bank statement that do NOT appear
        in ULAD liabilities, simulating undisclosed debt obligations.

        Returns:
            (mutated_bank, mutated_ulad, answer)
        """
        config = ULAD_MUTATION_CONFIGS["undisclosed_liabilities"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)

        if num_undisclosed is None:
            num_undisclosed = random.randint(*config["num_range"])

        # Get existing ULAD creditors to avoid overlap
        deal = self._get_deal(ulad)
        liabilities = deal.get("LIABILITIES", {}).get("LIABILITY", [])
        if not isinstance(liabilities, list):
            liabilities = [liabilities]
        existing_creditors = {
            lib.get("LIABILITY_HOLDER", {}).get("NAME", {}).get("FullName", "").lower()
            for lib in liabilities if isinstance(lib, dict)
        }

        available = [c for c in config["creditors"] if c.lower() not in existing_creditors]
        if len(available) < num_undisclosed:
            available = config["creditors"]

        creditors_chosen = random.sample(available, min(num_undisclosed, len(available)))
        added = []

        for creditor in creditors_chosen:
            amount = round(random.uniform(*config["amount_range"]), 2)
            date_transacted, date_posted = self._get_random_date()
            txn = {
                "description": f"ACH DEBIT {creditor} PMT",
                "amount": -amount,
                "currency": "USD",
                "transaction_id": "",
                "tag": "general transaction",
                "date_transacted": date_transacted,
                "date_posted": date_posted,
            }
            self._add_transaction_to_checking(bank, txn)
            added.append(txn)

        answer = self._format_transaction_list(added, answer_type)
        return bank, ulad, answer

    def mutate_rental_income_consistency(self, match: bool = None, answer_type: str = "boolean") -> Tuple[Dict, Dict, str]:
        """
        Add a REO property to ULAD with a set gross rental income amount, then add
        rental income deposits to the bank statement that match (or differ).

        Returns:
            (mutated_bank, mutated_ulad, answer)
        """
        config = ULAD_MUTATION_CONFIGS["rental_income_consistency"]
        bank = copy.deepcopy(self.base_bank_statement)
        ulad = copy.deepcopy(self.base_ulad)

        if match is None:
            match = random.random() < config["match_probability"]

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
        self._remove_transactions_by_description(bank, ["ZELLE PAYMENT FROM", "RENTAL INCOME DEPOSIT"])

        deposit_amount = rental_amount if match else rental_amount + random.choice([-500, 500, 1000, -1000])
        deposit_amount = max(100, deposit_amount)

        added = []
        for i in range(2):
            date_transacted, date_posted = self._get_random_date(90 - i * 30, max(0, 60 - i * 30))
            txn = {
                "description": f"ZELLE PAYMENT FROM {payer.upper()}",
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

        if joint_name is None:
            joint_name = random.choice(config["joint_names"])

        # Remove any pre-existing joint accounts
        bank["override_accounts"] = [
            a for a in bank["override_accounts"]
            if len(a.get("identity", {}).get("names", [])) <= 1
        ]

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

        if match is None:
            match = random.random() < config["match_probability"]

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

        ulad_employer = self._get_employer_name(ulad) or random.choice(config["ulad_employers"])
        self._set_employer_name(ulad, ulad_employer)

        bank_employers = [e for e in config["bank_employers"] if e != ulad_employer]
        bank_employer = random.choice(bank_employers)

        self._remove_transactions_by_description(bank, ["PAYROLL", "ACH CREDIT PAYROLL"])

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
