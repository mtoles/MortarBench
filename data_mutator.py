"""
Data Mutator for Mortgage ApplicationDataset Generation

This module provides mutation functions to create test cases from base bank statement
and ULAD files. Each mutation function corresponds to specific transaction tags and
generates appropriate ground truth answers.
"""

import json
import random
import copy
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Callable
import pandas as pd


# ==================== CONFIGURATION DICTIONARIES ====================

TRANSACTION_CONFIGS = {
    "bnpl": {
        "tag": "BNPL transactions",
        "keywords": ["Klarna", "Afterpay", "Affirm", "Sezzle", "Zip Co", "PayPal in 4"],
        "description_template": "ACH DEBIT - {keyword} PMT",
        "amount_range": (25, 150),
        "amount_sign": "negative",
        "default_count_range": (0, 4),
        "date_spacing": None,  # None = random dates
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
        "description_template": "{keyword}",  # Use keyword as-is
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
        "date_spacing": "monthly",  # Space 30 days apart
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
        "amount_range": None,  # Special: discrete values
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
        "keywords": ["Consulting Income", "Side Gig", "Freelance Payment", "SSA US TREASURY", "ACH CREDIT"],
        "description_template": "{keyword}",
        "amount_range": (200, 2500),
        "amount_sign": "positive",
        "default_count_range": (0, 6),
        "date_spacing": "bi-weekly",  # Space 15 days apart with variance
        "recurring": True,  # Use same keyword for all transactions
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
        "keywords": ["Alice Homeowner", "Bob Smith", "Jane Doe"], # Note: hardcoded
        "description_template": "Transfer from {keyword}",
        "amount_range": (200, 1000),
        "amount_sign": "positive",
        "default_count_range": (1, 2),
        "date_spacing": None,
    },
}

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
}


class DataMutator:
    """Handles mutation of bank statements and ULAD data for test case generation."""
    
    def __init__(self, bank_statement_path: str, ulad_path: str):
        """
        Initialize the mutator with base data files.
        
        Args:
            bank_statement_path: Path to base bank statement JSON
            ulad_path: Path to base ULAD JSON
        """
        with open(bank_statement_path, 'r') as f:
            self.base_bank_statement = json.load(f)
        
        with open(ulad_path, 'r') as f:
            self.base_ulad = json.load(f)
        
        self.transaction_counter = 1000
    
    def _generate_txn_id(self, dataset_id: str) -> str:
        """Generate a unique transaction ID."""
        self.transaction_counter += 1
        return f"plaid-{dataset_id}-{self.transaction_counter:05d}"
    
    def _get_random_date(self, start_days_ago: int = 90, end_days_ago: int = 0) -> Tuple[str, str]:
        """
        Generate random transaction and posted dates.
        
        Returns:
            Tuple of (date_transacted, date_posted)
        """
        days_ago = random.randint(end_days_ago, start_days_ago)
        date_transacted = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        date_posted = (datetime.now() - timedelta(days=days_ago - 1)).strftime("%Y-%m-%d")
        return date_transacted, date_posted
    
    def _get_spaced_dates(self, num_dates: int, spacing: str) -> List[Tuple[str, str]]:
        """
        Generate evenly spaced dates based on spacing type.
        
        Args:
            num_dates: Number of date pairs to generate
            spacing: Type of spacing ('monthly', 'bi-weekly', etc.)
            
        Returns:
            List of (date_transacted, date_posted) tuples
        """
        dates = []
        spacing_days = {
            "monthly": 30,
            "bi-weekly": 15,
            "weekly": 7,
        }
        
        days_between = spacing_days.get(spacing, 30)
        
        for i in range(num_dates):
            start = 90 - (i * days_between)
            end = max(0, start - days_between + 10)  # Allow some variance
            date_transacted, date_posted = self._get_random_date(start, end)
            dates.append((date_transacted, date_posted))
        
        return dates
    
    def _remove_transactions_by_tag(self, bank_statement: Dict, tag: str) -> List[Dict]:
        """
        Remove all transactions with a specific tag from bank statement.
        
        Args:
            bank_statement: Bank statement dict to modify
            tag: Tag to filter by
            
        Returns:
            List of removed transactions
        """
        removed = []
        for account in bank_statement["override_accounts"]:
            if "transactions" not in account:
                continue
            
            original_txns = account["transactions"]
            remaining_txns = []
            
            for txn in original_txns:
                if txn.get("tag") == tag:
                    removed.append(txn)
                else:
                    remaining_txns.append(txn)
            
            account["transactions"] = remaining_txns
        
        return removed
    
    def _add_transaction_to_checking(self, bank_statement: Dict, transaction: Dict) -> None:
        """Add a transaction to the main checking account."""
        dataset_id = bank_statement["seed"].split("-")[-1]
        transaction["transaction_id"] = self._generate_txn_id(dataset_id)
        
        # Find checking account (first depository/checking account)
        for account in bank_statement["override_accounts"]:
            if account.get("type") == "depository" and account.get("subtype") == "checking":
                account["transactions"].append(transaction)
                # Sort by date
                account["transactions"].sort(key=lambda x: x["date_transacted"], reverse=True)
                return
    
    def _format_transaction_list(self, transactions: List[Dict]) -> str:
        """Format a list of transactions for the answer."""
        if not transactions:
            return "None noted."
        
        result = f"{len(transactions)} transaction{'s' if len(transactions) > 1 else ''}:\n"
        for i, txn in enumerate(transactions, 1):
            date = txn.get("date_transacted", "")
            desc = txn.get("description", "")
            amount = txn.get("amount", 0)
            result += f"{i}) {date} - {desc} - ${abs(amount):,.2f}\n"
        
        return result.strip()
    
    # ====================  MUTATION FUNCTIONS ====================
    
    def mutate_transaction(self, transaction_type: str, num_transactions: int = None) -> Tuple[Dict, str]:
        """
        Transaction mutation function to add transactions to the bank statement.
        
        Args:
            transaction_type: Type of transaction from TRANSACTION_CONFIGS
            num_transactions: Number of transactions to add (random if None)
            
        Returns:
            Tuple of (mutated_bank_statement, answer)
        """
        if transaction_type not in TRANSACTION_CONFIGS:
            raise ValueError(f"Unknown transaction type: {transaction_type}")
        
        config = TRANSACTION_CONFIGS[transaction_type]
        bank_statement = copy.deepcopy(self.base_bank_statement)
        
        # Remove existing transactions with this tag
        self._remove_transactions_by_tag(bank_statement, config["tag"])
        
        # Determine number of transactions
        if num_transactions is None:
            min_count, max_count = config["default_count_range"]
            num_transactions = random.randint(min_count, max_count)
        
        added_transactions = []
        
        # Handle recurring transactions (use same keyword)
        if config.get("recurring", False) and num_transactions > 0:
            keyword = random.choice(config["keywords"])
            base_amount = round(random.uniform(*config["amount_range"]), 2)
        else:
            keyword = None
            base_amount = None
        
        # Generate dates
        if config.get("date_spacing") and num_transactions > 0:
            dates = self._get_spaced_dates(num_transactions, config["date_spacing"])
        else:
            dates = [self._get_random_date() for _ in range(num_transactions)]
        
        # Create transactions
        for i in range(num_transactions):
            # Select keyword
            if keyword is None:
                current_keyword = random.choice(config["keywords"])
            else:
                current_keyword = keyword
            
            # Generate amount
            if config.get("amount_discrete"):
                amount = float(random.choice(config["amount_discrete"]))
            elif config.get("recurring", False):
                # Add variance to base amount
                variance = round(random.uniform(-50, 50), 2)
                amount = base_amount + variance
            else:
                amount = round(random.uniform(*config["amount_range"]), 2)
            
            # Apply sign
            if config["amount_sign"] == "negative":
                amount = -abs(amount)
            else:
                amount = abs(amount)
            
            # Generate description
            description = config["description_template"].format(keyword=current_keyword)
            
            # Get dates
            date_transacted, date_posted = dates[i]
            
            # Create transaction
            txn = {
                "description": description,
                "amount": amount,
                "currency": "USD",
                "transaction_id": "",  # Will be set by _add_transaction_to_checking
                "tag": config["tag"],
                "date_transacted": date_transacted,
                "date_posted": date_posted
            }
            
            self._add_transaction_to_checking(bank_statement, txn)
            added_transactions.append(txn)
        
        # Generate answer
        answer = self._format_transaction_list(added_transactions)
        
        return bank_statement, answer
    
    def mutate_account(self, account_type: str, has_account: bool = None) -> Tuple[Dict, str]:
        """
        Account mutation function to add accounts to the bank statement.
        
        Args:
            account_type: Type of account from ACCOUNT_CONFIGS
            has_account: Whether to include account (random if None)
            
        Returns:
            Tuple of (mutated_bank_statement, answer)
        """
        if account_type not in ACCOUNT_CONFIGS:
            raise ValueError(f"Unknown account type: {account_type}")
        
        config = ACCOUNT_CONFIGS[account_type]
        bank_statement = copy.deepcopy(self.base_bank_statement)
        
        # Remove existing accounts of this type
        if account_type == "retirement":
            bank_statement["override_accounts"] = [
                acc for acc in bank_statement["override_accounts"]
                if acc.get("type") != config["type"] or acc.get("subtype") != config["subtype"]
            ]
        elif account_type == "custodial":
            bank_statement["override_accounts"] = [
                acc for acc in bank_statement["override_accounts"]
                if "Custodial" not in acc.get("official_name", "")
            ]
        
        # Determine whether to add account
        if has_account is None:
            has_account = random.random() < 0.5
        
        if has_account:
            dataset_id = bank_statement["seed"].split("-")[-1]
            account_num = f"{random.randint(10000000, 99999999)}"
            balance = round(random.uniform(*config["balance_range"]), 2)
            
            # Create account
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
                    "date_posted": self._get_random_date()[1]
                }],
                "identity": bank_statement["override_accounts"][0]["identity"]
            }
            
            # Add optional fields
            if config.get("official_name"):
                account["official_name"] = config["official_name"]
            
            if account_type == "retirement":
                account["tags"] = [config["tag"]]
            
            bank_statement["override_accounts"].append(account)
            
            # Format answer
            answer = config["answer_format"].format(
                account_num=account_num,
                balance=balance
            )
        else:
            answer = "None noted."
        
        return bank_statement, answer
    


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python data_mutator.py <bank_statement.json> <ulad.json>")
        sys.exit(1)
    
    bank_statement_path = sys.argv[1]
    ulad_path = sys.argv[2]
    
    mutator = DataMutator(bank_statement_path, ulad_path)
    
    # Example: Generate BNPL mutation using transaction mutation function
    print("Generating BNPL transactions...")
    mutated_bank, answer = mutator.mutate_transaction("bnpl", num_transactions=3)
    
    print("\nAnswer:")
    print(answer)
    
    print("\nSaving mutated bank statement...")
    with open("mutated_bank_statement.json", "w") as f:
        json.dump(mutated_bank, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()
