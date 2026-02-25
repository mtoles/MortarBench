import json
import random
import uuid
import datetime
import argparse
import os

from ulad_generator import UladGenerator

# --- Configuration & Constants ---

KEYWORDS = {
    "BNPL": ["Klarna", "Afterpay", "Affirm", "Sezzle", "Zip Co", "PayPal in 4"],
    "Large_Deposit": [5000, 10000, 20000], # thresholds or generating logic
    "Rental": ["Rent Payment", "Landlord", "Property Management", "Apt Rent"],
    "Child_Support": ["Child Support", "Alimony", "Wage Garnishment", "DC OAG"],
    "Gift": ["Wire In - Gift", "Gift from Relative"],
    "Secured_Loan": ["Loan Proceeds", "Secured Loan Deposit"],
    "Crypto": ["Coinbase", "Crypto.com", "Binance", "Gemini", "Kraken"],
    "Overdraft": ["Overdraft Fee", "NSF Fee", "Non-Sufficient Funds"],
    "Unsecured_Loan": ["Personal Loan", "LendingClub", "SoFi", "Upstart"],
    "Undisclosed_Debt": ["Chase Credit Card", "Auto Loan", "Navient", "Nelnet"],
    "Cash_Deposit": ["ATM Cash Deposit", "Cash Deposit", "Branch Deposit"],
    "Undisclosed_Income": ["Consulting Income", "Side Gig", "Venmo Cash Out"],
    "Payroll": ["ACH Credit Payroll"], # Will append Employer
    "Business_Account": ["Business Checking", "Business Savings"],
    "Savings_Club": ["Sou-sou", "Savings Club", "Partner Hand"],
    "Payday_Loan": ["Speedy Cash", "Check 'n Go", "Ace Cash Express"],
    "Custodial": ["UTMA", "UGMA", "Custodial Account"],
    "Foreign": ["International Wire", "Swift Transfer", "Deutsche Bank"],
    "Retirement": ["401k", "IRA", "Roth IRA", "Fidelity NetBenefits"],
    "Mortgage": ["Mortgage Payment", "Home Loan", "Flagstar Bank", "Mr. Cooper"]
}

TAG_OPTIONS = {
    "large_deposit": "large deposits",
    "rental": "rental payments",
    "bnpl": "BNPL transactions",
    "secured_loan": "secured loan",
    "crypto_deposit": "deposit from cryptocurrency source",
    "overdraft": "overdraft or NSF",
    "withdrawal": "withdrawal",
    "payday": "payday loan or high-interest lending source",
    "custodial": "custodial account",
    "foreign": "foreign origin",
    "retirement": "retirement accounts",
    "retirement_assets": "retirement assets",
    "additional_holder": "additional account holder",
    "undisclosed_housing": "undisclosed housing payments",
    "undisclosed_income": "undisclosed income source",
    "unexplained_deposit": "unexplained deposits",
    "excessive_cash": "excessive cash deposits",
    "default": "general transaction",
    "child_support": "child support",
    "undisclosed_debt": "undisclosed debt",
}

EMPLOYERS = ["Acme Corp", "Global Tech", "State University", "City Hospital"]


ECONOMIC_PROFILES = {
    "Lower": {
        "payroll": (1500, 2500),
        "rental": (600, 1200),
        "bnpl": (10, 50),
        "child_support": (100, 300),
        "crypto": (10, 50),
        "overdraft": [25, 35],
        "undisclosed_debt": (50, 150),
        "payday": (200, 500), # Amount received
        "foreign": (100, 500),
        "large_deposit": (0.55, 2.0), # Multiplier of monthly income (must be > 0.5)
        "cash_deposit": (100, 800),
        "earnest": (500, 2000),
        "unsecured": (500, 2500),
        "undisclosed_income": (50, 200),
        "gift": (500, 2500),
        "undisclosed_employment": (300, 800),
        "savings_club": (50, 200),
        "secured_loan": (500, 2000),
        "balances": {
            "checking": (100, 1500),
            "savings": (100, 2000),
            "retirement": (0, 5000),
            "joint": (100, 1000),
            "custodial": (50, 500),
            "business": (100, 1000)
        }
    },
    "Middle": {
        "payroll": (2800, 5000),
        "rental": (1200, 2500),
        "bnpl": (25, 150),
        "child_support": (300, 800),
        "crypto": (50, 500),
        "overdraft": [25, 35, 50],
        "undisclosed_debt": (100, 600),
        "payday": (300, 1000),
        "foreign": (500, 2000),
        "large_deposit": (0.55, 3.0), # Multiplier
        "cash_deposit": (200, 2000),
        "earnest": (2000, 10000),
        "unsecured": (2000, 10000),
        "undisclosed_income": (200, 600),
        "gift": (2000, 10000),
        "undisclosed_employment": (800, 2000),
        "savings_club": (100, 500),
        "secured_loan": (2000, 8000),
        "balances": {
            "checking": (1500, 10000),
            "savings": (5000, 30000),
            "retirement": (20000, 150000),
            "joint": (1000, 8000),
            "custodial": (500, 5000),
            "business": (1000, 10000)
        }
    },
    "Upper": {
        "payroll": (8000, 25000),
        "rental": (3000, 8000),
        "bnpl": (100, 1000), # Even rich people use BNPL for cash flow
        "child_support": (1500, 5000),
        "crypto": (1000, 50000),
        "overdraft": [35, 50, 100], # Occasional slip ups?
        "undisclosed_debt": (1000, 5000),
        "payday": (1000, 3000), # Rare but possible? Or maybe they don't use payday loans. 
        # But we need to inject it if the question asks. We'll simulate it as high cost short term lending.
        "foreign": (5000, 50000),
        "large_deposit": (0.55, 4.0), # Multiplier
        "cash_deposit": (1000, 10000),
        "earnest": (15000, 100000),
        "unsecured": (10000, 50000),
        "undisclosed_income": (1000, 5000),
        "gift": (15000, 100000),
        "undisclosed_employment": (3000, 10000),
        "savings_club": (1000, 5000), # Investment clubs?
        "secured_loan": (10000, 100000),
        "balances": {
            "checking": (10000, 100000),
            "savings": (50000, 500000),
            "retirement": (200000, 2500000),
            "joint": (10000, 100000),
            "custodial": (10000, 100000),
            "business": (10000, 200000)
        }
    }
}

class PlaidGenerator:
    def __init__(self, seed=None):
        if seed:
            random.seed(seed)
        self.transaction_counter = 0
        self.dataset_id = str(uuid.uuid4())[:8]
        self.used_account_numbers = set()

    def _get_date(self, offset_days=0):
        base = datetime.date.today()
        dt = base + datetime.timedelta(days=offset_days)
        return dt.isoformat()

    def _generate_txn_id(self):
        self.transaction_counter += 1
        return f"plaid-{self.dataset_id}-{self.transaction_counter:05d}"

    def _generate_account_id(self):
        while True:
            # Generate 8 digit account number
            acc_num = str(random.randint(10000000, 99999999))
            if acc_num not in self.used_account_numbers:
                self.used_account_numbers.add(acc_num)
                return acc_num

    def generate_single_dataset(self):
        employer = random.choice(EMPLOYERS)
        user_name = "John Homeowner" 
        
        # Select Economic Profile
        profile_name = random.choice(["Lower", "Middle", "Upper"])
        profile = ECONOMIC_PROFILES[profile_name]
        
        accounts = []
        
        # 1. Main Checking Account (Depository)
        checking_txns = []
        
        # Generate basic payroll (Credit/Deposit -> Positive)
        payroll_range = profile["payroll"]
        payroll_base = round(random.uniform(payroll_range[0], payroll_range[1]), 2)
        for i in range(1, 4): # 3 months
            date_offset = -30 * i
            checking_txns.append({
                "date_transacted": self._get_date(date_offset),
                "date_posted": self._get_date(date_offset + 1),
                "amount": payroll_base,  # Positive for deposit
                "description": f"ACH CREDIT PAYROLL - {employer}",
                "currency": "USD",
                "tag": TAG_OPTIONS["default"], # Using default as 'income' tag was removed
                "transaction_id": self._generate_txn_id()
            })

        # Inject Scenarios (Probabilistic or Forced)
        # We ensure coverage but use profile-specific ranges.
        
        # BNPL (Debit/Withdrawal -> Negative)
        r = profile["bnpl"]
        checking_txns.append(self._create_txn(random.choice(KEYWORDS["BNPL"]), -round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["bnpl"]))
        
        # Rental (Debit/Withdrawal -> Negative) or Mortgage
        r = profile["rental"]
        rand_val = random.random()
        if rand_val < 0.33:
             checking_txns.append(self._create_txn(random.choice(KEYWORDS["Rental"]), -round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["rental"]))
        elif rand_val < 0.66:
             checking_txns.append(self._create_txn(random.choice(KEYWORDS["Mortgage"]), -round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["default"]))
        else:
             checking_txns.append(self._create_txn("Housing Payment", -round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["undisclosed_housing"]))
        
        # Child Support (Liability) (Debit/Withdrawal -> Negative)
        r = profile["child_support"]
        checking_txns.append(self._create_txn(random.choice(KEYWORDS["Child_Support"]), -round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["child_support"]))
        
        # Crypto (Debit - Purchase -> Negative)
        r = profile["crypto"]
        checking_txns.append(self._create_txn("Purchase at " + random.choice(KEYWORDS["Crypto"]), -round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["withdrawal"]))
        
        # Crypto (Credit - Deposit -> Positive)
        r = profile["crypto"]
        checking_txns.append(self._create_txn("Deposit from " + random.choice(KEYWORDS["Crypto"]), round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["crypto_deposit"]))
        
        # Overdraft (Debit/Withdrawal -> Negative)
        r = profile["overdraft"]
        # Overdraft fees are expenses, so negative
        checking_txns.append(self._create_txn(random.choice(KEYWORDS["Overdraft"]), -float(random.choice(r)), tag=TAG_OPTIONS["overdraft"]))
        
        # Undisclosed Debt (Debit/Withdrawal -> Negative)
        r = profile["undisclosed_debt"]
        checking_txns.append(self._create_txn("Payment to " + random.choice(KEYWORDS["Undisclosed_Debt"]), -round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["undisclosed_debt"]))
        
        # Payday Loan (Income/Deposit -> Positive)
        r = profile["payday"]
        checking_txns.append(self._create_txn(random.choice(KEYWORDS["Payday_Loan"]), round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["payday"]))
        
        # Foreign Deposit (Credit/Deposit -> Positive)
        r = profile["foreign"]
        checking_txns.append(self._create_txn(random.choice(KEYWORDS["Foreign"]), round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["foreign"]))
        
        # Large Deposit (Unusual) (Credit/Deposit -> Positive)
        r = profile["large_deposit"]
        large_deposit_amount = round(payroll_base * random.uniform(r[0], r[1]), 2)
        checking_txns.append(self._create_txn("Wire Transfer", large_deposit_amount, tag=TAG_OPTIONS["large_deposit"])) 
        
        # Cash Deposit (Credit/Deposit -> Positive)
        # Excessive relative to usual income
        cash_deposit_amount = round(payroll_base * random.uniform(0.5, 0.9), 2)
        checking_txns.append(self._create_txn(random.choice(KEYWORDS["Cash_Deposit"]), cash_deposit_amount, tag=TAG_OPTIONS["excessive_cash"]))
        
        # Earnest Money (Withdrawal) (Debit -> Negative)
        r = profile["earnest"]
        checking_txns.append(self._create_txn("Earnest Money Deposit", -round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["withdrawal"]))

        # --- Missing Coverage Injections ---

        # Unsecured Loan (Credit/Deposit -> Positive)
        r = profile["unsecured"]
        checking_txns.append(self._create_txn(random.choice(KEYWORDS["Unsecured_Loan"]), round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["unexplained_deposit"]))

        # Undisclosed Income (Credit/Deposit -> Positive)
        r = profile["undisclosed_income"]
        side_gig_keyword = random.choice(KEYWORDS["Undisclosed_Income"])
        side_gig_amount = round(random.uniform(r[0], r[1]), 2)
        for _ in range(3):
            # Minor variance in amount
            amt = side_gig_amount + round(random.uniform(- (side_gig_amount * 0.1), (side_gig_amount * 0.1)), 2)
            checking_txns.append(self._create_txn(side_gig_keyword, amt, tag=TAG_OPTIONS["undisclosed_income"]))

        # Gift (Credit/Deposit -> Positive)
        r = profile["gift"]
        checking_txns.append(self._create_txn(random.choice(KEYWORDS["Gift"]), round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["large_deposit"]))

        # Undisclosed Employment (Credit/Deposit -> Positive)
        r = profile["undisclosed_employment"]
        other_employer = random.choice([e for e in EMPLOYERS if e != employer])
        checking_txns.append(self._create_txn(f"ACH CREDIT PAYROLL - {other_employer}", round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["undisclosed_income"]))

        # Additional Account Holder (Credit/Deposit -> Positive)
        checking_txns.append(self._create_txn("Transfer from Alice Homeowner", 500, tag=TAG_OPTIONS["additional_holder"]))

        # Missing Info (Fake transaction that would trigger this observation)
        # Treat as deposit
        checking_txns.append(self._create_txn("Check Deposit (No Image)", 1200, tag=TAG_OPTIONS["default"]))

        # 2. Savings Account (Probabilistic)
        has_savings = random.random() < 0.8
        savings_txns_to_move = []
        
        # Prepare Savings Transactions
        # Savings Club (Credit/Deposit -> Positive)
        r = profile["savings_club"]
        savings_txns_to_move.append(self._create_txn(random.choice(KEYWORDS["Savings_Club"]), round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["default"]))
        # Secured Loan Deposit (Credit/Deposit -> Positive)
        r = profile["secured_loan"]
        savings_txns_to_move.append(self._create_txn(random.choice(KEYWORDS["Secured_Loan"]), round(random.uniform(r[0], r[1]), 2), tag=TAG_OPTIONS["secured_loan"]))

        if has_savings:
            r = profile["balances"]["savings"]
            savings_account = {
                "type": "depository",
                "subtype": "savings",
                "starting_balance": round(random.uniform(r[0], r[1]), 2),
                "currency": "USD",
                "numbers": {"account": self._generate_account_id()},
                "transactions": savings_txns_to_move,
                "identity": self._create_identity(user_name)
            }
            # Fix dates for savings
            for t in savings_account["transactions"]:
                 days = random.randint(-90, 0)
                 t['date_transacted'] = self._get_date(days)
                 t['date_posted'] = self._get_date(days + 1)
            accounts.append(savings_account)
        else:
            # Move to checking if no savings
            checking_txns.extend(savings_txns_to_move)

        # Re-sort Checking transactions + injected ones by date
        for t in checking_txns:
            if 'date_transacted' not in t: 
                days = random.randint(-90, 0)
                t['date_transacted'] = self._get_date(days)
                t['date_posted'] = self._get_date(days + 1)
        
        # Sort checking transactions purely by date for realism
        checking_txns.sort(key=lambda x: x['date_transacted'], reverse=True)

        r = profile["balances"]["checking"]
        checking_account = {
            "type": "depository",
            "subtype": "checking",
            "starting_balance": round(random.uniform(r[0], r[1]), 2),
            "currency": "USD",
            "numbers": {"account": self._generate_account_id()},
            "transactions": checking_txns,
            "identity": self._create_identity(user_name)
        }
        accounts.append(checking_account)
        
        # 3. Retirement Account (Investment) - Probabilistic
        if random.random() < 0.8:
            r = profile["balances"]["retirement"]
            retirement_account = {
                "type": "investment",
                "subtype": "401k",
                "starting_balance": round(random.uniform(r[0], r[1]), 2),
                "currency": "USD",
                "numbers": {"account": self._generate_account_id()},
                "transactions": [],
                "identity": self._create_identity(user_name),
                "tags": [TAG_OPTIONS["retirement"]]
            }
            # Add a transaction for tagging purposes (Contribution = Withdrawal from income, but here it's INSIDE the 401k)
            # Inside 401k, a contribution is a Deposit/Credit -> Positive
            retirement_account["transactions"].append(
                 self._create_txn("Contribution", 500, tag=TAG_OPTIONS["retirement_assets"])
            )
            # Fix dates
            for t in retirement_account["transactions"]:
                 days = random.randint(-90, 0)
                 t['date_transacted'] = self._get_date(days)
                 t['date_posted'] = self._get_date(days + 1)
            accounts.append(retirement_account)

        # 4. Joint Account logic (Question 18) - Probabilistic
        if random.random() < 0.6:
            r = profile["balances"]["joint"]
            joint_account = {
                "type": "depository",
                "subtype": "checking",
                "starting_balance": round(random.uniform(r[0], r[1]), 2),
                "currency": "USD",
                "numbers": {"account": self._generate_account_id()},
                "transactions": [],
                "identity": self._create_identity(user_name, joint_user="Alice Homeowner")
            }
            # Add transaction (Transfer In -> Positive)
            joint_account["transactions"].append(
                self._create_txn("Transfer", 100, tag=TAG_OPTIONS["default"])
            )
            for t in joint_account["transactions"]:
                 days = random.randint(-90, 0)
                 t['date_transacted'] = self._get_date(days)
                 t['date_posted'] = self._get_date(days + 1)
            accounts.append(joint_account)

        # 5. Custodial Account (Question 25) - Probabilistic
        if random.random() < 0.6:
            r = profile["balances"]["custodial"]
            custodial_account = {
                "type": "depository",
                "subtype": "money market",
                "official_name": "Custodial Account for Jr",
                "starting_balance": round(random.uniform(r[0], r[1]), 2),
                "currency": "USD",
                "numbers": {"account": self._generate_account_id()},
                "transactions": [],
                "identity": self._create_identity(user_name)
            }
            # Transfer In -> Positive
            custodial_account["transactions"].append(
                 self._create_txn("Transfer In", 50, tag=TAG_OPTIONS["custodial"])
            )
            for t in custodial_account["transactions"]:
                 days = random.randint(-90, 0)
                 t['date_transacted'] = self._get_date(days)
                 t['date_posted'] = self._get_date(days + 1)
            accounts.append(custodial_account)

        # 6. Business Account (Question 22) - Probabilistic
        if random.random() < 0.4:
            r = profile["balances"]["business"]
            business_account = {
                "type": "depository",
                "subtype": "checking",
                "class": "business", 
                "official_name": "John's Business LLC",
                "starting_balance": round(random.uniform(r[0], r[1]), 2),
                "currency": "USD",
                "numbers": {"account": self._generate_account_id()},
                "transactions": [],
                "identity": {
                     "names": ["John's Business LLC"],
                     "emails": [{"data": "john@business.com", "primary": True, "type": "work"}],
                     "addresses": [{"data": {"city": "Washington", "country": "US", "postal_code": "20013", "region": "DC", "street": "123 Biz St"}, "primary": True}]
                }
            }
            accounts.append(business_account)

        # Calculate End Balance for each account
        for account in accounts:
            starting = account.get("starting_balance", 0.0)
            txns = account.get("transactions", [])
            net_flow = sum(t.get("amount", 0.0) for t in txns)
            account["end_balance"] = round(starting + net_flow, 2)

        return {
            "seed": f"generated-test-{self.dataset_id}",
            "override_accounts": accounts
        }

    def _create_txn(self, desc, amount, tag=None):
        # User Request: Deposits/Credits cannot be negative.
        # Implied Convention: 
        #   - Credits/Deposits: Positive (+)
        #   - Debits/Withdrawals: Negative (-)
        
        final_tag = tag if tag else TAG_OPTIONS["default"]
        
        txn = {
            "description": desc,
            "amount": amount,
            "currency": "USD",
            "transaction_id": self._generate_txn_id(),
            "tag": final_tag
        }
        return txn

    def _create_identity(self, name, joint_user=None):
        names = [name]
        if joint_user:
            names.append(joint_user)
        return {
            "names": names,
            "emails": [{"data": f"{name.replace(' ', '.').lower()}@testmail.com", "primary": True, "type": "primary"}],
            "addresses": [{
                "primary": True,
                "data": {
                    "country": "US",
                    "city": "Washington",
                    "street": "175 13th St",
                    "postal_code": "20013",
                    "region": "DC"
                }
            }]
        }

def main():
    parser = argparse.ArgumentParser(description="Generate Plaid-like datasets for Mortgage Benchmark.")
    parser.add_argument("--ulad-template", type=str, default="data/ulad_template.json", help="Input template file")
    parser.add_argument("-n", "--number", type=int, default=1, help="Number of datasets to generate")
    parser.add_argument("-o", "--output", type=str, default="generated_data", help="Output directory")
    
    import sys
    
    # Check if arguments were provided, otherwise ask interactively
    if len(sys.argv) == 1:
        try:
            num_input = input("Enter the number of datasets to generate (default 1): ").strip()
            args_number = int(num_input) if num_input else 1
            output_input = input("Enter output directory (default 'generated_data'): ").strip()
            args_output = output_input if output_input else "generated_data"
        except ValueError:
            print("Invalid number entered. Using default 1.")
            args_number = 1
            args_output = "generated_data"
    else:
        args = parser.parse_args()
        args_number = args.number
        args_output = args.output
    
    if not os.path.exists(args_output):
        os.makedirs(args_output)
        
    for i in range(args_number):
        plaid_generator = PlaidGenerator()
        plaid_data = plaid_generator.generate_single_dataset()
        ulad_generator = UladGenerator(args.ulad_template, plaid_data, args_output)
        ulad_data = ulad_generator.generate_ulad()

        plaid_filename = f"plaid_{plaid_data['seed']}.json"
        plaid_filepath = os.path.join(args_output, plaid_filename)
        ulad_filename = f"ulad_{plaid_data['seed']}.json"
        ulad_filepath = os.path.join(args_output, ulad_filename)
        
        with open(plaid_filepath, 'w') as f:
            json.dump(plaid_data, f, indent=2)
        with open(ulad_filepath, 'w') as f:
            json.dump(ulad_data, f, indent=2)
        print(f"Generated {plaid_filepath} and {ulad_filepath}")

if __name__ == "__main__":
    main()
