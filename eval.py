"""
Evaluation script for foundational models on bank statement question-answering tasks.


Features:
    - Preprocesses Excel data into JSONL format with bank statements and ULAD DU data
    - Evaluates models with configurable trials for statistical reliability
    - Calculates exact match accuracy and F1 scores for different answer types
    - Tracks token usage and costs per model
    - Generates detailed markdown summaries with error analysis
    - Supports filtering by loan IDs, row indexes, and downsampling

Supported Models:
    - OpenAI: gpt-5
    - Anthropic: claude-opus-4-1-20250805, claude-sonnet-4-5, claude-3-7-sonnet-20250219
    - Google: gemini-2.5-flash, gemini-2.5-pro, gemini-3-pro-preview
    - Solo: solo (special model with two-stage processing)

    To add a new model, add the model to the `MODEL_PRICING` dictionary 

Answer Types:
    - boolean: Yes/no questions
    - txn_id_list: Questions requiring transaction ID lists
    - account_id_list: Questions requiring account number lists (last 4 digits)

Usage:
    python eval.py --model_id gpt-5 --trials 3 --use_domain_expertise

See --help for full list of command-line options.
"""

import ast
import datetime
import os
import json
import argparse
import random
import re
from collections import defaultdict
import pandas as pd
import concurrent.futures
import threading
import time
from decimal import Decimal, InvalidOperation
from llm import call_llm_wrapper, clear_messages
from agents import SoloAgent, BaselineAgent, ExperimentalAgent


# Pricing per million tokens (as of November 2025)
MODEL_PRICING = {
    "gpt-5": {"input": 1.25, "output": 10.00},
    "claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},  # Using Sonnet 4 pricing
    "solo": {"input": 0.0, "output": 0.0},  # No cost for solo
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-3-pro-preview": {"input": 1.25, "output": 10.00},
}

domain_expertise = "A large deposit is defined as exceeding 50% of the borrower's total monthly qualifying income."

prompt = "{question}\n\nBank Statement: {context}\n\nULAD DU: {ulad_du}\n\nAnswer the question. {domain_expertise}Do not think out loud. {answer_instruction}."

model_answer_instruction = {
    "txn_id_list": "Describe the relevant transactions in text (titles/amounts/dates); do not guess or output transaction IDs.",
    "boolean": "Answer with yes or no.",
    "account_id_list": "Identify the relevant accounts in text (names/descriptions/last4 digits); do not guess or output account IDs.",
}

CLEANUP_MODEL_ID = "gpt-5"

cleaning_answer_instruction = {
    "txn_id_list": 'Return ONLY a valid JSON list of matching TransactionID values, e.g. `["plaid-2-00037", "plaid-2-00049"]`, or [] if there are no IDs.',
    "boolean": "Return only yes or no. DO NOT output any other text.",
    "account_id_list": 'Return ONLY a valid JSON list of account numbers (last 4 digits only). e.g. `["1234", "5678"]`, or [] if there are no account numbers.',
}

LOAN_IDXS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LOAN_IDS = [
    "85670492709",  # test case 1
    "86881713506",  # test case 2
    "84192307554",  # test case 3
    "80731120165",  # test case 4
    "89811904866",  # test case 5
    "86744679655",  # test case 6
    "81613557991",  # test case 7
    "84192307554",  # test case 8 (same as test case 3, has no questions)
    "83352063666",  # test case 9
    "81301535410",  # test case 10
]

test_case_map = dict(zip(LOAN_IDXS, LOAN_IDS))
loan_id_to_test_case = {loan_id: test_case for test_case, loan_id in test_case_map.items()}

answer_type_map = {
    "boolean": "boolean",
    "yes": "boolean",
    "no": "boolean",
    "true": "boolean",
    "false": "boolean",
    "y": "boolean",
    "n": "boolean",
    "id_list": "txn_id_list",
    "id_list_account": "account_id_list",
}


def load_plaid_bank_statement(test_case_number):
    """Load the plaid_with_ids file for a given test case number."""
    base_dir = f"data/Test Case {test_case_number} Docs"
    plaid_with_ids_path = os.path.join(base_dir, "plaid_with_ids.json")
    if os.path.exists(plaid_with_ids_path):
        with open(plaid_with_ids_path, "r") as f:
            return json.load(f)

    primary_plaid_path = os.path.join(base_dir, "plaid.json")
    if os.path.exists(primary_plaid_path):
        with open(primary_plaid_path, "r") as f:
            return json.load(f)

    fallback_files = sorted(
        [
            f
            for f in os.listdir(base_dir)
            if f.lower().startswith("plaid") and f.lower().endswith(".json")
        ]
    )
    if fallback_files:
        payloads = []
        for fname in fallback_files:
            path = os.path.join(base_dir, fname)
            with open(path, "r") as f:
                payloads.append(json.load(f))
        if len(payloads) == 1:
            return payloads[0]
        return {"plaid_files": payloads}

    raise FileNotFoundError(
        f"No plaid JSON found for Test Case {test_case_number} in {base_dir}"
    )


def iter_plaid_accounts(bank_statement):
    """Yield account objects regardless of single or multi-file plaid payloads."""
    if not bank_statement:
        return
    if isinstance(bank_statement, dict) and "override_accounts" in bank_statement:
        for account in bank_statement.get("override_accounts", []):
            yield account
    elif isinstance(bank_statement, dict) and "plaid_files" in bank_statement:
        for payload in bank_statement.get("plaid_files", []):
            for account in payload.get("override_accounts", []):
                yield account


def flatten_plaid_transactions(bank_statement):
    """Flatten plaid transactions to a simple list for prompting/cleanup."""
    flat = []
    for account in iter_plaid_accounts(bank_statement) or []:
        account_numbers = account.get("numbers", {}) or {}
        account_last4 = account_numbers.get("account")
        if account_last4:
            account_last4 = account_last4[-4:]

        base = {
            "account_type": account.get("type"),
            "account_subtype": account.get("subtype"),
            "account_last4": account_last4,
        }
        for txn in account.get("transactions", []):
            entry = {"transaction_id": txn.get("transaction_id")}
            entry.update(base)
            entry.update(
                {
                    "description": txn.get("description"),
                    "amount": txn.get("amount"),
                    "currency": txn.get("currency"),
                    "date_transacted": txn.get("date_transacted"),
                    "date_posted": txn.get("date_posted"),
                }
            )
            flat.append(entry)
    return flat


def extract_plaid_accounts(bank_statement):
    """Extract account metadata (including last 4 digits) from plaid payloads."""
    accounts = []
    for account in iter_plaid_accounts(bank_statement) or []:
        numbers = account.get("numbers", {}) or {}
        account_number = numbers.get("account")
        last4 = account_number[-4:] if account_number and len(account_number) >= 4 else None
        accounts.append(
            {
                "name": account.get("name") or account.get("type"),
                "type": account.get("type"),
                "subtype": account.get("subtype"),
                "account_number_last4": last4,
            }
        )
    return accounts


def _normalize_amount(amount):
    """Normalize numeric amounts for matching (absolute value, 2 decimal places)."""
    if amount is None:
        return None
    try:
        quantized = Decimal(str(amount)).copy_abs().quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None
    return str(quantized)


def _normalize_description(description):
    """Canonicalize descriptions for fuzzy matching."""
    if not description:
        return None
    cleaned = re.sub(r"\s+", " ", str(description).strip().lower())
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return cleaned or None


def _normalize_date(date_value):
    """Normalize date strings to YYYY-MM-DD."""
    if not date_value:
        return None
    date_str = str(date_value).strip()
    if not date_str:
        return None
    return date_str[:10]


def _parse_solo_transactions(raw_txn_info):
    """Extract Solo transaction dicts from the unstructured info block."""
    if not raw_txn_info:
        return []
    try:
        start = raw_txn_info.find("{")
        end = raw_txn_info.rfind("}")
        if start == -1 or end == -1:
            return []
        payload = raw_txn_info[start : end + 1]
        data = ast.literal_eval(payload)
    except (ValueError, SyntaxError):
        return []

    transactions = []

    def _collect(obj):
        if isinstance(obj, dict):
            if "Transactions" in obj and isinstance(obj["Transactions"], list):
                for item in obj["Transactions"]:
                    if isinstance(item, dict):
                        transactions.append(item)
            for value in obj.values():
                _collect(value)
        elif isinstance(obj, list):
            for entry in obj:
                _collect(entry)

    _collect(data)
    return transactions


def _match_plaid_transaction(solo_entry, plaid_transactions, used_ids):
    """Match a solo transaction entry to a plaid transaction_id."""
    solo_desc = _normalize_description(solo_entry.get("Description"))
    solo_amount = _normalize_amount(solo_entry.get("Amount"))
    solo_dates = []
    for key in ("Date", "ParsedDate"):
        normalized = _normalize_date(solo_entry.get(key))
        if normalized:
            solo_dates.append(normalized)

    best_candidate = None
    best_score = 0
    for txn in plaid_transactions:
        plaid_id = txn.get("transaction_id")
        if not plaid_id or plaid_id in used_ids:
            continue
        plaid_desc = _normalize_description(txn.get("description"))
        plaid_amount = _normalize_amount(txn.get("amount"))
        plaid_dates = [
            _normalize_date(txn.get("date_transacted")),
            _normalize_date(txn.get("date_posted")),
        ]

        score = 0
        if solo_amount and plaid_amount and solo_amount == plaid_amount:
            score += 3
        if solo_desc and plaid_desc and solo_desc == plaid_desc:
            score += 4
        if solo_dates:
            overlap = set(filter(None, plaid_dates)) & set(solo_dates)
            if overlap:
                score += 3

        if score > best_score:
            best_score = score
            best_candidate = plaid_id

    # Require at least a moderate score to avoid incorrect mappings
    if best_score >= 4:
        return best_candidate
    return None


def _find_plaid_by_hint(token, plaid_transactions, used_ids):
    """Fallback: try to map raw token to plaid ID via description substring."""
    normalized_token = _normalize_description(token)
    if not normalized_token:
        return None

    for txn in plaid_transactions:
        plaid_id = txn.get("transaction_id")
        if not plaid_id or plaid_id in used_ids:
            continue
        desc = _normalize_description(txn.get("description"))
        if desc and normalized_token in desc and abs(len(desc) - len(normalized_token)) <= 10:
            return plaid_id
    return None


def _map_predicted_ids_to_plaid(predicted_ids, plaid_transactions, solo_txn_info=None):
    """Convert predicted IDs to plaid IDs when possible."""
    if not predicted_ids:
        return []

    solo_lookup = {}
    if solo_txn_info:
        for entry in _parse_solo_transactions(solo_txn_info):
            solo_id = entry.get("TransactionID")
            if solo_id:
                solo_lookup[str(solo_id).strip()] = entry

    normalized_ids = []
    used_plaid_ids = set()

    for raw_id in predicted_ids:
        token = str(raw_id).strip()
        if not token or token.lower() == "none":
            continue
        if token.startswith("plaid-"):
            if token not in used_plaid_ids:
                normalized_ids.append(token)
                used_plaid_ids.add(token)
            continue

        mapped_id = None
        if token in solo_lookup:
            mapped_id = _match_plaid_transaction(
                solo_lookup[token], plaid_transactions, used_plaid_ids
            )
        if mapped_id is None:
            mapped_id = _find_plaid_by_hint(token, plaid_transactions, used_plaid_ids)

        if mapped_id:
            normalized_ids.append(mapped_id)
            used_plaid_ids.add(mapped_id)
        else:
            normalized_ids.append(token)

    # Preserve order while removing duplicate plaid IDs
    deduped = []
    seen = set()
    for item in normalized_ids:
        if item.startswith("plaid-"):
            if item in seen:
                continue
            seen.add(item)
        deduped.append(item)
    return deduped


def _dedupe_preserving_order(items):
    """Deduplicate while maintaining the original order."""
    seen = set()
    ordered = []
    for item in items or []:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def normalize_transaction_answer(cleaned_answer, answer_type, plaid_transactions, solo_txn_info=None):
    """Normalize cleaned transaction answers to ensure plaid ID output."""
    if answer_type != "txn_id_list":
        return cleaned_answer

    if cleaned_answer is None:
        return "[]"

    cleaned_answer = cleaned_answer.strip()
    list_match = re.search(r"\[[^\]]*\]", cleaned_answer, re.DOTALL)
    parsed_list = None

    if list_match:
        try:
            candidate = json.loads(list_match.group(0))
            if isinstance(candidate, list):
                parsed_list = candidate
        except json.JSONDecodeError:
            parsed_list = None

    if parsed_list is None:
        direct_ids = re.findall(r"plaid-[a-zA-Z0-9-]+", cleaned_answer)
        if direct_ids:
            parsed_list = direct_ids
        else:
            normalized_text = cleaned_answer.strip().lower()
            if re.match(r"^\s*no\b", normalized_text):
                return "[]"
            if any(
                phrase in normalized_text
                for phrase in [
                    "no matching",
                    "no transactions",
                    "no deposits",
                    "no relevant transactions",
                    "none",
                    "not found",
                    "does not appear",
                ]
            ):
                return "[]"
            return cleaned_answer

    mapped_ids = _map_predicted_ids_to_plaid(
        parsed_list, plaid_transactions, solo_txn_info=solo_txn_info
    )
    return json.dumps(mapped_ids)


def normalize_account_answer(cleaned_answer, valid_last4):
    """Normalize cleaned account answers to ensure last-4 digit output."""
    if cleaned_answer is None:
        return "[]"

    cleaned_answer = cleaned_answer.strip()
    list_match = re.search(r"\[[^\]]*\]", cleaned_answer, re.DOTALL)
    parsed_list = None

    if list_match:
        try:
            candidate = json.loads(list_match.group(0))
            if isinstance(candidate, list):
                parsed_list = [str(token).strip() for token in candidate if str(token).strip()]
        except json.JSONDecodeError:
            parsed_list = None

    if parsed_list is None:
        valid_set = {str(x).strip() for x in valid_last4 or [] if str(x).strip()}
        digit_matches = []
        for match in re.findall(r"\b\d{4}\b", cleaned_answer):
            if not valid_set or match in valid_set:
                digit_matches.append(match)
        if digit_matches:
            parsed_list = digit_matches
        else:
            normalized_text = cleaned_answer.lower()
            if any(
                phrase in normalized_text
                for phrase in [
                    "no accounts",
                    "no matching",
                    "no relevant accounts",
                    "none",
                    "not found",
                    "does not appear",
                ]
            ):
                return "[]"
            return cleaned_answer

    normalized = [token for token in parsed_list if token]
    return json.dumps(_dedupe_preserving_order(normalized))


def load_ulad_xml(test_case_number):
    """Load the ULAD DU XML file for a given test case number."""
    xml_path = f"data/Test Case {test_case_number} Docs/ulad.xml"
    with open(xml_path, "r") as f:
        return f.read()


def map_answer_type_from_column(excel_answer_type):
    """Map Excel Answer Type column values to internal answer types."""
    excel_answer_type = str(excel_answer_type).strip().lower()

    if excel_answer_type not in answer_type_map:
        raise ValueError(
            f"Unknown Answer Type value: {excel_answer_type}. "
            f"Expected one of: {list(answer_type_map.keys())}"
        )

    return answer_type_map[excel_answer_type]


def preprocess_data(
    question_col,
    excel_path="data/Labeled Questions and Answers.xlsx",
    metadata_path="data/benchmark_dataset_metadata.json",
    test_cases_path="data/benchmark_dataset_test_cases.jsonl",
):
    """Preprocess the Excel data into benchmark metadata and test case files."""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    df_p1 = pd.read_excel(excel_path, sheet_name="1-5", header=0)
    df_p2 = pd.read_excel(excel_path, sheet_name="6-10", header=1)
    df = pd.concat([df_p1, df_p2], ignore_index=True)

    print(f"Loaded {len(df)} rows from Excel file")
    print(f"Columns: {list(df.columns)}")

    question_counters = defaultdict(int)
    metadata_entries = []

    for _, row in df.iterrows():
        answer = str(row["Revised Answer V2"])
        test_case_number = row["Test Case Number"]
        if pd.isna(row["Answer Type"]):
            continue
        answer_type = map_answer_type_from_column(row["Answer Type"])
        question = row[question_col]

        if answer_type not in ["boolean", "txn_id_list", "account_id_list"]:
            raise ValueError(f"Invalid answer type: {answer_type}")

        if pd.isna(question) or str(question).strip().lower() in ["", "n/a", "na"]:
            continue

        test_case_number = int(test_case_number)
        question_counters[test_case_number] += 1
        question_id = f"{test_case_number}-{question_counters[test_case_number]}"

        pii_value = False
        if not pd.isna(row["PII"]):
            pii_value = str(row["PII"]).strip().lower() in {"true", "1", "yes", "y"}

        metadata_entries.append(
            {
                "question_id": question_id,
                "loan_id": test_case_map[test_case_number],
                "test_case_number": test_case_number,
                "question": str(question).strip(),
                "answer": answer,
                "answer_type": answer_type,
                "pii": pii_value,
            }
        )

    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata_entries, metadata_file, indent=2)

    test_case_numbers = sorted(test_case_map.keys())
    with open(test_cases_path, "w") as test_cases_file:
        for test_case_number in test_case_numbers:
            try:
                bank_statement = load_plaid_bank_statement(test_case_number)
            except FileNotFoundError as exc:
                print(f"Warning: {exc}")
                bank_statement = None
            try:
                ulad_du = load_ulad_xml(test_case_number)
            except FileNotFoundError:
                print(
                    f"Warning: ULAD DU file not found for Test Case {test_case_number}"
                )
                ulad_du = None

            entry = {
                "test_case_number": test_case_number,
                "bank_statement": bank_statement,
                "ulad_du": ulad_du,
            }
            test_cases_file.write(json.dumps(entry) + "\n")

    print(f"Successfully created {metadata_path}")
    print(f"Successfully created {test_cases_path}")
    return True


def load_dataset(
    metadata_path="data/benchmark_dataset_metadata.json",
    test_cases_path="data/benchmark_dataset_test_cases.jsonl",
):
    """Load the benchmark dataset by combining metadata and test case contexts."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not os.path.exists(test_cases_path):
        raise FileNotFoundError(f"Test cases file not found: {test_cases_path}")

    with open(metadata_path, "r") as metadata_file:
        metadata_entries = json.load(metadata_file)

    test_case_map_ctx = {}
    with open(test_cases_path, "r") as test_cases_file:
        for line in test_cases_file:
            if not line.strip():
                continue
            entry = json.loads(line)
            test_case_number = entry.get("test_case_number")
            if test_case_number is None:
                continue
            test_case_map_ctx[int(test_case_number)] = entry

    dataset = []
    for item in metadata_entries:
        test_case_number = int(item["test_case_number"])
        test_case_entry = test_case_map_ctx.get(test_case_number)
        if not test_case_entry:
            raise ValueError(
                f"Missing test case data for Test Case {test_case_number}."
            )
        bank_statement = test_case_entry.get("bank_statement")
        if not bank_statement:
            try:
                bank_statement = load_plaid_bank_statement(test_case_number)
            except FileNotFoundError as exc:
                print(f"Warning: {exc}")
                bank_statement = None
        ulad_du = test_case_entry.get("ulad_du")
        if ulad_du is None:
            try:
                ulad_du = load_ulad_xml(test_case_number)
            except FileNotFoundError:
                ulad_du = None
        dataset.append(
            {
                **item,
                "bank_statement": bank_statement,
                "ulad_du": ulad_du,
            }
        )

    return pd.DataFrame(dataset)

def get_answer_type(answer):
    """Get the answer type from the answer based on its format."""
    normalized_answer = answer.strip().lower()
    if normalized_answer in {"yes", "no"}:
        return "boolean"
    if normalized_answer == "none":
        return "txn_id_list"
    tokens = [token.strip() for token in answer.split(",") if token.strip()]
    if not tokens:
        raise ValueError("Empty answer provided")
    first_token = tokens[0]
    if all(token.isdigit() and len(token) == 4 for token in tokens):
        return "account_id_list"
    if all(len(token) == 20 for token in tokens):
        return "txn_id_list"
    if all(token.startswith("plaid-") for token in tokens):
        return "txn_id_list"
    raise ValueError(f"Unknown answer type: {answer}")


def create_agent(model_id, loan_id, cleared_loans, cleared_loans_lock, wait_for_loan_gap_func, prompt_template, model_type="baseline"):
    """Factory function to create the appropriate agent based on model_type."""
    if model_type == "solo":
        return SoloAgent(model_id, loan_id, cleared_loans, cleared_loans_lock, wait_for_loan_gap_func)
    elif model_type == "experimental":
        return ExperimentalAgent(model_id, loan_id, cleared_loans, cleared_loans_lock, wait_for_loan_gap_func, prompt_template)
    else:
        return BaselineAgent(model_id, loan_id, cleared_loans, cleared_loans_lock, wait_for_loan_gap_func, prompt_template)

def evaluate_model(
    model_id,
    df,
    use_domain_expertise,
    downsample_size=None,
    offset=0,
    trials=None,
    model_type="baseline",
):
    """Evaluate a model on the dataset."""
    domain_expertise_str = (
        f"Domain Expertise: {domain_expertise}\n\n" if use_domain_expertise else ""
    )

    # Apply offset first
    if offset > 0:
        df = df.iloc[offset:].reset_index(drop=True)

    # Then apply downsample if specified
    if downsample_size:
        df = df.sample(n=min(downsample_size, len(df)), random_state=42).reset_index(
            drop=True
        )

    results = [None] * len(df)
    exact_match_count = 0
    f1_sum = 0.0
    total_count = len(df)
    total_input_tokens = 0
    total_output_tokens = 0
    total_valid_trials = 0
    cleared_loans = set()
    cleared_loans_lock = threading.Lock()
    last_question_time = {}
    last_question_lock = threading.Lock()
    aggregate_lock = threading.Lock()

    def wait_for_loan_gap(loan_id, gap_seconds=50):
        """Enforce a minimum gap between questions for the same loan."""
        now = time.monotonic()
        with last_question_lock:
            last_time = last_question_time.get(loan_id, 0)
            elapsed = now - last_time
            wait = max(0.0, gap_seconds - elapsed)
            # Reserve the slot so other threads respect this timing
            last_question_time[loan_id] = now + wait
        if wait > 0:
            time.sleep(wait)

    def process_single_row(row_idx, row):
        """Process one row (one loan) and return result plus metrics."""
        loan_id = row["loan_id"]
        question = row["question"]
        gt_answer = row["answer"]
        answer_type = row.get("answer_type")
        if answer_type:
            answer_type = str(answer_type).strip().lower()
        else:
            answer_type = get_answer_type(row["answer"])
        if answer_type not in model_answer_instruction:
            raise ValueError(
                f"Unsupported answer_type '{answer_type}' for question_id {row.get('question_id')}"
            )
        test_case_number = row.get("test_case_number") or loan_id_to_test_case.get(
            loan_id
        )
        plaid_bank_statement_obj = row.get("bank_statement")
        if not plaid_bank_statement_obj:
            try:
                plaid_bank_statement_obj = load_plaid_bank_statement(
                    int(test_case_number)
                )
            except FileNotFoundError:
                plaid_bank_statement_obj = {}

        plaid_transactions_flat = flatten_plaid_transactions(plaid_bank_statement_obj)
        transactions_json = json.dumps(plaid_transactions_flat, indent=2)
        accounts_summary = extract_plaid_accounts(plaid_bank_statement_obj)
        accounts_json = json.dumps(accounts_summary, indent=2)
        account_last4_values = [
            account.get("account_number_last4")
            for account in accounts_summary
            if account.get("account_number_last4")
        ]

        if not test_case_number:
            raise ValueError(f"Missing test_case_number for loan_id {loan_id}")

        bank_statement_obj_to_use = json.loads(json.dumps(plaid_bank_statement_obj))

        bank_statement = json.dumps(bank_statement_obj_to_use, indent=2)
        pii_raw = row["pii"] if "pii" in row else False
        is_pii = bool(pii_raw) if not pd.isna(pii_raw) else False
        ulad_du = row["ulad_du"]

        print(f"is_pii: {is_pii}")

        model_answer_instruction_str = model_answer_instruction[answer_type]
        cleaning_answer_instruction_str = cleaning_answer_instruction[answer_type]

        # Create agent instance for this loan
        agent = create_agent(
            model_id, loan_id, cleared_loans, cleared_loans_lock, 
            wait_for_loan_gap, prompt, model_type=model_type
        )
        
        # Setup loan (clears messages, enforces spacing for solo)
        agent.setup_loan()

        predicted_answers = []
        raw_answers = []
        solo_answers = [] if isinstance(agent, SoloAgent) else None
        trial_metrics = []
        row_input_tokens = 0
        row_output_tokens = 0

        for trial_idx in range(trials):
            try:
                # First call: model produces a textual answer (no IDs expected)
                formatted_prompt = agent.get_initial_prompt(
                    question, bank_statement, ulad_du, domain_expertise_str, model_answer_instruction_str
                )
                raw_answer, input_tok, output_tok = call_llm_wrapper(
                    model_id=model_id,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    loan_id=loan_id,
                )
                
                if isinstance(agent, SoloAgent):
                    solo_answers.append(raw_answer)

                row_input_tokens += input_tok
                row_output_tokens += output_tok

                # Cleaning / normalization stage (extract IDs/accounts using agent methods)
                if answer_type == "boolean":
                    processed_raw_answer, cleaned_answer, clean_in_tok, clean_out_tok = agent.process_boolean(
                        question, raw_answer, loan_id
                    )
                elif answer_type == "txn_id_list":
                    processed_raw_answer, cleaned_answer, clean_in_tok, clean_out_tok = agent.process_txn_id_list(
                        question, raw_answer, loan_id, transactions_json, 
                        cleaning_answer_instruction_str, plaid_transactions_flat
                    )
                elif answer_type == "account_id_list":
                    processed_raw_answer, cleaned_answer, clean_in_tok, clean_out_tok = agent.process_account_id_list(
                        question, raw_answer, loan_id, accounts_json, 
                        cleaning_answer_instruction_str, account_last4_values
                    )
                else:
                    raise ValueError(f"Invalid answer type: {answer_type}")

                row_input_tokens += clean_in_tok
                row_output_tokens += clean_out_tok

                predicted_answers.append(cleaned_answer)
                raw_answers.append(processed_raw_answer)

                if is_pii:
                    metrics = {"exact_match": None, "f1_score": None}
                else:
                    metrics = is_correct(cleaned_answer, answer_type, gt_answer)
                trial_metrics.append(metrics)
            except Exception as e:
                print(f"Error encountered for loan_id {loan_id}, question: {question}")
                print(f"Error details: {str(e)}")
                
                error_msg = f"ERROR: {str(e)}"
                predicted_answers.append(error_msg)
                raw_answers.append(error_msg)
                if isinstance(agent, SoloAgent):
                    solo_answers.append(error_msg)
                
                trial_metrics.append({"exact_match": None, "f1_score": None})

        valid_trials = 0
        current_trial_f1_sum = 0
        current_trial_exact_match_count = 0
        
        for metrics in trial_metrics:
            if metrics["exact_match"] is not None:
                valid_trials += 1
                if metrics["exact_match"]:
                    current_trial_exact_match_count += 1
                current_trial_f1_sum += metrics["f1_score"]
        
        result = {
            "loan_id": loan_id,
            "question": question,
            "true_answer": gt_answer,
            "predicted_answer": predicted_answers,
            "raw_answer": raw_answers,
            "answer_type": answer_type,
            "exact_match": [m["exact_match"] for m in trial_metrics],
            "f1_score": [m["f1_score"] for m in trial_metrics],
        }

        if isinstance(agent, SoloAgent):
            result["solo_answer"] = solo_answers

        return (
            row_idx,
            result,
            current_trial_exact_match_count,
            current_trial_f1_sum,
            valid_trials,
            row_input_tokens,
            row_output_tokens,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(10, len(df))
    ) as executor:
        future_to_idx = {
            executor.submit(process_single_row, idx, row): idx
            for idx, (_, row) in enumerate(df.iterrows())
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            (
                row_idx,
                result,
                row_exact_match_count,
                row_f1_sum,
                valid_trials,
                row_input_tokens,
                row_output_tokens,
            ) = future.result()

            with aggregate_lock:
                results[row_idx] = result
                exact_match_count += row_exact_match_count
                f1_sum += row_f1_sum
                total_input_tokens += row_input_tokens
                total_output_tokens += row_output_tokens
                total_valid_trials += valid_trials
                completed += 1

            denom = total_valid_trials if total_valid_trials > 0 else 1
            avg_f1 = f1_sum / denom
            exact_match_rate = exact_match_count / denom
            print(
                f"Progress: {completed}/{total_count} - "
                f"Exact Match: {exact_match_rate:.3f} - Avg F1: {avg_f1:.3f}"
            )

    if total_valid_trials > 0:
        exact_match_accuracy = exact_match_count / total_valid_trials
        avg_f1_accuracy = f1_sum / total_valid_trials
    else:
        exact_match_accuracy = 0
        avg_f1_accuracy = 0

    pricing = MODEL_PRICING[model_id]
    input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
    output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    cost_info = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }

    return results, avg_f1_accuracy, df, cost_info


def calculate_f1_score(pred_set, gt_set):
    """Calculate F1 score between predicted and ground truth sets."""
    if len(pred_set) == 0 and len(gt_set) == 0:
        return 1.0  # Both empty sets are perfect match
    if len(pred_set) == 0 or len(gt_set) == 0:
        return 0.0  # One empty, one not = no overlap

    intersection = len(pred_set & gt_set)
    precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
    recall = intersection / len(gt_set) if len(gt_set) > 0 else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def is_correct(predicted_answer, answer_type, gt_answer):
    gt_answer = str(gt_answer)
    if answer_type == "boolean":
        # Convert answer to yes/no format for comparison
        answer_str = gt_answer.lower()
        if answer_str in ["yes", "true", "1"]:
            expected = "yes"
        elif answer_str in ["no", "false", "0"]:
            expected = "no"
        else:
            print(f"Warning: Invalid boolean answer value: {gt_answer}")
            return {"exact_match": False, "f1_score": 0.0}
        predicted_norm = (
            predicted_answer.strip().strip(".").strip().lower()
        )
        exact_match = predicted_norm == expected
        f1_score = 1.0 if exact_match else 0.0
        return {"exact_match": exact_match, "f1_score": f1_score}
    elif answer_type in ["txn_id_list", "account_id_list"]:
        match = re.search(r"\[(.*)\]", predicted_answer, re.DOTALL)
        if match is None:
            print(
                f"warning: no list found in predicted answer: {predicted_answer} "
                f"for answer: {gt_answer}"
            )
            return {"exact_match": False, "f1_score": 0.0}
        cleaned_answer = match.group(0)
        try:
            pred_as_list = json.loads(cleaned_answer)
        except json.JSONDecodeError:
            print(
                f"warning: invalid JSON in predicted answer: {cleaned_answer} "
                f"for answer: {gt_answer}"
            )
            return {"exact_match": False, "f1_score": 0.0}

        gt_tokens = [str(token).strip().lower() for token in gt_answer.split(",")]
        if len(gt_tokens) == 1 and gt_tokens[0] == "none":
            gt_answer = []
        else:
            gt_answer = [token for token in gt_tokens if token]

        normalized_pred = []
        for token in pred_as_list:
            if not isinstance(token, str):
                token = str(token)
            token = token.strip().lower()
            if token:
                normalized_pred.append(token)

        pred_set = set(normalized_pred)
        gt_set = set(gt_answer)

        exact_match = pred_set == gt_set
        f1_score = calculate_f1_score(pred_set, gt_set)

        return {"exact_match": exact_match, "f1_score": f1_score}
    else:
        raise ValueError(f"Invalid answer type: {answer_type}")


def save_results(
    results,
    f1_accuracy,
    model_id,
    output_dir,
    df,
    downsample_size=None,
    args=None,
    cost_info=None,
):
    """Save evaluation results."""
    # Save results as JSONL with original data plus pred and correct columns
    # Add prediction and correctness columns to the dataframe
    df_with_results = df.copy()
    df_with_results["pred"] = [result["predicted_answer"] for result in results]
    df_with_results["raw_answer"] = [result["raw_answer"] for result in results]
    df_with_results["exact_match"] = [result["exact_match"] for result in results]
    df_with_results["f1_score"] = [result["f1_score"] for result in results]

    # Add solo_answer field for solo model results
    if model_id == "solo":
        df_with_results["solo_answer"] = [
            result["solo_answer"] for result in results
        ]

    jsonl_path = f"{output_dir}/{model_id}_results.jsonl"
    df_with_results.to_json(jsonl_path, orient="records", lines=True)
    print(f"JSONL results saved to {jsonl_path}")

    # Create markdown summary
    summary_path = f"{output_dir}/{model_id}_summary.md"
    with open(summary_path, "w") as f:
        f.write(f"# Evaluation Results: {model_id}\n\n")
        f.write(
            f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        # Add all parser arguments at the top
        if args is not None:
            f.write("## Configuration Parameters\n\n")
            args_dict = vars(args)
            for key, value in args_dict.items():
                f.write(f"- **{key}:** {value}\n")
            f.write("\n")

        f.write(f"**Model:** {model_id}\n")
        if args is not None and hasattr(args, 'model_type'):
            f.write(f"**Agent Type:** {args.model_type}\n")
        f.write("\n")
        f.write(f"**Dataset Size:** {len(results)}")
        if downsample_size:
            f.write(f" (downsampled from original)")
        f.write("\n\n")
        # Calculate exact match accuracy
        exact_match_count = 0
        f1_score_sum = 0.0
        total_trials = 0
        for r in results:
            # Filter out None values (skipped trials)
            valid_exact_matches = [m for m in r["exact_match"] if m is not None]
            valid_f1_scores = [s for s in r["f1_score"] if s is not None]
            
            exact_match_count += sum(valid_exact_matches)
            f1_score_sum += sum(valid_f1_scores)
            total_trials += len(valid_exact_matches)
            
        exact_match_accuracy = (
            exact_match_count / total_trials if total_trials > 0 else 0
        )

        f.write(
            f"**Overall F1 Accuracy:** {f1_accuracy:.3f} ({f1_accuracy*100:.1f}%)\n\n"
        )
        f.write(
            f"**Overall Exact Match Accuracy:** {exact_match_accuracy:.3f} ({exact_match_accuracy*100:.1f}%)\n\n"
        )

        # Calculate metrics for PII==False subset
        # Results list is aligned with df by position (results[i] corresponds to df.iloc[i])
        pii_false_results = []
        for i in range(len(results)):
            if df.iloc[i]["pii"] == False:
                pii_false_results.append(results[i])
        
        if len(pii_false_results) > 0:
            pii_false_exact_match_count = 0
            pii_false_f1_sum = 0.0
            pii_false_total_trials = 0
            for r in pii_false_results:
                # Filter out None values
                valid_exact_matches = [m for m in r["exact_match"] if m is not None]
                valid_f1_scores = [s for s in r["f1_score"] if s is not None]

                pii_false_exact_match_count += sum(valid_exact_matches)
                pii_false_f1_sum += sum(valid_f1_scores)
                pii_false_total_trials += len(valid_exact_matches)

            pii_false_exact_match_accuracy = (
                pii_false_exact_match_count / pii_false_total_trials
                if pii_false_total_trials > 0
                else 0
            )
            pii_false_f1_accuracy = (
                pii_false_f1_sum / pii_false_total_trials
                if pii_false_total_trials > 0
                else 0
            )
            
            f.write("## Metrics for PII==False Subset\n\n")
            f.write(
                f"**F1 Accuracy (PII==False):** {pii_false_f1_accuracy:.3f} ({pii_false_f1_accuracy*100:.1f}%)\n\n"
            )
            f.write(
                f"**Exact Match Accuracy (PII==False):** {pii_false_exact_match_accuracy:.3f} ({pii_false_exact_match_accuracy*100:.1f}%)\n\n"
            )
            f.write(f"**Dataset Size (PII==False):** {len(pii_false_results)}\n\n")

        # Cost information
        if cost_info:
            f.write("## Cost Analysis\n\n")
            f.write(f"**Total Input Tokens:** {cost_info['total_input_tokens']:,}\n\n")
            f.write(f"**Total Output Tokens:** {cost_info['total_output_tokens']:,}\n\n")
            f.write(f"**Input Cost:** ${cost_info['input_cost']:.2f}\n\n")
            f.write(f"**Output Cost:** ${cost_info['output_cost']:.2f}\n\n")
            f.write(f"**Total Cost:** ${cost_info['total_cost']:.2f}\n\n")

        # Accuracy by answer type
        f.write("## Accuracy by Answer Type\n\n")
        answer_type_stats = {}
        for result in results:
            answer_type = result["answer_type"]
            if answer_type not in answer_type_stats:
                answer_type_stats[answer_type] = {
                    "exact_match": 0,
                    "f1_sum": 0.0,
                    "total": 0,
                }
            
            valid_exact_matches = [m for m in result["exact_match"] if m is not None]
            valid_f1_scores = [s for s in result["f1_score"] if s is not None]
            
            answer_type_stats[answer_type]["total"] += len(valid_exact_matches)
            answer_type_stats[answer_type]["exact_match"] += sum(valid_exact_matches)
            answer_type_stats[answer_type]["f1_sum"] += sum(valid_f1_scores)

        for answer_type, stats in answer_type_stats.items():
            if stats["total"] > 0:
                exact_acc = stats["exact_match"] / stats["total"]
                avg_f1 = stats["f1_sum"] / stats["total"]
            else:
                exact_acc = 0
                avg_f1 = 0
                
            f.write(
                f"- **{answer_type}:** Exact Match: {exact_acc:.3f} "
                f"({exact_acc*100:.1f}%) - {stats['exact_match']}/{stats['total']}\n"
            )
            f.write(f"  F1 Score: {avg_f1:.3f} ({avg_f1*100:.1f}%)\n")

        # Add detailed results section for solo model (errors only)
        if model_id == "solo":
            f.write("\n## Detailed Results with Raw Solo Output (Errors Only)\n\n")
            # Only count as error if valid trial and not exact match
            errors = []
            for r in results:
                has_error = False
                for i, match in enumerate(r["exact_match"]):
                    if match is not None and not match:
                        has_error = True
                        break
                if has_error:
                    errors.append(r)
                    
            for i, result in enumerate(errors):
                f.write(f"### Error {i+1}\n")
                f.write(f"**Loan ID:** {result['loan_id']}\n")
                f.write(f"**Question:** {result['question']}\n")
                f.write(
                    f"**Expected:** {result['true_answer']} ({result['answer_type']})\n"
                )
                f.write(f"**Predicted (cleaned):** {result['predicted_answer']}\n")
                f.write(f"**Raw Answer:** {result['raw_answer']}\n")
                f.write(f"**F1 Score:** {result['f1_score']}\n")

                if "solo_answer" in result and result["solo_answer"] is not None:
                    f.write(
                        f"**Raw Solo Output:**\n```\n{result['solo_answer']}\n```\n"
                    )

                f.write("\n")

        # Only show sample errors section for non-solo models (solo models already have detailed results above)
        if model_id != "solo":
            f.write("\n## Sample Errors\n\n")
            # Only count as error if valid trial and not exact match
            errors = []
            for r in results:
                has_error = False
                for i, match in enumerate(r["exact_match"]):
                    if match is not None and not match:
                        has_error = True
                        break
                if has_error:
                    errors.append(r)

            for i, error in enumerate(errors):
                f.write(f"### Error {i+1}\n")
                f.write(f"**Loan ID:** {error['loan_id']}\n")
                f.write(f"**Question:** {error['question']}\n")
                f.write(
                    f"**Expected:** {error['true_answer']} ({error['answer_type']})\n"
                )
                f.write(f"**Predicted (cleaned):** {error['predicted_answer']}\n")
                f.write(f"**Raw Answer:** {error['raw_answer']}\n")
                f.write("\n")

    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on bank statement QA dataset"
    )
    parser.add_argument(
        "--model_id", type=str, default="gpt-5", help="Model ID to evaluate"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="baseline",
        choices=["solo", "baseline", "experimental"],
        help="Type of agent to use: solo, baseline, or experimental",
    )
    parser.add_argument("--downsample_size", type=int, help="Number of samples to use")
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="data/benchmark_dataset_metadata.json",
        help="Path to question metadata JSON file",
    )
    parser.add_argument(
        "--test_cases_path",
        type=str,
        default="data/benchmark_dataset_test_cases.jsonl",
        help="Path to test case contexts JSONL file",
    )
    parser.add_argument(
        "--use_domain_expertise", action="store_true", help="Use domain expertise"
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        default="data/Labeled Questions and Answers.xlsx",
        help="Path to Excel file for preprocessing",
    )
    parser.add_argument(
        "--skip_preprocessing", action="store_true", help="Skip preprocessing step"
    )
    parser.add_argument(
        "--results_dir", default="eval", help="Directory name for results"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of samples to skip from the beginning",
    )
    parser.add_argument(
        "--trials", type=int, default=1, help="Number of trials to run"
    )
    parser.add_argument(
        "--loan_ids",
        type=str,
        nargs="*",
        default=None,
        help="Filter to only these loan IDs. Can pass numeric indices (0, 1, 2, etc.) or full loan ID strings. If not provided, use all loan IDs.",
    )
    parser.add_argument(
        "--question_col",
        type=str,
        default="Question",
        choices=["Question", "Rephrased Question"],
    )
    parser.add_argument(
        "--row_indexes",
        type=int,
        nargs="+",
        default=None,
        help="Run evaluation only on these row indexes (0-based). Can pass multiple indexes, e.g., --row_indexes 0 5 10",
    )

    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"results/{args.results_dir}/{args.question_col.replace(' ', '_')}/{now}"
    os.makedirs(output_dir, exist_ok=True)

    metadata_exists = os.path.exists(args.metadata_path)
    test_cases_exists = os.path.exists(args.test_cases_path)

    if not metadata_exists or not test_cases_exists:
        if args.skip_preprocessing:
            raise FileNotFoundError(
                "Benchmark dataset files are missing but --skip_preprocessing was set."
            )
        print("Preprocessing data...")
        if not preprocess_data(
            args.question_col,
            args.excel_path,
            args.metadata_path,
            args.test_cases_path,
        ):
            print("Preprocessing failed. Exiting.")
            return
    else:
        print(
            f"Found existing benchmark files at {args.metadata_path} and {args.test_cases_path}. "
            "Skipping preprocessing."
        )

    print(f"\n{'='*50}")
    print(f"Running evaluation")
    print(f"{'='*50}")

    print(f"Loading dataset...")
    dataset = load_dataset(args.metadata_path, args.test_cases_path)
    print(f"Loaded {len(dataset)} samples")

    if len(dataset) == 0:
        print(f"No samples found. Exiting.")
        return

    # Filter by loan IDs if provided
    starting_len = len(dataset)
    if args.loan_ids:
        # Convert numeric indices to loan IDs
        filtered_loan_ids = []
        for loan_id_arg in args.loan_ids:
            # Check if it's a numeric index
            idx = int(loan_id_arg)
            # Convert 0-indexed to loan ID
            filtered_loan_ids.append(test_case_map[idx])

        dataset = dataset[dataset["loan_id"].isin(filtered_loan_ids)]
        print(f"Filtered from {starting_len} to {len(dataset)} samples by loan IDs")

    # Filter by row indexes if provided
    if args.row_indexes:
        starting_len = len(dataset)
        # Validate row indexes are within bounds
        max_index = len(dataset) - 1
        invalid_indexes = [
            idx for idx in args.row_indexes if idx < 0 or idx > max_index
        ]
        if invalid_indexes:
            raise ValueError(
                f"Invalid row indexes: {invalid_indexes}. "
                f"Valid range is 0-{max_index}"
            )
        # Filter dataset to only the specified row indexes
        dataset = dataset.iloc[args.row_indexes].reset_index(drop=True)
        print(
            f"Filtered from {starting_len} to {len(dataset)} samples "
            f"by row indexes: {args.row_indexes}"
        )

    if args.downsample_size:
        print(f"Downsampling to {args.downsample_size} samples")
        random.seed(42)  # For reproducibility

    print(f"Evaluating model: {args.model_id} (type: {args.model_type})")
    results, f1_accuracy, df, cost_info = evaluate_model(
        args.model_id,
        dataset,
        args.use_domain_expertise,
        args.downsample_size,
        args.offset,
        args.trials,
        args.model_type,
    )

    # Calculate exact match accuracy for display
    exact_match_count = 0
    total_trials = 0
    for r in results:
        valid_exact_matches = [m for m in r["exact_match"] if m is not None]
        exact_match_count += sum(valid_exact_matches)
        total_trials += len(valid_exact_matches)
        
    exact_match_accuracy = exact_match_count / total_trials if total_trials > 0 else 0

    print(f"\nFinal F1 Accuracy: {f1_accuracy:.3f} ({f1_accuracy*100:.1f}%)")
    print(
        f"Final Exact Match Accuracy: {exact_match_accuracy:.3f} ({exact_match_accuracy*100:.1f}%)"
    )
    print(f"\nTotal Cost: ${cost_info['total_cost']:.2f}")
    print(
        f"  Input: {cost_info['total_input_tokens']:,} tokens "
        f"(${cost_info['input_cost']:.2f})"
    )
    print(
        f"  Output: {cost_info['total_output_tokens']:,} tokens "
        f"(${cost_info['output_cost']:.2f})"
    )

    # Save results
    save_results(
        results,
        f1_accuracy,
        args.model_id,
        output_dir,
        df,
        args.downsample_size,
        args,
        cost_info,
    )

    print(f"Results saved in {output_dir}")

    print(f"\n{'='*50}")
    print("Evaluation completed!")
    print(f"Results saved in {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
