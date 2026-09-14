"""
Prompt assembly + answer normalization utilities shared by `eval` and `agents`.

This module is intentionally leaf-level — it imports only stdlib so both
`eval.py` and `agents.py` can import from it without creating a circular
dependency.
"""

import ast
import json
import re
from decimal import Decimal, InvalidOperation


domain_expertise = (
    "A large deposit is defined as exceeding 50% of the borrower's total monthly qualifying income. "
    "Eligible income is calculated from the bank statement transactions, NOT from the ULAD DU's declared "
    "income fields (e.g. CurrentIncomeMonthlyTotalAmount). "
    # "To compute eligible income, filter transactions by their 'tag' field (not by description text): "
    # "qualifying deposits have tag 'payroll' or 'ssa_income'; "
    # "debt obligations have tag 'auto_loan', 'mortgage', or 'child_support'. "
    # "All other tags (e.g. 'general transaction', 'mortgage payments', 'cash', 'crypto') are excluded. "
    "Eligible income = sum(qualifying deposits) − sum(obligation payments)."
)

# Set from `eval.main` based on --include_tags. When True, build_prompt adds
# the tag-trust instruction so the model treats `tag`/`tags` fields as
# authoritative.
_INCLUDE_TAGS = False
TAG_TRUST_NOTE = (
    "Transaction data includes tags, such as \"BNPL transaction\". These tags "
    "should be considered perfectly complete and accurate. Tags describe all "
    "relevant properties of the transaction and no tags are missing."
)


def set_include_tags(flag):
    global _INCLUDE_TAGS
    _INCLUDE_TAGS = bool(flag)


def build_prompt(question, bank_statement, ulad_du, use_domain_expertise, answer_instruction, extra_context="", bank_statement_b=None):
    """Construct the model prompt."""
    parts = [question]
    if bank_statement_b:
        parts.append(f"Bank Statement 1: {bank_statement}")
        parts.append(f"Bank Statement 2: {bank_statement_b}")
    else:
        parts.append(f"Bank Statement: {bank_statement}")
    parts.append(f"ULAD DU: {ulad_du}")
    if _INCLUDE_TAGS:
        parts.append(TAG_TRUST_NOTE)
    if extra_context:
        parts.append(extra_context)
    closing = "Answer the question."
    if use_domain_expertise:
        closing += f" Domain Expertise: {domain_expertise}"
    closing += f" Do not think out loud. {answer_instruction}."
    parts.append(closing)
    return "\n\n".join(parts)


model_answer_instruction = {
    "txn_id_list": "Describe the relevant transactions in text (titles/amounts/dates); do not guess or output transaction IDs.",
    "boolean": "Answer with yes or no.",
    "account_id_list": "Identify the relevant accounts in text (names/descriptions/last4 digits); do not guess or output account IDs.",
    "dollar_amount": "Think step by step about which dollar amounts are relevant to the question and why. Then identify and list each relevant amount from the documents, stating what it represents. Do not calculate totals yourself.",
}

cleaning_answer_instruction = {
    "txn_id_list": 'Return ONLY a valid JSON list of matching TransactionID values, e.g. `["plaid-2-00037", "plaid-2-00049"]`, or [] if there are no IDs.',
    "boolean": "Return only yes or no. DO NOT output any other text.",
    "account_id_list": 'Return ONLY a valid JSON list of account numbers (last 4 digits only). e.g. `["1234", "5678"]`, or [] if there are no account numbers.',
    "dollar_amount": "Return ONLY the final dollar amount as a plain number with two decimal places (e.g., 1234.56). No $ sign, no commas, no other text.",
}


# ---------------------------------------------------------------------------
# Answer normalization helpers
# ---------------------------------------------------------------------------


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
