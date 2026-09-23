"""MortarBench: evaluating LLM agents on mortgage loan origination.

Each sample presents a synthetic bank statement (Plaid-format JSON) and a
completed loan application (ULAD/MISMO XML), then asks an underwriting question
that can only be answered by cross-referencing the two documents.

This task reproduces the paper's `BaselineAgent` protocol, which runs in two
model passes:

1. **Answer.** The model is shown both documents and told to describe the
   relevant transactions or accounts *in prose* - it is explicitly instructed
   not to emit IDs, because models transcribe them unreliably.
2. **Clean.** A second call maps that prose onto concrete TransactionID /
   account values, given the flattened reference JSON.

The result is then normalized and compared to the gold answer as a set.

Prompts are imported from the repository's own `prompts.py` rather than
restated here, so this task cannot drift from the numbers in the paper.

Paper: https://arxiv.org/abs/2606.19416
Dataset: https://huggingface.co/datasets/ManavMunjal/MortarBench

Usage:
    inspect eval inspect_evals/mortarbench --model openai/gpt-5
    inspect eval inspect_evals/mortarbench -T answer_types=boolean
"""

import ast
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure environment variables from .env are loaded
load_dotenv(override=True)

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, TaskState, generate, solver

try:
    from .prompts import (
        build_prompt,
        cleaning_answer_instruction,
        model_answer_instruction,
        normalize_account_answer,
        normalize_transaction_answer,
    )
except (ImportError, ModuleNotFoundError):
    try:
        from inspect_evals.mortarbench.prompts import (
            build_prompt,
            cleaning_answer_instruction,
            model_answer_instruction,
            normalize_account_answer,
            normalize_transaction_answer,
        )
    except (ImportError, ModuleNotFoundError):
        from prompts import (
            build_prompt,
            cleaning_answer_instruction,
            model_answer_instruction,
            normalize_account_answer,
            normalize_transaction_answer,
        )

HF_REPO = "ManavMunjal/MortarBench"


THRESHOLD_MODEL_INSTRUCTION = (
    "List every transaction in the bank statement that is plausibly related to "
    "the question, even if you are not sure. For each one, assign an integer "
    "confidence rating from 1 to 5 of how strongly the transaction matches the "
    "question's criteria: 1 = least likely to be relevant, 5 = clearly relevant. "
    "Do not mention any transactions that are definitely irrelevant (confidence 0). "
    "Think outloud and state what assumptions you are making. "
    "A confidence of 5 should require no assumptions whatsoever to be relevant."
    "After stating your assumptions, return a JSON list starting with ```json of the form"
    '`[{"transaction_id": "<TransactionID>", "confidence": <1-5>}, ...]`, or '
    "`[]` if nothing is even plausibly related. Do not output any other text."
)

THRESHOLD_CLEANING_INSTRUCTION = (
    "The answer above should be a JSON list of "
    '`{"transaction_id", "confidence"}` objects rating candidate transactions. '
    "Return ONLY a valid JSON list in EXACTLY that shape — preserve every "
    "transaction_id and its confidence value unchanged. If the answer is empty "
    "or no transactions are listed, return `[]`. Output ONLY the JSON list."
)

# ---------------------------------------------------------------------------
# Document flattening (mirrors flatten_plaid_transactions / extract_plaid_accounts
# in eval.py, restricted to the Plaid payload shapes the released data uses).
# ---------------------------------------------------------------------------


def _iter_accounts(statement: dict):
    """Test cases 4 and 9 bundle payloads under "plaid_files"; the rest are flat."""
    if "plaid_files" in statement:
        for payload in statement["plaid_files"]:
            yield from payload.get("override_accounts", []) or []
    else:
        yield from statement.get("override_accounts", []) or []


def _flatten_transactions(statement: dict) -> list[dict]:
    flat = []
    for account in _iter_accounts(statement):
        last4 = (account.get("numbers") or {}).get("account")
        base = {
            "account_type": account.get("type"),
            "account_subtype": account.get("subtype"),
            "account_last4": last4[-4:] if last4 else None,
        }
        for txn in account.get("transactions", []) or []:
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


def _extract_accounts(statement: dict) -> list[dict]:
    accounts = []
    for account in _iter_accounts(statement):
        last4 = (account.get("numbers") or {}).get("account")
        accounts.append(
            {
                "name": account.get("subtype"),
                "type": account.get("type"),
                "subtype": account.get("subtype"),
                "account_number_last4": last4[-4:] if last4 else None,
            }
        )
    return accounts


def create_record_to_sample(method: str = "baseline", confidence_threshold: int = 5):
    def record_to_sample(record: dict) -> Sample:
        statement = json.loads(record["bank_statement"])
        transactions = _flatten_transactions(statement)
        accounts = _extract_accounts(statement)
        answer_type = record["answer_type"]
        
        if method == "crit" and answer_type == "txn_id_list":
            instruction = THRESHOLD_MODEL_INSTRUCTION
        else:
            instruction = model_answer_instruction[answer_type]

        return Sample(
            id=record["question_id"],
            input=build_prompt(
                record["question"],
                json.dumps(statement, indent=2),
                record["ulad_du"],
                use_domain_expertise=False,
                answer_instruction=instruction,
            ),
            target=record["answer"],
            metadata={
                "answer_type": answer_type,
                "question": record["question"],
                "test_case_number": record["test_case_number"],
                "loan_id": record["loan_id"],
                "pii": record["pii"],
                "transactions_json": json.dumps(transactions, indent=2),
                "accounts_json": json.dumps(accounts, indent=2),
                "account_last4": [
                    a["account_number_last4"] for a in accounts if a["account_number_last4"]
                ],
                "transactions": transactions,
                "method": method,
                "confidence_threshold": confidence_threshold,
            },
        )
    return record_to_sample


# ---------------------------------------------------------------------------
# Cleaning pass - prompts copied from BaselineAgent in agents.py
# ---------------------------------------------------------------------------

TXN_CLEANING_PROMPT = (
    "Question: {question}\n\n"
    "Unformatted answer text (source of truth):\n{solo_text}\n\n"
    "Unformatted transaction JSON (may be incomplete or wrong):\n{txn_info}\n\n"
    "Reference bank statement transactions JSON:\n{transactions}\n\n"
    "Step-by-step:\n"
    "1) From the text only, count how many distinct transactions or payment occurrences are implied (call this N, allow that it might be N+ if frequency/pattern suggests more occurrences).\n"
    "2) Using the reference transactions JSON, find all matching transactions (titles/descriptions/amounts/dates). Do not stop at the first N; include additional matches if the pattern implies more than N.\n"
    "3) If the text says none / no matching transactions, return []. Otherwise return ONLY a JSON list of all matching TransactionID values (no prose, no extra text).\n"
    "Ignore any TransactionIDs in the unformatted JSON portion if they conflict with the text. "
    "If nothing matches, return an empty list ([]).\n"
    "Ignore any boilerplate such as 'analysis report is outdated' or suggestion/help sections; they are not part of the answer.\n\n"
    "{answer_instruction}"
)

ACCOUNT_CLEANING_PROMPT = (
    "Question: {question}\n\n"
    "Unformatted answer text (source of truth):\n{solo_text}\n\n"
    "Unformatted transaction/account JSON (may be incomplete or wrong):\n{txn_info}\n\n"
    "Reference bank statement accounts JSON:\n{accounts}\n\n"
    "Use the text portion to decide which accounts the answer refers to. "
    "Match the mentioned account names/descriptions to the BankStatementAccounts in the reference JSON and "
    "return ONLY a JSON list of the last 4 digits of the matching AccountNumber values (no prose, no extra text). "
    "Ignore any account IDs in the unformatted JSON if they conflict with the text. "
    "If nothing matches, return an empty list ([]).\n"
    "Ignore any boilerplate such as 'analysis report is outdated' or suggestion/help sections; they are not part of the answer.\n\n"
    "{answer_instruction}"
)

BOOLEAN_CLEANING_PROMPT = (
    "Question: {question}\n\nUnformatted answer: {raw_answer}\n\n"
    "The answer given should be either yes or no. "
    "Read the question and answer, and simplify the answer to yes or no. "
    "Ignore any boilerplate (e.g., 'analysis report is outdated' or suggestion/help sections); they are not part of the answer."
)


def _build_cleaning_prompt(metadata: dict, raw_answer: str) -> str:
    answer_type = metadata["answer_type"]
    question = metadata["question"]
    instruction = cleaning_answer_instruction[answer_type]

    if answer_type == "boolean":
        return BOOLEAN_CLEANING_PROMPT.format(
            question=question, raw_answer=raw_answer
        )
    if answer_type == "account_id_list":
        return ACCOUNT_CLEANING_PROMPT.format(
            question=question,
            solo_text=raw_answer,
            txn_info=metadata["accounts_json"],
            accounts=metadata["accounts_json"],
            answer_instruction=instruction,
        )
    if answer_type == "txn_id_list" and metadata.get("method") == "crit":
        return (
            f"Question: {question}\n\n"
            f"Unformatted answer text (source of truth):\n{raw_answer}\n\n"
            f"Reference bank statement transactions JSON:\n{metadata['transactions_json']}\n\n"
            f"{THRESHOLD_CLEANING_INSTRUCTION}"
        )
        
    return TXN_CLEANING_PROMPT.format(
        question=question,
        solo_text=raw_answer,
        txn_info=metadata["transactions_json"],
        transactions=metadata["transactions_json"],
        answer_instruction=instruction,
    )


@solver
def clean_answer():
    """Second pass: map the prose answer onto concrete IDs."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        metadata = state.metadata or {}
        raw_answer = state.output.completion
        state.store.set("raw_answer", raw_answer)

        state.messages = [
            ChatMessageUser(content=_build_cleaning_prompt(metadata, raw_answer))
        ]
        return await generate(state)

    return solve


def _extract_list(text: str) -> str:
    """Pull a JSON list out of the cleaning output, preferring a fenced block."""
    fenced = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if fenced:
        return fenced.group(1).strip()
    bare = re.search(r"\[[^\]]*\]", text, re.DOTALL)
    return bare.group(0).strip() if bare else text


# ---------------------------------------------------------------------------
# Scoring - mirrors is_correct() in eval.py
# ---------------------------------------------------------------------------


def _gold_tokens(gt_answer: str, answer_type: str) -> set[str]:
    gt_str = str(gt_answer).strip()
    raw: list = []
    if gt_str.startswith("[") and gt_str.endswith("]"):
        try:
            parsed = ast.literal_eval(gt_str)
            raw = parsed if isinstance(parsed, list) else gt_str.split(",")
        except (ValueError, SyntaxError):
            raw = gt_str.split(",")
    else:
        raw = gt_str.split(",")

    tokens = [str(t).strip().lower() for t in raw]
    if len(tokens) == 1 and tokens[0] in {"none", "[]", ""}:
        return set()

    tokens = [t for t in tokens if t]
    if answer_type == "account_id_list":
        tokens = [t[-4:] if len(t) >= 4 else t for t in tokens]
    return set(tokens)


@scorer(metrics=[accuracy(), stderr()])
def mortarbench_scorer():
    """Exact match, following eval.py: yes/no for booleans, set equality for lists."""

    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata or {}
        answer_type = metadata.get("answer_type", "boolean")
        completion = state.output.completion
        raw_answer = state.store.get("raw_answer", "")

        if answer_type == "boolean":
            expected = str(target.text).strip().lower()
            expected = {"true": "yes", "1": "yes", "false": "no", "0": "no"}.get(
                expected, expected
            )
            predicted = completion.strip().strip(".").strip().lower()
            correct = predicted == expected
            return Score(
                value=CORRECT if correct else INCORRECT,
                answer=predicted,
                explanation=f"expected {expected!r}\n\nprose answer:\n{raw_answer}",
            )

        cleaned = _extract_list(completion)
        if answer_type == "txn_id_list":
            if metadata.get("method") == "crit":
                try:
                    parsed = json.loads(cleaned)
                    kept_ids = []
                    for obj in parsed if isinstance(parsed, list) else []:
                        if isinstance(obj, dict):
                            conf = int(obj.get("confidence", 0))
                            if obj.get("transaction_id") is not None and conf >= metadata.get("confidence_threshold", 5):
                                kept_ids.append(str(obj["transaction_id"]))
                    cleaned = json.dumps(kept_ids)
                except (json.JSONDecodeError, ValueError, TypeError):
                    cleaned = "[]"
            cleaned = normalize_transaction_answer(
                cleaned, "txn_id_list", metadata.get("transactions") or []
            )
        else:
            cleaned = normalize_account_answer(
                cleaned, metadata.get("account_last4") or []
            )

        match = re.search(r"\[(.*)\]", cleaned, re.DOTALL)
        if match is None:
            return Score(
                value=INCORRECT,
                answer=cleaned,
                explanation=f"no JSON list in cleaned answer\n\nprose answer:\n{raw_answer}",
            )
        try:
            predicted_list = json.loads(match.group(0))
        except json.JSONDecodeError:
            return Score(
                value=INCORRECT,
                answer=cleaned,
                explanation=f"invalid JSON in cleaned answer\n\nprose answer:\n{raw_answer}",
            )

        predicted = set()
        for token in predicted_list:
            token = str(token).strip().lower()
            if not token:
                continue
            if answer_type == "account_id_list" and len(token) >= 4:
                token = token[-4:]
            predicted.add(token)

        expected = _gold_tokens(target.text, answer_type)
        return Score(
            value=CORRECT if predicted == expected else INCORRECT,
            answer=json.dumps(sorted(predicted)),
            explanation=f"expected {sorted(expected)}\n\nprose answer:\n{raw_answer}",
        )

    return score


@task
def mortarbench(answer_types: str | None = None, method: str = "baseline", confidence_threshold: int = 5) -> Task:
    """MortarBench mortgage underwriting benchmark (BaselineAgent protocol).

    Args:
        answer_types: Optional comma-separated filter, any of
            `boolean`, `txn_id_list`, `account_id_list`.
        method: Evaluation method, either `baseline` or `crit`.
        confidence_threshold: Confidence threshold for `crit` method (1-5).
    """
    dataset = hf_dataset(
        HF_REPO, 
        split="test", 
        sample_fields=create_record_to_sample(method, int(confidence_threshold))
    )

    if answer_types:
        wanted = {t.strip() for t in answer_types.split(",")}
        dataset = dataset.filter(
            lambda sample: (sample.metadata or {}).get("answer_type") in wanted
        )

    return Task(
        dataset=dataset,
        solver=[generate(), clean_answer()],
        scorer=mortarbench_scorer(),
    )
