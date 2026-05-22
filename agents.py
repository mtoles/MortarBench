"""Agent classes for handling different model types in evaluation.

Includes the baseline agent stack (Agent, SoloAgent, BaselineAgent,
ExperimentalAgent) as well as the reflection / numerical-reasoning agent
(ReflectionAgent) that was previously its own module. Co-locating them
avoids the agents ↔ reflection_agent circular import.
"""

import json
import math
import os
import re
import subprocess
import threading
from typing import Dict, List

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

from llm import call_llm_wrapper, clear_messages
from prompts import (
    build_prompt,
    cleaning_answer_instruction,
    normalize_account_answer,
    normalize_transaction_answer,
)
from rag_pipeline import (
    FAISS_INDEX_PATH,
    PDF_PATH,
    build_retrievers,
    load_and_split_documents,
    retrieve_and_rerank,
)

load_dotenv(override=True)


class Agent:
    """Base class for model agents."""

    def __init__(self, model_id, loan_id, cleared_loans, cleared_loans_lock, wait_for_loan_gap_func):
        self.model_id = model_id
        self.loan_id = loan_id
        self.cleared_loans = cleared_loans
        self.cleared_loans_lock = cleared_loans_lock
        self.wait_for_loan_gap_func = wait_for_loan_gap_func
        self.solo_answers = []

    def setup_loan(self):
        """Setup called once per loan before processing questions."""
        pass

    def get_initial_prompt(self, question, bank_statement, ulad_du, use_domain_expertise, answer_instruction, bank_statement_b=None):
        """Default: full prompt with bank statement + (optional) ULAD."""
        return build_prompt(
            question, bank_statement, ulad_du, use_domain_expertise, answer_instruction,
            bank_statement_b=bank_statement_b,
        )

    def process_initial_prompt(self, question, bank_statement, ulad_du,
                                use_domain_expertise, answer_instruction,
                                bank_statement_b=None):
        """Build the initial prompt via :meth:`get_initial_prompt` and run it
        through the configured LLM. Returns ``(raw_answer, input_tok, output_tok)``."""
        prompt = self.get_initial_prompt(
            question, bank_statement, ulad_du, use_domain_expertise, answer_instruction,
            bank_statement_b=bank_statement_b,
        )
        return call_llm_wrapper(
            model_id=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            loan_id=self.loan_id,
        )


class SoloAgent(Agent):
    """Agent for solo model with special handling for answer types."""
    
    def __init__(self, model_id, loan_id, cleared_loans, cleared_loans_lock, wait_for_loan_gap_func):
        super().__init__(model_id, loan_id, cleared_loans, cleared_loans_lock, wait_for_loan_gap_func)
        self.solo_answers = []
    
    def setup_loan(self):
        """Clear messages once per loan and enforce spacing."""
        with self.cleared_loans_lock:
            if self.loan_id not in self.cleared_loans:
                clear_messages(self.loan_id)
                self.cleared_loans.add(self.loan_id)
        self.wait_for_loan_gap_func(self.loan_id)
    
    def get_initial_prompt(self, question, *args, **kwargs):
        """Solo uses just the question."""
        return question
    
    def process_boolean(self, question, raw_answer, loan_id):
        """Process boolean answer type for solo agent."""
        cleaned_answer_prompt = (
            f"Question: {question}\n\nUnformatted answer: {raw_answer}\n\n"
            "The answer given should be either yes or no. "
            "Read the question and answer, and simplify the answer to yes or no. "
            "Ignore any boilerplate (e.g., 'analysis report is outdated' or suggestion/help sections); they are not part of the answer."
        )
        cleaned_answer, clean_in_tok, clean_out_tok = call_llm_wrapper(
            model_id=self.model_id,
            messages=[{"role": "user", "content": cleaned_answer_prompt}],
            loan_id=loan_id,
        )
        # For solo, raw_answer and cleaned_answer are the same
        return raw_answer, cleaned_answer, clean_in_tok, clean_out_tok
    
    def process_txn_id_list(self, question, raw_answer, loan_id, transactions_json, cleaning_answer_instruction_str, plaid_transactions_flat):
        """Process txn_id_list answer type for solo agent."""
        solo_answer_parts = raw_answer.split("==========\n")
        solo_text_answer = (
            solo_answer_parts[0].strip()
            if solo_answer_parts
            else raw_answer
        )
        txn_info_display = (
            "Solo agent referenced data intentionally omitted; rely exclusively on the narrative answer."
        )
        solo_txn_info_for_mapping = None
        
        cleaned_answer_prompt = (
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
        ).format(
            question=question,
            solo_text=solo_text_answer,
            txn_info=txn_info_display,
            transactions=transactions_json,
            answer_instruction=cleaning_answer_instruction_str,
        )
        cleaned_answer, clean_in_tok, clean_out_tok = call_llm_wrapper(
            model_id=self.model_id,
            messages=[{"role": "user", "content": cleaned_answer_prompt}],
            loan_id=loan_id,
        )
        fenced_match = re.search(
            r"```(?:json)?\s*(\[[\s\S]*?\])\s*```",
            cleaned_answer,
        )
        if fenced_match:
            cleaned_answer = fenced_match.group(1).strip()
        else:
            list_match = re.search(r"\[[^\]]*\]", cleaned_answer, re.DOTALL)
            if list_match:
                cleaned_answer = list_match.group(0).strip()
        cleaned_answer = normalize_transaction_answer(
            cleaned_answer,
            "txn_id_list",
            plaid_transactions_flat,
            solo_txn_info=solo_txn_info_for_mapping,
        )
        # For solo, raw_answer and cleaned_answer are the same
        return raw_answer, cleaned_answer, clean_in_tok, clean_out_tok
    
    def process_account_id_list(self, question, raw_answer, loan_id, accounts_json, cleaning_answer_instruction_str, account_last4_values):
        """Process account_id_list answer type for solo agent."""
        solo_answer_parts = raw_answer.split("==========\n")
        solo_text_answer = (
            solo_answer_parts[0].strip()
            if solo_answer_parts
            else raw_answer
        )
        txn_info = (
            "==========\n".join(solo_answer_parts[1:]).strip()
            if len(solo_answer_parts) > 1
            else raw_answer
        )
        
        cleaned_answer_prompt = (
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
        ).format(
            question=question,
            solo_text=solo_text_answer,
            txn_info=txn_info,
            accounts=accounts_json,
            answer_instruction=cleaning_answer_instruction_str,
        )
        cleaned_answer, clean_in_tok, clean_out_tok = call_llm_wrapper(
            model_id=self.model_id,
            messages=[{"role": "user", "content": cleaned_answer_prompt}],
            loan_id=loan_id,
        )
        fenced_match = re.search(
            r"```(?:json)?\s*(\[[\s\S]*?\])\s*```",
            cleaned_answer,
        )
        if fenced_match:
            cleaned_answer = fenced_match.group(1).strip()
        else:
            list_match = re.search(r"\[[^\]]*\]", cleaned_answer, re.DOTALL)
            if list_match:
                cleaned_answer = list_match.group(0).strip()
        cleaned_answer = normalize_account_answer(
            cleaned_answer,
            account_last4_values,
        )
        # For solo, raw_answer and cleaned_answer are the same
        return raw_answer, cleaned_answer, clean_in_tok, clean_out_tok


class BaselineAgent(Agent):
    """Agent for baseline/generic models (non-solo)."""

    def __init__(self, model_id, loan_id, cleared_loans, cleared_loans_lock, wait_for_loan_gap_func):
        super().__init__(model_id, loan_id, cleared_loans, cleared_loans_lock, wait_for_loan_gap_func)
        self._rag_context_str = ""

    def set_rag_context(self, question_str):
        """Retrieve Fannie Mae Selling Guide chunks for `question_str` and inject
        them into the initial prompt. Matches the retrieval + formatting used by
        ReflectionAgent."""
        if not question_str:
            self._rag_context_str = ""
            return
        retriever = get_shared_retriever(self.model_id)
        retrieved_docs = retrieve_and_rerank(self.model_id, question_str, retriever)
        self._rag_context_str = "\n\n".join([
            f"Content:\n{d.page_content}\nSource: {d.metadata.get('source', 'Unknown')} - Page: {d.metadata.get('page', 'Unknown')} - Score: {d.metadata.get('relevance_score', 0):.4f}"
            for d in retrieved_docs
        ])

    def get_initial_prompt(self, question, bank_statement, ulad_du, use_domain_expertise, answer_instruction, bank_statement_b=None):
        extra = (
            f"Fannie Mae Selling Guide (RAG Context):\n{self._rag_context_str}"
            if self._rag_context_str else ""
        )
        return build_prompt(
            question, bank_statement, ulad_du, use_domain_expertise, answer_instruction,
            extra_context=extra, bank_statement_b=bank_statement_b,
        )
        return build_prompt(question, bank_statement, ulad_du, use_domain_expertise, answer_instruction, extra_context=extra)
    
    def process_boolean(self, question, raw_answer, loan_id):
        """Process boolean answer type for baseline agent."""
        cleaned_answer_prompt = (
            f"Question: {question}\n\nUnformatted answer: {raw_answer}\n\n"
            "The answer given should be either yes or no. "
            "Read the question and answer, and simplify the answer to yes or no. "
            "Ignore any boilerplate (e.g., 'analysis report is outdated' or suggestion/help sections); they are not part of the answer."
        )
        cleaned_answer, clean_in_tok, clean_out_tok = call_llm_wrapper(
            model_id=self.model_id,
            messages=[{"role": "user", "content": cleaned_answer_prompt}],
            loan_id=loan_id,
        )
        # For baseline, raw_answer and cleaned_answer are the same
        return raw_answer, cleaned_answer, clean_in_tok, clean_out_tok
    
    def process_txn_id_list(self, question, raw_answer, loan_id, transactions_json, cleaning_answer_instruction_str, plaid_transactions_flat):
        """Process txn_id_list answer type for baseline agent."""
        solo_text_answer = raw_answer
        txn_info_display = transactions_json
        solo_txn_info_for_mapping = None

        cleaned_answer_prompt = (
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
        ).format(
            question=question,
            solo_text=solo_text_answer,
            txn_info=txn_info_display,
            transactions=transactions_json,
            answer_instruction=cleaning_answer_instruction_str,
        )
        cleaned_answer, clean_in_tok, clean_out_tok = call_llm_wrapper(
            model_id=self.model_id,
            messages=[{"role": "user", "content": cleaned_answer_prompt}],
            loan_id=loan_id,
        )
        fenced_match = re.search(
            r"```(?:json)?\s*(\[[\s\S]*?\])\s*```",
            cleaned_answer,
        )
        if fenced_match:
            cleaned_answer = fenced_match.group(1).strip()
        else:
            list_match = re.search(r"\[[^\]]*\]", cleaned_answer, re.DOTALL)
            if list_match:
                cleaned_answer = list_match.group(0).strip()
        cleaned_answer = normalize_transaction_answer(
            cleaned_answer,
            "txn_id_list",
            plaid_transactions_flat,
            solo_txn_info=solo_txn_info_for_mapping,
        )
        # For solo, raw_answer and cleaned_answer are the same
        return raw_answer, cleaned_answer, clean_in_tok, clean_out_tok
    
    def process_dollar_amounts(self, question, raw_answer, loan_id):
        """Process dollar_amounts answer type for baseline agent (simple cleanup pass)."""
        cleanup_prompt = (
            f"Question: {question}\n\nUnformatted answer: {raw_answer}\n\n"
            f"{cleaning_answer_instruction['dollar_amount']}"
        )
        cleaned_answer, clean_in_tok, clean_out_tok = call_llm_wrapper(
            model_id=self.model_id,
            messages=[{"role": "user", "content": cleanup_prompt}],
            loan_id=loan_id,
        )
        return raw_answer, cleaned_answer.strip(), clean_in_tok, clean_out_tok

    def process_account_id_list(self, question, raw_answer, loan_id, accounts_json, cleaning_answer_instruction_str, account_last4_values):
        """Process account_id_list answer type for baseline agent."""
        solo_text_answer = raw_answer
        txn_info = accounts_json

        cleaned_answer_prompt = (
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
        ).format(
            question=question,
            solo_text=solo_text_answer,
            txn_info=txn_info,
            accounts=accounts_json,
            answer_instruction=cleaning_answer_instruction_str,
        )
        cleaned_answer, clean_in_tok, clean_out_tok = call_llm_wrapper(
            model_id=self.model_id,
            messages=[{"role": "user", "content": cleaned_answer_prompt}],
            loan_id=loan_id,
        )
        fenced_match = re.search(
            r"```(?:json)?\s*(\[[\s\S]*?\])\s*```",
            cleaned_answer,
        )
        if fenced_match:
            cleaned_answer = fenced_match.group(1).strip()
        else:
            list_match = re.search(r"\[[^\]]*\]", cleaned_answer, re.DOTALL)
            if list_match:
                cleaned_answer = list_match.group(0).strip()
        cleaned_answer = normalize_account_answer(
            cleaned_answer,
            account_last4_values,
        )
        # For solo, raw_answer and cleaned_answer are the same
        return raw_answer, cleaned_answer, clean_in_tok, clean_out_tok


# ---------------------------------------------------------------------------
# ThresholdAgent prompt instructions
# ---------------------------------------------------------------------------
#
# Initial pass: ask the model to enumerate every plausibly-related transaction
# with a 1–5 confidence rating. Cleanup pass: re-emit the same shape so we can
# filter in Python by a threshold.

THRESHOLD_MODEL_INSTRUCTION = (
    "List every transaction in the bank statement that is plausibly related to "
    "the question, even if you are not sure. For each one, assign an integer "
    "confidence rating from 1 to 5 of how strongly the transaction matches the "
    "question's criteria: 1 = least likely to be relevant, 5 = clearly relevant. "
    "Do not mention any transactions that are definitely irrelevant (confidence 0). "
    # "Return ONLY a JSON list of objects of the form "
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


class ThresholdAgent(BaselineAgent):
    """Confidence-thresholded variant of BaselineAgent.

    The initial prompt asks the model to list every plausibly-related
    transaction with a 1-5 confidence rating; the cleanup pass re-emits the
    same JSON shape; we then drop any transaction with confidence below
    `self.confidence_threshold` in Python.
    """

    def __init__(self, model_id, loan_id, cleared_loans, cleared_loans_lock,
                 wait_for_loan_gap_func, confidence_threshold=3):
        super().__init__(model_id, loan_id, cleared_loans, cleared_loans_lock, wait_for_loan_gap_func)
        self.confidence_threshold = int(confidence_threshold)

    def get_initial_prompt(self, question, bank_statement, ulad_du, use_domain_expertise,
                           answer_instruction, bank_statement_b=None):
        """Swap in the confidence-rating instruction ONLY when the caller
        requested the txn_id_list instruction; for boolean / account_id_list /
        dollar_amount the agent behaves identically to BaselineAgent."""
        # The instruction strings live in prompts.py; we identify the
        # txn_id_list case by equality with the canonical string so we don't
        # have to thread answer_type through the call signature.
        from prompts import model_answer_instruction
        if answer_instruction == model_answer_instruction["txn_id_list"]:
            answer_instruction = THRESHOLD_MODEL_INSTRUCTION
        return super().get_initial_prompt(
            question, bank_statement, ulad_du, use_domain_expertise,
            answer_instruction, bank_statement_b=bank_statement_b,
        )

    def process_txn_id_list(self, question, raw_answer, loan_id, transactions_json,
                            cleaning_answer_instruction_str, plaid_transactions_flat):
        """Cleanup → JSON parse → filter by confidence → normalize."""
        cleaned_answer_prompt = (
            f"Question: {question}\n\n"
            f"Unformatted answer text (source of truth):\n{raw_answer}\n\n"
            f"Reference bank statement transactions JSON:\n{transactions_json}\n\n"
            f"{THRESHOLD_CLEANING_INSTRUCTION}"
        )
        cleaned_answer, clean_in_tok, clean_out_tok = call_llm_wrapper(
            model_id=self.model_id,
            messages=[{"role": "user", "content": cleaned_answer_prompt}],
            loan_id=loan_id,
        )

        # Extract the JSON list from fenced or bare form (mirrors BaselineAgent).
        fenced_match = re.search(
            r"```(?:json)?\s*(\[[\s\S]*?\])\s*```",
            cleaned_answer,
        )
        if fenced_match:
            json_blob = fenced_match.group(1).strip()
        else:
            list_match = re.search(r"\[[\s\S]*\]", cleaned_answer, re.DOTALL)
            json_blob = list_match.group(0).strip() if list_match else "[]"

        try:
            parsed = json.loads(json_blob)
            if not isinstance(parsed, list):
                parsed = []
        except (json.JSONDecodeError, ValueError):
            parsed = []

        kept_ids = []
        for obj in parsed:
            if not isinstance(obj, dict):
                continue
            try:
                conf = int(obj.get("confidence", 0))
            except (TypeError, ValueError):
                conf = 0
            tid = obj.get("transaction_id")
            if tid is not None and conf >= self.confidence_threshold:
                kept_ids.append(str(tid))

        cleaned_answer = normalize_transaction_answer(
            json.dumps(kept_ids),
            "txn_id_list",
            plaid_transactions_flat,
            solo_txn_info=None,
        )
        return raw_answer, cleaned_answer, clean_in_tok, clean_out_tok





# ---------------------------------------------------------------------------
# Shared FAISS retriever for reflection / RAG paths
# ---------------------------------------------------------------------------

_SHARED_RETRIEVER = None
_RETRIEVER_LOCK = threading.Lock()


def get_shared_retriever(model_id):
    global _SHARED_RETRIEVER
    if _SHARED_RETRIEVER is None:
        with _RETRIEVER_LOCK:
            if _SHARED_RETRIEVER is None:
                if os.path.exists(FAISS_INDEX_PATH):
                    splits = []
                else:
                    if not os.path.exists(PDF_PATH):
                        raise FileNotFoundError(
                            f"RAG requires either a prebuilt FAISS index at '{FAISS_INDEX_PATH}' "
                            f"or the source PDF at '{PDF_PATH}' to build one. Neither was found. "
                            f"Place the Selling Guide PDF at '{PDF_PATH}' or disable RAG with --use_rag off."
                        )
                    splits = load_and_split_documents(PDF_PATH)
                _SHARED_RETRIEVER = build_retrievers(model_id, splits, force_rebuild=False)
    return _SHARED_RETRIEVER


# ---------------------------------------------------------------------------
# Python tool definition + subprocess executor (used by ReflectionAgent for
# dollar_amount answer type)
# ---------------------------------------------------------------------------

PYTHON_TOOL = {
    "name": "run_python",
    "description": (
        "Execute Python code for arithmetic. Returns the printed output. "
        "Use ONLY for numerical computation — do not use for anything else."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Valid Python code. Must use print() to output the final result. "
                    "Use only the stdlib — no imports of third-party packages."
                ),
            }
        },
        "required": ["code"],
    },
}


def execute_python_safely(code: str, timeout: int = 5) -> str:
    """Run LLM-generated Python in a subprocess and return stdout or an error string."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"ERROR: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "ERROR: Code execution timed out"
    except Exception as exc:
        return f"ERROR: {exc}"


def call_with_python_tool(
    model_id: str,
    messages: List[Dict],
    loan_id: str = None,
) -> tuple:
    """LLM call with the run_python tool available; full generate → execute →
    feed-back-result → return loop. Anthropic and OpenAI use their native
    tool-calling APIs directly; other providers fall back to plain text."""
    print("calling python!")
    total_in_tok = 0
    total_out_tok = 0

    if model_id.startswith("claude-"):
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        anthropic_tool = {
            "name": PYTHON_TOOL["name"],
            "description": PYTHON_TOOL["description"],
            "input_schema": PYTHON_TOOL["input_schema"],
        }

        response = client.messages.create(
            model=model_id,
            max_tokens=4096,
            messages=messages,
            tools=[anthropic_tool],
            tool_choice={"type": "any"},
            temperature=0,
        )
        total_in_tok += response.usage.input_tokens
        total_out_tok += response.usage.output_tokens

        tool_use_block = next(
            (b for b in response.content if b.type == "tool_use" and b.name == "run_python"),
            None,
        )

        if tool_use_block is not None:
            code = tool_use_block.input.get("code", "")
            exec_result = execute_python_safely(code)

            continuation = list(messages) + [
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": exec_result,
                        }
                    ],
                },
            ]
            final = client.messages.create(
                model=model_id,
                max_tokens=4096,
                messages=continuation,
                tools=[anthropic_tool],
                temperature=0,
            )
            total_in_tok += final.usage.input_tokens
            total_out_tok += final.usage.output_tokens
            text = "".join(b.text for b in final.content if hasattr(b, "text"))
            return text.strip(), total_in_tok, total_out_tok

        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        return text.strip(), total_in_tok, total_out_tok

    elif model_id.startswith("gpt-"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        openai_tool = {
            "type": "function",
            "function": {
                "name": PYTHON_TOOL["name"],
                "description": PYTHON_TOOL["description"],
                "parameters": PYTHON_TOOL["input_schema"],
            },
        }

        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=[openai_tool],
            tool_choice="required",
        )
        total_in_tok += response.usage.prompt_tokens
        total_out_tok += response.usage.completion_tokens

        msg = response.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            code = json.loads(tc.function.arguments).get("code", "")
            exec_result = execute_python_safely(code)

            continuation = list(messages) + [
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": exec_result,
                },
            ]
            final = client.chat.completions.create(
                model=model_id,
                messages=continuation,
            )
            total_in_tok += final.usage.prompt_tokens
            total_out_tok += final.usage.completion_tokens
            return (final.choices[0].message.content or "").strip(), total_in_tok, total_out_tok

        return (msg.content or "").strip(), total_in_tok, total_out_tok

    else:
        content, in_tok, out_tok = call_llm_wrapper(model_id, messages, loan_id=loan_id)
        return content, in_tok, out_tok


# ---------------------------------------------------------------------------
# Dollar-amount normalizer (used by eval.py for scoring)
# ---------------------------------------------------------------------------

def normalize_dollar_answer(value_str: str):
    """Extract a numeric dollar amount; returns the LAST valid float found
    in the cleaned string (covers ranges like "≈ 5000 to 5234.56" → 5234.56)
    or None if no number can be parsed."""
    if not value_str:
        return None
    cleaned = re.sub(r"[$,\s]", "", str(value_str))
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# ReflectionAgent
# ---------------------------------------------------------------------------

class ReflectionAgent(BaselineAgent):
    """Extends BaselineAgent with self-correction and numerical reasoning via
    code execution (used for dollar_amount answers)."""

    def __init__(
        self,
        model_id,
        loan_id,
        cleared_loans,
        cleared_loans_lock,
        wait_for_loan_gap_func,
    ):
        super().__init__(
            model_id,
            loan_id,
            cleared_loans,
            cleared_loans_lock,
            wait_for_loan_gap_func,
        )
        self._bank_statement_str: str = ""
        self._bank_statement_b_str: str = ""
        self._ulad_du_str: str = ""

    def set_context(self, bank_statement_str, ulad_du_str, question_str=None, bank_statement_b=None) -> None:
        """Store source documents for use during reflection and numerical reasoning."""

        def _to_str(val):
            if val is None:
                return ""
            if isinstance(val, float) and math.isnan(val):
                return ""
            if isinstance(val, str):
                return val
            return json.dumps(val)

        self._bank_statement_str = _to_str(bank_statement_str)
        self._bank_statement_b_str = _to_str(bank_statement_b)
        self._ulad_du_str = _to_str(ulad_du_str)
        self._rag_context_str = ""

        if question_str:
            retriever = get_shared_retriever(self.model_id)
            retrieved_docs = retrieve_and_rerank(self.model_id, question_str, retriever)
            self._rag_context_str = "\n\n".join([
                f"Content:\n{d.page_content}\nSource: {d.metadata.get('source', 'Unknown')} - Page: {d.metadata.get('page', 'Unknown')} - Score: {d.metadata.get('relevance_score', 0):.4f}"
                for d in retrieved_docs
            ])

    def _reflect(self, question: str, initial_answer: str, answer_type: str) -> tuple:
        """One round of self-correction grounded in the source documents."""
        if self._bank_statement_b_str:
            source_doc = (
                f"Bank Statement 1:\n{self._bank_statement_str}\n\n"
                f"Bank Statement 2:\n{self._bank_statement_b_str}\n\n"
                f"ULAD DU:\n{self._ulad_du_str}"
            )
        else:
            source_doc = (
                f"Bank Statement:\n{self._bank_statement_str}\n\n"
                f"ULAD DU:\n{self._ulad_du_str}"
            )
        if self._rag_context_str:
            source_doc += f"\n\nFannie Mae Selling Guide (RAG Context):\n{self._rag_context_str}"

        if answer_type == "boolean":
            check_instruction = (
                "Find the exact sentence or field in the source documents that "
                "directly supports your YES or NO answer and quote it. "
                "If you cannot find a direct quote, reconsider your answer. "
                "Return the corrected answer with its supporting quote, or the "
                "original answer with its supporting quote if it is already correct."
            )
        elif answer_type == "txn_id_list":
            check_instruction = (
                "For each transaction in your answer, quote the exact field values "
                "from the source JSON (description, amount, date) that confirm it "
                "qualifies for this question. "
                "Then check whether any transactions in the source match the question "
                "criteria but are missing from your answer. "
                "Remove any transaction you cannot quote support for, and add any "
                "qualifying transactions that were missed. "
                "Return the corrected answer."
            )
        elif answer_type == "account_id_list":
            check_instruction = (
                "For each account in your answer, quote the exact field values from "
                "the source JSON (account name, type, last 4 digits) that confirm it "
                "qualifies for this question. "
                "Then check whether any accounts in the source match the question "
                "criteria but are missing from your answer. "
                "Remove any account you cannot quote support for, and add any "
                "qualifying accounts that were missed. "
                "Return the corrected answer."
            )
        else:
            check_instruction = (
                "Review the answer against the source documents. "
                "If the answer is incorrect or incomplete, return a corrected answer. "
                "If the answer is already correct, return it unchanged."
            )

        prompt = (
            f"Question: {question}\n\n"
            f"Answer: {initial_answer}\n\n"
            f"Source documents:\n{source_doc}\n\n"
            f"{check_instruction}"
        )
        refined, in_tok, out_tok = call_llm_wrapper(
            model_id=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            loan_id=self.loan_id,
        )
        return refined, in_tok, out_tok

    def process_boolean(self, question, raw_answer, loan_id):
        """Reflection pass → BaselineAgent boolean cleanup."""
        reflected, ref_in, ref_out = self._reflect(question, raw_answer, "boolean")
        _, cleaned, clean_in, clean_out = super().process_boolean(question, reflected, loan_id)
        return reflected, cleaned, ref_in + clean_in, ref_out + clean_out

    def process_txn_id_list(
        self, question, raw_answer, loan_id,
        transactions_json, cleaning_answer_instruction_str, plaid_transactions_flat,
    ):
        """Reflection pass → BaselineAgent txn_id_list cleanup."""
        reflected, ref_in, ref_out = self._reflect(question, raw_answer, "txn_id_list")
        _, cleaned, clean_in, clean_out = super().process_txn_id_list(
            question, reflected, loan_id,
            transactions_json, cleaning_answer_instruction_str, plaid_transactions_flat,
        )
        return reflected, cleaned, ref_in + clean_in, ref_out + clean_out

    def process_account_id_list(
        self, question, raw_answer, loan_id,
        accounts_json, cleaning_answer_instruction_str, account_last4_values,
    ):
        """Reflection pass → BaselineAgent account_id_list cleanup."""
        reflected, ref_in, ref_out = self._reflect(question, raw_answer, "account_id_list")
        _, cleaned, clean_in, clean_out = super().process_account_id_list(
            question, reflected, loan_id,
            accounts_json, cleaning_answer_instruction_str, account_last4_values,
        )
        return reflected, cleaned, ref_in + clean_in, ref_out + clean_out

    def process_dollar_amount(self, question: str, raw_answer: str, loan_id: str) -> tuple:
        """Reflection pass to verify numbers, then compute the result in Python."""
        reflected, ref_in, ref_out = self._reflect(question, raw_answer, "dollar_amount")

        if self._bank_statement_b_str:
            source_excerpt = (
                f"Bank Statement 1:\n{self._bank_statement_str}\n\n"
                f"Bank Statement 2:\n{self._bank_statement_b_str}\n\n"
                f"ULAD DU:\n{self._ulad_du_str}"
            )
        else:
            source_excerpt = (
                f"Bank Statement:\n{self._bank_statement_str}\n\n"
                f"ULAD DU:\n{self._ulad_du_str}"
            )
        if self._rag_context_str:
            source_excerpt += f"\n\nFannie Mae Selling Guide (RAG Context):\n{self._rag_context_str}"
        compute_prompt = (
            f"Question: {question}\n\n"
            f"Preliminary analysis:\n{reflected}\n\n"
            f"Source documents:\n{source_excerpt}\n\n"
            "Extract the specific numerical values from the source documents needed "
            "to answer the question. Call the run_python tool with Python code that "
            "computes the final dollar amount and prints it as a plain number with "
            "two decimal places (e.g., print(round(x, 2))). "
            "Do NOT calculate the result mentally — write it as executable Python."
        )
        computed_answer, comp_in, comp_out = call_with_python_tool(
            model_id=self.model_id,
            messages=[{"role": "user", "content": compute_prompt}],
            loan_id=loan_id,
        )

        cleaned = normalize_dollar_answer(computed_answer)
        cleaned_str = f"{cleaned:.2f}" if cleaned is not None else computed_answer.strip()

        return reflected, cleaned_str, ref_in + comp_in, ref_out + comp_out
