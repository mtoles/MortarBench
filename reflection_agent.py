"""
Reflection Agent module for mortgage benchmark evaluation.

Adds two capabilities on top of the vanilla LLM-based Q&A pipeline:

1. Self-Correction / Reflection Loop
   - After the agent produces an answer, one additional round checks the
     question + answer against the original source documents and refines
     the answer if needed.

2. Numerical Reasoning via Code Execution (dollar_amount answer type)
   - Instead of having the LLM compute arithmetic, it extracts operands and
     writes a Python expression; the code is executed in a subprocess and the
     result is fed back for a final answer.

Usage in eval.py:
    from reflection_agent import ReflectionAgent, normalize_dollar_answer

    agent = ReflectionAgent(model_id, loan_id, ...)
    agent.set_context(bank_statement_str, ulad_du_str)
"""

import json
import math
import os
import re
import subprocess
import threading
from typing import List, Dict

from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

from agents import BaselineAgent
from llm import call_llm_wrapper
from rag_pipeline import build_retrievers, load_and_split_documents, FAISS_INDEX_PATH, PDF_PATH

load_dotenv(override=True)

_SHARED_RETRIEVER = None
_RETRIEVER_LOCK = threading.Lock()

def get_shared_retriever():
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
                _SHARED_RETRIEVER = build_retrievers(splits, force_rebuild=False)
    return _SHARED_RETRIEVER

# ---------------------------------------------------------------------------
# Python tool definition (Anthropic schema; converted for OpenAI where needed)
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


# ---------------------------------------------------------------------------
# Subprocess-based Python executor
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# call_with_python_tool — full tool-use loop for Anthropic and OpenAI
# ---------------------------------------------------------------------------

def call_with_python_tool(
    model_id: str,
    messages: List[Dict],
    loan_id: str = None,
) -> tuple:
    """
    Call an LLM with the run_python tool available.

    Handles the full generate → execute → return loop:
      1. Call the LLM with the tool definition.
      2. If it emits a tool_use / function_call block, execute the Python code.
      3. Feed the result back and return the final text answer.

    Returns:
        (final_answer: str, total_input_tokens: int, total_output_tokens: int)
    """
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
    """
    Extract a numeric dollar amount from a string.

    Strips $, commas, and whitespace, then returns the LAST valid float found.
    The last number in a computed answer is almost always the precise result
    (e.g. "approximately 5000 to 5234.56" → 5234.56).
    Returns None if no number can be parsed.
    """
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
    """
    Extends BaselineAgent with:

    1. Self-Correction: after the initial answer, one additional round checks
       the question + answer against the source documents and refines if needed.

    2. Numerical Reasoning (dollar_amount): the LLM writes Python code to
       compute the answer; it is executed in a subprocess and the result is
       returned as the cleaned answer.
    """

    def __init__(
        self,
        model_id,
        loan_id,
        cleared_loans,
        cleared_loans_lock,
        wait_for_loan_gap_func,
        prompt_template,
    ):
        super().__init__(
            model_id,
            loan_id,
            cleared_loans,
            cleared_loans_lock,
            wait_for_loan_gap_func,
            prompt_template,
        )
        self._bank_statement_str: str = ""
        self._ulad_du_str: str = ""
        self._rag_context_str: str = ""

    # ------------------------------------------------------------------
    # Context setter — must be called before process_* methods
    # ------------------------------------------------------------------

    def set_context(self, bank_statement_str, ulad_du_str, question_str=None) -> None:
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
        self._ulad_du_str = _to_str(ulad_du_str)
        self._rag_context_str = ""
        
        if question_str:
            retriever = get_shared_retriever()
            retrieved_docs = retriever.invoke(question_str)
            self._rag_context_str = "\n\n".join([
                f"Content:\n{d.page_content}\nSource: {d.metadata.get('source', 'Unknown')} - Page: {d.metadata.get('page', 'Unknown')}" 
                for d in retrieved_docs
            ])

    # ------------------------------------------------------------------
    # Reflection pass
    # ------------------------------------------------------------------

    def _reflect(self, question: str, initial_answer: str, answer_type: str) -> tuple:
        """
        One round of self-correction grounded in the source documents.
        The check instruction is specific to the answer type so the model
        knows exactly what to verify rather than doing a generic review.

        Returns:
            (refined_answer: str, input_tokens: int, output_tokens: int)
        """
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

    # ------------------------------------------------------------------
    # process_* overrides — reflect then delegate cleanup to BaselineAgent
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Numerical reasoning via code execution
    # ------------------------------------------------------------------

    def process_dollar_amount(self, question: str, raw_answer: str, loan_id: str) -> tuple:
        """
        Reflection pass to verify numbers, then compute the result in Python.

        Returns:
            (reflected_answer, cleaned_dollar_str, total_input_tokens, total_output_tokens)
        """
        reflected, ref_in, ref_out = self._reflect(question, raw_answer, "dollar_amount")

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
