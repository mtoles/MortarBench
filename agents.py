"""Agent classes for handling different model types in evaluation."""

import re
from llm import call_llm_wrapper, clear_messages


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
        from eval import build_prompt
        return build_prompt(
            question, bank_statement, ulad_du, use_domain_expertise, answer_instruction,
            bank_statement_b=bank_statement_b,
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
        from eval import normalize_transaction_answer
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
        from eval import normalize_account_answer
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
        # Lazy import to avoid circular dependency with reflection_agent.
        from reflection_agent import get_shared_retriever
        from rag_pipeline import retrieve_and_rerank

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
        from eval import build_prompt
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
        from eval import normalize_transaction_answer
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
        from eval import cleaning_answer_instruction
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
        from eval import normalize_account_answer
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




class ExperimentalAgent(Agent):
    """Agent for experimental two-pass processing. Inherits the base prompt builder."""

    def process_boolean(self, question, raw_answer, loan_id):
        """Process boolean answer type for experimental agent."""
        answer_prompt = (
            f"Question: {question}\n\nUnformatted answer: {raw_answer}\n\n"
            "The answer given should be either yes or no. "
            "Read the question and answer, and then give an explanation followed by either yes or no. "
            "Ignore any boilerplate (e.g., 'analysis report is outdated' or suggestion/help sections); they are not part of the answer."
        )
        processed_answer, clean_in_tok, clean_out_tok = call_llm_wrapper(
            model_id=self.model_id,
            messages=[{"role": "user", "content": answer_prompt}],
            loan_id=loan_id,
        )
        # Extract yes/no from the processed answer
        match = re.search(r"yes|no", processed_answer, re.IGNORECASE)
        cleaned_answer = match.group(0).lower() if match else processed_answer.strip().lower()
        
        # For experimental, raw_answer is the explanation + answer, cleaned_answer is just yes/no
        return processed_answer, cleaned_answer, clean_in_tok, clean_out_tok
    
    def process_txn_id_list(self, question, raw_answer, loan_id, transactions_json, cleaning_answer_instruction_str, plaid_transactions_flat):
        """Process txn_id_list answer type for experimental agent."""
        from eval import normalize_transaction_answer
        solo_text_answer = raw_answer
        txn_info_display = transactions_json
        solo_txn_info_for_mapping = None

        answer_prompt = (
            "Question: {question}\n\n"
            "Unformatted answer text (source of truth):\n{solo_text}\n\n"
            "Unformatted transaction JSON (may be incomplete or wrong):\n{txn_info}\n\n"
            "Reference bank statement transactions JSON:\n{transactions}\n\n"
            "Step-by-step:\n"
            "1) From the text only, count how many distinct transactions or payment occurrences are implied (call this N, allow that it might be N+ if frequency/pattern suggests more occurrences).\n"
            "2) Using the reference transactions JSON, find all matching transactions (titles/descriptions/amounts/dates). Do not stop at the first N; include additional matches if the pattern implies more than N.\n"
            "3) Give an explanation for each transaction you chose, followed by a list of the transaction IDs. If the text says none / no matching transactions, return []. Otherwise return ONLY a JSON list of all matching TransactionID values (no prose, no extra text).\n"
            "Ignore any TransactionIDs in the unformatted JSON portion if they conflict with the text. "
            "If nothing matches, return an empty list ([]).\n"
            "Ignore any boilerplate such as 'analysis report is outdated' or suggestion/help sections; they are not part of the answer.\n\n"
            "You answer should be formatted as: <explanation>\n\n[list_item_1, list_item_2, ...]"
            "{answer_instruction}"
        ).format(
            question=question,
            solo_text=solo_text_answer,
            txn_info=txn_info_display,
            transactions=transactions_json,
            answer_instruction=cleaning_answer_instruction_str,
        )
        processed_answer, clean_in_tok, clean_out_tok = call_llm_wrapper(
            model_id=self.model_id,
            messages=[{"role": "user", "content": answer_prompt}],
            loan_id=loan_id,
        )
        txn_check_prompt = (
            "### Double Checking Prompt:\n"
            "Check each transaction in your previous response. For each transaction, double check that the transaction is correctly included based on their line of business. "
            "Additionally, check that the transaction should actually be included in the answer, even if it results in no transactions being included. " 
            "Common mistakes include:\n"
            "- Accidentally including additional vendors who do not provide the relevant service\n"
            "- Including transactions that are interesting but do not qualify under the question criteria\n\n"
            "Then generate a corrected output (or identical output if no corrections are needed). "
        )
        txn_check_answer, txn_check_in_tok, txn_check_out_tok = call_llm_wrapper(
            model_id=self.model_id,
            messages=[
                {"role": "user", "content": answer_prompt},
                {"role": "assistant", "content": processed_answer},
                {"role": "user", "content": txn_check_prompt}
                ],
            loan_id=loan_id,
            tools=[{"type": "web_search"}],
        )
        fenced_matches = list(re.finditer(
            r"```(?:json)?\s*(\[[\s\S]*?\])\s*```",
            txn_check_answer,
        ))
        if fenced_matches:
            extracted_list = fenced_matches[-1].group(1).strip()
        else:
            list_matches = list(re.finditer(r"\[[^\]]*\]", txn_check_answer, re.DOTALL))
            if list_matches:
                extracted_list = list_matches[-1].group(0).strip()
            else:
                extracted_list = txn_check_answer
        cleaned_answer = normalize_transaction_answer(
            extracted_list,
            "txn_id_list",
            plaid_transactions_flat,
            solo_txn_info=solo_txn_info_for_mapping,
        )
        # For experimental, raw_answer is the explanation + list, cleaned_answer is just the list
        # Sum tokens from both LLM calls
        total_in_tok = clean_in_tok + txn_check_in_tok
        total_out_tok = clean_out_tok + txn_check_out_tok
        return txn_check_answer, cleaned_answer, total_in_tok, total_out_tok
    
    def process_account_id_list(self, question, raw_answer, loan_id, accounts_json, cleaning_answer_instruction_str, account_last4_values):
        """Process account_id_list answer type for experimental agent."""
        from eval import normalize_account_answer
        solo_text_answer = raw_answer
        txn_info = accounts_json

        answer_prompt = (
            "Question: {question}\n\n"
            "Unformatted answer text (source of truth):\n{solo_text}\n\n"
            "Unformatted transaction/account JSON (may be incomplete or wrong):\n{txn_info}\n\n"
            "Reference bank statement accounts JSON:\n{accounts}\n\n"
            "Use the text portion to decide which accounts the answer refers to. "
            "Match the mentioned account names/descriptions to the BankStatementAccounts in the reference JSON and "
            "Give an explanation for each account you chose, followed by a list of the account IDs. Return a JSON list of the last 4 digits of the matching AccountNumber values (no prose, no extra text). "
            "Ignore any account IDs in the unformatted JSON if they conflict with the text. "
            "If nothing matches, return an empty list ([]).\n"
            "Ignore any boilerplate such as 'analysis report is outdated' or suggestion/help sections; they are not part of the answer.\n\n"
            "You answer should be formatted as: <explanation>\n\n[list_item_1, list_item_2, ...]"
            "{answer_instruction}"
        ).format(
            question=question,
            solo_text=solo_text_answer,
            txn_info=txn_info,
            accounts=accounts_json,
            answer_instruction=cleaning_answer_instruction_str,
        )
        processed_answer, clean_in_tok, clean_out_tok = call_llm_wrapper(
            model_id=self.model_id,
            messages=[{"role": "user", "content": answer_prompt}],
            loan_id=loan_id,
        )
        fenced_match = re.search(
            r"```(?:json)?\s*(\[[\s\S]*?\])\s*```",
            processed_answer,
        )
        if fenced_match:
            extracted_list = fenced_match.group(1).strip()
        else:
            list_match = re.search(r"\[[^\]]*\]", processed_answer, re.DOTALL)
            if list_match:
                extracted_list = list_match.group(0).strip()
            else:
                extracted_list = processed_answer
        cleaned_answer = normalize_account_answer(
            extracted_list,
            account_last4_values,
        )
        # For experimental, raw_answer is the explanation + list, cleaned_answer is just the list
        return processed_answer, cleaned_answer, clean_in_tok, clean_out_tok
