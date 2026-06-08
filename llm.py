"""
LLM wrapper module for handling model calls with caching.

This module provides a unified interface for calling different LLM models,
with disk-based caching to avoid redundant API calls.
"""

import json
from typing import List, Dict, Any
from openai import OpenAI
from anthropic import Anthropic
from joblib import Memory
import os
import requests
import json
import uuid
import time
import urllib.parse
from datetime import datetime, timezone
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

# Load environment variables from .env file
load_dotenv(override=True)

# Initialize joblib memory for caching
memory = Memory("joblib_cache", verbose=0)


# Module-level seed used both as a cache-key component and (where the provider
# supports it) as a request parameter. Set via `set_llm_seed`; defaults to None
# which keeps cache behavior identical to the pre-seed version.
_seed = None


def set_llm_seed(seed):
    """Set the seed used for caching and (where supported) sampling."""
    global _seed
    _seed = seed


gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Read Anthropic API key from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

### SOLO Stuff ###

# --- Configuration ---
STAGING_AUTH_TOKEN = os.getenv("STAGING_AUTH_TOKEN")
SOLO_ADMIN_TOKEN = (
    str(STAGING_AUTH_TOKEN).strip().replace("\r\n", "").replace("\n", "")
)  # Ensure it's a string and remove leading/trailing whitespace and newlines
AGENT_CLIENT_API_URL = "https://staging-api.tidalhq.com"
SOLO_ACCOUNT_ID = "d45upb94l9puh1k8kdrg"

admin_headers = {
    "Authorization": f"Bearer {SOLO_ADMIN_TOKEN}",
    "Content-Type": "application/json",
    "accept": "application/json",
}


@memory.cache
def _cached_llm_call(model_id: str, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None, seed=None) -> tuple:
    """
    Cached LLM call. Implementation lives directly inside the decorated function
    so joblib's source-fingerprint catches behavior changes (e.g. swapping SDKs)
    and invalidates stale entries automatically. To bypass the cache, call
    `_cached_llm_call.func(...)` — joblib exposes the un-decorated function
    there.

    Args:
        model_id: The model identifier (e.g., 'gpt-4', 'gpt-5', 'claude-3-opus-20240229')
        messages: List of message dictionaries with 'role' and 'content' keys
        tools: Optional list of tool dictionaries (e.g., [{"type": "web_search"}])

    Returns:
        Tuple of (response_content, input_tokens, output_tokens)
    """
    if model_id.startswith("gpt-"):
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=OPENAI_API_KEY)
        # Use Chat Completions API which supports messages directly
        kwargs = {"model": model_id, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if seed is not None:
            kwargs["seed"] = seed

        response = client.chat.completions.create(**kwargs)
        
        content = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        # Check for reasoning tokens if available
        if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
            if hasattr(response.usage.completion_tokens_details, 'reasoning_tokens'):
                reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
                if reasoning_tokens:
                    output_tokens += reasoning_tokens
        
        return content, input_tokens, output_tokens
    elif model_id.startswith("claude-"):
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=model_id,
            # Required by the Anthropic API. Set generously so Claude isn't
            # truncated relative to the GPT/Gemini paths, which don't cap output.
            max_tokens=8192,
            messages=messages,
            temperature=1.0,
        )
        content = response.content[0].text.strip()
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        return content, input_tokens, output_tokens
    elif model_id.startswith("gemini-"):
        contents = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                genai_types.Content(role=role, parts=[genai_types.Part(text=msg["content"])])
            )
        contents.append(
            genai_types.Content(role="user", parts=[genai_types.Part(text=messages[-1]["content"])])
        )

        config = genai_types.GenerateContentConfig(seed=seed) if seed is not None else None
        response = gemini_client.models.generate_content(
            model=model_id, contents=contents, config=config,
        )

        content = response.text.strip()
        usage = response.usage_metadata
        # SDK types every count as Optional[int]; we've observed total and
        # candidates fields come back as None for some Gemini 3.x responses.
        # Force ints so downstream `+=` arithmetic and joblib-cached tuples
        # never propagate None.
        in_tok = int(usage.prompt_token_count or 0)
        out_tok = int(
            (usage.candidates_token_count or 0)
            + (usage.thoughts_token_count or 0)
            + (usage.tool_use_prompt_token_count or 0)
        )
        return content, in_tok, out_tok
    else:
        raise ValueError(f"Invalid model ID: {model_id}")

def call_llm_wrapper(model_id: str, messages: List[Dict[str, str]], **kwargs) -> tuple:
    """
    Wrapper function for calling LLM models with caching.

    Args:
        model_id: The model identifier. Supports OpenAI models or 'solo' (raises NotImplementedError)
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        Tuple of (response_content, input_tokens, output_tokens)
        For solo model, returns (response_content, 0, 0) as it has no cost

    Raises:
        NotImplementedError: If model_id is 'solo'
    """
    if model_id == "solo":
        assert len(messages) == 1
        assert messages[-1]["role"] == "user"
        
        # Retry logic for timeout errors (up to 5 attempts)
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                message, txn_search_result = send_message(
                    messages[-1]["content"], kwargs["loan_id"]
                )
                break  # Success, exit retry loop
            except TimeoutError as e:
                if attempt < max_retries:
                    print(f"Timeout error on attempt {attempt}/{max_retries}. Retrying...")
                    time.sleep(2)  # Wait 2 seconds before retrying
                else:
                    print(f"Failed after {max_retries} attempts.")
                    raise  # Re-raise the timeout error after all retries exhausted
        # if txn_search_result.get("AssetTransactionSearchResult") and txn_search_result["AssetTransactionSearchResult"].get("Transactions"):
        #     transaction_ids = ", ".join([txn["TransactionID"] for txn in txn_search_result["AssetTransactionSearchResult"]["Transactions"]])
        # else:
        #     transaction_ids = "[No transaction data was referenced.]"
        if not txn_search_result:
            transaction_ids = "[No transaction data was referenced.]"

        result = (
            message["Content"]
            + "\n\nThe Solo Agent referenced the following data:\n\n"
            + "==========\n"
            + str(txn_search_result)
        )
        return result, 0, 0

    else:  # GPT and Claude Models
        # Validate messages format
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list of dictionaries")

        for message in messages:
            if (
                not isinstance(message, dict)
                or "role" not in message
                or "content" not in message
            ):
                raise ValueError(
                    "Each message must be a dict with 'role' and 'content' keys"
                )

        # Extract tools from kwargs if provided
        tools = kwargs.get("tools", None)
        if os.getenv("DISABLE_LLM_CACHE", "").lower() in ("1", "true", "yes"):
            return _cached_llm_call.func(model_id, messages, tools, _seed)
        return _cached_llm_call(model_id, messages, tools, _seed)


def clear_messages(loan_id):
    """Clears messages for a given loan ID."""
    loan_id = str(loan_id)
    print(f"Clearing messages for loan ID: {loan_id}")
    clear_url = f"{AGENT_CLIENT_API_URL}/api/chat/messages/clear"
    clear_payload = {
        "LoanID": loan_id
    }
    try:
        clear_response = requests.post(clear_url, headers=admin_headers, data=json.dumps(clear_payload))
        clear_response.raise_for_status()  # Raise an exception for bad status codes
        print("Messages cleared successfully!")
        print("Response:", clear_response.json())
        return clear_response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error clearing messages: {e}")
        print(f"Response content: {clear_response.text}")
        return None


# @memory.cache
def send_message(message, loan_id):
    """Sends a message to the chat API and polls for response."""
    print(f"Sending message: {message}")

    # Send Message Config
    send_url = f"{AGENT_CLIENT_API_URL}/api/chat/messages"
    message_id = str(uuid.uuid4())  # Generate a unique ID for the message
    current_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    send_payload = {
        "content": {
            "AccountID": SOLO_ACCOUNT_ID,
            "FromAccountID": SOLO_ACCOUNT_ID,
            "IsSent": False,
            "CreatedAt": current_time,
            "UpdatedAt": current_time,
            "ID": message_id,
            "UseChatBot": True,
            "text": message,
            "user": {"_id": SOLO_ACCOUNT_ID},
            "createdAt": current_time,
            "_id": str(uuid.uuid4()),  # Another unique ID
            "Content": message,
        },
        "context": {"loanID": str(loan_id), "page": "Loan"},
    }
    send_response = requests.post(
        send_url, headers=admin_headers, data=json.dumps(send_payload)
    )
    # try:
    _ = send_response.raise_for_status()  # Raise an exception for bad status codes
    #     print("Message sent successfully!")
    #     print("Response:", send_response.json())  # Print the JSON response
    # except requests.exceptions.RequestException as e:
    #     print(f"Error sending message: {e}")
    #     print(f"Response content: {send_response.text}")
    #     return None, None  # Indicate failure

    # Poll for response
    print("Polling for response...")
    TIMEOUT_SECONDS = 100
    start_time = time.time()
    response_received = False
    cnt = 0
    response = None
    transaction_search_result = {}

    context_dict = {"loanID": loan_id, "page": "Loan"}
    context_str = json.dumps(context_dict, separators=(",", ":"))
    context_encoded = urllib.parse.quote(context_str)
    poll_url = f"{AGENT_CLIENT_API_URL}/api/chat/messages?loanID={loan_id}&contextual=1&context={context_encoded}"

    while time.time() - start_time < TIMEOUT_SECONDS:
        # print("polling #", cnt+1)
        cnt += 1

        # try:
        get_response = requests.get(poll_url, headers=admin_headers)
        get_response.raise_for_status()
        get_response_data = get_response.json()

        # Check if second to last message is from user, and the last message is from SOLO, that is the message we want
        last_message = get_response_data["Messages"][-1]
        second_to_last_message = get_response_data["Messages"][-2]
        is_last_message_from_solo = last_message.get("FromSystem", False) == True
        last_message_type = last_message.get("SystemMessageType")
        last_message_content = last_message.get("Content", "")

        if (
            last_message_content != "Thinking..."
            and is_last_message_from_solo
            and last_message_type != "system"
            and last_message_type != "notification"
        ):
            # print("\n--- SOLO's Response Text ---")
            # print(last_message.get('Content', {}))
            # print("\n--- SOLO's Transaction Search Result ---")
            # print(json.dumps(last_message.get('Ext', {}), indent=2))
            response_received = True
            response = last_message
            transaction_search_result = last_message.get("Ext", {})
            break  # Found the response, exit the polling loop

        # except requests.exceptions.RequestException as e:
        #     # print(f"Error polling for message: {e}")
        #     raise Exception(f"Error polling for message: {e}")
        # Continue polling in case of transient errors

        time.sleep(3)  # Wait for 3 second before the next poll

    if not response_received:
        raise TimeoutError(
            "Timeout: Did not receive a response within the specified time."
        )
    return response, transaction_search_result


if __name__ == "__main__":
    LOAN_IDS = ["86744679655", "81613557991", "83352063666", "81301535410"]

    QUESTIONS = [
        "Identify any large deposits on the borrower’s bank statements.",
        "Which deposits on the bank statement, if any, appear to originate from a cryptocurrency source?",
        "Do the bank statements show evidence of any rental payments?",
    ]

    for loan_id in LOAN_IDS:
        for question in QUESTIONS:
            print(f"\n\nUsing loan {loan_id}, with question: {question}")
            clear_messages(loan_id)

            message, txn_search_result = send_message(question, loan_id)
            if message is None:
                print("Failed to send message")
                break

            print(message)
            print(json.dumps(txn_search_result, indent=2))

    # example call to the send_message function
    message, txn_search_result = send_message(
        # "Are there any small deposits under 50 dollars?", "301535410" # from doc
        "Are there any small deposits under 50 dollars?",
        "86744679655",
    )
