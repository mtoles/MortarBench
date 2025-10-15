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

# Initialize joblib memory for caching
memory = Memory("joblib_cache", verbose=0)


STAGING_AUTH_TOKEN = os.getenv("STAGING_AUTH_TOKEN")

### SOLO Stuff ###

# --- Configuration ---
SOLO_ADMIN_TOKEN = (
    str(STAGING_AUTH_TOKEN).strip().replace("\r\n", "").replace("\n", "")
)  # Ensure it's a string and remove leading/trailing whitespace and newlines
AGENT_CLIENT_API_URL = "https://staging-api.tidalhq.com"
SOLO_ACCOUNT_ID = "cn5qbp13nbp48h1trvq0"

admin_headers = {
    "Authorization": f"Bearer {SOLO_ADMIN_TOKEN}",
    "Content-Type": "application/json",
    "accept": "application/json",
}


@memory.cache
def _cached_llm_call(model_id: str, messages: List[Dict[str, str]]) -> str:
    """
    Cached LLM call function.

    Args:
        model_id: The model identifier (e.g., 'gpt-4', 'gpt-5', 'claude-3-opus-20240229')
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        The model's response content as a string
    """
    if model_id.startswith("gpt-"):
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    elif model_id.startswith("claude-"):
        client = Anthropic()
        response = client.messages.create(
            model=model_id,
            max_tokens=4096,
            messages=messages,
        )
        return response.content[0].text.strip()
    else:
        raise ValueError(f"Invalid model ID: {model_id}")


def call_llm_wrapper(model_id: str, messages: List[Dict[str, str]], **kwargs) -> str:
    """
    Wrapper function for calling LLM models with caching.

    Args:
        model_id: The model identifier. Supports OpenAI models or 'solo' (raises NotImplementedError)
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        The model's response content as a string

    Raises:
        NotImplementedError: If model_id is 'solo'
    """
    if model_id == "solo":
        assert len(messages) == 1
        assert messages[-1]["role"] == "user"
        message, txn_search_result = send_message(
            messages[-1]["content"], kwargs["loan_id"]
        )
        # if txn_search_result.get("AssetTransactionSearchResult") and txn_search_result["AssetTransactionSearchResult"].get("Transactions"):
        #     transaction_ids = ", ".join([txn["TransactionID"] for txn in txn_search_result["AssetTransactionSearchResult"]["Transactions"]])
        # else:
        #     transaction_ids = "[No transaction data was referenced.]"
        if not txn_search_result:
            transaction_ids = "[No transaction data was referenced.]"

        result = (
            message["Content"]
            + "\n\nThe Solo Agent referenced the following data:\n\n"
            + str(txn_search_result)
        )
        return result

    else:  # GPT Models
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

        return _cached_llm_call(model_id, messages)


@memory.cache
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
    send_response.raise_for_status()  # Raise an exception for bad status codes
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
        raise Exception(
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
