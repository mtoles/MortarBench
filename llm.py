"""
LLM wrapper module for handling model calls with caching.

This module provides a unified interface for calling different LLM models,
with disk-based caching to avoid redundant API calls.
"""

import json
from typing import List, Dict, Any
from openai import OpenAI
from joblib import Memory

# Initialize joblib memory for caching
memory = Memory("cache/llm_calls", verbose=0)

client = OpenAI()


@memory.cache
def _cached_llm_call(model_id: str, messages: List[Dict[str, str]]) -> str:
    """
    Cached LLM call function.

    Args:
        model_id: The model identifier (e.g., 'gpt-4', 'gpt-5')
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        The model's response content as a string
    """
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


def call_llm_wrapper(model_id: str, messages: List[Dict[str, str]]) -> str:
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
        raise NotImplementedError("Solo model not implemented yet")

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
