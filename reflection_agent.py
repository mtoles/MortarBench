"""Backwards-compatible re-exports.

The reflection agent and its helpers were merged into :mod:`agents` to break
the circular import between :mod:`agents` and :mod:`reflection_agent`. This
module is kept only so existing `from reflection_agent import ...` callers
continue to work; prefer importing directly from :mod:`agents`.
"""

from agents import (
    PYTHON_TOOL,
    ReflectionAgent,
    call_with_python_tool,
    execute_python_safely,
    get_shared_retriever,
    normalize_dollar_answer,
)

__all__ = [
    "PYTHON_TOOL",
    "ReflectionAgent",
    "call_with_python_tool",
    "execute_python_safely",
    "get_shared_retriever",
    "normalize_dollar_answer",
]
