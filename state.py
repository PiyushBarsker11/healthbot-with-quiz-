"""
State shared across all nodes in the HealthBot workflow.
Every node reads/writes to this; LangGraph merges partial
updates back into it automatically.
"""

from typing import TypedDict, Optional


class HealthBotState(TypedDict, total=False):
    topic: str
    search_results: str
    summary: str
    quiz_question: str
    quiz_answer: str
    grade: str
    explanation: str
    want_new_topic: Optional[bool]
