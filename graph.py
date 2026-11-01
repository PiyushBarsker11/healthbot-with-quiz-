"""
Wires the nodes into a LangGraph StateGraph with a checkpointer.
A checkpointer is REQUIRED for interrupt()/Command(resume=...) to work -
it's what lets the graph pause and be resumed later (even across
Streamlit reruns) instead of losing its place.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import HealthBotState
from nodes import (
    get_topic_node,
    search_node,
    summarize_node,
    comprehension_ready_node,
    quiz_node,
    get_answer_node,
    grade_node,
    continue_node,
    reset_node,
)


def _route_after_continue(state: HealthBotState) -> str:
    return "reset" if state.get("want_new_topic") else "end"


def build_graph():
    graph = StateGraph(HealthBotState)

    graph.add_node("get_topic", get_topic_node)
    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("comprehension_ready", comprehension_ready_node)
    graph.add_node("quiz", quiz_node)
    graph.add_node("get_answer", get_answer_node)
    graph.add_node("grade", grade_node)
    graph.add_node("continue_check", continue_node)
    graph.add_node("reset", reset_node)

    graph.add_edge(START, "get_topic")
    graph.add_edge("get_topic", "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", "comprehension_ready")
    graph.add_edge("comprehension_ready", "quiz")
    graph.add_edge("quiz", "get_answer")
    graph.add_edge("get_answer", "grade")
    graph.add_edge("grade", "continue_check")
    graph.add_conditional_edges(
        "continue_check",
        _route_after_continue,
        {"reset": "reset", "end": END},
    )
    graph.add_edge("reset", "get_topic")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
