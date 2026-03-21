"""
agent/graph.py
LangGraph graph definition — wires all nodes into the agent execution graph.

State flows: parse_goal → scan_inbox → check_memory → find_resources
               → find_calendar_slot → book_session → generate_brief → END

The TypedDict state is passed between nodes — each node reads what it needs
and adds its outputs. LangGraph persists this across the full run.
"""

from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END

from nodes import (
    parse_goal,
    scan_inbox,
    check_memory,
    find_resources,
    find_calendar_slot,
    book_session,
    generate_brief,
    send_daily_brief,
)


# ---------------------------------------------------------------------------
# State schema — the shared object passed between all nodes
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # Input
    goal: str

    # Populated by parse_goal
    sub_skills: list[str]

    # Populated by scan_inbox
    email_signals: list[Any]
    urgent_skills: list[str]

    # Populated by check_memory
    remaining_skills: list[str]
    next_topic: str
    skill_level: str                   # "beginner" | "intermediate" | "advanced"

    # Populated by find_resources
    resources: list[Any]

    # Populated by find_calendar_slot
    available_slot: Optional[Any]

    # Populated by book_session
    calendar_link: Optional[str]

    # Final output
    daily_brief: str
    email_sent: Optional[bool]


# ---------------------------------------------------------------------------
# Conditional edge — skip booking if no slot found
# ---------------------------------------------------------------------------

def should_book(state: AgentState) -> str:
    """
    Routes to book_session if a slot was found, else jumps straight to brief.
    """
    return "book_session" if state.get("available_slot") else "generate_brief"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("parse_goal",         parse_goal)
    graph.add_node("scan_inbox",         scan_inbox)
    graph.add_node("check_memory",       check_memory)
    graph.add_node("find_resources",     find_resources)
    graph.add_node("find_calendar_slot", find_calendar_slot)
    graph.add_node("book_session",       book_session)
    graph.add_node("generate_brief",     generate_brief)
    graph.add_node("send_daily_brief",   send_daily_brief)

    # Entry point
    graph.set_entry_point("parse_goal")

    # Linear edges
    graph.add_edge("parse_goal",         "scan_inbox")
    graph.add_edge("scan_inbox",         "check_memory")
    graph.add_edge("check_memory",       "find_resources")
    graph.add_edge("find_resources",     "find_calendar_slot")

    # Conditional: book only if slot exists
    graph.add_conditional_edges(
        "find_calendar_slot",
        should_book,
        {
            "book_session":   "book_session",
            "generate_brief": "generate_brief",
        },
    )

    graph.add_edge("book_session",     "generate_brief")
    graph.add_edge("generate_brief",   "send_daily_brief")
    graph.add_edge("send_daily_brief", END)

    return graph.compile()


# Singleton — import this anywhere
agent_graph = build_graph()
