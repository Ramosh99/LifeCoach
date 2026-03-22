"""
graph.py
LangGraph orchestration for LifeCoach.
Handles context gathering, memory, and timetable generation.
"""

import json
import re
from typing import Annotated, TypedDict
from datetime import date, timedelta

from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from settings import settings

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def _extract_json_array(text: str):
    clean = re.sub(r"```[a-z]*\n?", "", text).strip().strip("`").strip()
    start = clean.find("[")
    end = clean.rfind("]")
    if start != -1 and end != -1 and end > start:
        clean = clean[start:end + 1]
    return json.loads(clean)

@tool
def build_personalized_timetable(goals: list[str], user_schedule_constraints: str) -> str:
    """
    Call this tool ONLY when the user explicitly asks for, or agrees to, generating the study timetable!
    Pass in the list of goals/topics they want to learn, and comprehensively summarize their schedule constraints.
    Returns a JSON string representing the built timetable.
    """
    builder_llm = ChatOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.model,
        default_headers={
            "HTTP-Referer": settings.openrouter_app_url,
            "X-Title": settings.openrouter_app_name,
        },
        max_tokens=1500,
    )
    
    system = (
        "You are a JSON timetable builder. Output ONLY a raw JSON array. "
        "Each object must have keys: week (int), skill (str), duration (str), start_date (str like 'Apr 01'), end_date (str), daily_hours (str). "
        "Respect the scheduling constraints provided."
    )
    user_prompt = f"Goals to schedule: {goals}\n\nConstraints:\n{user_schedule_constraints}"
    
    response = builder_llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt}
    ])
    
    try:
        # Validate that it parses as JSON array
        data = _extract_json_array(response.content)
        for i, row in enumerate(data):
            row["week"] = row.get("week", i + 1)
        return json.dumps(data)
    except Exception as e:
        # Fallback to simple deterministic structure
        rows = []
        cursor = date.today()
        for i, skill in enumerate(goals):
            end_day = cursor + timedelta(days=6)
            rows.append({
                "week": i + 1,
                "skill": str(skill),
                "description": "",
                "duration": "1 week",
                "start_date": cursor.strftime("%b %d"),
                "end_date": end_day.strftime("%b %d"),
                "daily_hours": "1-2 hrs",
            })
            cursor = end_day + timedelta(days=1)
        return json.dumps(rows)

tools = [build_personalized_timetable]

# Initialize LLM
llm = ChatOpenAI(
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
    model=settings.model,
    default_headers={
        "HTTP-Referer": settings.openrouter_app_url,
        "X-Title": settings.openrouter_app_name,
    },
    max_tokens=1500,
)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: AgentState):
    system_prompt = (
        "You are LifeCoach AI, a warm, empathetic personal coach who communicates like a real therapist — "
        "calm, curious, and deeply personal.\n\n"
        "CRITICAL RULE: You must ALWAYS ask only ONE question per message. Never list multiple questions "
        "or use bullet points to ask several things at once. Ask one question, wait for the answer, then ask the next.\n\n"
        "Your flow:\n"
        "1. Warmly acknowledge the user's goal with genuine enthusiasm.\n"
        "2. Then, one question at a time, naturally learn about their daily life: "
        "   their work schedule, free time, preferred study session length, learning style, any commitments, etc. "
        "   Ask follow-up questions based on their answers, just like a real conversation.\n"
        "3. Once you feel you have a complete picture of their routine (after at least 3-4 exchanges), "
        "   tell them you now have everything you need, and ask ONE final question: "
        "   'Would you like me to generate your personalized timetable?'\n"
        "4. ONLY if they say YES, call the `build_personalized_timetable` tool.\n"
        "5. After the tool returns, celebrate their plan warmly, then ask if they'd like any changes."
    )
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Build the Graph
builder = StateGraph(AgentState)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", END: END})
builder.add_edge("tools", "chatbot")

memory = MemorySaver()
agent_graph = builder.compile(checkpointer=memory)
