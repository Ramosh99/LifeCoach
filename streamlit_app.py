"""
streamlit_app.py
A thin UI client connecting to the LangGraph orchestrator in graph.py.
"""
import streamlit as st
import pandas as pd
import json
import uuid
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from settings import settings
from graph import agent_graph

def _is_set(v: str) -> bool:
    return bool((v or "").strip())

def _mask(v: str) -> str:
    if not v: return "Not set"
    return f"{v[:4]}...{v[-4:]}"

st.set_page_config(page_title="LifeCoach AI", page_icon="🧭", layout="wide")

# Persistent thread ID for LangGraph memory
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- Sidebar ---
with st.sidebar:
    st.subheader("Runtime Status")
    st.write(f"OpenRouter: {'✅' if _is_set(settings.openrouter_api_key) else '❌ Missing'}")
    with st.expander("Debug"):
        st.write(f"Model: {settings.model}")
        st.write(f"Key: {_mask(settings.openrouter_api_key)}")
        st.write(f"Thread: {st.session_state.thread_id[:8]}")
    st.divider()
    if st.button("🔄 Start Over", width="stretch"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

st.title("🧭 LifeCoach AI")
st.caption("Your personal AI life and learning coach powered by LangGraph.")
st.divider()

# Get current state from LangGraph
try:
    state = agent_graph.get_state(config)
    messages = state.values.get("messages", [])
except Exception:
    messages = []

# If no messages exist yet, have the bot send a greeting
if not messages:
    greeting = "👋 Hi! I'm your AI LifeCoach. I can help you set goals, build study routines, or discuss career plans. What's on your mind today?"
    # We push an initial AIMessage to the graph to anchor the conversation
    agent_graph.update_state(config, {"messages": [AIMessage(content=greeting)]})
    state = agent_graph.get_state(config)
    messages = state.values.get("messages", [])

def _render_timetable(json_str: str):
    try:
        data = json.loads(json_str)
        if isinstance(data, list) and len(data) > 0:
            st.markdown("🗓️ **Your Personalized Study Timetable:**")
            df = pd.DataFrame([{
                "Week": f"Week {r.get('week', '')}",
                "Dates": f"{r.get('start_date', '')} → {r.get('end_date', '')}",
                "Topic": r.get("skill", ""),
                "Duration": r.get("duration", ""),
                "Notes": r.get("daily_hours", "")
            } for r in data])
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Failed to render timetable natively. Raw output:\n{json_str}")

# Render history
for msg in messages:
    if isinstance(msg, AIMessage):
        # Only render AI content if it exists
        if msg.content:
            with st.chat_message("assistant"):
                st.markdown(msg.content)
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, ToolMessage):
        if msg.name == "build_personalized_timetable":
            # The tool returned the raw JSON timetable, render it nicely
            with st.chat_message("assistant"):
                _render_timetable(msg.content)

# Input
if prompt := st.chat_input("Type your message..."):
    # Immediately render the user's box
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke Graph
                agent_graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]}, 
                    config=config
                )
                
                # Rerun to cleanly re-draw all messages including tool outputs
                st.rerun()
            except Exception as e:
                st.error(f"Error communicating with the AI: {e}")
