"""
streamlit_app.py
A pure, conversational AI LifeCoach chat bot. 
No external tools or APIs.
"""

import streamlit as st
from openai import OpenAI
from settings import settings

# Initialize OpenRouter / OpenAI client
_client = OpenAI(
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
    default_headers={
        "HTTP-Referer": settings.openrouter_app_url,
        "X-Title": settings.openrouter_app_name,
    },
)

def _is_set(v: str) -> bool:
    return bool((v or "").strip())

def _mask(v: str) -> str:
    if not v: return "Not set"
    if len(v) <= 8: return "*" * len(v)
    return f"{v[:4]}...{v[-4:]}"

# --- Page Config ---
st.set_page_config(page_title="LifeCoach AI", page_icon="🧭", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.subheader("Runtime Status")
    st.write(f"OpenRouter: {'✅' if _is_set(settings.openrouter_api_key) else '❌ Missing'}")
    with st.expander("Debug"):
        st.write(f"Model: {settings.model}")
        st.write(f"Key: {_mask(settings.openrouter_api_key)}")
    st.divider()
    if st.button("🔄 Start Over", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# --- Title ---
st.title("🧭 LifeCoach AI")
st.caption("Your personal AI life and learning coach. Let's chat!")
st.divider()

# --- Initial Greeting ---
if not st.session_state.messages:
    greeting = "👋 Hi! I'm your AI LifeCoach. I can help you set goals, build study routines, or discuss career plans. What's on your mind today?"
    st.session_state.messages.append({"role": "assistant", "content": greeting})

# --- Display History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                system_prompt = (
                    "You are LifeCoach AI, a friendly, encouraging, and insightful personal life and learning coach. "
                    "Your goal is to help the user identify their goals, break them down into actionable steps, "
                    "and provide ongoing motivation and context-aware advice. "
                    "Keep your responses conversational, engaging, and highly focused on the user's personal growth."
                )
                
                # Build message payload for the LLM
                api_messages = [{"role": "system", "content": system_prompt}]
                for m in st.session_state.messages:
                    api_messages.append({"role": m["role"], "content": m["content"]})

                # Stream response (or wait for full completion)
                response = _client.chat.completions.create(
                    model=settings.model,
                    max_tokens=1500,
                    messages=api_messages,
                )
                
                reply = response.choices[0].message.content
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
            except Exception as e:
                st.error(f"Error communicating with the AI: {e}")
