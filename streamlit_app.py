"""
streamlit_app.py
LifeCoach as a chat interface — all decisions happen via st.chat_input().

Conversation flow:
  🤖 "What's your learning goal?"
  👤 "Learn ML in 3 months"
  🤖 [runs workflow, shows brief + resources]
     "Would you like to create a timetable? (yes / no)"
  👤 "yes"
  🤖 [shows timetable]
     "Any refinements? Describe changes or type 'done'"
  👤 "done"  OR  "extend math to 4 weeks"
  🤖 [applies refinements or confirms]
     "Add these sessions to Google Calendar? (yes / no)"
  👤 "yes"
  🤖 [books calendar events, shows summary]
"""

import traceback
import streamlit as st
import pandas as pd

from settings import settings


# ─── Helpers ────────────────────────────────────────────────────────────────

def _is_set(v: str) -> bool:
    return bool((v or "").strip())

def _mask(v: str) -> str:
    if not v: return "Not set"
    if len(v) <= 8: return "*" * len(v)
    return f"{v[:4]}...{v[-4:]}"

def _init():
    defaults = {
        "messages":    [],   # [{role, content}]  — chat history
        "step":        "ask_goal",
        "result":      None,
        "timetable":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _bot(text: str, render: bool = True):
    """Append a bot message to history and (optionally) render it immediately."""
    st.session_state.messages.append({"role": "assistant", "content": text})
    if render:
        with st.chat_message("assistant"):
            st.markdown(text)

def _is_yes(text: str) -> bool:
    return text.strip().lower() in {"yes", "y", "yeah", "yep", "sure", "ok", "okay"}

def _is_no(text: str) -> bool:
    return text.strip().lower() in {"no", "n", "nope", "nah", "skip"}


# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="LifeCoach AI", page_icon="🧭", layout="wide")
_init()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Runtime Status")
    st.write(f"OpenRouter: {'✅' if _is_set(settings.openrouter_api_key) else '❌ Missing'}")
    st.write(f"Tavily: {'✅' if _is_set(__import__('os').getenv('TAVILY_API_KEY','')) else '❌ Missing'}")
    st.write(f"Google creds: {'✅' if __import__('os').path.exists(settings.google_credentials_path) else '❌ Missing'}")
    with st.expander("Debug"):
        st.write(f"Model: {settings.model}")
        st.write(f"OAuth port: {settings.google_oauth_port}")
        st.write(f"Key: {_mask(settings.openrouter_api_key)}")
    st.divider()
    if st.button("🔄 Start Over", width="stretch"):
        for k in ["messages", "step", "result", "timetable"]:
            st.session_state.pop(k, None)
        st.rerun()

# ─── Title ───────────────────────────────────────────────────────────────────

st.title("🧭 LifeCoach AI")
st.caption("Your personal AI learning coach. Just chat — I'll handle planning, resources, and scheduling.")
st.divider()

# ─── Replay existing message history ─────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Send opening message on first load ──────────────────────────────────────

if st.session_state.step == "ask_goal" and not st.session_state.messages:
    _bot("👋 Hi! I'm your LifeCoach AI.\n\n**What's your learning goal?**\n\n*Example: \"Learn ML fundamentals in 3 months\"*")


# ═══════════════════════════════════════════════════════════════════════════
# Main chat input handler
# ═══════════════════════════════════════════════════════════════════════════

prompt = st.chat_input("Type your message...")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    step = st.session_state.step

    # ── STEP 1: Receive goal, run agent ──────────────────────────────────────
    if step == "ask_goal":
        with st.chat_message("assistant"):
            with st.status("Running LifeCoach workflow...", expanded=True) as status:
                st.write("🔍 Parsing goal into sub-skills...")
                st.write("📬 Scanning Gmail inbox for signals...")
                st.write("🧠 Checking your learning memory...")
                st.write("🔗 Finding best resources...")
                st.write("📝 Generating your daily brief...")
                try:
                    from runner import run_agent
                    result = run_agent(prompt.strip())
                    st.session_state.result = result
                    status.update(label="✅ Done!", state="complete")
                except Exception as exc:
                    status.update(label="❌ Failed", state="error")
                    st.error(f"Error: {exc}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())
                    st.stop()

            # Show results inline
            resources = result.get("resources", [])
            res_lines = "\n".join(
                f"- [{r.get('title','Resource')}]({r.get('url','')})" if r.get('url')
                else f"- {r.get('title','Resource')}"
                for r in resources
            )
            summary = (
                f"**🎯 Goal:** {result.get('goal')}\n\n"
                f"**📖 Today's topic:** {result.get('next_topic')} *(level: {result.get('skill_level','beginner')})*\n\n"
                f"**📚 Resources:**\n{res_lines or '_None found_'}\n\n"
                f"**📋 Daily Brief:**\n> {result.get('daily_brief','')}"
            )
            st.markdown(summary)
            st.session_state.messages.append({"role": "assistant", "content": summary})

        # Ask timetable question
        _bot("Would you like me to create a **full study timetable** mapping each topic to specific weeks?\n\n👉 Reply **yes** or **no**")
        st.session_state.step = "ask_timetable"

    # ── STEP 2: Timetable decision ─────────────────────────────────────────
    elif step == "ask_timetable":
        if _is_yes(prompt):
            from nodes import generate_timetable
            sub_skills = st.session_state.result.get("sub_skills", [])
            timetable = generate_timetable(sub_skills)
            st.session_state.timetable = timetable

            with st.chat_message("assistant"):
                st.markdown("🗓️ **Here's your study timetable:**")
                df = pd.DataFrame([{
                    "Week": f"Week {r['week']}",
                    "Dates": f"{r['start_date']} → {r['end_date']}",
                    "Topic": r["skill"],
                    "Duration": r["duration"],
                    "Daily": f"{r['daily_hours']}h/day",
                } for r in timetable])
                st.dataframe(df, use_container_width=True, hide_index=True)
                note = "🗓️ **Timetable generated!** (see above)"
                st.session_state.messages.append({"role": "assistant", "content": note})

            _bot("Any **refinements** you'd like?\n\n*Examples: \"extend Mathematics to 4 weeks\", \"swap week 1 and 2\", \"remove Data Visualization\"*\n\nOr type **done** if it looks good.")
            st.session_state.step = "ask_refine"

        elif _is_no(prompt):
            _bot("No problem! Would you like to add **individual learning sessions** to your **Google Calendar**?\n\n👉 Reply **yes** or **no**")
            st.session_state.step = "ask_calendar"
        else:
            _bot("Please reply **yes** or **no** — would you like a timetable?")

    # ── STEP 3: Refinements ───────────────────────────────────────────────
    elif step == "ask_refine":
        if prompt.strip().lower() in {"done", "looks good", "good", "ok", "okay", "no", "none"}:
            _bot("Great! Would you like to add these sessions to **Google Calendar**?\n\n👉 Reply **yes** or **no**")
            st.session_state.step = "ask_calendar"
        else:
            # Apply refinements with LLM
            with st.chat_message("assistant"):
                with st.spinner("Applying your refinements..."):
                    import json, re
                    from nodes import _llm
                    timetable = st.session_state.timetable
                    response = _llm(
                        system=(
                            "You are a learning schedule optimizer. Given a JSON timetable and user refinements, "
                            "return an updated JSON array with the same structure keys: "
                            "week, skill, description, duration, start_date, end_date, daily_hours. "
                            "Return ONLY the valid JSON array."
                        ),
                        user=f"Timetable:\n{json.dumps(timetable, indent=2)}\n\nRefinements:\n{prompt}",
                    )
                    clean = re.sub(r"```json|```", "", response).strip()
                    try:
                        updated = json.loads(clean)
                        for i, row in enumerate(updated):
                            row["week"] = i + 1
                        st.session_state.timetable = updated
                        st.markdown("✅ **Updated timetable:**")
                        df = pd.DataFrame([{
                            "Week": f"Week {r['week']}",
                            "Dates": f"{r.get('start_date','')} → {r.get('end_date','')}",
                            "Topic": r["skill"],
                            "Duration": r["duration"],
                        } for r in updated])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.session_state.messages.append({"role": "assistant", "content": "✅ Timetable updated (see above)"})
                    except Exception:
                        st.warning("Couldn't parse the refined timetable — keeping original.")
                        st.session_state.messages.append({"role": "assistant", "content": "⚠️ Could not parse refinements, keeping original timetable."})

            _bot("Any more changes? Or type **done** to continue.")

    # ── STEP 4: Calendar decision ─────────────────────────────────────────
    elif step == "ask_calendar":
        if _is_yes(prompt):
            result = st.session_state.result
            timetable = st.session_state.timetable
            resources = result.get("resources", [])
            resource_url = resources[0].get("url", "") if resources else ""

            topics = []
            if timetable:
                topics = [r["skill"] for r in timetable]
            else:
                for s in result.get("sub_skills", []):
                    topics.append(s.get("skill", str(s)) if isinstance(s, dict) else str(s))

            booked_lines = []
            with st.chat_message("assistant"):
                with st.status("Booking calendar sessions...", expanded=True) as status:
                    try:
                        from calendar_tool import find_free_slots, book_learning_session
                        slots = find_free_slots(days_ahead=len(topics) * 2 + 7, min_duration_mins=30)
                        slot_idx = 0
                        for topic in topics:
                            if slot_idx >= len(slots):
                                st.write(f"⚠️ No slot for: {topic}")
                                continue
                            slot = slots[slot_idx]
                            try:
                                link = book_learning_session(
                                    topic=topic, slot=slot,
                                    resource_url=resource_url,
                                    duration_mins=settings.learning_slot_duration_mins,
                                )
                                time_str = slot.start.strftime("%a %b %d %H:%M")
                                st.write(f"✅ **{topic}** → {time_str}")
                                booked_lines.append(f"- [{topic}]({link}) — {time_str}" if link else f"- {topic} — {time_str}")
                                slot_idx += 1
                            except Exception as e:
                                st.write(f"❌ {topic}: {e}")
                                slot_idx += 1
                        status.update(label=f"✅ Booked {len(booked_lines)}/{len(topics)} sessions", state="complete")
                    except Exception as exc:
                        status.update(label="❌ Calendar unavailable", state="error")
                        st.error(f"Calendar connection failed: {exc}")
                        st.info("💡 Complete the Google OAuth flow first — add your email as a test user in Google Cloud Console.")
                        booked_lines = []

                summary_msg = (
                    "🎉 **All done! Your plan is set.**\n\n"
                    + ("**📅 Calendar sessions booked:**\n" + "\n".join(booked_lines) if booked_lines else "_No sessions booked._")
                )
                st.markdown(summary_msg)
                st.session_state.messages.append({"role": "assistant", "content": summary_msg})
                st.balloons()

            _bot("You're all set! Start a new goal anytime with the **🔄 Start Over** button in the sidebar.")
            st.session_state.step = "done"

        elif _is_no(prompt):
            st.session_state.messages.append({"role": "assistant", "content": "✅ No calendar booking — you're all set!"})
            with st.chat_message("assistant"):
                st.markdown("✅ No problem! You're all set. Use **🔄 Start Over** to plan a new goal.")
                st.balloons()
            st.session_state.step = "done"
        else:
            _bot("Please reply **yes** or **no** — add sessions to Google Calendar?")

    # ── STEP 5: Done ──────────────────────────────────────────────────────
    elif step == "done":
        _bot("You've completed your plan! 🎉 Use **🔄 Start Over** in the sidebar to plan a new goal.")

    st.rerun()
