"""
agent/nodes.py
LangGraph node functions — each function is one step in the agent graph.

Flow:
  parse_goal → scan_inbox → check_memory → find_resources
      → find_calendar_slot → book_session → generate_brief

LLM brain: OpenRouter (openai-compatible)
Swap settings.model to change model — nothing else changes.
"""

from typing import Any
from datetime import date, timedelta
from openai import OpenAI

from settings import settings
from vector_store import LearnedTopicsMemory
from gmail_tool import fetch_recent_emails, send_email
from calendar_tool import find_free_slots, book_learning_session
from search_tool import find_learning_resources

# OpenRouter uses the OpenAI SDK — just point base_url at OpenRouter
# and pass your OpenRouter key. That's the entire integration change.
_client = OpenAI(
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
    default_headers={
        "HTTP-Referer": settings.openrouter_app_url,
        "X-Title": settings.openrouter_app_name,
    },
)
_memory = LearnedTopicsMemory()


def _offline_llm(system: str, user: str) -> str:
    """
    Deterministic fallback used for local debugging when remote LLM auth
    is unavailable. Keeps the graph runnable without API keys.
    """
    if "JSON array" in system:
        return (
            '["Python basics", "Linear algebra for ML", "Statistics fundamentals", '
            '"Supervised learning", "Unsupervised learning", "Model evaluation", '
            '"Scikit-learn projects"]'
        )

    return (
        "Focus on one concrete step today: study the selected topic, take notes, "
        "and complete one small hands-on exercise. If a calendar slot is available, "
        "treat it as non-negotiable deep work time."
    )


# ---------------------------------------------------------------------------
# Helper — call the LLM via OpenRouter
# ---------------------------------------------------------------------------

def _llm(system: str, user: str) -> str:
    """
    Single call point for all LLM reasoning in the agent.
    Swap settings.model to change the model — no other changes needed.

    Examples:
      "anthropic/claude-sonnet-4-5"   <- default
      "openai/gpt-4o"
      "google/gemini-2.0-flash-001"
      "meta-llama/llama-3.3-70b-instruct"
    """


    try:
        response = _client.chat.completions.create(
            model=settings.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"[LLM] Remote model unavailable ({exc}) — using offline debug fallback")
        return _offline_llm(system=system, user=user)


# ---------------------------------------------------------------------------
# Node 1 — Parse the user's goal into concrete sub-skills
# ---------------------------------------------------------------------------

def parse_goal(state: dict) -> dict:
    goal = state["goal"]
    print(f"\n[Node 1] Parsing goal: '{goal}'")

    response = _llm(
        system=(
            "You are a learning path architect. Given a goal, return a JSON array "
            "of 5-8 ordered sub-skills the user must learn, from foundational to advanced. "
            "Return ONLY the JSON array, no explanation."
        ),
        user=f"Goal: {goal}",
    )

    import json, re
    clean = re.sub(r"```json|```", "", response).strip()
    try:
        sub_skills = json.loads(clean)
        if not isinstance(sub_skills, list) or not sub_skills:
            raise ValueError("Expected non-empty JSON array")
    except Exception:
        sub_skills = []

    print(f"[Node 1] Sub-skills: {sub_skills}")
    return {**state, "sub_skills": sub_skills}


# ---------------------------------------------------------------------------
# Node 2 — Scan inbox for urgency signals
# ---------------------------------------------------------------------------

def scan_inbox(state: dict) -> dict:
    print("\n[Node 2] Scanning Gmail inbox...")
    try:
        signals = fetch_recent_emails(max_results=15)
    except Exception as exc:
        print(f"[Node 2] Gmail unavailable ({exc}) — continuing without inbox signals")
        signals = []

    urgent_skills = []
    for sig in signals:
        if sig.is_recruiter:
            urgent_skills.extend(sig.mentioned_skills)
            print(f"[Node 2] Recruiter signal: {sig.sender} — skills: {sig.mentioned_skills}")

    sub_skills = state["sub_skills"]
    reranked = sorted(
        sub_skills,
        key=lambda s: -sum(1 for u in urgent_skills if u.lower() in s.lower()),
    )

    return {**state, "sub_skills": reranked, "email_signals": signals, "urgent_skills": urgent_skills}


# ---------------------------------------------------------------------------
# Node 3 — Check memory: what has the user already learned?
# ---------------------------------------------------------------------------

def check_memory(state: dict) -> dict:
    print("\n[Node 3] Checking learned topics memory...")
    remaining = []

    for skill in state["sub_skills"]:
        if _memory.already_knows(skill):
            print(f"[Node 3] Already knows: '{skill}' — skipping")
        else:
            remaining.append(skill)

    next_topic = remaining[0] if remaining else state["sub_skills"][0]
    skill_level = _memory.get_skill_level(next_topic) or "beginner"

    print(f"[Node 3] Next topic: '{next_topic}' at {skill_level} level")
    return {**state, "remaining_skills": remaining, "next_topic": next_topic, "skill_level": skill_level}


# ---------------------------------------------------------------------------
# Node 4 — Find learning resources
# ---------------------------------------------------------------------------

def find_resources(state: dict) -> dict:
    print(f"\n[Node 4] Finding resources for '{state['next_topic']}'...")
    resources = find_learning_resources(
        topic=state["next_topic"],
        skill_level=state["skill_level"],
        top_k=3,
    )
    for r in resources:
        print(f"[Node 4] Found: {r.title} ({r.url})")

    return {**state, "resources": resources}


# ---------------------------------------------------------------------------
# Node 5 — Find a free calendar slot
# ---------------------------------------------------------------------------

def find_calendar_slot(state: dict) -> dict:
    print("\n[Node 5] Finding free calendar slot...")
    try:
        slots = find_free_slots(days_ahead=3, min_duration_mins=30)
    except Exception as exc:
        print(f"[Node 5] Calendar unavailable ({exc}) — skipping booking")
        return {**state, "booked_slot": None, "calendar_link": None}

    if not slots:
        print("[Node 5] No free slots found in the next 3 days")
        return {**state, "booked_slot": None, "calendar_link": None}

    best_slot = slots[0]
    print(f"[Node 5] Best slot: {best_slot}")
    return {**state, "available_slot": best_slot}


# ---------------------------------------------------------------------------
# Node 6 — Book the calendar event
# ---------------------------------------------------------------------------

def book_session(state: dict) -> dict:
    slot = state.get("available_slot")
    if not slot:
        return {**state, "calendar_link": None}

    resources = state.get("resources", [])
    resource_url = resources[0].url if resources else ""

    print(f"\n[Node 6] Booking '{state['next_topic']}' at {slot}...")
    try:
        link = book_learning_session(
            topic=state["next_topic"],
            slot=slot,
            resource_url=resource_url,
            duration_mins=settings.learning_slot_duration_mins,
        )
    except Exception as exc:
        print(f"[Node 6] Booking failed ({exc}) — returning without calendar link")
        link = None
    return {**state, "calendar_link": link}


# ---------------------------------------------------------------------------
# Node 7 — Generate the daily brief
# ---------------------------------------------------------------------------

def generate_brief(state: dict) -> dict:
    print("\n[Node 7] Generating daily brief...")

    resources_text = "\n".join(
        f"- {r.title}: {r.url}" for r in state.get("resources", [])
    )
    urgent = state.get("urgent_skills", [])
    slot = state.get("available_slot")
    slot_str = str(slot) if slot else "No slot found — calendar is full"

    brief = _llm(
        system=(
            "You are a personal learning coach. Write a concise, motivating daily brief "
            "(max 150 words). Be specific — mention the topic, why it matters now "
            "(based on recruiter signals), when the session is, and the top resource. "
            "No fluff."
        ),
        user=(
            f"Goal: {state['goal']}\n"
            f"Today's topic: {state['next_topic']} ({state['skill_level']} level)\n"
            f"Urgent skills from recruiters: {urgent}\n"
            f"Session time: {slot_str}\n"
            f"Resources:\n{resources_text}"
        ),
    )

    print(f"\n{'='*60}\nDAILY BRIEF\n{'='*60}\n{brief}\n{'='*60}\n")
    return {**state, "daily_brief": brief}


# ---------------------------------------------------------------------------
# Utility — Generate a full timetable from sub-skills (called from Streamlit)
# ---------------------------------------------------------------------------

def generate_timetable(sub_skills: list, start_date: date = None) -> list[dict]:
    """
    Builds a week-by-week timetable from the list of sub-skills.
    Each entry: { week, dates, skill, description, duration, daily_hours }
    """
    if start_date is None:
        start_date = date.today()

    timetable = []
    cursor = start_date

    for i, skill in enumerate(sub_skills):
        # Support both dict and plain string sub-skills
        if isinstance(skill, dict):
            name = skill.get("skill", str(skill))
            desc = skill.get("description", "")
            duration_str = skill.get("duration", "1 week")
        else:
            name = str(skill)
            desc = ""
            duration_str = "1 week"

        # Parse duration string -> number of weeks
        try:
            weeks = float(duration_str.split()[0])
        except (ValueError, IndexError):
            weeks = 1.0
        num_days = max(1, int(weeks * 7))

        end_day = cursor + timedelta(days=num_days - 1)
        timetable.append({
            "week": i + 1,
            "skill": name,
            "description": desc,
            "duration": duration_str,
            "start_date": cursor.strftime("%b %d"),
            "end_date": end_day.strftime("%b %d"),
            "daily_hours": 1.5,
        })
        cursor = end_day + timedelta(days=1)

    return timetable


# ---------------------------------------------------------------------------
# Node 8 — Send the daily brief via Gmail
# ---------------------------------------------------------------------------

def send_daily_brief(state: dict) -> dict:
    """
    Emails the generated daily brief to the user's own Gmail address.
    Fails gracefully if Gmail is unavailable.
    """
    brief = state.get("daily_brief", "")
    if not brief:
        print("[Node 8] No brief to send — skipping email")
        return state

    print("\n[Node 8] Sending daily brief via Gmail...")
    try:
        send_email(
            subject="[LifeCoach] Your Daily Learning Brief",
            body=brief,
        )
        print("[Node 8] Brief sent successfully")
    except Exception as exc:
        print(f"[Node 8] Email unavailable ({exc}) — skipping")
    return state