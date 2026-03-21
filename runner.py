"""
agent/runner.py
Entry point — run the LifeCoach agent with a user goal.

Usage:
    python -m agent.runner --goal "Become a product manager in 6 months"

Or import and call run_agent() from your own code / scheduler.
"""

import argparse
from settings import enable_langsmith_tracing
from graph import agent_graph


def run_agent(goal: str) -> dict:
    """
    Runs the full LifeCoach agent pipeline for a given goal.
    Returns the final state including the daily_brief.

    LangSmith automatically traces every LLM call and tool use —
    view traces at https://smith.langchain.com
    """
    enable_langsmith_tracing()

    print(f"\n[LifeCoach] Starting agent for goal: '{goal}'\n")

    # Invoke the compiled LangGraph
    final_state = agent_graph.invoke({"goal": goal})

    return {
        "goal":          final_state["goal"],
        "sub_skills":    final_state.get("sub_skills", []),
        "next_topic":    final_state.get("next_topic"),
        "skill_level":   final_state.get("skill_level"),
        "calendar_link": final_state.get("calendar_link"),
        "daily_brief":   final_state.get("daily_brief"),
        "resources":     [
            {"title": r.title, "url": r.url}
            for r in final_state.get("resources", [])
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LifeCoach AI Agent")
    parser.add_argument(
        "--goal",
        type=str,
        default="Learn machine learning fundamentals in 3 months",
        help="Your learning/career goal",
    )
    args = parser.parse_args()

    result = run_agent(args.goal)

    print("\n--- RESULT ---")
    print(f"Next topic : {result['next_topic']} ({result['skill_level']})")
    print(f"Calendar   : {result['calendar_link'] or 'Not booked'}")
    print(f"\nBrief:\n{result['daily_brief']}")
