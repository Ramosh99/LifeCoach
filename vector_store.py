"""
memory/vector_store.py
Debug memory layer with no external vector DB dependencies.

This keeps the same public API as the original implementation so the
rest of the agent can run during local debugging without LlamaIndex,
Chroma, or OpenAI embeddings.
"""

from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Optional


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LearnedTopicsMemory:
    """
    Tracks topics the user has already studied.
    Used by the agent to skip basics and assign the right difficulty.
    """

    def __init__(self):
        self._topics: list[dict] = []

    def add_topic(self, topic: str, summary: str, skill_level: str = "beginner"):
        """
        Call after a learning session completes.
        skill_level: "beginner" | "intermediate" | "advanced"
        """
        self._topics.append(
            {
                "topic": topic,
                "summary": summary,
                "skill_level": skill_level,
                "learned_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        print(f"[Memory] Stored: '{topic}' at {skill_level} level")

    def get_skill_level(self, topic: str) -> Optional[str]:
        """
        Returns the user's known skill level for a topic, or None if unknown.
        """
        if not self._topics:
            return None
        best = max(self._topics, key=lambda t: _similarity(topic, t["topic"]))
        if _similarity(topic, best["topic"]) < 0.55:
            return None
        return best.get("skill_level")

    def already_knows(self, topic: str, threshold: float = 0.82) -> bool:
        """
        Returns True if the user has already studied this topic at
        a similarity score above the threshold.
        """
        if not self._topics:
            return False
        best_score = max(_similarity(topic, t["topic"]) for t in self._topics)
        return best_score >= threshold

    def get_all_topics(self) -> list[str]:
        return [t.get("topic", "") for t in self._topics]


class ResourceMemory:
    """
    Stores learning resources (URLs, articles, tutorials) found by the
    search tool. Enables the agent to avoid re-fetching the same content
    and to recall relevant resources by topic.
    """

    def __init__(self):
        self._resources: list[dict] = []

    def add_resource(self, title: str, url: str, topic: str, summary: str):
        self._resources.append(
            {
                "title": title,
                "url": url,
                "topic": topic,
                "summary": summary,
                "added_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def find_resources(self, topic: str, top_k: int = 3) -> list[dict]:
        scored = []
        for resource in self._resources:
            score = max(
                _similarity(topic, resource.get("topic", "")),
                _similarity(topic, resource.get("title", "")),
            )
            scored.append(
                {
                    "title": resource.get("title"),
                    "url": resource.get("url"),
                    "topic": resource.get("topic"),
                    "score": round(score, 3),
                }
            )

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]
