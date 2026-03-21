"""
tools/search_tool.py
Web search — finds learning resources for a given topic.
Uses Tavily (best for LLM agents) with a fallback to DuckDuckGo.
Results are stored in LlamaIndex ResourceMemory to avoid re-fetching.
"""

from dataclasses import dataclass
from typing import Any, Optional

from vector_store import ResourceMemory

import os

_resource_memory = ResourceMemory()


def _get_tavily_client() -> Optional[Any]:
    """
    Returns a Tavily client only when an API key is configured.
    Avoids import-time crashes when TAVILY_API_KEY is missing.
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from tavily import TavilyClient
    except ModuleNotFoundError:
        return None
    return TavilyClient(api_key=api_key)


@dataclass
class SearchResult:
    title: str
    url: str
    summary: str
    relevance_score: float


def find_learning_resources(
    topic: str,
    skill_level: str = "beginner",
    top_k: int = 3,
    use_cache: bool = True,
) -> list[SearchResult]:
    """
    Finds the best learning resources for `topic` at `skill_level`.

    1. Checks vector memory first (avoids duplicate searches)
    2. Falls back to Tavily web search
    3. Stores new results in memory for future sessions
    """

    # Step 1 — check memory cache
    if use_cache:
        cached = _resource_memory.find_resources(topic, top_k=top_k)
        if cached:
            print(f"[Search] Cache hit for '{topic}' — {len(cached)} results")
            return [
                SearchResult(
                    title=r["title"],
                    url=r["url"],
                    summary=r["topic"],
                    relevance_score=r["score"],
                )
                for r in cached
            ]

    # Step 2 — live search
    tavily_client = _get_tavily_client()
    if not tavily_client:
        print("[Search] TAVILY_API_KEY not set; skipping live web search")
        return []

    query = f"best {skill_level} tutorial for {topic} in 2025"
    print(f"[Search] Querying Tavily: '{query}'")

    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=top_k,
        include_answer=False,
    )

    results = []
    for r in response.get("results", []):
        result = SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            summary=r.get("content", "")[:300],
            relevance_score=r.get("score", 0.0),
        )
        results.append(result)

        # Step 3 — persist to memory
        _resource_memory.add_resource(
            title=result.title,
            url=result.url,
            topic=topic,
            summary=result.summary,
        )

    return results
