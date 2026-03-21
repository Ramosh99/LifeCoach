"""
config/settings.py
Central config — API keys, model names, vector DB path.

LLM brain: OpenRouter (openai-compatible endpoint)
Swap `model` to any OpenRouter model slug — no other code changes needed.

Popular options:
  "anthropic/claude-sonnet-4-5"
  "openai/gpt-4o"
  "google/gemini-2.0-flash-001"
  "meta-llama/llama-3.3-70b-instruct"
  "mistralai/mistral-large"
  "deepseek/deepseek-r1"

Full list → https://openrouter.ai/models
"""

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv


load_dotenv()


def _env(name: str, default: str = "") -> str:
    """
    Read an env var with small cleanup for common .env formatting mistakes.
    """
    value = os.getenv(name, default)
    if not isinstance(value, str):
        return default

    clean = value.strip()

    # Some .env lines accidentally include a shell command after the value.
    if " " in clean:
        clean = clean.split()[0]

    # If a var name is accidentally appended to the value, strip it.
    for suffix in ("OPENROUTER_API_KEY", "TAVILY_API_KEY", "LANGCHAIN_API_KEY"):
        if clean.endswith(suffix) and clean != suffix:
            clean = clean[: -len(suffix)].strip()

    return clean or default


def _env_port(name: str, default: int = 8080) -> int:
    """
    Read a port from env. Supports either:
    - raw port values (e.g. "8080")
    - URL values (e.g. "http://localhost:8501/")
    Falls back to default for invalid inputs.
    """
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError:
        parsed = urlparse(raw)
        if parsed.port is not None:
            return int(parsed.port)
    return default


@dataclass
class Settings:
    # ── OpenRouter (LLM brain) ───────────────────────────────────────────────
    openrouter_api_key: str = _env("OPENROUTER_API_KEY", _env("OPEN_ROUTER_API_KEY", ""))
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Swap this one string to change your model — nothing else changes
    model: str = _env("LIFECOACH_MODEL", "anthropic/claude-sonnet-4-5")

    # Optional: OpenRouter shows these in usage dashboards
    openrouter_app_name: str = "LifeCoach-AI"
    openrouter_app_url: str = "https://github.com/your-repo/lifecoach-agent"

    # ── LangSmith — set these to enable tracing automatically ────────────────
    langsmith_api_key: str = _env("LANGCHAIN_API_KEY", "")
    langsmith_project: str = _env("LANGCHAIN_PROJECT", "lifecoach-agent")

    # ── Google APIs (OAuth credentials JSON path) ────────────────────────────
    google_credentials_path: str = _env(
        "GOOGLE_CREDENTIALS_PATH", "credentials.json"
    )
    google_oauth_port: int = _env_port("GOOGLE_OAUTH_PORT", 8080)

    # ── LlamaIndex vector store ───────────────────────────────────────────────
    vector_db_path: str = "./memory/chroma_db"

    # Embeddings: OpenRouter doesn't do embeddings — use OpenAI directly
    # (free tier is plenty for this use case)
    openai_api_key: str = _env("OPENAI_API_KEY", "")
    embedding_model: str = "text-embedding-3-small"

    # ── Agent behaviour ───────────────────────────────────────────────────────
    max_iterations: int = 10
    learning_slot_duration_mins: int = 45


settings = Settings()


def enable_langsmith_tracing():
    """
    Call once at startup. LangSmith auto-traces all LangChain/LangGraph
    calls when these env vars are set — no code changes needed elsewhere.
    """
    if not settings.langsmith_api_key.strip():
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ.pop("LANGCHAIN_API_KEY", None)
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        print("[LangSmith] No API key set — tracing disabled")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    print(f"[LangSmith] Tracing enabled → project: {settings.langsmith_project}")