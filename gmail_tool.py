"""
tools/gmail_tool.py
Google API client — reads the user's Gmail inbox and extracts signals
relevant to their learning goal (recruiter emails, topic mentions, etc.)
"""

from dataclasses import dataclass

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from settings import settings

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",   # for sending brief email
]


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _get_gmail_service():
    """
    Returns an authenticated Gmail API service.
    Uses OAuth2 — on first run, opens browser for consent.
    Stores token in token.json for subsequent runs.
    """
    import os, json
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            settings.google_credentials_path, SCOPES
        )
        try:
            creds = flow.run_local_server(port=settings.google_oauth_port)
        except OSError:
            creds = flow.run_local_server(port=0)  # auto-pick free port
        with open("token.json", "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EmailSignal:
    sender: str
    subject: str
    snippet: str
    is_recruiter: bool
    mentioned_skills: list[str]
    urgency: str   # "high" | "medium" | "low"


# ---------------------------------------------------------------------------
# Core tool functions (called by LangGraph nodes)
# ---------------------------------------------------------------------------

def fetch_recent_emails(max_results: int = 20) -> list[EmailSignal]:
    """
    Fetches recent emails and extracts learning-relevant signals.
    Returns a list of EmailSignal objects.
    """
    service = _get_gmail_service()
    results = service.users().messages().list(
        userId="me",
        maxResults=max_results,
        labelIds=["INBOX"],
    ).execute()

    messages = results.get("messages", [])
    signals = []

    for msg in messages:
        full = service.users().messages().get(
            userId="me", id=msg["id"], format="metadata",
            metadataHeaders=["From", "Subject"]
        ).execute()

        headers = {h["name"]: h["value"] for h in full["payload"]["headers"]}
        sender = headers.get("From", "")
        subject = headers.get("Subject", "")
        snippet = full.get("snippet", "")

        # Heuristic: detect recruiter emails
        recruiter_keywords = ["recruiter", "hiring", "opportunity", "role", "position"]
        is_recruiter = any(kw in sender.lower() + subject.lower() for kw in recruiter_keywords)

        # Extract tech skills mentioned
        skill_keywords = [
            "python", "pytorch", "tensorflow", "llm", "ml", "machine learning",
            "data science", "product manager", "pm", "agile", "sql", "react",
            "langchain", "vector", "rag",
        ]
        text = (subject + " " + snippet).lower()
        mentioned = [s for s in skill_keywords if s in text]

        urgency = "high" if is_recruiter else ("medium" if mentioned else "low")

        signals.append(EmailSignal(
            sender=sender,
            subject=subject,
            snippet=snippet,
            is_recruiter=is_recruiter,
            mentioned_skills=mentioned,
            urgency=urgency,
        ))

    return signals


def send_email(subject: str, body: str, to: str = "me") -> None:
    """
    Sends an email from the authenticated Gmail account to itself (or `to`).
    Uses 'me' as both sender and recipient by default — self-email daily brief.
    """
    import base64
    from email.mime.text import MIMEText

    service = _get_gmail_service()

    # Get authenticated user's email address
    profile = service.users().getProfile(userId="me").execute()
    user_email = profile.get("emailAddress", "me")
    recipient = user_email if to == "me" else to

    msg = MIMEText(body)
    msg["to"] = recipient
    msg["from"] = user_email
    msg["subject"] = subject

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(
        userId="me",
        body={"raw": raw},
    ).execute()


def label_email(message_id: str, label_name: str = "LifeCoach/Processed"):
    """
    Applies a Gmail label to a processed email so the agent doesn't re-read it.
    Creates the label if it doesn't exist.
    """
    service = _get_gmail_service()

    # Get or create label
    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    label_id = next((l["id"] for l in labels if l["name"] == label_name), None)

    if not label_id:
        new_label = service.users().labels().create(
            userId="me", body={"name": label_name}
        ).execute()
        label_id = new_label["id"]

    service.users().messages().modify(
        userId="me",
        id=message_id,
        body={"addLabelIds": [label_id]},
    ).execute()
