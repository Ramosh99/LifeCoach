"""
tools/calendar_tool.py
Google API client — finds free time slots and books learning sessions
on the user's Google Calendar.
"""

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from settings import settings

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]


# ---------------------------------------------------------------------------
# Auth (reuses token.json from gmail_tool if scopes overlap)
# ---------------------------------------------------------------------------

def _get_calendar_service():
    import os
    creds = None
    if os.path.exists("token_calendar.json"):
        creds = Credentials.from_authorized_user_file("token_calendar.json", SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            settings.google_credentials_path, SCOPES
        )
        try:
            creds = flow.run_local_server(port=settings.google_oauth_port)
        except OSError:
            creds = flow.run_local_server(port=0)  # auto-pick free port
        with open("token_calendar.json", "w") as f:
            f.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FreeSlot:
    start: datetime
    end: datetime
    duration_mins: int

    def __str__(self):
        return f"{self.start.strftime('%a %b %d %H:%M')} → {self.end.strftime('%H:%M')} ({self.duration_mins}m)"


# ---------------------------------------------------------------------------
# Core tool functions
# ---------------------------------------------------------------------------

def find_free_slots(
    days_ahead: int = 3,
    min_duration_mins: int = 30,
    preferred_hours: tuple[int, int] = (8, 20),   # 8am–8pm
) -> list[FreeSlot]:
    """
    Scans the next `days_ahead` days and returns free slots
    longer than `min_duration_mins` within preferred working hours.
    """
    service = _get_calendar_service()
    now = datetime.now(timezone.utc)
    end_window = now + timedelta(days=days_ahead)

    # Fetch busy intervals from all calendars
    body = {
        "timeMin": now.isoformat(),
        "timeMax": end_window.isoformat(),
        "items": [{"id": "primary"}],
    }
    freebusy = service.freebusy().query(body=body).execute()
    busy_periods = freebusy["calendars"]["primary"]["busy"]

    busy = [
        (
            datetime.fromisoformat(p["start"]),
            datetime.fromisoformat(p["end"]),
        )
        for p in busy_periods
    ]

    # Walk each day and find gaps
    free_slots = []
    cursor = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    while cursor < end_window:
        day_start = cursor.replace(hour=preferred_hours[0], minute=0)
        day_end = cursor.replace(hour=preferred_hours[1], minute=0)
        slot_start = max(cursor, day_start)

        # Collect busy blocks for this day
        day_busy = sorted(
            [(s, e) for s, e in busy if s.date() == cursor.date()],
            key=lambda x: x[0],
        )

        for busy_start, busy_end in day_busy:
            if slot_start < busy_start:
                gap_mins = int((busy_start - slot_start).total_seconds() / 60)
                if gap_mins >= min_duration_mins:
                    free_slots.append(FreeSlot(
                        start=slot_start,
                        end=busy_start,
                        duration_mins=gap_mins,
                    ))
            slot_start = max(slot_start, busy_end)

        # Remaining time after last meeting
        if slot_start < day_end:
            gap_mins = int((day_end - slot_start).total_seconds() / 60)
            if gap_mins >= min_duration_mins:
                free_slots.append(FreeSlot(
                    start=slot_start,
                    end=day_end,
                    duration_mins=gap_mins,
                ))

        cursor += timedelta(days=1)

    return free_slots


def book_learning_session(
    topic: str,
    slot: FreeSlot,
    resource_url: str = "",
    duration_mins: int = None,
) -> str:
    """
    Creates a calendar event for a learning session.
    Returns the event URL.
    """
    service = _get_calendar_service()
    duration = duration_mins or min(slot.duration_mins, settings.learning_slot_duration_mins)
    end_time = slot.start + timedelta(minutes=duration)

    description = f"LifeCoach AI scheduled this session.\nTopic: {topic}"
    if resource_url:
        description += f"\nResource: {resource_url}"

    event = {
        "summary": f"[LifeCoach] {topic}",
        "description": description,
        "start": {"dateTime": slot.start.isoformat(), "timeZone": "UTC"},
        "end":   {"dateTime": end_time.isoformat(),   "timeZone": "UTC"},
        "colorId": "2",   # Sage green — visually distinct from regular meetings
        "reminders": {
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": 10}],
        },
    }

    created = service.events().insert(calendarId="primary", body=event).execute()
    print(f"[Calendar] Booked: '{topic}' on {slot.start.strftime('%a %b %d %H:%M')}")
    return created.get("htmlLink", "")
