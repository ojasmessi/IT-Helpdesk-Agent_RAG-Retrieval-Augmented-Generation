"""
agent/tools.py — actions the agent can take in the "real world".

Here tickets are saved to a local JSON file. In a real company you would swap
these function bodies for API calls to ServiceNow, Jira Service Management,
Freshservice etc. — the agent graph would not need to change.

CONCEPT: @tool turns a function into a LangChain Tool (name + description + typed
arguments). We call them directly from graph nodes here, but the same tools can
be handed to an LLM for tool-calling (see the "next steps" in the README).
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

from config import TICKETS_FILE


def _load() -> dict:
    path = Path(TICKETS_FILE)
    return json.loads(path.read_text()) if path.exists() else {}


def _save(tickets: dict) -> None:
    Path(TICKETS_FILE).write_text(json.dumps(tickets, indent=2))


@tool
def create_ticket(summary: str, description: str, priority: str = "normal") -> str:
    """Create an IT support ticket. priority is 'normal' or 'high'. Returns the ticket ID."""
    tickets = _load()
    ticket_id = f"INC-{uuid.uuid4().hex[:6].upper()}"
    tickets[ticket_id] = {
        "summary": summary[:120],
        "description": description,
        "priority": priority,
        "status": "Open",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save(tickets)
    return ticket_id


@tool
def get_ticket_status(ticket_id: str) -> str:
    """Look up an existing ticket by ID (e.g. INC-1A2B3C)."""
    ticket = _load().get(ticket_id.upper())
    if not ticket:
        return f"I couldn't find a ticket with ID {ticket_id}."
    return (f"Ticket {ticket_id.upper()} — status: {ticket['status']}, priority: {ticket['priority']}, "
            f"created {ticket['created']}. Summary: {ticket['summary']}")
