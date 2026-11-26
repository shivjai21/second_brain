"""LangChain tools exposing BrainDB habits, routines, notes, and planning context."""

from datetime import date, datetime
import json
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.storage.db import BrainDB


class UpsertRoutineInput(BaseModel):
    name: str = Field(..., description="Short snake_case routine name, e.g. 'strength_training' or 'protein_intake'.")
    category: str = Field(default="general", description="Category such as health, nutrition, career, or mindfulness.")
    target_value: Optional[float] = Field(default=None, description="Numeric daily target, e.g. 110 for grams of protein.")
    unit: str = Field(default="", description="Unit such as grams, blocks, minutes, or session.")
    preferred_time: str = Field(default="anytime", description="Preferred schedule window, e.g. '07:00' or 'morning'.")
    frequency: str = Field(default="daily", description="Recurrence such as daily or 5x_week.")
    notes: str = Field(default="", description="Optional rule or preference for this routine.")


class LogHabitInput(BaseModel):
    habit_name: str = Field(..., description="Habit name matching a routine if applicable, e.g. 'protein_intake'.")
    value: float = Field(default=1.0, description="Numeric quantity completed, e.g. 45 or 1.")
    unit: str = Field(default="", description="Unit such as grams, minutes, blocks, or session.")
    note: str = Field(default="", description="Context extracted from what the user said, e.g. 'push day, low focus'.")
    log_date: Optional[str] = Field(default=None, description="ISO date YYYY-MM-DD; omit for today.")


class SaveNoteInput(BaseModel):
    title: str = Field(..., description="Concise, searchable title for the note.")
    content: str = Field(..., description="Detailed note content to store in the second brain.")
    tags: list[str] = Field(default_factory=list, description="2 to 4 lowercase keyword tags.")


class SearchNotesInput(BaseModel):
    query: str = Field(..., description="Natural-language question or keywords to search saved notes.")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of top matching notes to return.")


def create_brain_tools(db: BrainDB) -> list:
    """Bind BrainDB methods as LangGraph-compatible tools."""

    @tool("upsert_routine", args_schema=UpsertRoutineInput)
    def upsert_routine(
        name: str,
        category: str = "general",
        target_value: Optional[float] = None,
        unit: str = "",
        preferred_time: str = "anytime",
        frequency: str = "daily",
        notes: str = "",
    ) -> str:
        """Create or update a recurring daily routine or target."""
        res = db.upsert_routine(name, category, target_value, unit, preferred_time, frequency, notes)
        return json.dumps(res, indent=2)

    @tool("log_habit", args_schema=LogHabitInput)
    def log_habit(
        habit_name: str,
        value: float = 1.0,
        unit: str = "",
        note: str = "",
        log_date: Optional[str] = None,
    ) -> str:
        """Log a completed habit, workout, protein intake, meditation, or study block into SQLite."""
        res = db.log_habit(habit_name, value, unit, note, log_date)
        return json.dumps(res, indent=2)

    @tool("get_today_dashboard")
    def get_today_dashboard(target_date: Optional[str] = None) -> str:
        """Get today's logged habits, streaks, active routines, and remaining targets."""
        res = db.get_daily_dashboard(target_date)
        return json.dumps(res, indent=2)

    @tool("save_note", args_schema=SaveNoteInput)
    def save_note(title: str, content: str, tags: list[str] | None = None) -> str:
        """Save a thought, idea, learning, or decision into the Second Brain (SQLite + Markdown + vector index)."""
        res = db.save_note(title, content, tags or [])
        return json.dumps(res, indent=2)

    @tool("search_notes", args_schema=SearchNotesInput)
    def search_notes(query: str, top_k: int = 5) -> str:
        """Search saved Second Brain notes using Hybrid Search (BM25 + local nomic-embed-text embeddings)."""
        hits = db.search_notes(query, top_k=top_k)
        if not hits:
            return "No matching notes found in the second brain."
        return json.dumps(hits, indent=2)

    @tool("get_planning_context")
    def get_planning_context(days_history: int = 5) -> str:
        """Fetch current time, active routines, today's progress, recent habit logs, and recent notes to build a realistic plan."""
        payload = {
            "current_datetime": datetime.now().strftime("%A, %Y-%m-%d %H:%M"),
            "today": date.today().isoformat(),
            "dashboard": db.get_daily_dashboard(),
            "recent_habit_history": db.get_recent_habit_history(days=days_history),
            "recent_notes": db.list_recent_notes(limit=6),
        }
        return json.dumps(payload, indent=2)

    return [
        upsert_routine,
        log_habit,
        get_today_dashboard,
        save_note,
        search_notes,
        get_planning_context,
    ]

