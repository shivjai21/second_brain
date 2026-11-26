"""LangGraph Personal OS agent with AES-encrypted SQLite checkpoints and voice-aware prompting."""

from datetime import datetime
import os
import sqlite3
from langchain_core.messages import SystemMessage, trim_messages
from langchain_ollama import ChatOllama
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.config import settings
from src.storage.db import BrainDB
from src.agent.tools import create_brain_tools


class BrainState(MessagesState):
    """Conversation state extending MessagesState with runtime voice/deep flags."""

    voice_mode: bool
    deep_mode: bool


def _build_checkpointer() -> SqliteSaver:
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.checkpoint_db_path), check_same_thread=False)

    raw_key = os.getenv("LANGGRAPH_AES_KEY") or ""
    key_bytes = raw_key.encode("utf-8")
    if len(key_bytes) not in (16, 24, 32):
        raise ValueError(
            "LANGGRAPH_AES_KEY in .env must be 16, 24, or 32 bytes "
            "(generate with: python -c \"import secrets; print(secrets.token_hex(16))\")."
        )

    serde = EncryptedSerializer.from_pycryptodome_aes(key=key_bytes)
    saver = SqliteSaver(conn=conn, serde=serde)
    saver.setup()
    return saver


def _build_system_prompt(voice_mode: bool) -> str:
    now_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    style_rules = (
        "VOICE MODE ACTIVE: Respond in 2 to 4 crisp, natural spoken sentences. "
        "Do NOT use Markdown tables, code fences, bullet symbols, or raw IDs unless asked."
        if voice_mode
        else "Respond concisely in clean Markdown. Include exact numbers, streaks, and note titles."
    )

    return (
        f"You are a 100% local, privacy-first Personal Intelligence Agent (Second Brain & Daily OS). "
        f"Current local time: {now_str}.\n\n"
        "OPERATING RULES:\n"
        "1. HABITS & ROUTINES: When the user mentions completing any habit, workout, meal/protein, "
        "meditation, or study block, immediately call `log_habit` for each item (you may call it multiple times). "
        "When they define or change a recurring schedule/target, call `upsert_routine`.\n"
        "2. SECOND BRAIN NOTES: When the user says 'note', 'remember', or shares an idea/decision to store, "
        "call `save_note` with a clear title and tags. When they ask about something they noted or decided earlier, "
        "ALWAYS call `search_notes` before answering—never guess from memory.\n"
        "3. ROUTINE-AWARE PLANNING: When asked to plan the day, review progress, or figure out what to do next, "
        "ALWAYS call `get_planning_context` first so your plan respects their real routines, remaining targets, "
        "recent streaks, and current time of day.\n"
        "4. HONESTY: If a search returns no notes or a habit has no logs, say so plainly.\n\n"
        f"{style_rules}"
    )


def build_brain_graph(db: BrainDB):
    """Compile the persistent LangGraph agent with encrypted SQLite checkpointing."""
    tools = create_brain_tools(db)
    checkpointer = _build_checkpointer()

    fast_llm = ChatOllama(
        model=settings.primary_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
        num_ctx=settings.ollama_num_ctx,
        reasoning=False,  # Keeps local tool calling & voice latency snappy
    ).bind_tools(tools)

    deep_llm = ChatOllama(
        model=settings.deep_model,
        base_url=settings.ollama_base_url,
        temperature=0.3,
        num_ctx=settings.ollama_num_ctx,
    ).bind_tools(tools)

    def agent_node(state: BrainState) -> dict:
        voice_mode = bool(state.get("voice_mode", False))
        deep_mode = bool(state.get("deep_mode", False))

        trimmed = trim_messages(
            state["messages"],
            max_tokens=16,  # Keep last 16 messages cleanly inside 12K context
            token_counter=len,
            strategy="last",
            start_on="human",
            include_system=False,
            allow_partial=False,
        )

        llm = deep_llm if deep_mode else fast_llm
        prompt_msgs = [SystemMessage(content=_build_system_prompt(voice_mode)), *trimmed]
        response = llm.invoke(prompt_msgs)
        return {"messages": [response]}

    builder = StateGraph(BrainState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)

