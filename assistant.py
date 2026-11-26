"""Local Voice Assistant with Smart Memory Consolidation (Save / Ask-to-Update / Delete)."""

import argparse
from datetime import datetime
from pathlib import Path
import queue
import re
import sqlite3
import subprocess
import tempfile
import time

import mlx_whisper
import numpy as np
import ollama
import sounddevice as sd
import soundfile as sf

# --- Configuration ---
MODEL = "qwen3.5:4b"  # Much stricter tool discipline than llama3.2:3b (already pulled)
WHISPER_MODEL = "mlx-community/whisper-small-mlx"
DB_PATH = Path("memory.db")
KOKORO_MODEL = Path("data/models/kokoro-v1.0.fp16.onnx")
KOKORO_VOICES = Path("data/models/voices-v1.0.bin")

SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.015  # Raise to 0.03 if fan/AC noise triggers recording early
SILENCE_SECONDS = 1.3
EXIT_PHRASES = {"exit", "quit", "stop listening", "goodbye", "bye"}
THINKING_PREFIXES = ("qwen3", "deepseek-r1", "gemma4")


# --- 1. SQLite Memory with IDs, Deduplication, Update & Delete ---
def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def save_note(content: str) -> str:
    """Save a BRAND-NEW durable personal fact, task, or reminder to SQLite.

    Do NOT use this for casual conversation, questions, or updates to existing notes.

    Args:
        content: A clear, standalone sentence describing the new information to remember.
    """
    clean = content.strip()
    if not clean:
        return "Nothing saved because the note was empty."

    norm_new = _normalize(clean)
    with sqlite3.connect(DB_PATH) as conn:
        existing = conn.execute("SELECT id, content FROM notes").fetchall()
        for note_id, old_text in existing:
            if _normalize(old_text) == norm_new:
                print(f"  [duplicate skipped] Already saved as ID {note_id}")
                return f"Already stored under ID {note_id}: {old_text}. No duplicate created."

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        cur = conn.execute(
            "INSERT INTO notes (created_at, content) VALUES (?, ?)",
            (now, clean),
        )
        new_id = cur.lastrowid

    print(f"  [saved ID {new_id}] {clean}")
    return f"Saved new note ID {new_id}: {clean}"


def update_note(note_id: int, new_content: str) -> str:
    """Overwrite an existing SQLite note by its ID ONLY after the user confirms the update.

    Args:
        note_id: The integer ID of the existing note to overwrite.
        new_content: The complete updated sentence replacing the old note content.
    """
    clean = new_content.strip()
    if not clean:
        return "Update failed: new content was empty."

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "UPDATE notes SET content = ?, created_at = ? WHERE id = ?",
            (clean, now, int(note_id)),
        )
        if cur.rowcount == 0:
            return f"No note found with ID {note_id}."

    print(f"  [updated ID {note_id}] {clean}")
    return f"Updated note ID {note_id} to: {clean}"


def delete_note(note_id: int) -> str:
    """Delete an existing SQLite note by its ID when the user asks to forget or remove it.

    Args:
        note_id: The integer ID of the note to delete.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("DELETE FROM notes WHERE id = ?", (int(note_id),))
        if cur.rowcount == 0:
            return f"No note found with ID {note_id}."

    print(f"  [deleted ID {note_id}]")
    return f"Deleted note ID {note_id}."


def load_all_notes(limit: int = 100) -> str:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, created_at, content FROM notes ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    if not rows:
        return "No saved notes yet."
    return "\n".join(f"- [ID: {nid} | {ts}] {text}" for nid, ts, text in reversed(rows))


# --- 2. Local Speech-to-Text (mlx-whisper) ---
def listen_until_silence() -> np.ndarray:
    block_size = int(SAMPLE_RATE * 0.1)
    q: queue.Queue[np.ndarray] = queue.Queue()

    def _cb(indata, frames, time_info, status):
        q.put(indata.copy())

    frames, pre_roll = [], []
    speaking = False
    silent_chunks = 0
    needed_silence = int(SILENCE_SECONDS / 0.1)

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block_size, callback=_cb
    ):
        while len(frames) < 300:
            chunk = q.get()
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if not speaking:
                pre_roll.append(chunk)
                if len(pre_roll) > 4:
                    pre_roll.pop(0)
                if rms >= VAD_THRESHOLD:
                    speaking = True
                    frames.extend(pre_roll)
                    silent_chunks = 0
            else:
                frames.append(chunk)
                if rms < VAD_THRESHOLD:
                    silent_chunks += 1
                    if silent_chunks >= needed_silence:
                        break
                else:
                    silent_chunks = 0

    return np.concatenate(frames, axis=0).flatten() if frames else np.array([], dtype=np.float32)


def transcribe(audio: np.ndarray) -> str:
    if audio.size < int(SAMPLE_RATE * 0.35):
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(tmp_path, audio, SAMPLE_RATE)
        result = mlx_whisper.transcribe(str(tmp_path), path_or_hf_repo=WHISPER_MODEL, language="en")
        return str(result.get("text", "")).strip()
    finally:
        tmp_path.unlink(missing_ok=True)


# --- 3. Local Text-to-Speech (kokoro-onnx with macOS 'say' fallback) ---
class Speaker:
    def __init__(self) -> None:
        self._kokoro = None
        if KOKORO_MODEL.exists() and KOKORO_VOICES.exists():
            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))

    def speak(self, text: str) -> None:
        clean = re.sub(r"[#*_`>]+", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        if not clean:
            return
        if self._kokoro is None:
            subprocess.run(["say", clean], check=False)
            return
        samples, sr = self._kokoro.create(clean, voice="af_heart", speed=1.05, lang="en-us")
        sd.play(samples, sr)
        sd.wait()
        time.sleep(0.25)


# --- 4. Ollama Brain (Disciplined Tool Calling + Ask-Before-Overwrite) ---
def _chat_ollama(messages: list[dict], tools: list | None = None):
    kwargs: dict = {
        "model": MODEL,
        "messages": messages,
        "options": {"temperature": 0.1},
    }
    if tools:
        kwargs["tools"] = tools
    if MODEL.lower().startswith(THINKING_PREFIXES):
        kwargs["think"] = False  # Fast voice replies without long thinking blocks
    return ollama.chat(**kwargs)


def _execute_tool(name: str, args: dict) -> str:
    try:
        if name == "save_note":
            content = args.get("content") or next(iter(args.values()), "")
            return save_note(str(content))
        if name == "update_note":
            return update_note(int(args["note_id"]), str(args["new_content"]))
        if name == "delete_note":
            return delete_note(int(args["note_id"]))
        return f"Unknown tool: {name}"
    except (KeyError, ValueError, TypeError) as exc:
        return f"Tool argument error for {name}: {exc}"


def ask_llm(user_text: str, history: list[dict]) -> str:
    system_prompt = (
        f"You are a concise, private local voice assistant. Current time: "
        f"{datetime.now().strftime('%A, %Y-%m-%d %H:%M')}.\n\n"
        "SPEECH STYLE:\n"
        "- Reply in 1 to 3 short, natural spoken sentences (no markdown, no bullet lists, no raw IDs unless clarifying).\n\n"
        "STRICT MEMORY RULES (CHECK SAVED NOTES BELOW BEFORE ACTING):\n"
        "1. CASUAL CHAT & QUESTIONS -> DO NOT CALL ANY TOOL.\n"
        "   - Never save greetings, normal conversation, questions ('do I have any bills?'), "
        "your own answers, or meta-comments about how to chat.\n"
        "2. OVERLAPPING OR UPDATED INFO -> ASK FIRST, DO NOT SAVE A DUPLICATE.\n"
        "   - If the user's message adds detail to or changes an existing saved note "
        "(e.g., adding a date to an existing trip, or updating a birthday/graduation note), "
        "DO NOT call save_note or update_note yet.\n"
        "   - Briefly mention the existing note and ask: 'Should I update that note with the new detail?'\n"
        "   - ONLY AFTER the user confirms in the next turn ('yes', 'go ahead', 'update it'), "
        "call `update_note(note_id, new_content)` with the merged, complete note.\n"
        "3. BRAND-NEW FACT, TASK, OR REMINDER -> CALL `save_note` ONCE.\n"
        "   - Only when the user shares genuinely new personal info, a reminder, or explicitly asks to note/remember something "
        "that does NOT overlap any saved note below.\n"
        "4. FORGETTING -> Call `delete_note(note_id)` when the user asks to delete or forget a stored note.\n"
        "5. ANSWERING QUESTIONS -> Use SAVED NOTES below. If not stored, simply say you don't have that saved (and do NOT save that).\n\n"
        f"SAVED NOTES:\n{load_all_notes()}"
    )

    tools = [save_note, update_note, delete_note]
    messages = [
        {"role": "system", "content": system_prompt},
        *history[-10:],
        {"role": "user", "content": user_text},
    ]

    response = _chat_ollama(messages, tools=tools)

    if response.message.tool_calls:
        messages.append(response.message)
        for call in response.message.tool_calls:
            result = _execute_tool(call.function.name, call.function.arguments or {})
            messages.append({"role": "tool", "content": result})
        response = _chat_ollama(messages)

    reply = (response.message.content or "").strip()
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    return reply


# --- 5. Main Loop ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Local Voice Assistant with Smart SQLite Memory")
    parser.add_argument("--text", action="store_true", help="Test via keyboard instead of microphone.")
    args = parser.parse_args()

    init_db()
    speaker = Speaker()
    history: list[dict] = []

    print(f"Assistant running | Model: {MODEL} | DB: {DB_PATH}")
    if not args.text:
        print("Voice mode active. Speak anytime (say 'stop listening' or press Ctrl+C to quit).")
        speaker.speak("I am ready. Go ahead.")

    while True:
        try:
            if args.text:
                user_text = input("\nYou > ").strip()
            else:
                print("\nListening...")
                user_text = transcribe(listen_until_silence())
                if not user_text:
                    continue
                print(f"You: {user_text}")

            if user_text.lower().strip(" .!?") in EXIT_PHRASES:
                if not args.text:
                    speaker.speak("Goodbye.")
                break

            reply = ask_llm(user_text, history)
            print(f"Assistant: {reply}")
            if not args.text:
                speaker.speak(reply)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
