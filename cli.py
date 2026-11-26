"""Interactive Text & Hands-Free Voice CLI for Local Second Brain."""

import argparse
from datetime import date
from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent.graph import build_brain_graph
from src.config import settings
from src.storage.db import BrainDB

console = Console()
EXIT_PHRASES = {"exit", "quit", "stop listening", "goodbye", "bye"}


def _run_turn(
    graph,
    thread_id: str,
    user_text: str,
    voice_mode: bool = False,
    deep_mode: bool = False,
) -> str:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 12,
    }
    state_input = {
        "messages": [HumanMessage(content=user_text)],
        "voice_mode": voice_mode,
        "deep_mode": deep_mode,
    }

    final_reply = ""
    for event in graph.stream(state_input, config=config, stream_mode="updates"):
        for node_name, update in event.items():
            msgs = update.get("messages", [])
            if node_name == "tools":
                for m in msgs:
                    console.print(f"[dim]⚙ tool: {getattr(m, 'name', 'tool')} completed[/dim]")
            elif node_name == "agent" and msgs:
                last = msgs[-1]
                if isinstance(last, AIMessage) and last.content:
                    final_reply = str(last.content).strip()
    return final_reply


def _run_voice_loop(graph, thread_id: str, push_to_talk: bool, deep_mode: bool) -> None:
    from src.voice.stt import LocalSTT
    from src.voice.tts import LocalTTS

    console.print("[cyan]Loading local voice engines (mlx-whisper + kokoro-onnx)...[/cyan]")
    stt = LocalSTT()
    tts = LocalTTS()

    mode_label = "Push-to-Talk" if push_to_talk else "Hands-Free Auto-VAD"
    console.print(
        Panel.fit(
            f"[bold green]Voice Mode Ready[/bold green] ({mode_label})\n"
            f"STT: [cyan]{stt.model_repo}[/cyan] | TTS: [magenta]{tts.backend_name}[/magenta]\n"
            f"Say [bold]'stop listening'[/bold] or press Ctrl+C to exit.",
            title="Local Second Brain — Voice",
        )
    )
    tts.speak("Second brain is ready. Go ahead.")

    while True:
        try:
            if push_to_talk:
                audio = stt.record_push_to_talk()
            else:
                console.print("\n[bold green]🎙 Listening...[/bold green] (speak naturally)")
                audio = stt.record_until_silence()

            transcript = stt.transcribe(audio)
            if not transcript:
                continue

            console.print(f"[bold cyan]You:[/bold cyan] {transcript}")
            normalized = transcript.lower().strip(" .!?")
            if normalized in EXIT_PHRASES:
                tts.speak("Signing off. Talk to you later.")
                break

            reply = _run_turn(
                graph,
                thread_id=thread_id,
                user_text=transcript,
                voice_mode=True,
                deep_mode=deep_mode,
            )
            if reply:
                console.print(Panel(Markdown(reply), title="Second Brain", border_style="green"))
                tts.speak(reply)
        except KeyboardInterrupt:
            console.print("\n[yellow]Voice session ended.[/yellow]")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Second Brain — Daily OS, Notes & Voice Agent")
    parser.add_argument("prompt", nargs="*", help="Optional one-shot command (omit for interactive mode).")
    parser.add_argument("--voice", action="store_true", help="Start in hands-free local voice mode.")
    parser.add_argument("--ptt", action="store_true", help="Use push-to-talk in voice mode.")
    parser.add_argument("--deep", action="store_true", help=f"Use {settings.deep_model} for deeper reasoning.")
    parser.add_argument("--thread", default=f"daily-{date.today().isoformat()}", help="Persistent thread ID.")
    args = parser.parse_args()

    db = BrainDB()
    graph = build_brain_graph(db)

    if args.voice:
        _run_voice_loop(graph, args.thread, push_to_talk=args.ptt, deep_mode=args.deep)
        return

    if args.prompt:
        query = " ".join(args.prompt)
        reply = _run_turn(graph, args.thread, query, voice_mode=False, deep_mode=args.deep)
        console.print(Panel(Markdown(reply), title="Second Brain", border_style="cyan"))
        return

    deep_mode = args.deep
    console.print(
        Panel.fit(
            f"[bold]Thread:[/bold] {args.thread} (AES-256 Encrypted SQLite)\n"
            f"[bold]Model:[/bold] {settings.deep_model if deep_mode else settings.primary_model}\n"
            "Commands: [cyan]/voice[/cyan] | [cyan]/today[/cyan] | [cyan]/notes[/cyan] | "
            "[cyan]/deep[/cyan] | [cyan]/quit[/cyan]",
            title="Local Second Brain v1",
        )
    )

    while True:
        try:
            user_input = console.input("\n[bold cyan]You > [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            break
        if user_input == "/voice":
            _run_voice_loop(graph, args.thread, push_to_talk=False, deep_mode=deep_mode)
            continue
        if user_input == "/deep":
            deep_mode = not deep_mode
            active = settings.deep_model if deep_mode else settings.primary_model
            console.print(f"[yellow]Switched active model to: {active}[/yellow]")
            continue
        if user_input == "/today":
            console.print(db.get_daily_dashboard())
            continue
        if user_input == "/notes":
            console.print(db.list_recent_notes(limit=10))
            continue

        reply = _run_turn(graph, args.thread, user_input, voice_mode=False, deep_mode=deep_mode)
        console.print(Panel(Markdown(reply), title="Second Brain", border_style="green"))


if __name__ == "__main__":
    main()

