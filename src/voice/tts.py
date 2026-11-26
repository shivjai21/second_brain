"""Local Text-to-Speech using kokoro-onnx with sentence playback and macOS fallback."""

import re
import subprocess
from kokoro_onnx import Kokoro
import sounddevice as sd

from src.config import settings


def clean_for_speech(text: str) -> str:
    """Strip Markdown syntax, URLs, and code fences so TTS sounds natural."""
    cleaned = re.sub(r"```[\s\S]*?```", " Code block shown on screen. ", text)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"[#*_~>|]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


class LocalTTS:
    """Fast local neural TTS backed by Kokoro-82M ONNX (falls back to macOS 'say')."""

    def __init__(self) -> None:
        self._kokoro: Kokoro | None = None
        if settings.kokoro_model_path.exists() and settings.kokoro_voices_path.exists():
            self._kokoro = Kokoro(
                str(settings.kokoro_model_path),
                str(settings.kokoro_voices_path),
            )

    @property
    def backend_name(self) -> str:
        return f"kokoro-onnx ({settings.kokoro_voice})" if self._kokoro else "macOS say (fallback)"

    def speak(self, text: str) -> None:
        spoken = clean_for_speech(text)
        if not spoken:
            return

        if self._kokoro is None:
            subprocess.run(["say", spoken], check=False)
            return

        for sentence in _split_sentences(spoken):
            samples, sample_rate = self._kokoro.create(
                sentence,
                voice=settings.kokoro_voice,
                speed=settings.kokoro_speed,
                lang="en-us",
            )
            sd.play(samples, sample_rate)
            sd.wait()

