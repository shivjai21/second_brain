"""Centralized configuration for the Local Second Brain agent."""

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(env_key: str, default_rel: str) -> Path:
    raw = os.getenv(env_key) or default_rel
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    # Local Ollama models
    primary_model: str = os.getenv("PRIMARY_MODEL") or "qwen3.5:4b"
    fast_model: str = os.getenv("FAST_MODEL") or "qwen3.5:0.8b"
    deep_model: str = os.getenv("DEEP_MODEL") or "gemma4:e4b"
    embed_model: str = os.getenv("EMBED_MODEL") or "nomic-embed-text"

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX") or "12288")

    # Local persistence paths
    data_dir: Path = _resolve_path("DATA_DIR", "data")
    brain_db_path: Path = _resolve_path("BRAIN_DB_PATH", "data/brain.db")
    checkpoint_db_path: Path = _resolve_path("CHECKPOINT_DB_PATH", "data/checkpoints.db")
    notes_export_dir: Path = _resolve_path("NOTES_EXPORT_DIR", "data/notes")

    # Local voice pipeline (STT: mlx-whisper, TTS: kokoro-onnx)
    whisper_model: str = os.getenv("WHISPER_MODEL") or "mlx-community/whisper-small-mlx"
    kokoro_model_path: Path = _resolve_path("KOKORO_MODEL_PATH", "data/models/kokoro-v1.0.fp16.onnx")
    kokoro_voices_path: Path = _resolve_path("KOKORO_VOICES_PATH", "data/models/voices-v1.0.bin")
    kokoro_voice: str = os.getenv("KOKORO_VOICE") or "af_heart"
    kokoro_speed: float = float(os.getenv("KOKORO_SPEED") or "1.05")
    vad_silence_seconds: float = float(os.getenv("VAD_SILENCE_SECONDS") or "1.4")
    vad_energy_threshold: float = float(os.getenv("VAD_ENERGY_THRESHOLD") or "0.015")


settings = Settings()

