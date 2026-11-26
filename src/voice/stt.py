"""Local Speech-to-Text using sounddevice VAD capture and Apple Silicon mlx-whisper."""

from pathlib import Path
import queue
import tempfile
import mlx_whisper
import numpy as np
import sounddevice as sd
import soundfile as sf

from src.config import settings

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.1  # 100ms audio frames


class LocalSTT:
    """Captures microphone audio locally and transcribes with mlx-whisper."""

    def __init__(self, model_repo: str | None = None) -> None:
        self.model_repo = model_repo or settings.whisper_model

    def record_until_silence(
        self,
        energy_threshold: float | None = None,
        silence_seconds: float | None = None,
        max_seconds: float = 30.0,
    ) -> np.ndarray:
        """Listen hands-free until speech starts, then stop after sustained silence."""
        threshold = energy_threshold or settings.vad_energy_threshold
        silence_limit = silence_seconds or settings.vad_silence_seconds
        block_size = int(SAMPLE_RATE * BLOCK_DURATION)

        audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        def _callback(indata, frames, time_info, status):
            audio_queue.put(indata.copy())

        frames_buffer: list[np.ndarray] = []
        pre_roll: list[np.ndarray] = []
        speaking = False
        silent_chunks = 0
        max_chunks = int(max_seconds / BLOCK_DURATION)
        silence_chunks_needed = int(silence_limit / BLOCK_DURATION)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            callback=_callback,
        ):
            while len(frames_buffer) < max_chunks:
                chunk = audio_queue.get()
                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if not speaking:
                    pre_roll.append(chunk)
                    if len(pre_roll) > 4:  # Keep 400ms before speech onset
                        pre_roll.pop(0)
                    if rms >= threshold:
                        speaking = True
                        frames_buffer.extend(pre_roll)
                        silent_chunks = 0
                else:
                    frames_buffer.append(chunk)
                    if rms < threshold:
                        silent_chunks += 1
                        if silent_chunks >= silence_chunks_needed:
                            break
                    else:
                        silent_chunks = 0

        if not frames_buffer:
            return np.array([], dtype=np.float32)

        return np.concatenate(frames_buffer, axis=0).flatten()

    def record_push_to_talk(self) -> np.ndarray:
        """Press Enter to start and Enter again to stop (best for noisy rooms)."""
        audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        def _callback(indata, frames, time_info, status):
            audio_queue.put(indata.copy())

        input("Press [Enter] to start speaking...")
        frames_buffer: list[np.ndarray] = []
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=_callback,
        ):
            input("Listening... Press [Enter] to stop.")
            while not audio_queue.empty():
                frames_buffer.append(audio_queue.get())

        if not frames_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(frames_buffer, axis=0).flatten()

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a 16kHz float32 mono numpy array via mlx-whisper."""
        if audio.size < int(SAMPLE_RATE * 0.35):
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            sf.write(tmp_path, audio, SAMPLE_RATE)
            result = mlx_whisper.transcribe(
                str(tmp_path),
                path_or_hf_repo=self.model_repo,
                language="en",
            )
            return str(result.get("text", "")).strip()
        finally:
            tmp_path.unlink(missing_ok=True)

