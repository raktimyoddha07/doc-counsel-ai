"""
Voice transcription via faster-whisper (CPU).

The browser records audio as WebM/Opus via the MediaRecorder API.
faster-whisper handles audio decoding internally through its bundled
ffmpeg-based reader (via ctranslate2), so no external conversion library
is needed — just pass the raw file path and it handles webm, opus, mp3,
wav, ogg, flac, and any other format ffmpeg supports.

The WhisperModel is loaded lazily on first use so the server still boots
instantly even before the model (~145MB for ``base``) is downloaded. Model size
and device are configurable via env vars so they can be changed without code
edits (e.g. upgrade to GPU later).
"""
import os
from typing import Optional


class TranscriptionError(RuntimeError):
    """Raised when transcription cannot be performed (model/load/decode failure)."""


_transcriber = None


def get_transcriber():
    """
    Lazily load a module-level faster-whisper WhisperModel singleton.

    Honors WHISPER_MODEL (default ``base``) and WHISPER_DEVICE (default ``cpu``).
    int8 compute type keeps CPU inference fast and memory-light.
    """
    global _transcriber
    if _transcriber is not None:
        return _transcriber

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper is not installed. Run `pip install faster-whisper` and restart the server."
        ) from exc

    model_name = os.getenv("WHISPER_MODEL", "base")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    try:
        _transcriber = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:
        raise TranscriptionError(
            f"Could not load Whisper model '{model_name}' on device '{device}': {exc}"
        ) from exc
    return _transcriber


def transcribe_audio_file(path: str, language: Optional[str] = None) -> str:
    """
    Transcribe the audio file at ``path`` and return the joined text.

    faster-whisper decodes the audio internally (via its bundled ffmpeg reader),
    so any format ffmpeg supports (webm/opus, mp3, wav, ogg, flac, etc.) works
    directly — no pre-conversion needed.

    ``language`` is an optional ISO code (e.g. ``"en"``) to skip auto-detection
    and speed up transcription. If None, faster-whisper auto-detects.
    """
    model = get_transcriber()
    segments, _info = model.transcribe(
        path,
        language=language,
        vad_filter=True,
        beam_size=5,
    )
    parts = [seg.text for seg in segments if seg.text and seg.text.strip()]
    return " ".join(parts).strip()
