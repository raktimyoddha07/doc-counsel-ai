"""
Voice transcription via faster-whisper (CPU).

The browser records audio as WebM/Opus via the MediaRecorder API. faster-whisper
works best on 16kHz mono WAV, so we normalize the input with pydub (which needs
the ``ffmpeg`` binary on PATH) before handing it to the model.

The WhisperModel is loaded lazily on first use so the server still boots
instantly even before the model (~145MB for ``base``) is downloaded. Model size
and device are configurable via env vars so they can be changed without code
edits (e.g. upgrade to GPU later).
"""
import os
import tempfile
from typing import Optional

from pydub import AudioSegment

# Make pydub's ffmpeg probe robust on Windows where the binary may not be on PATH.
# If FFMPEG_BINARY is set, use it directly.
if os.getenv("FFMPEG_BINARY"):
    AudioSegment.converter = os.getenv("FFMPEG_BINARY")


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


def _normalize_to_wav(input_path: str) -> str:
    """
    Convert any audio the browser sends (webm/opus, mp3, wav, ogg) to a
    16kHz mono WAV file that faster-whisper handles well. Returns the temp path.
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    audio.export(wav_path, format="wav")
    return wav_path


def transcribe_audio_file(path: str, language: Optional[str] = None) -> str:
    """
    Transcribe the audio file at ``path`` and return the joined text.

    ``language`` is an optional ISO code (e.g. ``"en"``) to skip auto-detection
    and speed up transcription. If None, faster-whisper auto-detects.
    """
    model = get_transcriber()
    wav_path: Optional[str] = None
    try:
        wav_path = _normalize_to_wav(path)
        segments, _info = model.transcribe(
            wav_path,
            language=language,
            vad_filter=True,
            beam_size=5,
        )
        parts = [seg.text for seg in segments if seg.text and seg.text.strip()]
        return " ".join(parts).strip()
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass
