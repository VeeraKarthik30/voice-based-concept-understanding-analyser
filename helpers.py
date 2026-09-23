from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


FILLER_PATTERN = re.compile(r"\b(um|uh|like|you know|actually|basically)\b", re.IGNORECASE)
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".ogg"}
MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024


@dataclass(slots=True)
class AnalysisPayload:
    user_id: int
    concept_name: str
    original_filename: str
    stored_audio_path: str
    transcription: str
    scores: dict[str, float]
    report_path: str


def current_timestamp(pattern: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.now().strftime(pattern)


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    return cleaned.strip("_").lower() or "audio"


def save_uploaded_file(uploaded_file: Any, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path(str(uploaded_file.name))
    extension = source_name.suffix.lower()
    filename = f"{current_timestamp('%Y%m%d_%H%M%S')}_{safe_slug(source_name.stem)}{extension}"
    target = destination_dir / filename
    with open(target, "wb") as file_obj:
        file_obj.write(uploaded_file.getbuffer())
    return target


def tokenize_keywords(value: str) -> list[str]:
    parts = [part.strip().lower() for part in re.split(r"[,;\n]", value) if part.strip()]
    return list(dict.fromkeys(parts))


def estimate_word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def detect_fillers(text: str) -> list[str]:
    return [match.group(0).lower() for match in FILLER_PATTERN.finditer(text)]


def default_improvement_suggestions(
    semantic_result: dict[str, Any],
    audio_result: dict[str, Any],
    transcript_text: str,
) -> list[str]:
    suggestions = []
    missing = semantic_result.get("missing_concepts", [])
    if missing:
        suggestions.append(f"Include these concepts more explicitly: {', '.join(missing[:5])}.")
    if audio_result.get("speech_rate_wpm", 0) > 170:
        suggestions.append("Reduce your speaking speed slightly to improve clarity and emphasis.")
    if audio_result.get("speech_rate_wpm", 0) < 110:
        suggestions.append("Increase speaking pace moderately so the answer sounds more confident and fluent.")
    if audio_result.get("pause_count", 0) > 4:
        suggestions.append("Practice speaking in short thought groups to reduce long or repeated pauses.")
    if audio_result.get("filler_count", 0) > 2:
        suggestions.append("Replace filler words with brief silent pauses to sound more composed.")
    if not suggestions:
        suggestions.append("Maintain the same balance of concept coverage, pacing, and low filler usage.")
    return suggestions


def coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def sanitize_html_text(value: Any) -> str:
    return html.escape(str(value or ""), quote=True).replace("\n", "<br>")


def format_items_for_display(items: list[str]) -> str:
    cleaned = [item.strip() for item in items if str(item).strip()]
    return ", ".join(sanitize_html_text(item) for item in cleaned)


def infer_audio_format(file_name: str, file_bytes: bytes) -> str | None:
    extension = Path(file_name).suffix.lower()
    header = file_bytes[:16]

    if header.startswith(b"RIFF") and b"WAVE" in header:
        return "wav"
    if header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
        return "mp3"
    if header.startswith(b"OggS"):
        return "ogg"
    if b"ftyp" in header and extension == ".m4a":
        return "m4a"
    if b"ftyp" in header and extension == ".mp4":
        return "mp4"

    return extension.lstrip(".") if extension in ALLOWED_AUDIO_EXTENSIONS else None


def validate_audio_upload(uploaded_file: Any, max_size_bytes: int = MAX_AUDIO_UPLOAD_BYTES) -> tuple[bool, str]:
    source_name = Path(str(getattr(uploaded_file, "name", "")))
    extension = source_name.suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        return False, "Unsupported file type. Please upload WAV, MP3, M4A, MP4, or OGG audio."

    file_bytes = bytes(uploaded_file.getbuffer())
    if not file_bytes:
        return False, "The uploaded audio file is empty."
    if len(file_bytes) > max_size_bytes:
        max_size_mb = max_size_bytes // (1024 * 1024)
        return False, f"Audio file is too large. Upload a file up to {max_size_mb} MB."

    detected_format = infer_audio_format(source_name.name, file_bytes)
    if detected_format is None:
        return False, "The uploaded file content does not match a supported audio format."
    if detected_format != extension.lstrip("."):
        return False, "The uploaded file extension does not match its audio content."

    return True, "Audio file validated successfully."
