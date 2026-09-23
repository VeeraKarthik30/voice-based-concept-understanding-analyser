from __future__ import annotations

import math
import sys
import wave
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def sample_audio_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample.wav"
    sample_rate = 16_000
    duration = 2.0
    amplitude = 18_000
    frames = bytearray()

    for index in range(int(sample_rate * duration)):
        value = int(amplitude * math.sin(2 * math.pi * 220 * index / sample_rate))
        frames.extend(int(value).to_bytes(2, byteorder="little", signed=True))

    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))
    return file_path
