"""Utility functions for audio processing."""

import logging
import struct


def set_detailed_logging(enabled: bool) -> None:
    """Set package logging verbosity for Gemini Live."""
    level = logging.DEBUG if enabled else logging.ERROR
    logging.getLogger("custom_components.gemini_live").setLevel(level)

def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw 16-bit signed PCM mono audio in a WAV container."""
    num_channels = 1
    sample_width = 2  # 16-bit

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm_data),
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format code
        num_channels,
        sample_rate,
        sample_rate * num_channels * sample_width,
        num_channels * sample_width,
        sample_width * 8,
        b"data",
        len(pcm_data),
    )
    return header + pcm_data


def streaming_wav_header(sample_rate: int = 24000) -> bytes:
    """Return a WAV header whose data length is terminated by end-of-stream."""
    num_channels = 1
    sample_width = 2
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        sample_rate * num_channels * sample_width,
        num_channels * sample_width,
        sample_width * 8,
        b"data",
        0xFFFFFFFF,
    )
