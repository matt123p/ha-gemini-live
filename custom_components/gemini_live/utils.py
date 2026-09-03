"""Utility functions for audio processing."""

import logging
import struct

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is supplied by Home Assistant
    np = None


def set_detailed_logging(enabled: bool) -> None:
    """Set package logging verbosity for live model providers."""
    level = logging.DEBUG if enabled else logging.ERROR
    logging.getLogger("custom_components.gemini_live").setLevel(level)

def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
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


def streaming_wav_header(sample_rate: int = 16000) -> bytes:
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


def resample_24k_to_16k(data: bytes) -> bytes:
    """Resample raw 16-bit signed PCM mono audio from 24kHz to 16kHz."""
    if np is not None:
        return _resample_24k_to_16k_numpy(data)
    return _resample_24k_to_16k_pure(data)


def resample_16k_to_24k(data: bytes) -> bytes:
    """Resample raw 16-bit signed PCM mono audio from 16kHz to 24kHz."""
    num_samples = len(data) // 2
    if num_samples == 0:
        return b""
    samples = struct.unpack(f"<{num_samples}h", data[: num_samples * 2])
    output: list[int] = []
    for index in range(0, num_samples - 1, 2):
        first = samples[index]
        second = samples[index + 1]
        output.extend((first, (first + second) // 2, second))
    if num_samples % 2:
        output.append(samples[-1])
    return struct.pack(f"<{len(output)}h", *output)


def _resample_24k_to_16k_numpy(data: bytes) -> bytes:
    """Resample using NumPy while matching the pure-Python implementation."""
    num_samples = len(data) // 2
    if num_samples == 0:
        return b""

    samples = np.frombuffer(data[: num_samples * 2], dtype="<i2")
    triplets = num_samples // 3
    if triplets == 0:
        return data[:2]

    triples = samples[: triplets * 3].reshape(triplets, 3)
    output = np.empty(triplets * 2, dtype="<i2")
    output[0::2] = triples[:, 0]
    output[1::2] = (
        (triples[:, 1].astype(np.int32) + triples[:, 2]) // 2
    ).astype(np.int16)
    if num_samples > triplets * 3:
        output = np.concatenate(
            (output, samples[triplets * 3 : triplets * 3 + 1])
        )
    return output.tobytes()


def _resample_24k_to_16k_pure(data: bytes) -> bytes:
    """Resample using the dependency-free linear interpolation fallback."""
    num_samples = len(data) // 2
    if num_samples == 0:
        return b""

    samples = struct.unpack(f"<{num_samples}h", data[: num_samples * 2])
    output = []
    i = 0
    while i < num_samples - 2:
        output.append(samples[i])
        output.append((samples[i+1] + samples[i+2]) // 2)
        i += 3
    if i < num_samples:
        output.append(samples[i])

    return struct.pack(f"<{len(output)}h", *output)
